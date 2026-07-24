from __future__ import annotations

import copy
import functools
import hashlib
import importlib.metadata
import inspect
import json
import platform
from dataclasses import dataclass, replace
from typing import Any, Callable, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from config.config import PHYSICAL_BOUNDS
from modules.ai_prediction import FeatureBuilder, ModelBundle


_TARGET_COLUMNS = {
    "wind_speed": {
        "physical": (
            "wind_speed_local",
            "wind_speed_terrain_corrected",
            "wind_speed_corrected",
        ),
        "truth": ("wind_speed_truth", "truth_wind_speed"),
    },
    "ambient_temp": {
        "physical": (
            "ambient_temp_local",
            "ambient_temp_terrain_corrected",
            "ambient_temp_corrected",
        ),
        "truth": ("ambient_temp_truth", "truth_ambient_temp"),
    },
}
_TRAINING_CONTRACT_VERSION = "residual-training-v1"
_TRAINING_LINEAGE_COLUMNS = (
    "source_file_hash",
    "source_file_hash_physical",
    "source_file_hash_truth",
    "dataset_id",
    "dataset_role",
)
_DEFAULT_ESTIMATOR_PARAMETERS = (
    ("objective", "reg:squarederror"),
    ("n_estimators", 120),
    ("max_depth", 3),
    ("learning_rate", 0.05),
    ("subsample", 0.9),
    ("colsample_bytree", 0.9),
    ("random_state", 42),
    ("n_jobs", 1),
)


class ConstantResidualEstimator:
    def __init__(self, value: float):
        self.value = float(value)

    def fit(self, features, target):
        return self

    def predict(self, features):
        return np.full(len(features), self.value, dtype=float)


@dataclass(frozen=True)
class TrainingResult:
    target: str
    line_id: str
    tower_id: str
    bundle: ModelBundle
    metrics: dict[str, float]
    metadata: dict
    training_contract_hash: str

    def __post_init__(self) -> None:
        if (
            not isinstance(self.training_contract_hash, str)
            or not self.training_contract_hash.strip()
        ):
            raise ValueError("training_contract_hash must be a non-empty string")
        if not isinstance(self.metadata, Mapping):
            raise ValueError("training result metadata must be a mapping")
        if not isinstance(self.bundle.metadata, Mapping):
            raise ValueError("training bundle metadata must be a mapping")
        if (
            self.metadata.get("training_contract_hash")
            != self.training_contract_hash
            or self.bundle.metadata.get("training_contract_hash")
            != self.training_contract_hash
        ):
            raise ValueError("training result contract fields must match")


@dataclass(frozen=True)
class TrainingPreparation:
    target: str
    line_id: str
    tower_id: str
    physical_col: str
    truth_col: str
    working: pd.DataFrame
    physical: np.ndarray
    truth: np.ndarray
    residual: np.ndarray
    feature_frame: pd.DataFrame
    feature_columns: tuple[str, ...]
    model_features: pd.DataFrame
    split: Optional[tuple[np.ndarray, np.ndarray]]
    input_data_hash: str
    evaluation_mode: str
    evaluation_set_hash: Optional[str]
    cadence_minutes: float
    training_contract_hash: str
    snapshot_hash: str


@dataclass(frozen=True)
class TrainingContract:
    version: str
    trainer_descriptor_json: str
    estimator_descriptor_json: str
    dependency_versions: tuple[tuple[str, str], ...]
    random_seed: int

    def scoped_hash(
        self,
        *,
        target: str,
        physical_col: str,
        truth_col: str,
        feature_columns: Sequence[str],
        cadence_minutes: float,
    ) -> str:
        return _stable_json_hash(
            {
                "version": self.version,
                "trainer": json.loads(self.trainer_descriptor_json),
                "estimator": json.loads(self.estimator_descriptor_json),
                "dependencies": dict(self.dependency_versions),
                "random_seed": self.random_seed,
                "target": target,
                "physical_col": physical_col,
                "truth_col": truth_col,
                "feature_columns": list(feature_columns),
                "cadence_minutes": float(cadence_minutes),
            }
        )


def _load_xgb_regressor():
    from xgboost import XGBRegressor

    return XGBRegressor


def default_estimator():
    estimator_class = _load_xgb_regressor()
    return estimator_class(**dict(_DEFAULT_ESTIMATOR_PARAMETERS))


def _metric_values(
    physical: np.ndarray,
    truth: np.ndarray,
    corrected: np.ndarray,
) -> dict[str, float]:
    baseline_error = physical - truth
    corrected_error = corrected - truth
    return {
        "baseline_mae": float(np.mean(np.abs(baseline_error))),
        "baseline_rmse": float(np.sqrt(np.mean(np.square(baseline_error)))),
        "corrected_mae": float(np.mean(np.abs(corrected_error))),
        "corrected_rmse": float(np.sqrt(np.mean(np.square(corrected_error)))),
    }


def _robust_residual_bounds(residual: np.ndarray) -> tuple[float, float]:
    values = np.asarray(residual, dtype=float)
    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError("residual history must contain finite values")
    if values.size == 1:
        value = float(values[0])
        return value, value

    quantile_lower, quantile_upper = np.quantile(values, [0.01, 0.99])
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    if mad > 0.0:
        scale = 1.4826 * mad
        lower = max(float(quantile_lower), median - 5.0 * scale)
        upper = min(float(quantile_upper), median + 5.0 * scale)
    else:
        lower = float(quantile_lower)
        upper = float(quantile_upper)
    if lower > upper:
        return median, median
    return lower, upper


def _stable_training_data_hash(
    frame: pd.DataFrame,
    columns: Sequence[str],
    *,
    target: str,
) -> str:
    selected_columns = list(dict.fromkeys(columns))
    canonical = frame.loc[:, selected_columns].copy(deep=True)
    row_hashes = np.sort(
        pd.util.hash_pandas_object(
            canonical,
            index=False,
            categorize=False,
        ).to_numpy(dtype=np.uint64)
    )
    digest = hashlib.sha256()
    digest.update(
        json.dumps(
            {"target": target, "columns": selected_columns},
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )
    digest.update(row_hashes.tobytes())
    return digest.hexdigest()


def _ordered_training_snapshot_hash(
    frame: pd.DataFrame,
    columns: Sequence[str],
    *,
    target: str,
) -> str:
    selected_columns = list(dict.fromkeys(columns))
    ordered = frame.loc[:, selected_columns].copy(deep=True)
    row_hashes = pd.util.hash_pandas_object(
        ordered,
        index=False,
        categorize=False,
    ).to_numpy(dtype=np.uint64)
    digest = hashlib.sha256()
    digest.update(
        _canonical_json(
            {"target": target, "columns": selected_columns, "ordered": True}
        ).encode("utf-8")
    )
    digest.update(row_hashes.tobytes())
    return digest.hexdigest()


def _canonical_json(payload: Any) -> str:
    return json.dumps(
        payload,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _stable_json_hash(payload: Any) -> str:
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _qualified_name(value: Any) -> str:
    module = getattr(value, "__module__", type(value).__module__)
    qualname = getattr(value, "__qualname__", type(value).__qualname__)
    return f"{module}.{qualname}"


def _contract_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if np.isfinite(value) else str(value)
    if isinstance(value, np.generic):
        return _contract_value(value.item())
    if isinstance(value, (list, tuple)):
        return [_contract_value(item) for item in value]
    if isinstance(value, (set, frozenset)):
        normalized = [_contract_value(item) for item in value]
        return sorted(normalized, key=_canonical_json)
    if isinstance(value, Mapping):
        return {
            str(key): _contract_value(item)
            for key, item in sorted(value.items(), key=lambda item: str(item[0]))
        }
    return {"type": _qualified_name(type(value))}


def _implementation_hash(value: Any) -> str:
    target = getattr(value, "__func__", value)
    try:
        source = inspect.getsource(target)
    except (OSError, TypeError):
        source = _qualified_name(target)
    return hashlib.sha256(source.encode("utf-8")).hexdigest()


def _callable_contract(value: Any) -> dict[str, Any]:
    if isinstance(value, functools.partial):
        return {
            "kind": "partial",
            "callable": _callable_contract(value.func),
            "args": _contract_value(tuple(value.args)),
            "keywords": _contract_value(value.keywords or {}),
        }
    descriptor = {
        "kind": "class" if inspect.isclass(value) else "callable",
        "identity": _qualified_name(value),
        "implementation_hash": _implementation_hash(value),
    }
    defaults = getattr(value, "__defaults__", None)
    keyword_defaults = getattr(value, "__kwdefaults__", None)
    if defaults:
        descriptor["defaults"] = _contract_value(tuple(defaults))
    if keyword_defaults:
        descriptor["keyword_defaults"] = _contract_value(keyword_defaults)
    closure = getattr(value, "__closure__", None)
    if closure:
        descriptor["closure"] = [
            _contract_value(cell.cell_contents) for cell in closure
        ]
    if value is default_estimator:
        descriptor["declared_parameters"] = dict(
            _DEFAULT_ESTIMATOR_PARAMETERS
        )
    return descriptor


def training_runtime_contract_hash(
    trainer: Any,
    preparation: TrainingPreparation,
) -> str:
    if not isinstance(preparation, TrainingPreparation):
        raise TypeError("preparation must be a TrainingPreparation")
    trainer_type = type(trainer)
    methods = {}
    for name in ("prepare_target", "train_prepared", "train_target"):
        method = getattr(trainer_type, name, None)
        if method is not None:
            methods[name] = _callable_contract(method)
    return _stable_json_hash(
        {
            "preparation_contract_hash": preparation.training_contract_hash,
            "trainer_type": _qualified_name(trainer_type),
            "methods": methods,
        }
    )


def bind_training_result_contract(
    result: TrainingResult,
    training_contract_hash: str,
) -> TrainingResult:
    if not isinstance(result, TrainingResult):
        raise TypeError("result must be a TrainingResult")
    if (
        not isinstance(training_contract_hash, str)
        or not training_contract_hash.strip()
    ):
        raise ValueError("training_contract_hash must be a non-empty string")
    metadata = {
        **dict(result.metadata),
        "training_contract_hash": training_contract_hash,
    }
    bundle = replace(
        result.bundle,
        metadata={
            **dict(result.bundle.metadata),
            "training_contract_hash": training_contract_hash,
        },
    )
    return replace(
        result,
        bundle=bundle,
        metadata=metadata,
        training_contract_hash=training_contract_hash,
    )


def _json_safe_training_value(value):
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if np.isfinite(value) else str(value)
    if isinstance(value, np.generic):
        return _json_safe_training_value(value.item())
    if isinstance(value, dict):
        return {
            str(key): _json_safe_training_value(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_json_safe_training_value(item) for item in value]
    return repr(value)


def _estimator_training_params(estimator: object) -> dict:
    estimator_class = type(estimator)
    metadata = {
        "estimator_class": (
            f"{estimator_class.__module__}.{estimator_class.__qualname__}"
        )
    }
    get_params = getattr(estimator, "get_params", None)
    if callable(get_params):
        try:
            metadata["parameters"] = _json_safe_training_value(
                get_params(deep=False)
            )
        except Exception as exc:
            metadata["parameter_read_error"] = type(exc).__name__
    return metadata


def _dependency_versions() -> dict[str, str]:
    versions = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
    }
    for distribution in ("joblib", "xgboost"):
        try:
            versions[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            versions[distribution] = "unavailable"
    return versions


class ResidualTrainer:
    """Train one weather residual model per line, tower, and target."""

    def __init__(
        self,
        estimator_factory: Optional[Callable[[], object]] = None,
        feature_builder: Optional[FeatureBuilder] = None,
    ):
        self.estimator_factory = estimator_factory or default_estimator
        self.feature_builder = feature_builder or FeatureBuilder()
        dependencies = _dependency_versions()
        trainer_type = type(self)
        trainer_descriptor = {
            "type": _qualified_name(trainer_type),
            "prepare_target": _callable_contract(trainer_type.prepare_target),
            "train_prepared": _callable_contract(trainer_type.train_prepared),
        }
        self.training_contract = TrainingContract(
            version=_TRAINING_CONTRACT_VERSION,
            trainer_descriptor_json=_canonical_json(trainer_descriptor),
            estimator_descriptor_json=_canonical_json(
                _callable_contract(self.estimator_factory)
            ),
            dependency_versions=tuple(sorted(dependencies.items())),
            random_seed=42,
        )

    @staticmethod
    def _target_columns(
        frame: pd.DataFrame,
        target: str,
        physical_col: Optional[str],
        truth_col: Optional[str],
    ) -> tuple[str, str]:
        if target not in _TARGET_COLUMNS:
            raise ValueError(
                "target must be wind_speed or ambient_temp residual weather"
            )
        allowed_physical_columns = _TARGET_COLUMNS[target]["physical"]
        if (
            physical_col is not None
            and physical_col not in allowed_physical_columns
        ):
            raise ValueError(
                "physical_col must name terrain-corrected weather"
            )
        allowed_truth_columns = _TARGET_COLUMNS[target]["truth"]
        if truth_col is not None and truth_col not in allowed_truth_columns:
            raise ValueError(
                "truth_col must name truth weather for the requested target"
            )
        if physical_col is None:
            physical_col = next(
                (
                    column
                    for column in allowed_physical_columns
                    if column in frame.columns
                ),
                None,
            )
        if truth_col is None:
            truth_col = next(
                (
                    column
                    for column in allowed_truth_columns
                    if column in frame.columns
                ),
                None,
            )
        if physical_col is None:
            raise ValueError(
                f"missing terrain-corrected physical column for {target}"
            )
        if truth_col is None:
            raise ValueError(f"missing truth column for {target}")
        if physical_col not in frame.columns or truth_col not in frame.columns:
            raise ValueError("configured physical or truth column is missing")
        if physical_col == truth_col:
            raise ValueError("physical and truth columns must be distinct")
        return physical_col, truth_col

    @staticmethod
    def _validated_scope(frame: pd.DataFrame) -> tuple[pd.DataFrame, str, str]:
        if not isinstance(frame, pd.DataFrame):
            raise TypeError("frame must be a pandas DataFrame")
        if frame.empty:
            raise ValueError("training frame cannot be empty")
        working = frame.copy(deep=True)
        for column in ("line_id", "tower_id"):
            if column not in working.columns:
                raise ValueError(f"missing required column: {column}")
            if working[column].isna().any():
                raise ValueError(f"{column} cannot contain missing values")
            values = working[column].map(lambda value: str(value).strip())
            if values.eq("").any():
                raise ValueError(f"{column} cannot contain empty values")
            working[column] = values
            if values.nunique(dropna=False) != 1:
                raise ValueError(
                    "train_target accepts a single line and single tower only"
                )
        return working, working["line_id"].iloc[0], working["tower_id"].iloc[0]

    def _canonicalize_working(
        self,
        frame: pd.DataFrame,
        *,
        physical_col: str,
        truth_col: str,
    ) -> pd.DataFrame:
        working = frame.copy(deep=True)
        original_attrs = copy.deepcopy(frame.attrs)
        working["timestamp"] = self.feature_builder._timestamps(working)
        for column in _TRAINING_LINEAGE_COLUMNS:
            if column not in working.columns:
                continue
            if working[column].isna().any():
                raise ValueError(f"{column} cannot contain missing values")
            working[column] = working[column].map(
                lambda value: str(value).strip()
            )
        try:
            working["__canonical_row_hash__"] = (
                pd.util.hash_pandas_object(
                    working.loc[
                        :,
                        [
                            "line_id",
                            "tower_id",
                            *(
                                column
                                for column in _TRAINING_LINEAGE_COLUMNS
                                if column in working.columns
                            ),
                            "timestamp",
                            physical_col,
                            truth_col,
                        ],
                    ],
                    index=False,
                    categorize=False,
                ).to_numpy(dtype=np.uint64)
            )
        except (TypeError, ValueError) as exc:
            raise ValueError("training rows must contain stable scalar values") from exc
        sort_columns = [
            "line_id",
            "tower_id",
            *(
                column
                for column in _TRAINING_LINEAGE_COLUMNS
                if column in working.columns
            ),
            "timestamp",
            "__canonical_row_hash__",
        ]
        working = (
            working.sort_values(sort_columns, kind="mergesort")
            .drop(columns="__canonical_row_hash__")
            .reset_index(drop=True)
        )
        working.attrs = original_attrs
        return working

    @staticmethod
    def _finite_values(frame: pd.DataFrame, column: str) -> np.ndarray:
        values = pd.to_numeric(frame[column], errors="coerce").to_numpy(
            dtype=float
        )
        if not np.isfinite(values).all():
            raise ValueError(f"{column} must contain finite values")
        return values

    @staticmethod
    def _fallback_estimator(target: np.ndarray) -> ConstantResidualEstimator:
        return ConstantResidualEstimator(float(np.median(target)))

    def _fit_estimator(
        self,
        features: pd.DataFrame,
        residual: np.ndarray,
    ) -> tuple[object, str]:
        if np.allclose(residual, residual[0], rtol=0.0, atol=1e-12):
            return self._fallback_estimator(residual), "constant_residual"
        try:
            estimator = self.estimator_factory()
        except ImportError:
            return self._fallback_estimator(residual), "xgboost_unavailable"
        except Exception as exc:
            return (
                self._fallback_estimator(residual),
                f"estimator_factory_failed:{type(exc).__name__}",
            )
        if not hasattr(estimator, "fit") or not hasattr(estimator, "predict"):
            return self._fallback_estimator(residual), "invalid_estimator"
        try:
            estimator.fit(features, residual)
        except Exception as exc:
            return (
                self._fallback_estimator(residual),
                f"estimator_fit_failed:{type(exc).__name__}",
            )
        try:
            fitted_prediction = np.asarray(
                estimator.predict(features), dtype=float
            ).reshape(-1)
        except Exception as exc:
            return (
                self._fallback_estimator(residual),
                f"estimator_prediction_failed:{type(exc).__name__}",
            )
        if (
            len(fitted_prediction) != len(features)
            or not np.isfinite(fitted_prediction).all()
        ):
            return self._fallback_estimator(residual), "non_finite_prediction"
        return estimator, ""

    @staticmethod
    def _predict_corrected(
        estimator: object,
        features: pd.DataFrame,
        physical: np.ndarray,
        target: str,
        residual_bounds: tuple[float, float],
    ) -> tuple[np.ndarray, str]:
        fallback_reason = ""
        try:
            prediction = np.asarray(
                estimator.predict(features), dtype=float
            ).reshape(-1)
        except Exception as exc:
            prediction = np.full(len(features), np.nan, dtype=float)
            fallback_reason = f"prediction_failed:{type(exc).__name__}"
        if len(prediction) != len(features):
            prediction = np.full(len(features), np.nan, dtype=float)
            fallback_reason = "unexpected_prediction_length"
        finite = np.isfinite(prediction)
        if not finite.all() and not fallback_reason:
            fallback_reason = "non_finite_prediction"
        residual = np.zeros(len(features), dtype=float)
        residual[finite] = np.clip(
            prediction[finite], residual_bounds[0], residual_bounds[1]
        )
        with np.errstate(over="ignore", invalid="ignore"):
            corrected = physical + residual
        valid = finite & np.isfinite(corrected)
        lower, upper = PHYSICAL_BOUNDS[target]
        bounds_exceeded = valid & (
            (corrected < lower) | (corrected > upper)
        )
        valid &= ~bounds_exceeded
        corrected[~valid] = physical[~valid]
        if bounds_exceeded.any():
            if fallback_reason:
                fallback_reason = (
                    f"{fallback_reason};physical_bounds_exceeded"
                )
            else:
                fallback_reason = "physical_bounds_exceeded"
        return corrected, fallback_reason

    @staticmethod
    def _time_split(
        frame: pd.DataFrame,
        segments: pd.Series,
    ) -> Optional[tuple[np.ndarray, np.ndarray]]:
        segment_values = np.asarray(segments, dtype=np.int64)
        unique_segments = np.unique(segment_values)
        if unique_segments.size < 2:
            return None
        timestamps = pd.to_datetime(frame["timestamp"], errors="coerce")
        starts = {
            segment: timestamps.iloc[segment_values == segment].min()
            for segment in unique_segments
        }
        holdout_segment = max(
            unique_segments,
            key=lambda segment: (starts[segment], int(segment)),
        )
        holdout_mask = segment_values == holdout_segment
        holdout_start = timestamps.iloc[holdout_mask].min()
        train_mask = (~holdout_mask) & (timestamps < holdout_start).to_numpy()
        if not train_mask.any() or not holdout_mask.any():
            return None
        return np.flatnonzero(train_mask), np.flatnonzero(holdout_mask)

    def prepare_target(
        self,
        frame: pd.DataFrame,
        target: str,
        *,
        physical_col: Optional[str] = None,
        truth_col: Optional[str] = None,
    ) -> TrainingPreparation:
        working, line_id, tower_id = self._validated_scope(frame)
        physical_col, truth_col = self._target_columns(
            working, target, physical_col, truth_col
        )
        working = self._canonicalize_working(
            working,
            physical_col=physical_col,
            truth_col=truth_col,
        )
        physical = self._finite_values(working, physical_col).copy()
        truth = self._finite_values(working, truth_col).copy()
        residual = (truth - physical).copy()
        if not np.isfinite(residual).all():
            raise ValueError("weather residual must contain finite values")

        feature_frame = self.feature_builder.transform(
            working, physical_col=physical_col
        ).copy(deep=True)
        feature_columns = tuple(
            self.feature_builder.feature_columns(physical_col)
        )
        model_features = feature_frame.loc[:, list(feature_columns)].copy(deep=True)
        if not np.isfinite(model_features.to_numpy(dtype=float)).all():
            raise ValueError("model features must contain finite values")
        segments = self.feature_builder.continuous_segments(working)
        split = self._time_split(feature_frame, segments)
        if split is not None:
            split = tuple(np.array(indices, dtype=np.int64, copy=True) for indices in split)
        lineage_columns = [
            column
            for column in _TRAINING_LINEAGE_COLUMNS
            if column in feature_frame.columns
        ]
        hash_columns = (
            "line_id",
            "tower_id",
            "timestamp",
            *lineage_columns,
            *feature_columns,
            truth_col,
        )
        input_data_hash = _stable_training_data_hash(
            feature_frame,
            hash_columns,
            target=target,
        )
        evaluation_mode = "full_fit" if split is None else "temporal_holdout"
        evaluation_set_hash = None
        if split is not None:
            _, holdout_positions = split
            evaluation_set_hash = _stable_training_data_hash(
                feature_frame.iloc[holdout_positions],
                hash_columns,
                target=target,
            )
        training_contract_hash = self.training_contract.scoped_hash(
            target=target,
            physical_col=physical_col,
            truth_col=truth_col,
            feature_columns=feature_columns,
            cadence_minutes=self.feature_builder.cadence_minutes,
        )
        snapshot_hash = _ordered_training_snapshot_hash(
            feature_frame,
            hash_columns,
            target=target,
        )
        return TrainingPreparation(
            target=target,
            line_id=line_id,
            tower_id=tower_id,
            physical_col=physical_col,
            truth_col=truth_col,
            working=working,
            physical=physical,
            truth=truth,
            residual=residual,
            feature_frame=feature_frame,
            feature_columns=feature_columns,
            model_features=model_features,
            split=split,
            input_data_hash=input_data_hash,
            evaluation_mode=evaluation_mode,
            evaluation_set_hash=evaluation_set_hash,
            cadence_minutes=self.feature_builder.cadence_minutes,
            training_contract_hash=training_contract_hash,
            snapshot_hash=snapshot_hash,
        )

    @staticmethod
    def _validate_preparation_split(
        split: Optional[tuple[np.ndarray, np.ndarray]],
        row_count: int,
    ) -> None:
        if split is None:
            return
        if not isinstance(split, tuple) or len(split) != 2:
            raise ValueError("preparation split must contain train and holdout indices")
        normalized = []
        for name, positions in zip(("train", "holdout"), split):
            values = np.asarray(positions)
            if values.ndim != 1 or values.dtype.kind not in "iu":
                raise ValueError(f"preparation {name} split indices must be integers")
            if values.size == 0:
                raise ValueError(f"preparation {name} split cannot be empty")
            if (values < 0).any() or (values >= row_count).any():
                raise ValueError(f"preparation {name} split indices are out of bounds")
            if np.unique(values).size != values.size:
                raise ValueError(f"preparation {name} split indices must be unique")
            normalized.append(values.astype(np.int64, copy=False))
        train_positions, holdout_positions = normalized
        if np.intersect1d(train_positions, holdout_positions).size:
            raise ValueError("preparation split indices must be mutually exclusive")

    @staticmethod
    def _assert_preparation_frame(
        actual: pd.DataFrame,
        expected: pd.DataFrame,
        name: str,
    ) -> None:
        if not isinstance(actual, pd.DataFrame):
            raise ValueError(f"preparation {name} must be a DataFrame")
        try:
            pd.testing.assert_frame_equal(
                actual,
                expected,
                check_dtype=True,
                check_exact=True,
                check_like=False,
            )
        except AssertionError as exc:
            raise ValueError(f"preparation {name} failed integrity validation") from exc

    @staticmethod
    def _assert_preparation_array(
        actual: np.ndarray,
        expected: np.ndarray,
        name: str,
    ) -> None:
        if not isinstance(actual, np.ndarray):
            raise ValueError(f"preparation {name} must be an ndarray")
        if actual.dtype != expected.dtype or not np.array_equal(actual, expected):
            raise ValueError(f"preparation {name} failed integrity validation")

    def _validated_preparation(
        self,
        preparation: TrainingPreparation,
    ) -> TrainingPreparation:
        if not isinstance(preparation, TrainingPreparation):
            raise TypeError("preparation must be a TrainingPreparation")
        self._validate_preparation_split(preparation.split, len(preparation.working))
        if not np.isclose(
            preparation.cadence_minutes,
            self.feature_builder.cadence_minutes,
            rtol=0.0,
            atol=1e-12,
        ):
            raise ValueError("preparation cadence does not match trainer")

        trusted = self.prepare_target(
            preparation.working,
            preparation.target,
            physical_col=preparation.physical_col,
            truth_col=preparation.truth_col,
        )
        scalar_fields = (
            "target",
            "line_id",
            "tower_id",
            "physical_col",
            "truth_col",
            "feature_columns",
            "input_data_hash",
            "evaluation_mode",
            "evaluation_set_hash",
            "cadence_minutes",
            "training_contract_hash",
            "snapshot_hash",
        )
        for field in scalar_fields:
            if getattr(preparation, field) != getattr(trusted, field):
                raise ValueError(f"preparation {field} failed integrity validation")

        self._assert_preparation_frame(
            preparation.working,
            trusted.working,
            "working",
        )
        self._assert_preparation_frame(
            preparation.feature_frame,
            trusted.feature_frame,
            "feature_frame",
        )
        self._assert_preparation_frame(
            preparation.model_features,
            trusted.model_features,
            "model_features",
        )
        for field in ("physical", "truth", "residual"):
            self._assert_preparation_array(
                getattr(preparation, field),
                getattr(trusted, field),
                field,
            )

        if (preparation.split is None) != (trusted.split is None):
            raise ValueError("preparation split failed integrity validation")
        if preparation.split is not None and trusted.split is not None:
            for actual, expected in zip(preparation.split, trusted.split):
                if not np.array_equal(actual, expected):
                    raise ValueError("preparation split failed integrity validation")
        return trusted

    def train_prepared(
        self,
        preparation: TrainingPreparation,
    ) -> TrainingResult:
        preparation = self._validated_preparation(preparation)

        target = preparation.target
        line_id = preparation.line_id
        tower_id = preparation.tower_id
        physical_col = preparation.physical_col
        truth_col = preparation.truth_col
        working = preparation.working
        physical = preparation.physical
        truth = preparation.truth
        residual = preparation.residual
        feature_frame = preparation.feature_frame
        feature_columns = preparation.feature_columns
        model_features = preparation.model_features
        split = preparation.split
        input_data_hash = preparation.input_data_hash

        if split is None:
            evaluation_mode = preparation.evaluation_mode
            independent_evaluation = False
            final_estimator, final_reason = self._fit_estimator(
                model_features, residual
            )
            final_bounds = _robust_residual_bounds(residual)
            corrected, prediction_reason = self._predict_corrected(
                final_estimator,
                model_features,
                physical,
                target,
                final_bounds,
            )
            if prediction_reason:
                final_reason = prediction_reason
            full_fit_metrics = _metric_values(physical, truth, corrected)
            metrics = {}
            evaluation_set_hash = preparation.evaluation_set_hash
            evaluation_reason = final_reason
        else:
            evaluation_mode = preparation.evaluation_mode
            independent_evaluation = True
            train_positions, holdout_positions = split
            evaluation_estimator, evaluation_reason = self._fit_estimator(
                model_features.iloc[train_positions], residual[train_positions]
            )
            evaluation_bounds = _robust_residual_bounds(
                residual[train_positions]
            )
            corrected, prediction_reason = self._predict_corrected(
                evaluation_estimator,
                model_features.iloc[holdout_positions],
                physical[holdout_positions],
                target,
                evaluation_bounds,
            )
            if prediction_reason:
                evaluation_reason = prediction_reason
            metrics = _metric_values(
                physical[holdout_positions],
                truth[holdout_positions],
                corrected,
            )
            evaluation_set_hash = preparation.evaluation_set_hash
            final_estimator, final_reason = self._fit_estimator(
                model_features, residual
            )
            final_bounds = _robust_residual_bounds(residual)
            full_fit_metrics = None

        timestamp_values = feature_frame["timestamp"]
        metadata = {
            "line_id": line_id,
            "tower_id": tower_id,
            "target": target,
            "residual_name": f"{target}_residual",
            "physical_col": physical_col,
            "truth_col": truth_col,
            "sample_count": len(working),
            "time_start": timestamp_values.min().isoformat(),
            "time_end": timestamp_values.max().isoformat(),
            "evaluation_mode": evaluation_mode,
            "independent_evaluation": independent_evaluation,
            "metric_domain": "weather_vs_truth",
            "input_data_hash": input_data_hash,
            "evaluation_set_hash": evaluation_set_hash,
            "feature_columns": list(feature_columns),
            "cadence_minutes": self.feature_builder.cadence_minutes,
            "training_contract_hash": preparation.training_contract_hash,
            "training_snapshot_hash": preparation.snapshot_hash,
            "random_state": 42,
            "training_params": _estimator_training_params(final_estimator),
            "dependency_versions": dict(
                self.training_contract.dependency_versions
            ),
            "fallback_reason": final_reason,
            "evaluation_fallback_reason": evaluation_reason,
            "residual_bounds": final_bounds,
        }
        if full_fit_metrics is not None:
            metadata["full_fit_metrics"] = full_fit_metrics

        bundle = ModelBundle(
            target_name=target,
            feature_columns=list(feature_columns),
            model=final_estimator,
            cadence_minutes=self.feature_builder.cadence_minutes,
            residual_bounds=final_bounds,
            line_id=line_id,
            tower_id=tower_id,
            metadata=metadata.copy(),
        )
        return TrainingResult(
            target=target,
            line_id=line_id,
            tower_id=tower_id,
            bundle=bundle,
            metrics=metrics,
            metadata=metadata,
            training_contract_hash=preparation.training_contract_hash,
        )

    def train_target(
        self,
        frame: pd.DataFrame,
        target: str,
        *,
        physical_col: Optional[str] = None,
        truth_col: Optional[str] = None,
    ) -> TrainingResult:
        preparation = self.prepare_target(
            frame,
            target,
            physical_col=physical_col,
            truth_col=truth_col,
        )
        return self.train_prepared(preparation)

    def train_many(
        self,
        frame: pd.DataFrame,
        targets: Sequence[str] = ("wind_speed", "ambient_temp"),
    ) -> dict[tuple[str, str, str], TrainingResult]:
        if not isinstance(frame, pd.DataFrame):
            raise TypeError("frame must be a pandas DataFrame")
        if frame.empty:
            return {}
        for column in ("line_id", "tower_id"):
            if column not in frame.columns:
                raise ValueError(f"missing required column: {column}")
            if frame[column].isna().any():
                raise ValueError(f"{column} cannot contain missing values")
        working = frame.copy(deep=True)
        working["line_id"] = working["line_id"].map(
            lambda value: str(value).strip()
        )
        working["tower_id"] = working["tower_id"].map(
            lambda value: str(value).strip()
        )
        if working["line_id"].eq("").any() or working["tower_id"].eq("").any():
            raise ValueError("line_id and tower_id cannot contain empty values")

        results = {}
        grouped = working.groupby(
            ["line_id", "tower_id"], sort=True, dropna=False
        )
        for (line_id, tower_id), tower_frame in grouped:
            for target in targets:
                result = self.train_target(tower_frame, target)
                results[(str(line_id), str(tower_id), target)] = result
        return results
