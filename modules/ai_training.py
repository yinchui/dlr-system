from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Sequence

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


def _load_xgb_regressor():
    from xgboost import XGBRegressor

    return XGBRegressor


def default_estimator():
    estimator_class = _load_xgb_regressor()
    return estimator_class(
        objective="reg:squarederror",
        n_estimators=120,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42,
        n_jobs=1,
    )


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


class ResidualTrainer:
    """Train one weather residual model per line, tower, and target."""

    def __init__(
        self,
        estimator_factory: Optional[Callable[[], object]] = None,
        feature_builder: Optional[FeatureBuilder] = None,
    ):
        self.estimator_factory = estimator_factory or default_estimator
        self.feature_builder = feature_builder or FeatureBuilder()

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
        corrected[~valid] = physical[~valid]
        lower, upper = PHYSICAL_BOUNDS[target]
        corrected = np.clip(corrected, lower, upper)
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

    def train_target(
        self,
        frame: pd.DataFrame,
        target: str,
        *,
        physical_col: Optional[str] = None,
        truth_col: Optional[str] = None,
    ) -> TrainingResult:
        working, line_id, tower_id = self._validated_scope(frame)
        physical_col, truth_col = self._target_columns(
            working, target, physical_col, truth_col
        )
        physical = self._finite_values(working, physical_col)
        truth = self._finite_values(working, truth_col)
        residual = truth - physical
        if not np.isfinite(residual).all():
            raise ValueError("weather residual must contain finite values")

        feature_frame = self.feature_builder.transform(
            working, physical_col=physical_col
        )
        feature_columns = self.feature_builder.feature_columns(physical_col)
        model_features = feature_frame.loc[:, feature_columns]
        if not np.isfinite(model_features.to_numpy(dtype=float)).all():
            raise ValueError("model features must contain finite values")
        segments = self.feature_builder.continuous_segments(working)
        split = self._time_split(feature_frame, segments)

        if split is None:
            evaluation_mode = "full_fit"
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
            evaluation_reason = final_reason
        else:
            evaluation_mode = "temporal_holdout"
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
            "feature_columns": feature_columns.copy(),
            "cadence_minutes": self.feature_builder.cadence_minutes,
            "random_state": 42,
            "fallback_reason": final_reason,
            "evaluation_fallback_reason": evaluation_reason,
            "residual_bounds": final_bounds,
        }
        if full_fit_metrics is not None:
            metadata["full_fit_metrics"] = full_fit_metrics

        bundle = ModelBundle(
            target_name=target,
            feature_columns=feature_columns.copy(),
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
        )

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
