from __future__ import annotations

import copy
import dis
import functools
import hashlib
import importlib.metadata
import inspect
import json
import platform
import re
import sys
import types
from dataclasses import asdict, dataclass, fields, is_dataclass, replace
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from config.config import PHYSICAL_BOUNDS
from modules.ai_prediction import (
    FeatureBuilder,
    ModelBundle,
    missing_value_collision_rows,
)


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
    ("missing", -1.0e30),
)
_SEALED_XGBOOST_ESTIMATOR_PATH = "xgboost.XGBRegressor"
_SEALED_XGBOOST_IMPLEMENTATION_MODULE = "xgboost.sklearn"
_SEALED_XGBOOST_ESTIMATOR_NAME = "XGBRegressor"
SEALED_XGBOOST_BACKEND_ID = "xgboost-residual-v1"
_NON_PRODUCTION_BACKEND_ID = "non-production-training-backend-v0"
_MAX_CONTRACT_DEPTH = 12
_MAX_CONTRACT_ITEMS = 256
_MAX_CONTRACT_NODES = 2_048
_EXTERNAL_CONTRACT_ROOTS = frozenset(
    {"joblib", "numpy", "pandas", "scipy", "sklearn", "xgboost"}
)
_DYNAMIC_GLOBAL_ACCESS_NAMES = frozenset(
    {"__import__", "eval", "exec", "globals", "locals"}
)
_TRAINING_OUTCOMES = frozenset(
    {"trained", "data_fallback", "operational_fallback"}
)
_OPERATIONAL_FALLBACK_REASONS = frozenset(
    {
        "invalid_estimator",
        "non_finite_prediction",
        "unexpected_prediction_length",
        "xgboost_unavailable",
    }
)
_OPERATIONAL_FALLBACK_PREFIXES = (
    "estimator_factory_failed:",
    "estimator_fit_failed:",
    "estimator_prediction_failed:",
    "prediction_failed:",
)
_MAX_TRAINING_METADATA_DEPTH = 6
_MAX_TRAINING_METADATA_ITEMS = 64
_MAX_TRAINING_METADATA_STRING = 256
_MAX_TRAINING_METADATA_NODES = 512
_MAX_TRAINING_METADATA_INTEGER_BITS = 1_024
_SENSITIVE_PARAMETER_MARKERS = frozenset(
    {
        "accesskey",
        "apikey",
        "credential",
        "password",
        "passwd",
        "privatekey",
        "secret",
        "token",
    }
)


class TrainingContractError(RuntimeError):
    """Raised when executable training configuration cannot be frozen safely."""


@dataclass(frozen=True)
class SealedEstimatorSpec:
    schema_version: int
    backend_id: str
    estimator_path: str
    parameters_json: str
    random_seed: int
    distributions: tuple[tuple[str, str], ...]
    implementation_sha256: str
    policy_version: str

    @property
    def parameters(self) -> Mapping[str, Any]:
        return MappingProxyType(json.loads(self.parameters_json))

    def digest(self) -> str:
        return _stable_json_hash(asdict(self))


class _FactoryContractContext:
    def __init__(self) -> None:
        self.active: set[int] = set()
        self.remaining_nodes = _MAX_CONTRACT_NODES

    def consume(self) -> None:
        if self.remaining_nodes <= 0:
            raise TrainingContractError("training contract exceeds node limit")
        self.remaining_nodes -= 1


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
    backend_id: str
    training_contract_hash: str
    training_outcome: str

    def __post_init__(self) -> None:
        if not isinstance(self.backend_id, str) or not self.backend_id.strip():
            raise ValueError("backend_id must be a non-empty string")
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
            self.metadata.get("backend_id") != self.backend_id
            or self.bundle.metadata.get("backend_id") != self.backend_id
        ):
            raise ValueError("training result backend fields must match")
        if (
            self.metadata.get("training_contract_hash")
            != self.training_contract_hash
            or self.bundle.metadata.get("training_contract_hash")
            != self.training_contract_hash
        ):
            raise ValueError("training result contract fields must match")
        if self.training_outcome not in _TRAINING_OUTCOMES:
            raise ValueError("unsupported training_outcome")
        if (
            self.metadata.get("training_outcome") != self.training_outcome
            or self.bundle.metadata.get("training_outcome")
            != self.training_outcome
        ):
            raise ValueError("training result outcome fields must match")


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


@dataclass(frozen=True)
class FrozenCallableContract:
    descriptor_json: str

    @classmethod
    def capture(cls, value: Any) -> "FrozenCallableContract":
        return cls(_canonical_json(_callable_contract(value)))

    def verify(self, value: Any) -> None:
        try:
            current = _canonical_json(_callable_contract(value))
        except TrainingContractError:
            raise
        except Exception as exc:
            raise TrainingContractError(
                "factory contract could not be verified"
            ) from exc
        if current != self.descriptor_json:
            raise TrainingContractError("factory contract changed after initialization")


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


def _training_outcome(*fallback_reasons: str) -> str:
    normalized = [
        part.strip()
        for reason in fallback_reasons
        for part in str(reason or "").split(";")
        if part.strip()
    ]
    if any(
        reason in _OPERATIONAL_FALLBACK_REASONS
        or reason.startswith(_OPERATIONAL_FALLBACK_PREFIXES)
        for reason in normalized
    ):
        return "operational_fallback"
    if "constant_residual" in normalized:
        return "data_fallback"
    return "trained"


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


def _strict_json_parameter_value(value: Any) -> bool:
    if value is None or type(value) in {bool, int, str}:
        return True
    if type(value) is float:
        return bool(np.isfinite(value))
    if type(value) is list:
        return all(_strict_json_parameter_value(item) for item in value)
    if type(value) is dict:
        return all(
            type(key) is str
            and bool(key)
            and _strict_json_parameter_value(item)
            for key, item in value.items()
        )
    return False


def _strict_estimator_parameters(estimator: object) -> tuple[dict[str, Any], str]:
    try:
        raw_parameters = estimator.get_params(deep=False)
    except Exception as exc:
        raise TrainingContractError(
            "estimator parameters could not be read"
        ) from exc
    if not isinstance(raw_parameters, Mapping):
        raise TrainingContractError("estimator parameters must be a mapping")
    try:
        parameters = dict(raw_parameters)
    except Exception as exc:
        raise TrainingContractError(
            "estimator parameters could not be copied"
        ) from exc
    try:
        strict_json = all(
            type(key) is str
            and bool(key)
            and _strict_json_parameter_value(value)
            for key, value in parameters.items()
        )
    except Exception as exc:
        raise TrainingContractError(
            "estimator parameters are not strict JSON"
        ) from exc
    if not strict_json:
        raise TrainingContractError("estimator parameters are not strict JSON")
    try:
        parameters_json = _canonical_json(parameters)
        if json.loads(parameters_json) != parameters:
            raise ValueError("parameter JSON round trip changed values")
    except Exception as exc:
        raise TrainingContractError(
            "estimator parameters could not be frozen"
        ) from exc
    return parameters, parameters_json


_DISTRIBUTION_NAME_PATTERN = re.compile(
    r"[A-Za-z0-9](?:[A-Za-z0-9._-]*[A-Za-z0-9])?"
)


def _normalized_distribution_name(value: Any) -> str:
    if (
        not isinstance(value, str)
        or value != value.strip()
        or _DISTRIBUTION_NAME_PATTERN.fullmatch(value) is None
    ):
        raise TrainingContractError("estimator distribution name is invalid")
    return re.sub(r"[-_.]+", "-", value).lower()


def _distribution_version(distribution: str) -> str:
    try:
        version = importlib.metadata.version(distribution)
    except Exception as exc:
        raise TrainingContractError(
            f"distribution version is unreadable for {distribution}"
        ) from exc
    if not isinstance(version, str) or not version.strip():
        raise TrainingContractError(
            f"distribution version is unreadable for {distribution}"
        )
    return version.strip()


def _resolved_distribution_versions(module_name: str) -> dict[str, str]:
    if not isinstance(module_name, str) or not module_name.strip():
        raise TrainingContractError("estimator import root has no distribution")
    import_root = module_name.partition(".")[0]
    try:
        package_map = importlib.metadata.packages_distributions()
    except Exception as exc:
        raise TrainingContractError(
            "estimator distribution mapping could not be read"
        ) from exc
    if not isinstance(package_map, Mapping):
        raise TrainingContractError("estimator distribution mapping is invalid")
    try:
        distributions = package_map.get(import_root)
    except Exception as exc:
        raise TrainingContractError(
            "estimator distribution mapping could not be read"
        ) from exc
    if not isinstance(distributions, (list, tuple)) or not distributions:
        raise TrainingContractError(
            f"distribution mapping is missing for import root {import_root}"
        )
    normalized = [
        _normalized_distribution_name(distribution)
        for distribution in distributions
    ]
    names = tuple(sorted(set(normalized)))
    if len(names) != 1:
        raise TrainingContractError(
            f"distribution mapping is ambiguous for import root {import_root}"
        )
    distribution = names[0]
    return {distribution: _distribution_version(distribution)}


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise TrainingContractError(
            "estimator implementation file could not be read"
        ) from exc
    return digest.hexdigest()


def _estimator_implementation_sha256(estimator_type: type) -> str:
    try:
        implementation_path = Path(inspect.getfile(estimator_type)).resolve(
            strict=True
        )
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        raise TrainingContractError(
            "estimator implementation file could not be resolved"
        ) from exc
    if not implementation_path.is_file():
        raise TrainingContractError("estimator implementation path is not a file")
    return _sha256_path(implementation_path)


def _sealed_xgboost_estimator_type() -> type:
    try:
        import xgboost
        from xgboost.sklearn import XGBRegressor as supported_estimator_type

        estimator_type = _load_xgb_regressor()
        public_module, separator, public_name = (
            _SEALED_XGBOOST_ESTIMATOR_PATH.rpartition(".")
        )
        identity_matches = (
            separator == "."
            and public_module == xgboost.__name__
            and public_name == _SEALED_XGBOOST_ESTIMATOR_NAME
            and supported_estimator_type.__module__
            == _SEALED_XGBOOST_IMPLEMENTATION_MODULE
            and supported_estimator_type.__name__
            == _SEALED_XGBOOST_ESTIMATOR_NAME
            and getattr(xgboost, public_name) is supported_estimator_type
            and estimator_type is supported_estimator_type
        )
    except Exception as exc:
        raise TrainingContractError(
            "sealed xgboost estimator class identity could not be verified"
        ) from exc
    if not identity_matches:
        raise TrainingContractError(
            "sealed xgboost estimator class identity does not match policy"
        )
    return supported_estimator_type


def _sealed_xgboost_backend() -> tuple[SealedEstimatorSpec, type]:
    estimator_type = _sealed_xgboost_estimator_type()
    try:
        estimator = estimator_type(**dict(_DEFAULT_ESTIMATOR_PARAMETERS))
    except Exception as exc:
        raise TrainingContractError(
            "sealed xgboost estimator is unavailable"
        ) from exc
    parameters, parameters_json = _strict_estimator_parameters(estimator)
    if parameters.get("random_state") != 42:
        raise TrainingContractError(
            "sealed xgboost random_state does not match policy"
        )
    distributions = _resolved_distribution_versions(estimator_type.__module__)
    return (
        SealedEstimatorSpec(
            schema_version=1,
            backend_id=SEALED_XGBOOST_BACKEND_ID,
            estimator_path=_SEALED_XGBOOST_ESTIMATOR_PATH,
            parameters_json=parameters_json,
            random_seed=42,
            distributions=tuple(sorted(distributions.items())),
            implementation_sha256=_estimator_implementation_sha256(estimator_type),
            policy_version="weather-residual-training-v1",
        ),
        estimator_type,
    )


def sealed_xgboost_spec() -> SealedEstimatorSpec:
    spec, _ = _sealed_xgboost_backend()
    return spec


def _sealed_estimator_parameters_match_policy(
    estimator: object,
    estimator_type: type,
) -> None:
    try:
        expected_estimator = estimator_type(
            **dict(_DEFAULT_ESTIMATOR_PARAMETERS)
        )
        expected_parameters = estimator_type.get_params(
            expected_estimator,
            deep=False,
        )
        actual_parameters = estimator_type.get_params(
            estimator,
            deep=False,
        )
    except Exception as exc:
        raise TrainingContractError(
            "sealed estimator parameters could not be verified"
        ) from exc
    if type(expected_parameters) is not dict or type(actual_parameters) is not dict:
        raise TrainingContractError(
            "sealed estimator parameters must be an exact mapping"
        )
    if len(actual_parameters) != len(expected_parameters):
        raise TrainingContractError(
            "sealed estimator parameters failed attestation"
        )

    missing = object()
    for name, expected in expected_parameters.items():
        if type(name) is not str:
            raise TrainingContractError(
                "sealed estimator parameter names failed attestation"
            )
        actual = dict.get(actual_parameters, name, missing)
        if actual is missing or type(actual) is not type(expected):
            raise TrainingContractError(
                "sealed estimator parameters failed attestation"
            )
        if expected is not None and actual != expected:
            label = "random seed" if name == "random_state" else "parameters"
            raise TrainingContractError(
                f"sealed estimator {label} failed attestation"
            )


def verify_sealed_training_artifact(
    model: object,
    *,
    backend_id: str,
    training_outcome: str,
    random_seed: object,
) -> None:
    """Verify a persistable model without inspecting arbitrary callables."""
    if backend_id != SEALED_XGBOOST_BACKEND_ID:
        raise TrainingContractError(
            "model artifact is not from the sealed production backend"
        )
    if type(random_seed) is not int or random_seed != 42:
        raise TrainingContractError(
            "model artifact random seed does not match sealed policy"
        )
    if training_outcome == "legacy":
        training_outcome = (
            "data_fallback"
            if type(model) is ConstantResidualEstimator
            else "trained"
        )
    if training_outcome == "data_fallback":
        if type(model) is not ConstantResidualEstimator:
            raise TrainingContractError(
                "data fallback requires the exact constant residual estimator"
            )
        value = object.__getattribute__(model, "value")
        if type(value) is not float or not np.isfinite(value):
            raise TrainingContractError(
                "data fallback value must be finite"
            )
        return
    if training_outcome != "trained":
        raise TrainingContractError(
            f"{training_outcome} training outcome cannot be published"
        )

    estimator_type = _sealed_xgboost_estimator_type()
    if type(model) is not estimator_type:
        raise TrainingContractError(
            "trained artifact requires the exact sealed XGBoost estimator"
        )
    _sealed_estimator_parameters_match_policy(model, estimator_type)


def _bounded_contract_items(values, *, label: str) -> list:
    items = []
    for item in values:
        if len(items) >= _MAX_CONTRACT_ITEMS:
            raise TrainingContractError(f"{label} exceeds contract item limit")
        items.append(item)
    return items


def _factory_state(value: Any) -> dict[str, Any]:
    state = dict(getattr(value, "__dict__", {}))
    for owner in type(value).__mro__:
        slots = owner.__dict__.get("__slots__", ())
        if isinstance(slots, str):
            slots = (slots,)
        for name in slots:
            if name in {"__dict__", "__weakref__"} or name in state:
                continue
            try:
                state[name] = getattr(value, name)
            except AttributeError:
                continue
    return {str(name): item for name, item in state.items()}


def _explicit_contract_descriptor(value: Any) -> Optional[Any]:
    provider = getattr(value, "training_contract_descriptor", None)
    if provider is None:
        return None
    if not callable(provider):
        raise TrainingContractError(
            "training_contract_descriptor must be callable"
        )
    try:
        descriptor = provider()
    except Exception as exc:
        raise TrainingContractError(
            "explicit training contract could not be read"
        ) from exc
    if descriptor is None:
        raise TrainingContractError(
            "explicit training contract cannot be None"
        )
    return descriptor


def _contract_value(
    value: Any,
    *,
    _depth: int = 0,
    _context: Optional[_FactoryContractContext] = None,
) -> Any:
    if _depth > _MAX_CONTRACT_DEPTH:
        raise TrainingContractError("training contract exceeds recursion limit")
    _context = _context or _FactoryContractContext()
    _context.consume()
    if value is None or isinstance(value, (bool, int)):
        return value
    if isinstance(value, float):
        return value if np.isfinite(value) else {"float": str(value)}
    if isinstance(value, str):
        if len(value) <= 1024:
            return value
        return {
            "type": "str",
            "length": len(value),
            "content_hash": hashlib.sha256(value.encode("utf-8")).hexdigest(),
        }
    if isinstance(value, bytes):
        return {
            "kind": "bytes",
            "length": len(value),
            "content_hash": hashlib.sha256(value).hexdigest(),
        }
    if isinstance(value, np.generic):
        return _contract_value(
            value.item(), _depth=_depth + 1, _context=_context
        )
    if isinstance(value, (np.ndarray, pd.DataFrame, pd.Series, Path)):
        raise TrainingContractError(
            "complex factory configuration requires an explicit descriptor"
        )
    if isinstance(value, types.ModuleType):
        return _module_contract(value, _context=_context)
    if isinstance(value, types.CodeType):
        return {"kind": "code", "value": _code_contract(value, _context)}
    if callable(value):
        return {"kind": "callable", "value": _callable_contract(value, _context)}

    identity = id(value)
    if identity in _context.active:
        raise TrainingContractError("cyclic factory configuration is unsupported")
    _context.active.add(identity)
    try:
        if isinstance(value, (list, tuple, frozenset)):
            items = [
                _contract_value(item, _depth=_depth + 1, _context=_context)
                for item in _bounded_contract_items(
                    value, label=type(value).__name__
                )
            ]
            if isinstance(value, frozenset):
                items.sort(key=_canonical_json)
            return {"kind": type(value).__name__, "items": items}
        if isinstance(value, Mapping):
            entries = [
                {"key": _contract_value(key, _depth=_depth + 1, _context=_context),
                 "value": _contract_value(item, _depth=_depth + 1, _context=_context)}
                for key, item in _bounded_contract_items(
                    value.items(), label="mapping"
                )
            ]
            entries.sort(key=lambda item: _canonical_json(item["key"]))
            return {
                "kind": "mapping",
                "type": _qualified_name(type(value)),
                "entries": entries,
            }
        state = _factory_state(value)
        if not state:
            raise TrainingContractError(
                "opaque factory configuration requires an explicit descriptor"
            )
        items = _bounded_contract_items(
            sorted(state.items()), label="object configuration"
        )
        return {
            "kind": "object",
            "type": _qualified_name(type(value)),
            "state": {
                name: _contract_value(item, _depth=_depth + 1, _context=_context)
                for name, item in items
            },
        }
    finally:
        _context.active.remove(identity)


def _code_contract(
    code: types.CodeType,
    context: _FactoryContractContext,
) -> dict[str, Any]:
    return {
        "bytecode_hash": hashlib.sha256(code.co_code).hexdigest(),
        "exception_table_hash": hashlib.sha256(code.co_exceptiontable).hexdigest(),
        "names": list(code.co_names),
        "constants": _contract_value(tuple(code.co_consts), _context=context),
    }


def _referenced_global_dependencies(
    code: types.CodeType,
) -> dict[str, tuple[tuple[str, ...], ...]]:
    dependencies: dict[str, set[tuple[str, ...]]] = {}
    instructions = list(dis.get_instructions(code))
    for index, instruction in enumerate(instructions):
        if instruction.opname not in {"LOAD_GLOBAL", "LOAD_NAME"}:
            continue
        name = str(instruction.argval)
        attribute_path = []
        for following in instructions[index + 1 :]:
            if following.opname not in {"LOAD_ATTR", "LOAD_METHOD"}:
                break
            attribute_path.append(str(following.argval))
        dependencies.setdefault(name, set()).add(tuple(attribute_path))
    for constant in code.co_consts:
        if isinstance(constant, types.CodeType):
            for name, paths in _referenced_global_dependencies(constant).items():
                dependencies.setdefault(name, set()).update(paths)
    return {
        name: tuple(sorted(paths))
        for name, paths in sorted(dependencies.items())
    }


def _module_namespace(value: types.ModuleType) -> dict[str, Any]:
    try:
        namespace = object.__getattribute__(value, "__dict__")
    except Exception as exc:
        raise TrainingContractError("module contract cannot be read safely") from exc
    if not isinstance(namespace, dict):
        raise TrainingContractError("module contract has no static namespace")
    return namespace


def _symbol_module_name(value: Any) -> str:
    if inspect.isclass(value) or inspect.isfunction(value) or inspect.ismethod(value):
        module_name = getattr(value, "__module__", "")
    else:
        module_name = getattr(type(value), "__module__", "")
    return module_name if isinstance(module_name, str) else ""


def _is_external_contract_symbol(value: Any) -> bool:
    module_name = _symbol_module_name(value)
    root = module_name.partition(".")[0]
    return bool(
        module_name == "builtins"
        or root in sys.stdlib_module_names
        or root in _EXTERNAL_CONTRACT_ROOTS
    )


def _external_symbol_contract(value: Any) -> dict[str, Any]:
    module_name = _symbol_module_name(value)
    root = module_name.partition(".")[0]
    descriptor = {
        "kind": "external_symbol",
        "identity": _qualified_name(value if inspect.isroutine(value) or inspect.isclass(value) else type(value)),
    }
    if root in sys.stdlib_module_names or module_name == "builtins":
        descriptor["version"] = platform.python_version()
    else:
        try:
            descriptor["version"] = importlib.metadata.version(root)
        except (importlib.metadata.PackageNotFoundError, ValueError):
            descriptor["version"] = "unavailable"
    return descriptor


def _static_dependency_value(value: Any, path: tuple[str, ...]) -> Any:
    current = value
    for name in path:
        if isinstance(current, types.ModuleType):
            namespace = _module_namespace(current)
            if name not in namespace:
                raise TrainingContractError(
                    f"module dependency {name} is not statically available"
                )
            current = namespace[name]
            continue
        try:
            current = inspect.getattr_static(current, name)
        except AttributeError as exc:
            raise TrainingContractError(
                f"dependency attribute {name} is not statically available"
            ) from exc
        if isinstance(current, (staticmethod, classmethod)):
            current = current.__func__
    return current


def _module_contract(
    value: types.ModuleType,
    *,
    attribute_paths: Sequence[tuple[str, ...]] = (),
    _context: Optional[_FactoryContractContext] = None,
) -> dict[str, Any]:
    _context = _context or _FactoryContractContext()
    namespace = _module_namespace(value)
    module_name = namespace.get("__name__", "")
    descriptor = {
        "kind": "module",
        "identity": module_name if type(module_name) is str else "",
    }
    version = namespace.get("__version__")
    if type(version) is str:
        descriptor["version"] = (
            version
            if len(version) <= 128
            else hashlib.sha256(version.encode("utf-8")).hexdigest()
        )
    accessed_attributes = []
    for path in sorted(set(attribute_paths)):
        if not path:
            continue
        dependency = _static_dependency_value(value, path)
        accessed_attributes.append(
            {
                "path": list(path),
                "value": _global_contract_value(
                    dependency,
                    attribute_paths=(),
                    _context=_context,
                ),
            }
        )
    if accessed_attributes:
        descriptor["accessed_attributes"] = accessed_attributes
    return descriptor


def _function_core_contract(
    value: Any,
    context: _FactoryContractContext,
) -> tuple[Any, types.CodeType, dict[str, Any]]:
    target = getattr(value, "__func__", value)
    code = getattr(target, "__code__", None)
    if not isinstance(code, types.CodeType):
        raise TrainingContractError("factory function has no inspectable bytecode")
    return target, code, {
        "module": str(getattr(target, "__module__", "")),
        "qualname": str(getattr(target, "__qualname__", "")),
        "implementation": _code_contract(code, context),
        "defaults": _contract_value(
            tuple(getattr(target, "__defaults__", ()) or ()), _context=context
        ),
        "keyword_defaults": _contract_value(
            getattr(target, "__kwdefaults__", None) or {}, _context=context
        ),
    }


def _global_contract_value(
    value: Any,
    *,
    attribute_paths: Sequence[tuple[str, ...]] = (),
    _context: _FactoryContractContext,
) -> Any:
    if isinstance(value, types.ModuleType):
        return _module_contract(
            value,
            attribute_paths=attribute_paths,
            _context=_context,
        )
    if callable(value) and _is_external_contract_symbol(value):
        return _external_symbol_contract(value)
    if inspect.isclass(value):
        return {
            "kind": "class_reference",
            "contract": _callable_contract(value, _context),
        }
    if inspect.isbuiltin(value) or inspect.ismethoddescriptor(value):
        return {"kind": "builtin_callable", "identity": _qualified_name(value)}
    if inspect.isfunction(value) or inspect.ismethod(value):
        return {
            "kind": "function_reference",
            "contract": _callable_contract(value, _context),
        }
    if callable(value):
        explicit = _explicit_contract_descriptor(value)
        if explicit is None:
            raise TrainingContractError(
                "global callable instance requires an explicit descriptor"
            )
        return {
            "kind": "callable_reference",
            "identity": _qualified_name(type(value)),
            "declared_configuration": _contract_value(
                explicit, _context=_context
            ),
        }
    descriptor = _contract_value(value, _context=_context)
    if not attribute_paths:
        return descriptor
    accessed_attributes = []
    for path in sorted(set(attribute_paths)):
        if not path:
            continue
        dependency = _static_dependency_value(value, path)
        accessed_attributes.append(
            {
                "path": list(path),
                "value": _global_contract_value(
                    dependency,
                    attribute_paths=(),
                    _context=_context,
                ),
            }
        )
    if not accessed_attributes:
        return descriptor
    return {
        "kind": "referenced_value",
        "value": descriptor,
        "accessed_attributes": accessed_attributes,
    }


def _function_contract(
    value: Any,
    *,
    _context: _FactoryContractContext,
) -> dict[str, Any]:
    target, code, descriptor = _function_core_contract(value, _context)
    explicit = _explicit_contract_descriptor(value)
    if explicit is not None:
        descriptor["declared_configuration"] = _contract_value(
            explicit, _context=_context
        )
    closure = None if explicit is not None else getattr(target, "__closure__", None)
    if closure:
        closure_values = []
        for name, cell in zip(code.co_freevars, closure):
            try:
                cell_value = cell.cell_contents
            except ValueError as exc:
                raise TrainingContractError("callable has an empty closure cell") from exc
            if name == "__class__" and inspect.isclass(cell_value):
                closure_values.append(
                    {"name": name, "class": _qualified_name(cell_value)}
                )
                continue
            closure_values.append(
                {
                    "name": name,
                    "value": _contract_value(
                        cell_value,
                        _context=_context,
                    ),
                }
            )
        descriptor["closure"] = closure_values
    global_values = getattr(target, "__globals__", {})
    builtin_values = getattr(target, "__builtins__", {})
    if not isinstance(builtin_values, Mapping):
        builtin_values = vars(builtin_values)
    references = {}
    for name, attribute_paths in _referenced_global_dependencies(code).items():
        if name in global_values:
            scope, global_value = "global", global_values[name]
        elif name in builtin_values:
            if name in _DYNAMIC_GLOBAL_ACCESS_NAMES:
                raise TrainingContractError(
                    f"dynamic global access through {name} is unsupported"
                )
            scope, global_value = "builtin", builtin_values[name]
        else:
            references[name] = {"scope": "unbound"}
            continue
        references[name] = {
            "scope": scope,
            "value": _global_contract_value(
                global_value,
                attribute_paths=attribute_paths,
                _context=_context,
            ),
        }
    if references:
        descriptor["globals"] = references
    return descriptor


def _class_contract(
    value: type,
    *,
    _context: _FactoryContractContext,
) -> dict[str, Any]:
    explicit = _explicit_contract_descriptor(value)
    if explicit is not None:
        return {
            "kind": "class",
            "identity": _qualified_name(value),
            "declared_configuration": _contract_value(
                explicit, _context=_context
            ),
        }

    members = {}
    method_count = 0
    owners = [owner for owner in reversed(value.__mro__) if owner is not object]
    for owner in owners:
        if _is_external_contract_symbol(owner):
            members[f"{_qualified_name(owner)}::__external__"] = (
                _external_symbol_contract(owner)
            )
            continue
        for name, item in sorted(vars(owner).items()):
            if name in {
                "__classcell__",
                "__dict__",
                "__doc__",
                "__module__",
                "__weakref__",
            }:
                continue
            if name.startswith("__") and name not in {
                "__call__",
                "__init__",
                "__repr__",
                "__str__",
            }:
                continue
            if isinstance(item, (staticmethod, classmethod)):
                item = item.__func__
            if inspect.isfunction(item):
                members[name] = {
                    "kind": "method",
                    "contract": _callable_contract(item, _context),
                }
                method_count += 1
            elif not name.startswith("_") and not inspect.isdatadescriptor(item):
                members[name] = {
                    "kind": "configuration",
                    "value": _contract_value(item, _context=_context),
                }
    if not method_count:
        raise TrainingContractError(
            f"class factory {_qualified_name(value)} has no inspectable methods"
        )
    return {
        "kind": "class",
        "identity": _qualified_name(value),
        "mro": [_qualified_name(owner) for owner in reversed(owners)],
        "members": members,
    }


def _callable_contract(
    value: Any,
    _context: Optional[_FactoryContractContext] = None,
) -> dict[str, Any]:
    _context = _context or _FactoryContractContext()
    _context.consume()
    if value is default_estimator:
        return {
            "kind": "default_estimator",
            "symbol": "xgboost.XGBRegressor",
            "declared_parameters": dict(_DEFAULT_ESTIMATOR_PARAMETERS),
        }
    if not isinstance(value, functools.partial) and _is_external_contract_symbol(
        value
    ):
        return _external_symbol_contract(value)
    identity = id(value)
    if identity in _context.active:
        raise TrainingContractError("cyclic callable factory is unsupported")
    if len(_context.active) >= _MAX_CONTRACT_DEPTH:
        raise TrainingContractError("training contract exceeds recursion limit")
    _context.active.add(identity)
    try:
        if isinstance(value, functools.partial):
            return {
                "kind": "partial",
                "callable": _callable_contract(value.func, _context),
                "args": _contract_value(tuple(value.args), _context=_context),
                "keywords": _contract_value(
                    value.keywords or {}, _context=_context
                ),
            }
        if (
            inspect.isbuiltin(value)
            or inspect.ismethoddescriptor(value)
            or (inspect.isclass(value) and value.__module__ == "builtins")
        ):
            return {"kind": "builtin_callable", "identity": _qualified_name(value)}
        if inspect.isclass(value):
            return _class_contract(value, _context=_context)
        if inspect.isfunction(value) or inspect.ismethod(value):
            descriptor = {
                "kind": "function",
                **_function_contract(value, _context=_context),
            }
            if inspect.ismethod(value) and not inspect.isclass(value.__self__):
                descriptor["bound_state"] = _contract_value(
                    value.__self__, _context=_context
                )
            return descriptor
        if not callable(value):
            raise TrainingContractError("estimator factory must be callable")
        explicit = _explicit_contract_descriptor(value)
        descriptor = {
            "kind": "callable_instance",
            "identity": _qualified_name(type(value)),
        }
        if explicit is not None:
            descriptor["declared_configuration"] = _contract_value(
                explicit, _context=_context
            )
        else:
            descriptor["class"] = _callable_contract(type(value), _context)
            descriptor["configuration"] = _contract_value(
                _factory_state(value), _context=_context
            )
        return descriptor
    finally:
        _context.active.remove(identity)


def _trainer_runtime_descriptor(trainer: Any) -> dict[str, Any]:
    explicit = _explicit_contract_descriptor(trainer)
    if explicit is None:
        raise TrainingContractError(
            "trainer must implement training_contract_descriptor()"
        )
    descriptor = {
        "trainer_type": _qualified_name(type(trainer)),
        "configuration": _contract_value(explicit),
    }
    if isinstance(trainer, ResidualTrainer):
        descriptor["builtin_runtime"] = _builtin_trainer_runtime_descriptor(
            trainer
        )
    return descriptor


def _builtin_trainer_runtime_descriptor(trainer: Any) -> dict[str, Any]:
    context = _FactoryContractContext()
    trainer_type = type(trainer)

    def method_code(name: str) -> dict[str, Any]:
        try:
            value = inspect.getattr_static(trainer_type, name)
        except AttributeError as exc:
            raise TrainingContractError(
                f"builtin trainer dependency {name} is missing"
            ) from exc
        if isinstance(value, (staticmethod, classmethod)):
            value = value.__func__
        code = getattr(value, "__code__", None)
        if not isinstance(code, types.CodeType):
            raise TrainingContractError(
                f"builtin trainer dependency {name} is not inspectable"
            )
        return _code_contract(code, context)

    return {
        "methods": {
            "prepare_target": method_code("prepare_target"),
            "train_prepared": method_code("train_prepared"),
            "_fit_estimator": method_code("_fit_estimator"),
            "_fallback_estimator": method_code("_fallback_estimator"),
            "_predict_corrected": method_code("_predict_corrected"),
            "_time_split": method_code("_time_split"),
        },
        "callables": {
            "_metric_values": _callable_contract(_metric_values, context),
            "_robust_residual_bounds": _callable_contract(
                _robust_residual_bounds, context
            ),
            "_training_outcome": _callable_contract(
                _training_outcome, context
            ),
            "missing_value_collision_rows": _callable_contract(
                missing_value_collision_rows, context
            ),
        },
        "configuration": _contract_value(
            {
                "physical_bounds": PHYSICAL_BOUNDS,
                "training_outcomes": _TRAINING_OUTCOMES,
                "operational_fallback_reasons": _OPERATIONAL_FALLBACK_REASONS,
                "operational_fallback_prefixes": _OPERATIONAL_FALLBACK_PREFIXES,
            },
            _context=context,
        ),
    }


def _runtime_contract_hash(
    trainer: Any,
    preparation_contract_hash: str,
) -> str:
    return _stable_json_hash(
        {
            "preparation_contract_hash": preparation_contract_hash,
            "trainer": _trainer_runtime_descriptor(trainer),
        }
    )


def training_runtime_contract_hash_for_scope(
    trainer: Any,
    *,
    target: str,
    physical_col: str,
    truth_col: str,
    feature_columns: Sequence[str],
    cadence_minutes: float,
) -> str:
    training_contract = getattr(trainer, "training_contract", None)
    if not isinstance(training_contract, TrainingContract):
        raise TypeError("trainer must provide a TrainingContract")
    preparation_contract_hash = training_contract.scoped_hash(
        target=target,
        physical_col=physical_col,
        truth_col=truth_col,
        feature_columns=feature_columns,
        cadence_minutes=cadence_minutes,
    )
    return _runtime_contract_hash(trainer, preparation_contract_hash)


def training_runtime_contract_hash(
    trainer: Any,
    preparation: TrainingPreparation,
) -> str:
    if not isinstance(preparation, TrainingPreparation):
        raise TypeError("preparation must be a TrainingPreparation")
    return _runtime_contract_hash(trainer, preparation.training_contract_hash)


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


def _safe_training_type(value: Any) -> str:
    return _qualified_name(type(value))


def _training_digest(value: Any) -> str:
    if isinstance(value, bytes):
        payload = value
    elif isinstance(value, (str, Path)):
        payload = str(value).encode("utf-8")
    elif isinstance(value, int) and not isinstance(value, bool):
        magnitude = abs(value)
        byte_count = max(1, (magnitude.bit_length() + 7) // 8)
        payload = (b"-" if value < 0 else b"+") + magnitude.to_bytes(
            byte_count, "big"
        )
    elif value is None or isinstance(value, (bool, float)):
        scalar = value if not isinstance(value, float) or np.isfinite(value) else str(value)
        payload = _canonical_json(
            {"type": _safe_training_type(value), "value": scalar}
        ).encode()
    else:
        payload = _safe_training_type(value).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _pandas_training_summary(
    value: pd.DataFrame | pd.Series,
) -> dict[str, Any]:
    shape = [int(item) for item in value.shape]
    if isinstance(value, pd.DataFrame):
        column_count = int(value.shape[1])
        visible = min(column_count, _MAX_TRAINING_METADATA_ITEMS)
        schema = {
            "shape": shape,
            "column_count": column_count,
            "column_hashes": [
                _training_digest(str(column)) for column in value.columns[:visible]
            ],
            "dtype_hashes": [
                _training_digest(
                    _canonical_json(_training_dtype_summary(dtype))
                )
                for dtype in value.dtypes.iloc[:visible]
            ],
        }
        if column_count > visible:
            schema["truncated"] = "max_columns"
    else:
        schema = {
            "shape": shape,
            "dtype": _training_dtype_summary(value.dtype),
            "name_hash": _training_digest(str(value.name)),
        }
    return {
        "type": _safe_training_type(value),
        **schema,
        "content_hash": _training_digest(_canonical_json(schema)),
    }


def _ndarray_training_summary(value: np.ndarray) -> dict[str, Any]:
    array = np.asarray(value)
    schema = {
        "shape": [int(item) for item in array.shape],
        "dtype": _training_dtype_summary(array.dtype),
    }
    digest = hashlib.sha256(_canonical_json(schema).encode())
    if not array.dtype.hasobject:
        digest.update(np.ascontiguousarray(array).tobytes())
    return {
        "type": _safe_training_type(value),
        **schema,
        "content_hash": digest.hexdigest(),
    }


def _training_dtype_summary(value: Any) -> dict[str, Any]:
    dtype_type = type(value)
    module_root = dtype_type.__module__.partition(".")[0]
    summary = {"type": _safe_training_type(value)}
    if module_root == "numpy" and isinstance(value, np.dtype):
        name = value.name
        kind = value.kind
        if type(name) is str:
            summary["name"] = name
        if type(kind) is str:
            summary["kind"] = kind
        summary["itemsize"] = int(value.itemsize)
    elif module_root == "pandas" and isinstance(
        value, pd.api.extensions.ExtensionDtype
    ):
        name = value.name
        if type(name) is str:
            summary["name"] = name
    return summary


def _is_sensitive_parameter_name(name: Any) -> bool:
    if isinstance(name, str):
        lowered = str.lower(name)
    elif isinstance(name, bytes):
        try:
            lowered = str.lower(bytes.decode(name, "ascii"))
        except UnicodeDecodeError:
            return True
    else:
        return False
    normalized = "".join(
        character
        for character in lowered
        if "a" <= character <= "z" or "0" <= character <= "9"
    )
    return any(marker in normalized for marker in _SENSITIVE_PARAMETER_MARKERS)


def _safe_mapping_key(value: Any) -> str:
    if isinstance(value, str):
        safe_value = str.__str__(value)
        if str.__len__(value) <= 128:
            return safe_value
        return f"str_sha256:{_training_digest(safe_value)}"
    return f"{_safe_training_type(value)}_sha256:{_training_digest(value)}"


def _sensitive_training_summary(value: Any) -> dict[str, Any]:
    if isinstance(value, (pd.DataFrame, pd.Series)):
        return _pandas_training_summary(value)
    if isinstance(value, np.ndarray):
        return _ndarray_training_summary(value)
    summary = {"type": _safe_training_type(value)}
    try:
        summary["length"] = len(value)
    except (TypeError, OverflowError):
        pass
    summary["content_hash"] = _training_digest(value)
    return summary


def _json_safe_training_value(
    value: Any,
    *,
    _depth: int = 0,
    _active: Optional[set[int]] = None,
    _budget: Optional[list[int]] = None,
):
    if _active is None:
        _active = set()
    if _budget is None:
        _budget = [_MAX_TRAINING_METADATA_NODES]
    if _budget[0] <= 0:
        return {"type": _safe_training_type(value), "truncated": "max_nodes"}
    _budget[0] -= 1
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, int):
        if abs(value).bit_length() <= _MAX_TRAINING_METADATA_INTEGER_BITS:
            return value
        return {
            "type": _safe_training_type(value),
            "bit_length": abs(value).bit_length(),
            "content_hash": _training_digest(value),
        }
    if isinstance(value, str):
        if len(value) <= _MAX_TRAINING_METADATA_STRING:
            return value
        return {
            "type": _safe_training_type(value),
            "length": len(value),
            "content_hash": _training_digest(value),
        }
    if isinstance(value, bytes):
        return {
            "type": _safe_training_type(value),
            "length": len(value),
            "content_hash": _training_digest(value),
        }
    if isinstance(value, float):
        if np.isfinite(value):
            return value
        return {"type": _safe_training_type(value), "value": str(value)}
    if isinstance(value, np.generic):
        return _json_safe_training_value(
            value.item(),
            _depth=_depth,
            _active=_active,
            _budget=_budget,
        )
    if isinstance(value, (pd.DataFrame, pd.Series)):
        return _pandas_training_summary(value)
    if isinstance(value, np.ndarray):
        return _ndarray_training_summary(value)
    if isinstance(value, Path):
        return {
            "type": _safe_training_type(value),
            "content_hash": _training_digest(value),
        }
    dataclass_value = is_dataclass(value) and not inspect.isclass(value)
    container = isinstance(value, (Mapping, list, tuple, set, frozenset))
    if not dataclass_value and not container:
        return {"type": _safe_training_type(value)}

    identity = id(value)
    if identity in _active:
        return {"type": _safe_training_type(value), "cycle": True}
    try:
        size = len(value)
    except (TypeError, OverflowError):
        size = None
    if _depth >= _MAX_TRAINING_METADATA_DEPTH:
        return {
            "type": _safe_training_type(value),
            "length": size,
            "truncated": "max_depth",
        }
    if (
        type(value) in {dict, list, tuple, set, frozenset}
        and size is not None
        and size > _MAX_TRAINING_METADATA_ITEMS
    ):
        return {
            "type": _safe_training_type(value),
            "length": size,
            "truncated": "max_items",
        }

    def sanitize(item):
        return _json_safe_training_value(
            item,
            _depth=_depth + 1,
            _active=_active,
            _budget=_budget,
        )

    _active.add(identity)
    try:
        mapping_value = isinstance(value, Mapping)
        if dataclass_value:
            source = (
                (field.name, getattr(value, field.name)) for field in fields(value)
            )
        elif mapping_value:
            source = value.items()
        else:
            source = ((None, item) for item in value)
        result = {} if dataclass_value or mapping_value else []
        truncated = None
        try:
            for index, (key, item) in enumerate(source):
                if index >= _MAX_TRAINING_METADATA_ITEMS or _budget[0] <= 0:
                    truncated = (
                        "max_items"
                        if index >= _MAX_TRAINING_METADATA_ITEMS
                        else "max_nodes"
                    )
                    break
                sanitized = (
                    _sensitive_training_summary(item)
                    if mapping_value
                    and _is_sensitive_parameter_name(key)
                    else sanitize(item)
                )
                if isinstance(result, dict):
                    result[
                        key if dataclass_value else _safe_mapping_key(key)
                    ] = sanitized
                else:
                    result.append(sanitized)
        except Exception as exc:
            truncated = f"unreadable_{type(exc).__name__}"
        if dataclass_value:
            summary = {
                "type": _safe_training_type(value),
                "field_names": list(result),
                "content_hash": _training_digest(_canonical_json(result)),
            }
            if truncated:
                summary["truncated"] = truncated
            return summary
        if mapping_value:
            if truncated:
                result["__training_metadata_truncated__"] = truncated
            return result
        if truncated:
            result.append(
                {"type": _safe_training_type(value), "truncated": truncated}
            )
        if isinstance(value, (set, frozenset)):
            result.sort(key=_canonical_json)
        return result
    finally:
        _active.remove(identity)


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
        if estimator_factory is None:
            self.estimator_factory = default_estimator
            (
                self.sealed_estimator_spec,
                self._sealed_estimator_type,
            ) = _sealed_xgboost_backend()
            self.production_eligible = True
        else:
            self.estimator_factory = estimator_factory
            self.sealed_estimator_spec = None
            self._sealed_estimator_type = None
            self.production_eligible = False
        self._attested_distribution_names: Optional[tuple[str, ...]] = None
        self.feature_builder = feature_builder or FeatureBuilder()
        self._factory_contract = FrozenCallableContract.capture(
            self.estimator_factory
        )
        dependencies = _dependency_versions()
        trainer_type = type(self)
        contract_context = _FactoryContractContext()
        trainer_descriptor = {
            "type": _qualified_name(trainer_type),
            "prepare_target": _code_contract(
                trainer_type.prepare_target.__code__, contract_context
            ),
            "train_prepared": _code_contract(
                trainer_type.train_prepared.__code__, contract_context
            ),
        }
        self.training_contract = TrainingContract(
            version=_TRAINING_CONTRACT_VERSION,
            trainer_descriptor_json=_canonical_json(trainer_descriptor),
            estimator_descriptor_json=self._factory_contract.descriptor_json,
            dependency_versions=tuple(sorted(dependencies.items())),
            random_seed=42,
        )

    def _assert_factory_contract(self) -> None:
        self._factory_contract.verify(self.estimator_factory)

    def attest_estimator(self, estimator: object) -> None:
        spec = self.sealed_estimator_spec
        estimator_type = self._sealed_estimator_type
        if (
            not self.production_eligible
            or spec is None
            or estimator_type is None
        ):
            raise TrainingContractError(
                "estimator attestation requires the sealed production backend"
            )
        if type(estimator) is not estimator_type:
            raise TrainingContractError("estimator type failed attestation")
        parameters, parameters_json = _strict_estimator_parameters(estimator)
        if parameters.get("random_state") != spec.random_seed:
            raise TrainingContractError(
                "estimator random_state failed attestation"
            )
        if parameters_json != spec.parameters_json:
            raise TrainingContractError("estimator parameters failed attestation")
        cached_distribution_names = self._attested_distribution_names
        distribution_names_to_cache = None
        if cached_distribution_names is None:
            distributions = tuple(
                sorted(
                    _resolved_distribution_versions(
                        estimator_type.__module__
                    ).items()
                )
            )
            distribution_names_to_cache = tuple(
                distribution for distribution, _ in distributions
            )
        else:
            distributions = tuple(
                (distribution, _distribution_version(distribution))
                for distribution in cached_distribution_names
            )
        if distributions != spec.distributions:
            raise TrainingContractError(
                "estimator distribution versions failed attestation"
            )
        implementation_sha256 = _estimator_implementation_sha256(estimator_type)
        if implementation_sha256 != spec.implementation_sha256:
            raise TrainingContractError(
                "estimator implementation hash failed attestation"
            )
        if distribution_names_to_cache is not None:
            self._attested_distribution_names = distribution_names_to_cache

    def training_contract_descriptor(self) -> dict[str, Any]:
        self._assert_factory_contract()
        return {
            "version": _TRAINING_CONTRACT_VERSION,
            "trainer_descriptor_hash": hashlib.sha256(
                self.training_contract.trainer_descriptor_json.encode("utf-8")
            ).hexdigest(),
            "estimator_descriptor_hash": hashlib.sha256(
                self.training_contract.estimator_descriptor_json.encode("utf-8")
            ).hexdigest(),
            "dependencies": dict(self.training_contract.dependency_versions),
            "random_seed": self.training_contract.random_seed,
            "feature_builder": {
                "type": _qualified_name(type(self.feature_builder)),
                "cadence_minutes": float(self.feature_builder.cadence_minutes),
                "derived_features": list(
                    self.feature_builder._DERIVED_FEATURE_COLUMNS
                ),
            },
        }

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
        if self.production_eligible:
            self.attest_estimator(estimator)
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
        self._assert_factory_contract()
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
        if self.production_eligible:
            missing = self.sealed_estimator_spec.parameters.get("missing")
            if missing_value_collision_rows(model_features, missing).any():
                raise ValueError(
                    "model features collide with estimator missing sentinel"
                )
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
            invalid = (
                values.ndim != 1
                or values.dtype.kind not in "iu"
                or values.size == 0
                or (values < 0).any()
                or (values >= row_count).any()
                or np.unique(values).size != values.size
            )
            if invalid:
                raise ValueError(f"preparation {name} split indices are invalid")
            normalized.append(values.astype(np.int64, copy=False))
        train_positions, holdout_positions = normalized
        if np.intersect1d(train_positions, holdout_positions).size:
            raise ValueError("preparation split indices must be mutually exclusive")

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

        for field in ("working", "feature_frame", "model_features"):
            actual, expected = getattr(preparation, field), getattr(trusted, field)
            if not isinstance(actual, pd.DataFrame):
                raise ValueError(f"preparation {field} must be a DataFrame")
            try:
                pd.testing.assert_frame_equal(actual, expected, check_exact=True)
            except AssertionError as exc:
                raise ValueError(
                    f"preparation {field} failed integrity validation"
                ) from exc
        for field in ("physical", "truth", "residual"):
            actual, expected = getattr(preparation, field), getattr(trusted, field)
            if (
                not isinstance(actual, np.ndarray)
                or actual.dtype != expected.dtype
                or not np.array_equal(actual, expected)
            ):
                raise ValueError(f"preparation {field} failed integrity validation")

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
        self._assert_factory_contract()
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

        self._assert_factory_contract()

        timestamp_values = feature_frame["timestamp"]
        training_outcome = _training_outcome(final_reason, evaluation_reason)
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
            "backend_id": (
                self.sealed_estimator_spec.backend_id
                if self.sealed_estimator_spec is not None
                else _NON_PRODUCTION_BACKEND_ID
            ),
            "training_contract_hash": preparation.training_contract_hash,
            "training_snapshot_hash": preparation.snapshot_hash,
            "random_state": 42,
            "training_params": _estimator_training_params(final_estimator),
            "dependency_versions": dict(
                self.training_contract.dependency_versions
            ),
            "fallback_reason": final_reason,
            "evaluation_fallback_reason": evaluation_reason,
            "training_outcome": training_outcome,
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
            backend_id=metadata["backend_id"],
            training_contract_hash=preparation.training_contract_hash,
            training_outcome=training_outcome,
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
