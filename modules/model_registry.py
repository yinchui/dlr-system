from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import shutil
import stat
import uuid
from dataclasses import InitVar, asdict, dataclass, field, replace
from pathlib import Path, PureWindowsPath
from typing import Any, Mapping, Optional

import joblib
from filelock import FileLock

from modules.ai_prediction import ModelBundle
from modules.ai_training import SEALED_XGBOOST_BACKEND_ID


_ALLOWED_TARGETS = frozenset({"wind_speed", "ambient_temp"})
_WEATHER_METRIC_NAMES = frozenset(
    {
        "baseline_mae",
        "baseline_rmse",
        "corrected_mae",
        "corrected_rmse",
    }
)
_EVALUATION_MODES = frozenset(
    {"full_fit", "temporal_holdout", "rolling_validation"}
)
_MODEL_STATUSES = frozenset(
    {"candidate", "active_provisional", "active"}
)
_TRAINING_OUTCOMES = frozenset(
    {"trained", "data_fallback", "operational_fallback", "legacy"}
)
_KEY_FIELDS = ("project_id", "line_id", "tower_id", "target")
_COMPATIBILITY_FIELDS = (
    "dem_hash",
    "crs_hash",
    "coordinate_hash",
    "conductor_hash",
    "feature_version",
    "correction_config_hash",
)
_MANIFEST_FIELDS = frozenset(
    {
        "schema_version",
        *_KEY_FIELDS,
        "model_version",
        "model_file",
        "metadata_file",
        "model_checksum",
        "metadata_checksum",
    }
)
_GENERATION_ARTIFACTS = ("model.joblib", "metadata.json", "manifest.json")
_PRIVATE_DIRECTORY_MODE = 0o700
_PRIVATE_FILE_MODE = 0o600
_LEGACY_TRAINING_CONTRACT_HASH = "legacy-training-contract-v0"
_LEGACY_BACKEND_ID = "legacy-training-backend-v0"
_ATTEMPT_POLICY_VERSION = "weather-promotion-v1"
_ATTEMPT_LEDGER_FIELDS = frozenset(
    {"schema_version", *_KEY_FIELDS, "entries"}
)
_ATTEMPT_ENTRY_FIELDS = frozenset(
    {
        *_KEY_FIELDS,
        "input_data_hash",
        "evaluation_set_hash",
        "policy_version",
        "min_mae_improvement",
        "training_contract_hash",
        "feature_version",
        "champion_context_hash",
        "fingerprint",
        "reason",
    }
)
_DETERMINISTIC_REJECTION_REASONS = frozenset(
    {
        "missing_evaluation_set_hash",
        "candidate_not_better_than_physical",
        "full_fit_cannot_replace_champion",
        "insufficient_mae_improvement",
        "champion_has_no_independent_evaluation",
        "evaluation_set_mismatch",
    }
)
_MAX_ATTEMPT_RECORD_LIMIT = 256


class FrozenMetadata(dict):
    """Pickle-safe immutable mapping used inside persisted model bundles."""

    def __init__(self, source=()):
        normalized = {
            copy.deepcopy(key): _freeze_metadata_value(value)
            for key, value in dict(source).items()
        }
        dict.__init__(self, normalized)

    @staticmethod
    def _immutable(*args, **kwargs):
        raise TypeError("model metadata is immutable")

    __setitem__ = _immutable
    __delitem__ = _immutable
    clear = _immutable
    pop = _immutable
    popitem = _immutable
    setdefault = _immutable
    update = _immutable
    __ior__ = _immutable

    def __copy__(self):
        return self

    def __deepcopy__(self, memo):
        return self

    def __reduce__(self):
        return type(self), (dict(self),)


def _freeze_metadata_value(value: Any) -> Any:
    if isinstance(value, FrozenMetadata):
        return value
    if isinstance(value, Mapping):
        return FrozenMetadata(value)
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_metadata_value(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return frozenset(_freeze_metadata_value(item) for item in value)
    return copy.deepcopy(value)


def _validate_identifier(value: str, name: str) -> None:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    if not value.strip():
        raise ValueError(f"{name} cannot be empty")
    if value != value.strip():
        raise ValueError(f"{name} cannot have surrounding whitespace")
    if value in {".", ".."}:
        raise ValueError(f"{name} cannot be a relative path component")
    if (
        Path(value).is_absolute()
        or os.path.isabs(value)
        or PureWindowsPath(value).drive
    ):
        raise ValueError(f"{name} cannot be an absolute path")
    if "/" in value or "\\" in value or "\x00" in value:
        raise ValueError(f"{name} cannot contain path separators")


@dataclass(frozen=True)
class ModelKey:
    project_id: str
    line_id: str
    tower_id: str
    target: str

    def __post_init__(self) -> None:
        for name in _KEY_FIELDS:
            _validate_identifier(getattr(self, name), name)
        if self.target not in _ALLOWED_TARGETS:
            raise ValueError("target must be wind_speed or ambient_temp")


def _require_nonempty_string(value: str, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")


@dataclass(frozen=True)
class ModelCompatibility:
    dem_hash: str
    crs_hash: str
    coordinate_hash: str
    conductor_hash: str
    feature_version: str
    correction_config_hash: str

    def __post_init__(self) -> None:
        for name in _COMPATIBILITY_FIELDS:
            _require_nonempty_string(getattr(self, name), name)

    def to_dict(self) -> dict[str, str]:
        return {name: getattr(self, name) for name in _COMPATIBILITY_FIELDS}


def _validated_metrics(
    metrics: Optional[Mapping[str, Any]], name: str
) -> Optional[dict[str, float]]:
    if metrics is None:
        return None
    if not isinstance(metrics, Mapping):
        raise ValueError(f"{name} must be a mapping")
    if not metrics:
        return {}
    if set(metrics) != _WEATHER_METRIC_NAMES:
        raise ValueError(f"{name} must contain only weather MAE/RMSE metrics")
    result = {}
    for metric_name, value in metrics.items():
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) < 0.0
        ):
            raise ValueError(f"{name}.{metric_name} must be finite and non-negative")
        result[metric_name] = float(value)
    return result


@dataclass(frozen=True)
class ModelMetadata:
    key: ModelKey
    model_version: str
    feature_columns: tuple[str, ...]
    training_params: Mapping[str, Any]
    random_seed: Optional[int]
    time_start: str
    time_end: str
    sample_count: int
    evaluation_mode: str
    metrics: Mapping[str, float]
    full_fit_metrics: Optional[Mapping[str, float]]
    residual_bounds: tuple[float, float]
    input_data_hash: str
    evaluation_set_hash: Optional[str]
    compatibility: ModelCompatibility
    dependency_versions: Mapping[str, str]
    cadence_minutes: float
    training_contract_hash: str
    backend_id: str
    _allow_legacy_training_contract: InitVar[bool] = False
    _allow_legacy_backend: InitVar[bool] = False
    training_outcome: str = "trained"
    checksum: str = ""
    status: str = "candidate"
    metric_domain: str = "weather_vs_truth"
    last_attempted_input_data_hash: Optional[str] = None

    def __post_init__(
        self,
        _allow_legacy_training_contract: bool,
        _allow_legacy_backend: bool,
    ) -> None:
        if not isinstance(self.key, ModelKey):
            raise TypeError("key must be a ModelKey")
        _validate_identifier(self.model_version, "model_version")
        if not self.feature_columns:
            raise ValueError("feature_columns cannot be empty")
        normalized_features = tuple(self.feature_columns)
        for feature in normalized_features:
            _require_nonempty_string(feature, "feature_columns")
        if len(set(normalized_features)) != len(normalized_features):
            raise ValueError("feature_columns cannot contain duplicates")
        object.__setattr__(self, "feature_columns", normalized_features)

        if not isinstance(self.training_params, Mapping):
            raise ValueError("training_params must be a mapping")
        training_params = dict(self.training_params)
        if not training_params:
            raise ValueError("training_params cannot be empty")
        try:
            json.dumps(training_params, allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise ValueError("training_params must be JSON serializable") from exc
        object.__setattr__(
            self, "training_params", FrozenMetadata(training_params)
        )

        if self.random_seed is not None and (
            isinstance(self.random_seed, bool)
            or not isinstance(self.random_seed, int)
        ):
            raise ValueError("random_seed must be an integer or None")
        _require_nonempty_string(self.time_start, "time_start")
        _require_nonempty_string(self.time_end, "time_end")
        if (
            isinstance(self.sample_count, bool)
            or not isinstance(self.sample_count, int)
            or self.sample_count < 1
        ):
            raise ValueError("sample_count must be a positive integer")
        if self.evaluation_mode not in _EVALUATION_MODES:
            raise ValueError("unsupported evaluation_mode")

        metrics = _validated_metrics(self.metrics, "metrics")
        full_fit_metrics = _validated_metrics(
            self.full_fit_metrics, "full_fit_metrics"
        )
        if self.evaluation_mode == "full_fit":
            if metrics:
                raise ValueError("full_fit metrics must remain empty")
            if not full_fit_metrics:
                raise ValueError("full_fit requires full_fit_metrics")
            if self.evaluation_set_hash is not None:
                raise ValueError("full_fit cannot have evaluation_set_hash")
        elif not metrics:
            raise ValueError("independent evaluation requires weather metrics")
        object.__setattr__(self, "metrics", FrozenMetadata(metrics or {}))
        object.__setattr__(
            self,
            "full_fit_metrics",
            None
            if full_fit_metrics is None
            else FrozenMetadata(full_fit_metrics),
        )

        if len(self.residual_bounds) != 2:
            raise ValueError("residual_bounds must contain lower and upper")
        lower, upper = self.residual_bounds
        if (
            isinstance(lower, bool)
            or isinstance(upper, bool)
            or not isinstance(lower, (int, float))
            or not isinstance(upper, (int, float))
            or not math.isfinite(float(lower))
            or not math.isfinite(float(upper))
            or float(lower) > float(upper)
        ):
            raise ValueError("residual_bounds must be finite and ordered")
        object.__setattr__(self, "residual_bounds", (float(lower), float(upper)))

        _require_nonempty_string(self.input_data_hash, "input_data_hash")
        _require_nonempty_string(
            self.training_contract_hash,
            "training_contract_hash",
        )
        if (
            self.training_contract_hash == _LEGACY_TRAINING_CONTRACT_HASH
            and not _allow_legacy_training_contract
        ):
            raise ValueError(
                "legacy training contract is reserved for metadata migration"
            )
        _require_nonempty_string(self.backend_id, "backend_id")
        if self.backend_id == _LEGACY_BACKEND_ID and not _allow_legacy_backend:
            raise ValueError("legacy backend is reserved for metadata migration")
        if self.last_attempted_input_data_hash is not None:
            _require_nonempty_string(
                self.last_attempted_input_data_hash,
                "last_attempted_input_data_hash",
            )
        if self.evaluation_set_hash is not None:
            _require_nonempty_string(
                self.evaluation_set_hash, "evaluation_set_hash"
            )
        if not isinstance(self.compatibility, ModelCompatibility):
            raise TypeError("compatibility must be ModelCompatibility")
        if not isinstance(self.dependency_versions, Mapping):
            raise ValueError("dependency_versions must be a mapping")
        dependencies = dict(self.dependency_versions)
        if not dependencies:
            raise ValueError("dependency_versions cannot be empty")
        for package, version in dependencies.items():
            _require_nonempty_string(package, "dependency package")
            _require_nonempty_string(version, f"dependency_versions.{package}")
        object.__setattr__(
            self, "dependency_versions", FrozenMetadata(dependencies)
        )
        if (
            isinstance(self.cadence_minutes, bool)
            or not isinstance(self.cadence_minutes, (int, float))
            or not math.isfinite(float(self.cadence_minutes))
            or float(self.cadence_minutes) <= 0.0
        ):
            raise ValueError("cadence_minutes must be positive and finite")
        object.__setattr__(self, "cadence_minutes", float(self.cadence_minutes))
        if self.training_outcome not in _TRAINING_OUTCOMES:
            raise ValueError("unsupported training_outcome")
        if self.checksum:
            if len(self.checksum) != 64 or any(
                character not in "0123456789abcdef" for character in self.checksum
            ):
                raise ValueError("checksum must be a lowercase SHA-256 digest")
        if self.status not in _MODEL_STATUSES:
            raise ValueError("unsupported model status")
        if self.metric_domain != "weather_vs_truth":
            raise ValueError("metric_domain must be weather_vs_truth")

    def to_dict(self) -> dict[str, Any]:
        values = asdict(self)
        key = values.pop("key")
        compatibility = values.pop("compatibility")
        values["feature_columns"] = list(self.feature_columns)
        values["residual_bounds"] = list(self.residual_bounds)
        return {"schema_version": 1, **key, **values, **compatibility}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ModelMetadata":
        if not isinstance(payload, Mapping):
            raise ValueError("metadata must be an object")
        values = dict(payload)
        if values.pop("schema_version", None) != 1:
            raise ValueError("unsupported metadata schema")
        key = ModelKey(**{name: values.pop(name) for name in _KEY_FIELDS})
        compatibility = ModelCompatibility(
            **{name: values.pop(name) for name in _COMPATIBILITY_FIELDS}
        )
        values["feature_columns"] = tuple(values["feature_columns"])
        values["residual_bounds"] = tuple(values["residual_bounds"])
        values.setdefault("full_fit_metrics", None)
        values.setdefault("evaluation_set_hash", None)
        values.setdefault("checksum", "")
        values.setdefault("status", "candidate")
        values.setdefault("metric_domain", "weather_vs_truth")
        values.setdefault("last_attempted_input_data_hash", None)
        values.setdefault("training_outcome", "legacy")
        missing_training_contract = "training_contract_hash" not in values
        if missing_training_contract:
            values["training_contract_hash"] = _LEGACY_TRAINING_CONTRACT_HASH
        missing_backend = "backend_id" not in values
        if missing_backend:
            values["backend_id"] = _LEGACY_BACKEND_ID
        return cls(
            key=key,
            compatibility=compatibility,
            _allow_legacy_training_contract=missing_training_contract,
            _allow_legacy_backend=missing_backend,
            **values,
        )


@dataclass(frozen=True)
class ModelCandidate:
    key: ModelKey
    bundle: ModelBundle
    metadata: ModelMetadata
    _integrity_hash: str = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not isinstance(self.key, ModelKey):
            raise TypeError("key must be a ModelKey")
        if not isinstance(self.bundle, ModelBundle):
            raise TypeError("bundle must be a ModelBundle")
        if not isinstance(self.metadata, ModelMetadata):
            raise TypeError("metadata must be ModelMetadata")
        if self.metadata.key != self.key:
            raise ValueError("candidate metadata scope does not match key")
        self.bundle.metadata = FrozenMetadata(self.bundle.metadata)
        _validate_bundle(self.key, self.bundle, self.metadata)
        object.__setattr__(self, "_integrity_hash", self._current_integrity_hash())

    def _current_integrity_hash(self) -> str:
        payload = {
            "key": {name: getattr(self.key, name) for name in _KEY_FIELDS},
            "metadata": self.metadata.to_dict(),
            "bundle": {
                "target_name": self.bundle.target_name,
                "feature_columns": list(self.bundle.feature_columns),
                "cadence_minutes": self.bundle.cadence_minutes,
                "residual_bounds": self.bundle.residual_bounds,
                "line_id": self.bundle.line_id,
                "tower_id": self.bundle.tower_id,
                "metadata": dict(self.bundle.metadata),
            },
        }
        return hashlib.sha256(_json_bytes(payload)).hexdigest()

    def validate_integrity(self) -> None:
        _validate_bundle(self.key, self.bundle, self.metadata)
        if self._current_integrity_hash() != self._integrity_hash:
            raise ValueError("candidate integrity validation failed")


def candidate_from_training_result(
    result,
    *,
    project_id: str,
    model_version: str,
    compatibility: ModelCompatibility,
) -> ModelCandidate:
    for attribute in (
        "target",
        "line_id",
        "tower_id",
        "bundle",
        "metrics",
        "metadata",
        "backend_id",
    ):
        if not hasattr(result, attribute):
            raise TypeError(f"training result is missing {attribute}")
    if not isinstance(result.metadata, Mapping):
        raise TypeError("training result metadata must be a mapping")
    key = ModelKey(
        project_id=project_id,
        line_id=str(result.line_id),
        tower_id=str(result.tower_id),
        target=str(result.target),
    )
    source = result.metadata
    result_contract_hash = getattr(result, "training_contract_hash", None)
    result_backend_id = getattr(result, "backend_id", None)
    result_training_outcome = getattr(result, "training_outcome", None)
    source_contract_hash = source.get("training_contract_hash")
    _require_nonempty_string(
        result_contract_hash,
        "training result training_contract_hash",
    )
    if source_contract_hash != result_contract_hash:
        raise ValueError("training result and metadata contracts differ")
    bundle_contract_hash = result.bundle.metadata.get(
        "training_contract_hash"
    )
    if bundle_contract_hash != result_contract_hash:
        raise ValueError("training result and bundle training contracts differ")
    _require_nonempty_string(result_backend_id, "training result backend_id")
    if (
        source.get("backend_id") != result_backend_id
        or result.bundle.metadata.get("backend_id") != result_backend_id
    ):
        raise ValueError("training result backend fields differ")
    if result_backend_id != SEALED_XGBOOST_BACKEND_ID:
        raise ValueError("training result is not from the sealed production backend")
    if (
        result_training_outcome not in _TRAINING_OUTCOMES - {"legacy"}
        or source.get("training_outcome") != result_training_outcome
        or result.bundle.metadata.get("training_outcome")
        != result_training_outcome
    ):
        raise ValueError("training result outcome fields differ")
    metadata = ModelMetadata(
        key=key,
        model_version=model_version,
        feature_columns=tuple(result.bundle.feature_columns),
        training_params=source["training_params"],
        random_seed=source.get("random_state"),
        time_start=source["time_start"],
        time_end=source["time_end"],
        sample_count=source["sample_count"],
        evaluation_mode=source["evaluation_mode"],
        metrics=dict(result.metrics),
        full_fit_metrics=source.get("full_fit_metrics"),
        residual_bounds=tuple(result.bundle.residual_bounds),
        input_data_hash=source["input_data_hash"],
        evaluation_set_hash=source.get("evaluation_set_hash"),
        compatibility=compatibility,
        dependency_versions=source["dependency_versions"],
        cadence_minutes=result.bundle.cadence_minutes,
        training_contract_hash=result_contract_hash,
        backend_id=result_backend_id,
        training_outcome=result_training_outcome,
    )
    return ModelCandidate(key=key, bundle=result.bundle, metadata=metadata)


@dataclass(frozen=True)
class PromotionDecision:
    promoted: bool
    reason: str
    metadata: Optional[ModelMetadata] = None


@dataclass(frozen=True)
class ModelLoadResult:
    bundle: Optional[ModelBundle]
    metadata: Optional[ModelMetadata]
    fallback_reason: str = ""


@dataclass(frozen=True)
class ModelAttempt:
    key: ModelKey
    input_data_hash: str
    evaluation_set_hash: Optional[str]
    policy_version: str
    min_mae_improvement: float
    training_contract_hash: str
    feature_version: str
    champion_context_hash: Optional[str] = None

    def __post_init__(self) -> None:
        if not isinstance(self.key, ModelKey):
            raise TypeError("key must be a ModelKey")
        for name in (
            "input_data_hash",
            "policy_version",
            "training_contract_hash",
            "feature_version",
        ):
            _require_nonempty_string(getattr(self, name), name)
        for name in ("evaluation_set_hash", "champion_context_hash"):
            value = getattr(self, name)
            if value is not None:
                _require_nonempty_string(value, name)
        if (
            isinstance(self.min_mae_improvement, bool)
            or not isinstance(self.min_mae_improvement, (int, float))
            or not math.isfinite(float(self.min_mae_improvement))
            or float(self.min_mae_improvement) < 0.0
        ):
            raise ValueError("min_mae_improvement must be finite and non-negative")
        object.__setattr__(
            self,
            "min_mae_improvement",
            float(self.min_mae_improvement),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            **{name: getattr(self.key, name) for name in _KEY_FIELDS},
            "input_data_hash": self.input_data_hash,
            "evaluation_set_hash": self.evaluation_set_hash,
            "policy_version": self.policy_version,
            "min_mae_improvement": self.min_mae_improvement,
            "training_contract_hash": self.training_contract_hash,
            "feature_version": self.feature_version,
            "champion_context_hash": self.champion_context_hash,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ModelAttempt":
        values = dict(payload)
        key = ModelKey(**{name: values.pop(name) for name in _KEY_FIELDS})
        return cls(key=key, **values)

    @property
    def fingerprint(self) -> str:
        return hashlib.sha256(_json_bytes(self.to_dict())).hexdigest()


def candidate_admission_reason(
    *,
    evaluation_mode: str,
    evaluation_set_hash: Optional[str],
    metrics: Mapping[str, float],
    full_fit_metrics: Optional[Mapping[str, float]],
    training_outcome: str = "trained",
) -> str:
    if training_outcome not in _TRAINING_OUTCOMES:
        raise ValueError("unsupported training_outcome")
    if training_outcome == "operational_fallback":
        return "operational_training_fallback"
    if evaluation_mode != "full_fit" and evaluation_set_hash is None:
        return "missing_evaluation_set_hash"
    candidate_metrics = (
        full_fit_metrics if evaluation_mode == "full_fit" else metrics
    ) or {}
    if candidate_metrics["corrected_mae"] >= candidate_metrics["baseline_mae"]:
        return "candidate_not_better_than_physical"
    return ""


def _validate_bundle(
    key: ModelKey, bundle: ModelBundle, metadata: ModelMetadata
) -> None:
    if bundle.target_name != key.target:
        raise ValueError("bundle target does not match key")
    if str(bundle.line_id) != key.line_id:
        raise ValueError("bundle line scope does not match key")
    if str(bundle.tower_id) != key.tower_id:
        raise ValueError("bundle tower scope does not match key")
    if tuple(bundle.feature_columns) != metadata.feature_columns:
        raise ValueError("bundle feature columns do not match metadata")
    if not math.isclose(
        float(bundle.cadence_minutes),
        metadata.cadence_minutes,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError("bundle cadence does not match metadata")
    if bundle.residual_bounds is None or tuple(bundle.residual_bounds) != (
        metadata.residual_bounds
    ):
        raise ValueError("bundle residual bounds do not match metadata")
    if not isinstance(bundle.metadata, Mapping):
        raise ValueError("bundle metadata must be a mapping")
    bundle_contract_hash = bundle.metadata.get("training_contract_hash")
    if (
        metadata.training_contract_hash != _LEGACY_TRAINING_CONTRACT_HASH
        and bundle_contract_hash != metadata.training_contract_hash
    ):
        raise ValueError("bundle training contract does not match metadata")
    bundle_backend_id = bundle.metadata.get("backend_id")
    if (
        metadata.backend_id != _LEGACY_BACKEND_ID
        and bundle_backend_id != metadata.backend_id
    ):
        raise ValueError("bundle backend does not match metadata")
    bundle_training_outcome = bundle.metadata.get("training_outcome", "trained")
    if (
        metadata.training_outcome != "legacy"
        and bundle_training_outcome != metadata.training_outcome
    ):
        raise ValueError("bundle training outcome does not match metadata")
    if not hasattr(bundle.model, "predict"):
        raise ValueError("bundle model must provide predict")


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_bytes(payload: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            payload,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")


def _write_bytes(path: Path, payload: bytes) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags, _PRIVATE_FILE_MODE)
    try:
        stream = os.fdopen(descriptor, "wb")
    except Exception:
        os.close(descriptor)
        raise
    with stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    _enforce_private_mode(path, _PRIVATE_FILE_MODE)


def _fsync_directory(path: Path) -> bool:
    if os.name != "posix":
        return False
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return True


class UnsafeModelPathError(OSError):
    pass


def _enforce_private_mode(path: Path, expected_mode: int) -> bool:
    if os.name != "posix":
        return False
    try:
        os.chmod(path, expected_mode)
    except Exception:
        pass
    try:
        actual_mode = stat.S_IMODE(path.lstat().st_mode)
    except OSError as exc:
        raise UnsafeModelPathError("cannot validate private permissions") from exc
    if actual_mode != expected_mode:
        raise UnsafeModelPathError("model path permissions are not private")
    return True


def _verify_private_regular_file(path: Path, expected_mode: int) -> None:
    try:
        _enforce_private_mode(path, expected_mode)
    except Exception:
        # A post-commit helper may fail even when the replaced inode is valid.
        # The direct lstat below remains the authority for the committed path.
        pass
    try:
        info = path.lstat()
    except OSError as exc:
        raise UnsafeModelPathError(
            "cannot validate replaced model artifact"
        ) from exc
    if (
        not stat.S_ISREG(info.st_mode)
        or stat.S_ISLNK(info.st_mode)
        or stat.S_IMODE(info.st_mode) != expected_mode
    ):
        raise UnsafeModelPathError(
            "replaced model artifact is not a private regular file"
        )


def _ensure_private_regular_file(path: Path) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags, _PRIVATE_FILE_MODE)
    except FileExistsError:
        pass
    except OSError as exc:
        raise UnsafeModelPathError("cannot create private model file") from exc
    else:
        os.close(descriptor)
    try:
        info = path.lstat()
    except OSError as exc:
        raise UnsafeModelPathError("cannot validate private model file") from exc
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
        raise UnsafeModelPathError("model file must be a regular file")
    _enforce_private_mode(path, _PRIVATE_FILE_MODE)


class ModelRegistry:
    def __init__(
        self,
        model_dir: Path | str,
        *,
        min_mae_improvement: float = 0.0,
        max_generations: int = 2,
        max_attempt_records: int = 64,
    ):
        if (
            isinstance(max_generations, bool)
            or not isinstance(max_generations, int)
            or max_generations < 1
        ):
            raise ValueError("max_generations must be a positive integer")
        if (
            isinstance(max_attempt_records, bool)
            or not isinstance(max_attempt_records, int)
            or not 1 <= max_attempt_records <= _MAX_ATTEMPT_RECORD_LIMIT
        ):
            raise ValueError(
                "max_attempt_records must be an integer between 1 and "
                f"{_MAX_ATTEMPT_RECORD_LIMIT}"
            )
        configured_root = Path(model_dir).expanduser()
        try:
            canonical_root = configured_root.resolve(strict=False)
            canonical_root.mkdir(
                mode=_PRIVATE_DIRECTORY_MODE,
                parents=True,
                exist_ok=True,
            )
            canonical_root = canonical_root.resolve(strict=True)
        except OSError as exc:
            raise ValueError("model_dir must resolve to a writable directory") from exc
        if not canonical_root.is_dir():
            raise ValueError("model_dir must resolve to a directory")
        self.model_dir = canonical_root
        try:
            self._safe_directory(self.model_dir, create=False)
        except UnsafeModelPathError as exc:
            raise ValueError(
                "model_dir must resolve to a private directory"
            ) from exc
        if (
            isinstance(min_mae_improvement, bool)
            or not isinstance(min_mae_improvement, (int, float))
            or not math.isfinite(float(min_mae_improvement))
            or float(min_mae_improvement) < 0.0
        ):
            raise ValueError("min_mae_improvement must be finite and non-negative")
        self.min_mae_improvement = float(min_mae_improvement)
        self.max_generations = max_generations
        self.max_attempt_records = max_attempt_records

    def _safe_directory(self, path: Path, *, create: bool) -> bool:
        try:
            relative = path.relative_to(self.model_dir)
        except ValueError as exc:
            raise UnsafeModelPathError("path escapes canonical model root") from exc
        try:
            root_info = self.model_dir.lstat()
            if stat.S_ISLNK(root_info.st_mode) or not stat.S_ISDIR(
                root_info.st_mode
            ):
                raise UnsafeModelPathError("canonical model root is unsafe")
            if self.model_dir.resolve(strict=True) != self.model_dir:
                raise UnsafeModelPathError("canonical model root changed")
            _enforce_private_mode(
                self.model_dir,
                _PRIVATE_DIRECTORY_MODE,
            )

            current = self.model_dir
            for component in relative.parts:
                current = current / component
                try:
                    info = current.lstat()
                except FileNotFoundError:
                    if not create:
                        return False
                    try:
                        current.mkdir(mode=_PRIVATE_DIRECTORY_MODE)
                    except FileExistsError:
                        pass
                    info = current.lstat()
                if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
                    raise UnsafeModelPathError(
                        f"unsafe model directory component: {component}"
                    )
                if current.resolve(strict=True) != current:
                    raise UnsafeModelPathError(
                        f"model directory component escaped root: {component}"
                    )
                _enforce_private_mode(current, _PRIVATE_DIRECTORY_MODE)
        except UnsafeModelPathError:
            raise
        except OSError as exc:
            raise UnsafeModelPathError("cannot validate model directory") from exc
        return True

    def _target_dir(self, key: ModelKey) -> Path:
        if not isinstance(key, ModelKey):
            raise TypeError("key must be a ModelKey")
        return (
            self.model_dir
            / key.project_id
            / key.line_id
            / key.tower_id
            / key.target
        )

    def path_for(self, key: ModelKey) -> Path:
        return self._target_dir(key) / "model.joblib"

    def metadata_path_for(self, key: ModelKey) -> Path:
        return self._target_dir(key) / "metadata.json"

    def manifest_path_for(self, key: ModelKey) -> Path:
        return self._target_dir(key) / "manifest.json"

    def attempt_path_for(self, key: ModelKey) -> Path:
        target_dir = self._target_dir(key)
        return target_dir.parent / f".{key.target}.attempts.json"

    def _generation_root(self, key: ModelKey) -> Path:
        target_dir = self._target_dir(key)
        return target_dir.parent / f".{key.target}.generations"

    def _lock_for(self, key: ModelKey) -> FileLock:
        lock_path = (
            self.model_dir
            / ".locks"
            / key.project_id
            / key.line_id
            / key.tower_id
            / f"{key.target}.lock"
        )
        self._safe_directory(lock_path.parent, create=True)
        _ensure_private_regular_file(lock_path)
        return FileLock(lock_path, mode=_PRIVATE_FILE_MODE)

    @staticmethod
    def _champion_context_hash(
        champion: Optional[ModelMetadata],
    ) -> Optional[str]:
        if champion is None:
            return None
        if not isinstance(champion, ModelMetadata):
            raise TypeError("champion must be ModelMetadata or None")
        payload = {
            "model_version": champion.model_version,
            "checksum": champion.checksum,
            "status": champion.status,
            "evaluation_mode": champion.evaluation_mode,
            "metrics": champion.metrics,
            "full_fit_metrics": champion.full_fit_metrics,
            "input_data_hash": champion.input_data_hash,
            "evaluation_set_hash": champion.evaluation_set_hash,
            "training_contract_hash": champion.training_contract_hash,
            "compatibility": champion.compatibility.to_dict(),
        }
        return hashlib.sha256(_json_bytes(payload)).hexdigest()

    def build_attempt(
        self,
        key: ModelKey,
        *,
        input_data_hash: str,
        evaluation_set_hash: Optional[str],
        training_contract_hash: str,
        feature_version: str,
        champion: Optional[ModelMetadata] = None,
    ) -> ModelAttempt:
        if not isinstance(key, ModelKey):
            raise TypeError("key must be a ModelKey")
        if champion is not None and champion.key != key:
            raise ValueError("champion scope does not match attempt key")
        return ModelAttempt(
            key=key,
            input_data_hash=input_data_hash,
            evaluation_set_hash=evaluation_set_hash,
            policy_version=_ATTEMPT_POLICY_VERSION,
            min_mae_improvement=self.min_mae_improvement,
            training_contract_hash=training_contract_hash,
            feature_version=feature_version,
            champion_context_hash=self._champion_context_hash(champion),
        )

    def _read_attempt_ledger_locked(self, key: ModelKey) -> list[dict[str, Any]]:
        path = self.attempt_path_for(key)
        if not self._safe_directory(path.parent, create=False):
            return []
        try:
            self._validate_artifact_path(path)
            if not path.exists():
                return []
            payload = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict) or set(payload) != _ATTEMPT_LEDGER_FIELDS:
                raise ValueError("invalid attempt ledger schema")
            if payload["schema_version"] != 1:
                raise ValueError("unsupported attempt ledger schema")
            if tuple(payload[name] for name in _KEY_FIELDS) != tuple(
                getattr(key, name) for name in _KEY_FIELDS
            ):
                raise ValueError("attempt ledger scope mismatch")
            entries = payload["entries"]
            if (
                not isinstance(entries, list)
                or len(entries) > _MAX_ATTEMPT_RECORD_LIMIT
            ):
                raise ValueError("invalid attempt ledger entries")
            normalized = []
            fingerprints = set()
            for raw_entry in entries:
                if not isinstance(raw_entry, dict) or set(raw_entry) != (
                    _ATTEMPT_ENTRY_FIELDS
                ):
                    raise ValueError("invalid attempt ledger entry")
                entry = dict(raw_entry)
                reason = entry.pop("reason")
                fingerprint = entry.pop("fingerprint")
                attempt = ModelAttempt.from_dict(entry)
                if (
                    attempt.key != key
                    or fingerprint != attempt.fingerprint
                    or fingerprint in fingerprints
                    or reason not in _DETERMINISTIC_REJECTION_REASONS
                ):
                    raise ValueError("invalid attempt ledger entry contract")
                fingerprints.add(fingerprint)
                normalized.append(raw_entry)
            return normalized
        except (
            KeyError,
            OSError,
            TypeError,
            ValueError,
            UnicodeDecodeError,
            json.JSONDecodeError,
            UnsafeModelPathError,
        ):
            return []

    def _write_attempt_ledger_locked(
        self,
        key: ModelKey,
        entries: list[dict[str, Any]],
    ) -> None:
        path = self.attempt_path_for(key)
        self._safe_directory(path.parent, create=True)
        self._validate_artifact_path(path)
        token = uuid.uuid4().hex
        temp_path = path.parent / f".{key.target}.attempts.{token}.tmp"
        payload = {
            "schema_version": 1,
            **{name: getattr(key, name) for name in _KEY_FIELDS},
            "entries": entries[-self.max_attempt_records :],
        }
        committed = False
        try:
            _write_bytes(temp_path, _json_bytes(payload))
            os.replace(temp_path, path)
            committed = True
            try:
                _verify_private_regular_file(path, _PRIVATE_FILE_MODE)
            except Exception:
                try:
                    path.unlink()
                    _fsync_directory(path.parent)
                except Exception:
                    pass
                raise
            try:
                _fsync_directory(path.parent)
            except Exception:
                pass
        finally:
            if not committed:
                try:
                    if temp_path.exists() or temp_path.is_symlink():
                        temp_path.unlink()
                except Exception:
                    pass

    def _record_attempt_rejection_locked(
        self,
        attempt: ModelAttempt,
        reason: str,
    ) -> None:
        entries = self._read_attempt_ledger_locked(attempt.key)
        entries = [
            entry
            for entry in entries
            if entry["fingerprint"] != attempt.fingerprint
        ]
        entries.append(
            {
                **attempt.to_dict(),
                "fingerprint": attempt.fingerprint,
                "reason": reason,
            }
        )
        self._write_attempt_ledger_locked(attempt.key, entries)

    def was_rejected(self, attempt: ModelAttempt) -> bool:
        if not isinstance(attempt, ModelAttempt):
            raise TypeError("attempt must be a ModelAttempt")
        try:
            with self._lock_for(attempt.key):
                return any(
                    entry["fingerprint"] == attempt.fingerprint
                    for entry in self._read_attempt_ledger_locked(attempt.key)
                )
        except Exception:
            return False

    @staticmethod
    def _fallback(reason: str) -> ModelLoadResult:
        return ModelLoadResult(None, None, reason)

    def _validate_generation_location(
        self, key: ModelKey, generation_dir: Path
    ) -> bool:
        generation_root = self._generation_root(key)
        if not self._safe_directory(generation_root, create=False):
            return False
        try:
            info = generation_dir.lstat()
        except OSError:
            return False
        if (
            generation_dir.parent != generation_root
            or stat.S_ISLNK(info.st_mode)
            or not stat.S_ISDIR(info.st_mode)
            or generation_dir.resolve(strict=True) != generation_dir
        ):
            return False
        _enforce_private_mode(generation_dir, _PRIVATE_DIRECTORY_MODE)
        return True

    def _active_generation(self, key: ModelKey) -> Path:
        target_dir = self._target_dir(key)
        generation_root = self._generation_root(key)
        if not self._safe_directory(generation_root, create=False):
            raise UnsafeModelPathError("active generation root is missing")
        try:
            link_value = os.readlink(target_dir)
        except OSError as exc:
            raise UnsafeModelPathError("active model pointer is unreadable") from exc
        relative_link = Path(link_value)
        if (
            relative_link.is_absolute()
            or len(relative_link.parts) != 2
            or relative_link.parts[0] != generation_root.name
            or relative_link.parts[1] in {".", ".."}
            or "\\" in link_value
        ):
            raise UnsafeModelPathError("active model pointer is unsafe")
        generation_dir = target_dir.parent / relative_link
        if not self._validate_generation_location(key, generation_dir):
            raise UnsafeModelPathError("active generation escaped model root")
        return generation_dir

    @staticmethod
    def _complete_generation(path: Path) -> bool:
        try:
            for artifact_name in _GENERATION_ARTIFACTS:
                if not stat.S_ISREG((path / artifact_name).lstat().st_mode):
                    return False
            return not any(
                child.name.endswith(".tmp") for child in path.iterdir()
            )
        except OSError:
            return False

    def _remove_generation_entry(self, key: ModelKey, path: Path) -> None:
        generation_root = self._generation_root(key)
        if path.parent != generation_root:
            raise UnsafeModelPathError("generation cleanup escaped its root")
        if not self._safe_directory(generation_root, create=False):
            raise UnsafeModelPathError("generation root is missing")
        try:
            info = path.lstat()
        except FileNotFoundError:
            return
        if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
            path.unlink()
            return
        if path.resolve(strict=True) != path:
            raise UnsafeModelPathError("generation cleanup target escaped root")
        shutil.rmtree(path)

    def _prune_generations(
        self, key: ModelKey, active_generation: Path
    ) -> None:
        generation_root = self._generation_root(key)
        if not self._validate_generation_location(key, active_generation):
            raise UnsafeModelPathError("active generation is unsafe")
        history = []
        with os.scandir(generation_root) as entries:
            for entry in entries:
                path = generation_root / entry.name
                if path == active_generation:
                    continue
                if entry.is_symlink() or not entry.is_dir(follow_symlinks=False):
                    self._remove_generation_entry(key, path)
                    continue
                if not self._complete_generation(path):
                    self._remove_generation_entry(key, path)
                    continue
                history.append(
                    (entry.stat(follow_symlinks=False).st_mtime_ns, path)
                )
        history.sort(key=lambda item: item[0], reverse=True)
        for _, obsolete in history[self.max_generations - 1 :]:
            self._remove_generation_entry(key, obsolete)

    @staticmethod
    def _validate_artifact_path(path: Path) -> None:
        try:
            info = path.lstat()
        except FileNotFoundError:
            return
        except OSError as exc:
            raise UnsafeModelPathError(
                "model artifact cannot be validated"
            ) from exc
        if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
            raise UnsafeModelPathError(
                "model artifact must be a regular file"
            )
        _enforce_private_mode(path, _PRIVATE_FILE_MODE)

    def _read_generation(
        self,
        key: ModelKey,
        generation_dir: Path,
        expected_compatibility: ModelCompatibility,
        expected_training_contract_hash: str,
        expected_backend_id: str,
    ) -> ModelLoadResult:
        if not self._validate_generation_location(key, generation_dir):
            return self._fallback("corrupt_manifest")
        manifest_path = generation_dir / "manifest.json"
        metadata_path = generation_dir / "metadata.json"
        model_path = generation_dir / "model.joblib"
        for artifact_path in (manifest_path, metadata_path, model_path):
            self._validate_artifact_path(artifact_path)
        try:
            manifest_bytes = manifest_path.read_bytes()
            manifest = json.loads(manifest_bytes.decode("utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            return self._fallback("corrupt_manifest")
        if not isinstance(manifest, dict) or not _MANIFEST_FIELDS.issubset(
            manifest
        ):
            return self._fallback("corrupt_manifest")
        if (
            manifest["schema_version"] != 1
            or manifest["model_file"] != "model.joblib"
            or manifest["metadata_file"] != "metadata.json"
        ):
            return self._fallback("corrupt_manifest")
        manifest_key = tuple(manifest[name] for name in _KEY_FIELDS)
        expected_key = tuple(getattr(key, name) for name in _KEY_FIELDS)
        if manifest_key != expected_key:
            return self._fallback("manifest_scope_mismatch")

        try:
            metadata_bytes = metadata_path.read_bytes()
        except OSError:
            return self._fallback("corrupt_metadata")
        if hashlib.sha256(metadata_bytes).hexdigest() != manifest["metadata_checksum"]:
            return self._fallback("corrupt_metadata")
        try:
            metadata = ModelMetadata.from_dict(
                json.loads(metadata_bytes.decode("utf-8"))
            )
        except (
            KeyError,
            TypeError,
            ValueError,
            UnicodeDecodeError,
            json.JSONDecodeError,
        ):
            return self._fallback("corrupt_metadata")
        if metadata.key != key:
            return self._fallback("metadata_scope_mismatch")
        if metadata.model_version != manifest["model_version"]:
            return self._fallback("metadata_version_mismatch")
        if metadata.checksum != manifest["model_checksum"]:
            return self._fallback("metadata_checksum_mismatch")

        for field_name, actual in metadata.compatibility.to_dict().items():
            expected = getattr(expected_compatibility, field_name)
            if actual != expected:
                return self._fallback(f"incompatible_{field_name}")

        if metadata.training_contract_hash != expected_training_contract_hash:
            return self._fallback("incompatible_training_contract_hash")
        if metadata.backend_id != expected_backend_id:
            return self._fallback("incompatible_backend_id")

        try:
            model_checksum = _sha256_path(model_path)
        except OSError:
            return self._fallback("corrupt_model")
        if model_checksum != manifest["model_checksum"]:
            return self._fallback("corrupt_model")
        try:
            bundle = joblib.load(model_path)
        except Exception:
            return self._fallback("corrupt_model")
        if not isinstance(bundle, ModelBundle):
            return self._fallback("corrupt_model")
        try:
            _validate_bundle(key, bundle, metadata)
        except (TypeError, ValueError, OverflowError):
            return self._fallback("invalid_model_contract")
        return ModelLoadResult(bundle, metadata, "")

    def _load_locked(
        self,
        key: ModelKey,
        expected_compatibility: ModelCompatibility,
        expected_training_contract_hash: str,
        expected_backend_id: str,
    ) -> ModelLoadResult:
        target_dir = self._target_dir(key)
        if not self._safe_directory(target_dir.parent, create=False):
            return self._fallback("model_not_found")
        if not target_dir.is_symlink():
            if target_dir.exists() or target_dir.is_symlink():
                return self._fallback("corrupt_manifest")
            return self._fallback("model_not_found")
        generation_dir = self._active_generation(key)
        return self._read_generation(
            key,
            generation_dir,
            expected_compatibility,
            expected_training_contract_hash,
            expected_backend_id,
        )

    def load(
        self,
        key: ModelKey,
        *,
        expected_compatibility: ModelCompatibility,
        expected_training_contract_hash: str,
        expected_backend_id: str,
    ) -> ModelLoadResult:
        if not isinstance(key, ModelKey):
            raise TypeError("key must be a ModelKey")
        if not isinstance(expected_compatibility, ModelCompatibility):
            raise TypeError(
                "expected_compatibility must be provided as ModelCompatibility"
            )
        _require_nonempty_string(
            expected_training_contract_hash,
            "expected_training_contract_hash",
        )
        _require_nonempty_string(expected_backend_id, "expected_backend_id")
        try:
            with self._lock_for(key):
                return self._load_locked(
                    key,
                    expected_compatibility,
                    expected_training_contract_hash,
                    expected_backend_id,
                )
        except UnsafeModelPathError:
            return self._fallback("unsafe_model_path")

    def load_many(
        self,
        keys,
        *,
        expected_compatibility: Mapping[ModelKey, ModelCompatibility],
        expected_training_contract_hash: Mapping[ModelKey, str],
        expected_backend_id: Mapping[ModelKey, str],
    ) -> dict[ModelKey, ModelLoadResult]:
        if not isinstance(expected_compatibility, Mapping):
            raise TypeError("expected_compatibility must map every ModelKey")
        if not isinstance(expected_training_contract_hash, Mapping):
            raise TypeError(
                "expected_training_contract_hash must map every ModelKey"
            )
        if not isinstance(expected_backend_id, Mapping):
            raise TypeError("expected_backend_id must map every ModelKey")
        normalized_keys = list(keys)
        if len(set(normalized_keys)) != len(normalized_keys):
            raise ValueError("keys cannot contain duplicates")
        for key in normalized_keys:
            if not isinstance(key, ModelKey):
                raise TypeError("keys must contain only ModelKey values")
            if key not in expected_compatibility:
                raise ValueError(f"missing compatibility expectation for {key}")
            if not isinstance(
                expected_compatibility[key], ModelCompatibility
            ):
                raise TypeError(
                    "each compatibility expectation must be ModelCompatibility"
                )
            if key not in expected_training_contract_hash:
                raise ValueError(
                    f"missing training contract expectation for {key}"
                )
            _require_nonempty_string(
                expected_training_contract_hash[key],
                f"expected_training_contract_hash[{key}]",
            )
            if key not in expected_backend_id:
                raise ValueError(f"missing backend expectation for {key}")
            _require_nonempty_string(
                expected_backend_id[key],
                f"expected_backend_id[{key}]",
            )

        loaded = {}
        for key in normalized_keys:
            try:
                loaded[key] = self.load(
                    key,
                    expected_compatibility=expected_compatibility[key],
                    expected_training_contract_hash=(
                        expected_training_contract_hash[key]
                    ),
                    expected_backend_id=expected_backend_id[key],
                )
            except Exception as exc:
                loaded[key] = self._fallback(
                    f"load_failed:{type(exc).__name__}"
                )
        return loaded

    def _publish_locked(
        self,
        candidate: ModelCandidate,
        status: str,
    ) -> ModelMetadata:
        key = candidate.key
        target_dir = self._target_dir(key)
        target_parent = target_dir.parent
        generation_root = self._generation_root(key)
        self._safe_directory(target_parent, create=True)
        self._safe_directory(generation_root, create=True)
        token = uuid.uuid4().hex
        generation_name = f"{candidate.metadata.model_version}-{token}"
        generation_dir = generation_root / generation_name
        generation_dir.mkdir(mode=_PRIVATE_DIRECTORY_MODE)
        _enforce_private_mode(generation_dir, _PRIVATE_DIRECTORY_MODE)
        temp_model = generation_dir / f".model.{token}.tmp"
        temp_metadata = generation_dir / f".metadata.{token}.tmp"
        temp_manifest = generation_dir / f".manifest.{token}.tmp"
        final_model = generation_dir / "model.joblib"
        final_metadata = generation_dir / "metadata.json"
        final_manifest = generation_dir / "manifest.json"
        temp_link = target_parent / f".{key.target}.{token}.tmp"
        committed = False
        try:
            _ensure_private_regular_file(temp_model)
            joblib.dump(candidate.bundle, temp_model, compress=3)
            _enforce_private_mode(temp_model, _PRIVATE_FILE_MODE)
            with temp_model.open("rb") as stream:
                os.fsync(stream.fileno())
            model_checksum = _sha256_path(temp_model)
            active_metadata = replace(
                candidate.metadata,
                checksum=model_checksum,
                status=status,
            )
            metadata_bytes = _json_bytes(active_metadata.to_dict())
            _write_bytes(temp_metadata, metadata_bytes)
            manifest = {
                "schema_version": 1,
                **{name: getattr(key, name) for name in _KEY_FIELDS},
                "model_version": active_metadata.model_version,
                "model_file": "model.joblib",
                "metadata_file": "metadata.json",
                "model_checksum": model_checksum,
                "metadata_checksum": hashlib.sha256(metadata_bytes).hexdigest(),
            }
            _write_bytes(temp_manifest, _json_bytes(manifest))
            os.replace(temp_model, final_model)
            os.replace(temp_metadata, final_metadata)
            os.replace(temp_manifest, final_manifest)
            _fsync_directory(generation_dir)
            _fsync_directory(generation_root)

            validated = self._read_generation(
                key,
                generation_dir,
                active_metadata.compatibility,
                active_metadata.training_contract_hash,
                active_metadata.backend_id,
            )
            if validated.bundle is None or validated.metadata != active_metadata:
                raise ValueError(
                    f"candidate validation failed: {validated.fallback_reason}"
                )
            relative_generation = os.path.relpath(
                generation_dir, start=target_parent
            )
            os.symlink(relative_generation, temp_link, target_is_directory=True)
            os.replace(temp_link, target_dir)
            committed = True
            try:
                _fsync_directory(target_parent)
            except Exception:
                pass
            try:
                self._prune_generations(key, generation_dir)
                _fsync_directory(generation_root)
            except Exception:
                pass
            return active_metadata
        finally:
            try:
                if temp_link.is_symlink() or temp_link.exists():
                    temp_link.unlink()
            except Exception:
                pass
            if not committed:
                try:
                    self._remove_generation_entry(key, generation_dir)
                except Exception:
                    pass

    def _publish_decision(
        self,
        candidate: ModelCandidate,
        *,
        status: str,
        reason: str,
        champion: Optional[ModelMetadata] = None,
    ) -> PromotionDecision:
        try:
            metadata = self._publish_locked(candidate, status=status)
        except UnsafeModelPathError:
            return PromotionDecision(False, "unsafe_model_path", champion)
        except Exception as exc:
            return PromotionDecision(
                False,
                f"publish_failed:{type(exc).__name__}",
                champion,
            )
        return PromotionDecision(True, reason, metadata)

    def _record_rejection_locked(
        self,
        current: ModelLoadResult,
        candidate: ModelCandidate,
        reason: str,
        attempt: Optional[ModelAttempt],
    ) -> PromotionDecision:
        champion = current.metadata
        if (
            attempt is None
            or reason not in _DETERMINISTIC_REJECTION_REASONS
        ):
            return PromotionDecision(False, reason, champion)
        current_attempt = self.build_attempt(
            candidate.key,
            input_data_hash=attempt.input_data_hash,
            evaluation_set_hash=attempt.evaluation_set_hash,
            training_contract_hash=attempt.training_contract_hash,
            feature_version=attempt.feature_version,
            champion=champion if current.bundle is not None else None,
        )
        try:
            self._record_attempt_rejection_locked(current_attempt, reason)
        except Exception as exc:
            return PromotionDecision(
                False,
                f"attempt_record_failed:{type(exc).__name__}",
                champion,
            )
        return PromotionDecision(False, reason, champion)

    def _validate_promotion_contract(
        self,
        candidate: ModelCandidate,
        attempt: Optional[ModelAttempt],
    ) -> None:
        candidate.validate_integrity()
        if (
            candidate.metadata.training_contract_hash
            == _LEGACY_TRAINING_CONTRACT_HASH
        ):
            raise ValueError("legacy training contract cannot be promoted")
        bundle_contract_hash = candidate.bundle.metadata.get(
            "training_contract_hash"
        )
        if bundle_contract_hash != candidate.metadata.training_contract_hash:
            raise ValueError("bundle training contract does not match metadata")
        if attempt is None:
            return
        if not isinstance(attempt, ModelAttempt):
            raise TypeError("attempt must be a ModelAttempt or None")
        if (
            attempt.key != candidate.key
            or attempt.input_data_hash != candidate.metadata.input_data_hash
            or attempt.evaluation_set_hash
            != candidate.metadata.evaluation_set_hash
            or attempt.training_contract_hash
            != candidate.metadata.training_contract_hash
            or attempt.training_contract_hash != bundle_contract_hash
            or attempt.feature_version
            != candidate.metadata.compatibility.feature_version
            or attempt.policy_version != _ATTEMPT_POLICY_VERSION
            or not math.isclose(
                attempt.min_mae_improvement,
                self.min_mae_improvement,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
        ):
            raise ValueError(
                "attempt contract does not match candidate, bundle, or registry"
            )

    def promote(
        self,
        candidate: ModelCandidate,
        *,
        attempt: Optional[ModelAttempt] = None,
    ) -> PromotionDecision:
        if not isinstance(candidate, ModelCandidate):
            raise TypeError("candidate must be a ModelCandidate")
        self._validate_promotion_contract(candidate, attempt)
        try:
            self._safe_directory(
                self._target_dir(candidate.key).parent, create=False
            )
            lock = self._lock_for(candidate.key)
        except UnsafeModelPathError:
            return PromotionDecision(False, "unsafe_model_path")
        with lock:
            self._validate_promotion_contract(candidate, attempt)
            admission_reason = candidate_admission_reason(
                evaluation_mode=candidate.metadata.evaluation_mode,
                evaluation_set_hash=candidate.metadata.evaluation_set_hash,
                metrics=candidate.metadata.metrics,
                full_fit_metrics=candidate.metadata.full_fit_metrics,
                training_outcome=candidate.metadata.training_outcome,
            )
            try:
                current = self._load_locked(
                    candidate.key,
                    candidate.metadata.compatibility,
                    candidate.metadata.training_contract_hash,
                    candidate.metadata.backend_id,
                )
            except UnsafeModelPathError:
                return PromotionDecision(False, "unsafe_model_path")
            if admission_reason:
                return self._record_rejection_locked(
                    current,
                    candidate,
                    admission_reason,
                    attempt,
                )
            if current.bundle is not None:
                champion = current.metadata
                if candidate.metadata.evaluation_mode == "full_fit":
                    return self._record_rejection_locked(
                        current,
                        candidate,
                        "full_fit_cannot_replace_champion",
                        attempt,
                    )
                if (
                    champion.status == "active_provisional"
                    and champion.evaluation_mode == "full_fit"
                ):
                    if (
                        candidate.metadata.model_version
                        == champion.model_version
                    ):
                        return self._record_rejection_locked(
                            current,
                            candidate,
                            "model_version_conflict",
                            attempt,
                        )
                    physical_improvement = (
                        candidate.metadata.metrics["baseline_mae"]
                        - candidate.metadata.metrics["corrected_mae"]
                    )
                    if (
                        physical_improvement
                        <= self.min_mae_improvement + 1e-12
                    ):
                        return self._record_rejection_locked(
                            current,
                            candidate,
                            "insufficient_mae_improvement",
                            attempt,
                        )
                    return self._publish_decision(
                        candidate,
                        status="active",
                        reason="promoted_from_provisional",
                        champion=champion,
                    )
                if (
                    champion.evaluation_mode == "full_fit"
                    or champion.evaluation_set_hash is None
                ):
                    return self._record_rejection_locked(
                        current,
                        candidate,
                        "champion_has_no_independent_evaluation",
                        attempt,
                    )
                if (
                    candidate.metadata.evaluation_set_hash
                    != champion.evaluation_set_hash
                ):
                    return self._record_rejection_locked(
                        current,
                        candidate,
                        "evaluation_set_mismatch",
                        attempt,
                    )
                if candidate.metadata.model_version == champion.model_version:
                    return self._record_rejection_locked(
                        current,
                        candidate,
                        "model_version_conflict",
                        attempt,
                    )
                improvement = (
                    champion.metrics["corrected_mae"]
                    - candidate.metadata.metrics["corrected_mae"]
                )
                if improvement <= self.min_mae_improvement + 1e-12:
                    return self._record_rejection_locked(
                        current,
                        candidate,
                        "insufficient_mae_improvement",
                        attempt,
                    )
                return self._publish_decision(
                    candidate,
                    status="active",
                    reason="promoted",
                    champion=champion,
                )
            return self._publish_decision(
                candidate,
                status="active_provisional",
                reason="promoted_provisional",
            )
