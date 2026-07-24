from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import stat
import uuid
from dataclasses import asdict, dataclass, replace
from pathlib import Path, PureWindowsPath
from typing import Any, Mapping, Optional

import joblib
from filelock import FileLock

from modules.ai_prediction import ModelBundle


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
    checksum: str = ""
    status: str = "candidate"
    metric_domain: str = "weather_vs_truth"
    last_attempted_input_data_hash: Optional[str] = None

    def __post_init__(self) -> None:
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
        object.__setattr__(self, "training_params", training_params)

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
        object.__setattr__(self, "metrics", metrics or {})
        object.__setattr__(self, "full_fit_metrics", full_fit_metrics)

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
        object.__setattr__(self, "dependency_versions", dependencies)
        if (
            isinstance(self.cadence_minutes, bool)
            or not isinstance(self.cadence_minutes, (int, float))
            or not math.isfinite(float(self.cadence_minutes))
            or float(self.cadence_minutes) <= 0.0
        ):
            raise ValueError("cadence_minutes must be positive and finite")
        object.__setattr__(self, "cadence_minutes", float(self.cadence_minutes))
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
        return cls(key=key, compatibility=compatibility, **values)


@dataclass(frozen=True)
class ModelCandidate:
    key: ModelKey
    bundle: ModelBundle
    metadata: ModelMetadata

    def __post_init__(self) -> None:
        if not isinstance(self.key, ModelKey):
            raise TypeError("key must be a ModelKey")
        if not isinstance(self.bundle, ModelBundle):
            raise TypeError("bundle must be a ModelBundle")
        if not isinstance(self.metadata, ModelMetadata):
            raise TypeError("metadata must be ModelMetadata")
        if self.metadata.key != self.key:
            raise ValueError("candidate metadata scope does not match key")
        _validate_bundle(self.key, self.bundle, self.metadata)


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


def candidate_admission_reason(
    *,
    evaluation_mode: str,
    evaluation_set_hash: Optional[str],
    metrics: Mapping[str, float],
    full_fit_metrics: Optional[Mapping[str, float]],
) -> str:
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
    ):
        if (
            isinstance(max_generations, bool)
            or not isinstance(max_generations, int)
            or max_generations < 1
        ):
            raise ValueError("max_generations must be a positive integer")
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

        for field, actual in metadata.compatibility.to_dict().items():
            expected = getattr(expected_compatibility, field)
            if actual != expected:
                return self._fallback(f"incompatible_{field}")

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
    ) -> ModelLoadResult:
        target_dir = self._target_dir(key)
        if not self._safe_directory(target_dir.parent, create=False):
            return self._fallback("model_not_found")
        if not target_dir.is_symlink():
            if target_dir.exists() or target_dir.is_symlink():
                return self._fallback("corrupt_manifest")
            return self._fallback("model_not_found")
        generation_dir = self._active_generation(key)
        return self._read_generation(key, generation_dir, expected_compatibility)

    def load(
        self,
        key: ModelKey,
        *,
        expected_compatibility: ModelCompatibility,
    ) -> ModelLoadResult:
        if not isinstance(key, ModelKey):
            raise TypeError("key must be a ModelKey")
        if not isinstance(expected_compatibility, ModelCompatibility):
            raise TypeError(
                "expected_compatibility must be provided as ModelCompatibility"
            )
        try:
            with self._lock_for(key):
                return self._load_locked(key, expected_compatibility)
        except UnsafeModelPathError:
            return self._fallback("unsafe_model_path")

    def load_many(
        self,
        keys,
        *,
        expected_compatibility: Mapping[ModelKey, ModelCompatibility],
    ) -> dict[ModelKey, ModelLoadResult]:
        if not isinstance(expected_compatibility, Mapping):
            raise TypeError("expected_compatibility must map every ModelKey")
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

        loaded = {}
        for key in normalized_keys:
            try:
                loaded[key] = self.load(
                    key,
                    expected_compatibility=expected_compatibility[key],
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
                key, generation_dir, active_metadata.compatibility
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
    ) -> PromotionDecision:
        champion = current.metadata
        if current.bundle is None or champion is None:
            return PromotionDecision(False, reason, champion)
        attempted_metadata = replace(
            champion,
            last_attempted_input_data_hash=candidate.metadata.input_data_hash,
        )
        try:
            recorded = self._publish_locked(
                ModelCandidate(
                    key=candidate.key,
                    bundle=current.bundle,
                    metadata=attempted_metadata,
                ),
                status=champion.status,
            )
        except UnsafeModelPathError:
            return PromotionDecision(False, "unsafe_model_path", champion)
        except Exception as exc:
            return PromotionDecision(
                False,
                f"attempt_record_failed:{type(exc).__name__}",
                champion,
            )
        return PromotionDecision(False, reason, recorded)

    def promote(self, candidate: ModelCandidate) -> PromotionDecision:
        if not isinstance(candidate, ModelCandidate):
            raise TypeError("candidate must be a ModelCandidate")
        admission_reason = candidate_admission_reason(
            evaluation_mode=candidate.metadata.evaluation_mode,
            evaluation_set_hash=candidate.metadata.evaluation_set_hash,
            metrics=candidate.metadata.metrics,
            full_fit_metrics=candidate.metadata.full_fit_metrics,
        )
        try:
            self._safe_directory(
                self._target_dir(candidate.key).parent, create=False
            )
            lock = self._lock_for(candidate.key)
        except UnsafeModelPathError:
            return PromotionDecision(False, "unsafe_model_path")
        with lock:
            try:
                current = self._load_locked(
                    candidate.key, candidate.metadata.compatibility
                )
            except UnsafeModelPathError:
                return PromotionDecision(False, "unsafe_model_path")
            if admission_reason:
                return self._record_rejection_locked(
                    current,
                    candidate,
                    admission_reason,
                )
            if current.bundle is not None:
                champion = current.metadata
                if candidate.metadata.evaluation_mode == "full_fit":
                    return self._record_rejection_locked(
                        current,
                        candidate,
                        "full_fit_cannot_replace_champion",
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
                    )
                if (
                    candidate.metadata.evaluation_set_hash
                    != champion.evaluation_set_hash
                ):
                    return self._record_rejection_locked(
                        current,
                        candidate,
                        "evaluation_set_mismatch",
                    )
                if candidate.metadata.model_version == champion.model_version:
                    return self._record_rejection_locked(
                        current,
                        candidate,
                        "model_version_conflict",
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
