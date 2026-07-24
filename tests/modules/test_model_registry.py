import json
import multiprocessing
import os
import stat
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import modules.model_registry as model_registry
from modules.ai_prediction import ModelBundle
from modules.ai_training import ResidualTrainer
from modules.model_registry import (
    ModelCandidate,
    ModelCompatibility,
    ModelKey,
    ModelMetadata,
    ModelRegistry,
    candidate_from_training_result,
)


WEATHER_METRICS = {
    "baseline_mae": 2.0,
    "baseline_rmse": 2.5,
    "corrected_mae": 1.0,
    "corrected_rmse": 1.5,
}

POSIX_ONLY = pytest.mark.skipif(
    os.name != "posix", reason="POSIX permission semantics required"
)


def private_mode(path):
    return stat.S_IMODE(Path(path).lstat().st_mode)


def compatible_hashes(suffix="a"):
    return ModelCompatibility(
        dem_hash=f"dem-{suffix}",
        crs_hash=f"crs-{suffix}",
        coordinate_hash=f"coordinates-{suffix}",
        conductor_hash=f"conductor-{suffix}",
        feature_version=f"features-{suffix}",
        correction_config_hash=f"correction-{suffix}",
    )


def model_metadata(
    key,
    *,
    model_version="version-1",
    evaluation_mode="temporal_holdout",
    corrected_mae=1.0,
    evaluation_set_hash="evaluation-a",
    compatibility=None,
):
    metrics = dict(WEATHER_METRICS, corrected_mae=corrected_mae)
    full_fit_metrics = None
    if evaluation_mode == "full_fit":
        full_fit_metrics = metrics
        metrics = {}
        evaluation_set_hash = None
    physical_col = (
        "wind_speed_local"
        if key.target == "wind_speed"
        else "ambient_temp_local"
    )
    return ModelMetadata(
        key=key,
        model_version=model_version,
        feature_columns=(physical_col, "lag_1"),
        training_params={"max_depth": 3},
        random_seed=42,
        time_start="2025-01-01T00:00:00+08:00",
        time_end="2025-01-02T00:00:00+08:00",
        sample_count=4,
        evaluation_mode=evaluation_mode,
        metrics=metrics,
        full_fit_metrics=full_fit_metrics,
        residual_bounds=(-2.0, 2.0),
        input_data_hash="input-a",
        evaluation_set_hash=evaluation_set_hash,
        compatibility=compatibility or compatible_hashes(),
        dependency_versions={"python": "3.11", "joblib": "1.5.3"},
        cadence_minutes=30.0,
    )


class ConstantResidualModel:
    def __init__(self, value=0.5):
        self.value = float(value)

    def predict(self, features):
        return np.full(len(features), self.value, dtype=float)


class MeanResidualModel:
    def fit(self, features, target):
        self.value = float(np.mean(target))
        return self

    def predict(self, features):
        return np.full(len(features), self.value, dtype=float)


def model_candidate(key, *, model_value=0.5, **metadata_options):
    metadata = model_metadata(key, **metadata_options)
    bundle = ModelBundle(
        target_name=key.target,
        feature_columns=list(metadata.feature_columns),
        model=ConstantResidualModel(model_value),
        cadence_minutes=metadata.cadence_minutes,
        residual_bounds=metadata.residual_bounds,
        line_id=key.line_id,
        tower_id=key.tower_id,
        metadata={"training_contract": "task-9"},
    )
    return ModelCandidate(key=key, bundle=bundle, metadata=metadata)


def training_frame():
    physical = np.array([2.0, 3.0, 4.0, 5.0])
    return pd.DataFrame(
        {
            "line_id": ["line-a"] * 4,
            "tower_id": ["001"] * 4,
            "timestamp": pd.to_datetime(
                [
                    "2025-01-01 00:00",
                    "2025-01-01 00:30",
                    "2025-01-02 00:00",
                    "2025-01-02 00:30",
                ],
                utc=True,
            ),
            "source_file_hash_physical": ["physical-a"] * 4,
            "source_file_hash_truth": ["truth-a"] * 4,
            "wind_speed_local": physical,
            "wind_speed_truth": physical + np.array([0.0, 1.0, 2.0, 3.0]),
        }
    )


def _process_promote_worker(
    model_dir,
    model_version,
    corrected_mae,
    model_value,
    start_event,
    result_queue,
):
    if not start_event.wait(10):
        raise TimeoutError("promotion start event was not released")
    key = ModelKey("project-a", "line-a", "001", "wind_speed")
    decision = ModelRegistry(model_dir).promote(
        model_candidate(
            key,
            model_version=model_version,
            corrected_mae=corrected_mae,
            model_value=model_value,
        )
    )
    result_queue.put((model_version, decision.promoted, decision.reason))


def _loaded_model_snapshot(model_dir):
    key = ModelKey("project-a", "line-a", "001", "wind_speed")
    loaded = ModelRegistry(model_dir).load(
        key, expected_compatibility=compatible_hashes()
    )
    if loaded.bundle is None:
        return (None, None, loaded.fallback_reason)
    prediction = loaded.bundle.model.predict(np.zeros((1, 2))).item()
    return (loaded.metadata.model_version, prediction, loaded.fallback_reason)


def _process_reader_worker(
    model_dir,
    ready_queue,
    start_event,
    stop_event,
    result_queue,
):
    observations = [_loaded_model_snapshot(model_dir)]
    ready_queue.put("reader-ready")
    if not start_event.wait(10):
        raise TimeoutError("reader start event was not released")
    for _ in range(500):
        if stop_event.is_set():
            break
        observations.append(_loaded_model_snapshot(model_dir))
    if not stop_event.wait(10):
        raise TimeoutError("writer did not finish")
    observations.append(_loaded_model_snapshot(model_dir))
    result_queue.put(("reader", observations))


def _process_writer_worker(model_dir, start_event, stop_event, result_queue):
    if not start_event.wait(10):
        raise TimeoutError("writer start event was not released")
    key = ModelKey("project-a", "line-a", "001", "wind_speed")
    decision = ModelRegistry(model_dir).promote(
        model_candidate(
            key,
            model_version="version-2",
            corrected_mae=0.5,
            model_value=0.9,
        )
    )
    result_queue.put(("writer", decision.promoted, decision.reason))
    stop_event.set()


def _join_processes(processes):
    for process in processes:
        process.join(15)
    alive = [process for process in processes if process.is_alive()]
    for process in alive:
        process.terminate()
        process.join(5)
    assert not alive, "child process exceeded timeout"
    assert [process.exitcode for process in processes] == [0] * len(processes)


def test_same_tower_on_different_lines_never_shares_model(tmp_path):
    registry = ModelRegistry(tmp_path)

    first = registry.path_for(
        ModelKey("project-a", "line-a", "001", "wind_speed")
    )
    second = registry.path_for(
        ModelKey("project-a", "line-b", "001", "wind_speed")
    )

    assert first == (
        tmp_path / "project-a" / "line-a" / "001" / "wind_speed"
        / "model.joblib"
    )
    assert second == (
        tmp_path / "project-a" / "line-b" / "001" / "wind_speed"
        / "model.joblib"
    )
    assert first != second


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("project_id", ""),
        ("project_id", "   "),
        ("project_id", "."),
        ("project_id", ".."),
        ("project_id", "/absolute"),
        ("project_id", "C:escape"),
        ("project_id", "contains\x00nul"),
        ("project_id", "../escape"),
        ("line_id", "line/child"),
        ("line_id", r"line\child"),
        ("tower_id", "../001"),
        ("target", "dlr"),
    ],
)
def test_model_key_rejects_empty_unsafe_or_unsupported_identifiers(
    field, value
):
    values = {
        "project_id": "project-a",
        "line_id": "line-a",
        "tower_id": "001",
        "target": "wind_speed",
    }
    values[field] = value

    with pytest.raises(ValueError):
        ModelKey(**values)


def test_registry_root_is_resolved_without_changing_key_layout(tmp_path):
    relative_root = Path(tmp_path.name) / "models"
    registry = ModelRegistry(tmp_path.parent / relative_root)
    key = ModelKey("project-a", "line-a", "001", "ambient_temp")

    assert registry.path_for(key).name == "model.joblib"
    assert registry.metadata_path_for(key).name == "metadata.json"
    assert registry.manifest_path_for(key).name == "manifest.json"


def test_configured_model_root_symlink_is_canonicalized_and_allowed(tmp_path):
    actual_root = tmp_path / "actual-models"
    actual_root.mkdir()
    actual_root.chmod(0o777)
    configured_root = tmp_path / "configured-models"
    configured_root.symlink_to(actual_root, target_is_directory=True)
    registry = ModelRegistry(configured_root)
    key = ModelKey("project-a", "line-a", "001", "wind_speed")

    decision = registry.promote(
        model_candidate(key, evaluation_mode="full_fit")
    )

    assert registry.model_dir == actual_root.resolve()
    if os.name == "posix":
        assert private_mode(actual_root) == 0o700
    assert decision.promoted is True
    assert registry.path_for(key).is_relative_to(actual_root.resolve())


@POSIX_ONLY
def test_registry_persists_models_and_locks_with_private_permissions(
    tmp_path,
):
    model_root = tmp_path / "models"
    model_tower = model_root / "project-a" / "line-a" / "001"
    generation_root = model_tower / ".wind_speed.generations"
    lock_tower = model_root / ".locks" / "project-a" / "line-a" / "001"
    generation_root.mkdir(parents=True)
    lock_tower.mkdir(parents=True)
    existing_directories = [
        model_root,
        model_root / "project-a",
        model_root / "project-a" / "line-a",
        model_tower,
        generation_root,
        model_root / ".locks",
        model_root / ".locks" / "project-a",
        model_root / ".locks" / "project-a" / "line-a",
        lock_tower,
    ]
    for directory in existing_directories:
        directory.chmod(0o777)

    previous_umask = os.umask(0)
    try:
        registry = ModelRegistry(model_root)
        key = ModelKey("project-a", "line-a", "001", "wind_speed")
        decision = registry.promote(
            model_candidate(key, evaluation_mode="full_fit")
        )
    finally:
        os.umask(previous_umask)

    assert decision.promoted is True
    generation_dir = registry.path_for(key).parent.resolve(strict=True)
    directories = [*existing_directories, generation_dir]
    assert {private_mode(path) for path in directories} == {0o700}
    artifacts = [
        generation_dir / "model.joblib",
        generation_dir / "metadata.json",
        generation_dir / "manifest.json",
    ]
    assert {private_mode(path) for path in artifacts} == {0o600}
    lock_path = lock_tower / "wind_speed.lock"
    with registry._lock_for(key):
        assert private_mode(lock_path) == 0o600


@POSIX_ONLY
def test_registry_rejects_public_root_when_permissions_cannot_be_tightened(
    tmp_path, monkeypatch
):
    model_root = tmp_path / "models"
    model_root.mkdir()
    model_root.chmod(0o777)
    real_chmod = model_registry.os.chmod

    def leave_model_root_public(path, mode, *args, **kwargs):
        if Path(path) == model_root:
            return None
        return real_chmod(path, mode, *args, **kwargs)

    monkeypatch.setattr(model_registry.os, "chmod", leave_model_root_public)

    with pytest.raises(ValueError, match="model_dir"):
        ModelRegistry(model_root)


@POSIX_ONLY
def test_promote_rejects_symlinked_lock_file_without_external_writes(
    tmp_path,
):
    model_root = tmp_path / "models"
    registry = ModelRegistry(model_root)
    lock_tower = model_root / ".locks" / "project-a" / "line-a" / "001"
    lock_tower.mkdir(parents=True)
    outside_lock = tmp_path / "outside.lock"
    outside_lock.write_text("must survive", encoding="utf-8")
    (lock_tower / "wind_speed.lock").symlink_to(outside_lock)
    key = ModelKey("project-a", "line-a", "001", "wind_speed")

    decision = registry.promote(
        model_candidate(key, evaluation_mode="full_fit")
    )

    assert decision.promoted is False
    assert decision.reason == "unsafe_model_path"
    assert outside_lock.read_text("utf-8") == "must survive"


@POSIX_ONLY
@pytest.mark.parametrize(
    "entry_name",
    ["generation_dir", "model.joblib", "metadata.json", "manifest.json"],
)
def test_load_rejects_public_generation_entry_when_mode_cannot_be_tightened(
    tmp_path, monkeypatch, entry_name
):
    registry = ModelRegistry(tmp_path / "models")
    key = ModelKey("project-a", "line-a", "001", "wind_speed")
    registry.promote(model_candidate(key, evaluation_mode="full_fit"))
    generation_dir = registry.path_for(key).parent.resolve(strict=True)
    unsafe_path = (
        generation_dir
        if entry_name == "generation_dir"
        else generation_dir / entry_name
    )
    unsafe_path.chmod(0o777 if unsafe_path.is_dir() else 0o666)
    real_chmod = model_registry.os.chmod

    def leave_generation_entry_public(path, mode, *args, **kwargs):
        if Path(path) == unsafe_path:
            return None
        return real_chmod(path, mode, *args, **kwargs)

    monkeypatch.setattr(
        model_registry.os, "chmod", leave_generation_entry_public
    )

    loaded = registry.load(
        key, expected_compatibility=compatible_hashes()
    )

    assert loaded.bundle is None
    assert loaded.fallback_reason == "unsafe_model_path"


@pytest.mark.parametrize("symlink_component", ["project", "line", "tower"])
def test_promote_rejects_symlinked_key_ancestor_without_external_writes(
    tmp_path, symlink_component
):
    model_root = tmp_path / "models"
    model_root.mkdir()
    outside = tmp_path / f"outside-{symlink_component}"
    outside.mkdir()
    components = ["project-a", "line-a", "001"]
    component_index = {"project": 0, "line": 1, "tower": 2}[
        symlink_component
    ]
    parent = model_root.joinpath(*components[:component_index])
    parent.mkdir(parents=True, exist_ok=True)
    (parent / components[component_index]).symlink_to(
        outside, target_is_directory=True
    )
    registry = ModelRegistry(model_root)
    key = ModelKey("project-a", "line-a", "001", "wind_speed")

    decision = registry.promote(
        model_candidate(key, evaluation_mode="full_fit")
    )

    assert decision.promoted is False
    assert decision.reason == "unsafe_model_path"
    assert list(outside.iterdir()) == []


def test_promote_rejects_symlinked_lock_root_without_external_writes(tmp_path):
    model_root = tmp_path / "models"
    model_root.mkdir()
    outside = tmp_path / "outside-locks"
    outside.mkdir()
    (model_root / ".locks").symlink_to(outside, target_is_directory=True)
    registry = ModelRegistry(model_root)
    key = ModelKey("project-a", "line-a", "001", "wind_speed")

    decision = registry.promote(
        model_candidate(key, evaluation_mode="full_fit")
    )

    assert decision.promoted is False
    assert decision.reason == "unsafe_model_path"
    assert list(outside.iterdir()) == []


def test_promote_rejects_symlinked_generation_root_without_external_writes(
    tmp_path,
):
    model_root = tmp_path / "models"
    tower_dir = model_root / "project-a" / "line-a" / "001"
    tower_dir.mkdir(parents=True)
    outside = tmp_path / "outside-generations"
    outside.mkdir()
    (tower_dir / ".wind_speed.generations").symlink_to(
        outside, target_is_directory=True
    )
    registry = ModelRegistry(model_root)
    key = ModelKey("project-a", "line-a", "001", "wind_speed")

    decision = registry.promote(
        model_candidate(key, evaluation_mode="full_fit")
    )

    assert decision.promoted is False
    assert decision.reason == "unsafe_model_path"
    assert list(outside.iterdir()) == []


def test_load_rejects_key_ancestor_symlink_to_external_bundle(tmp_path):
    model_root = tmp_path / "models"
    registry = ModelRegistry(model_root)
    key = ModelKey("project-a", "line-a", "001", "wind_speed")
    registry.promote(model_candidate(key, evaluation_mode="full_fit"))
    outside = tmp_path / "outside-project"
    outside.mkdir()
    project_dir = model_root / "project-a"
    moved_project = outside / "project-a"
    project_dir.rename(moved_project)
    project_dir.symlink_to(moved_project, target_is_directory=True)

    loaded = registry.load(
        key, expected_compatibility=compatible_hashes()
    )

    assert loaded.bundle is None
    assert loaded.fallback_reason == "unsafe_model_path"


def test_load_rejects_generation_root_symlink_to_external_bundle(tmp_path):
    model_root = tmp_path / "models"
    registry = ModelRegistry(model_root)
    key = ModelKey("project-a", "line-a", "001", "wind_speed")
    registry.promote(model_candidate(key, evaluation_mode="full_fit"))
    target_dir = registry.path_for(key).parent
    generation_dir = target_dir.resolve(strict=True)
    generation_root = generation_dir.parent
    outside = tmp_path / "outside-generations"
    outside.mkdir()
    generation_dir.rename(outside / generation_dir.name)
    generation_root.rmdir()
    generation_root.symlink_to(outside, target_is_directory=True)

    loaded = registry.load(
        key, expected_compatibility=compatible_hashes()
    )

    assert loaded.bundle is None
    assert loaded.fallback_reason == "unsafe_model_path"


def test_full_fit_metadata_keeps_independent_metrics_empty():
    key = ModelKey("project-a", "line-a", "001", "wind_speed")

    metadata = model_metadata(key, evaluation_mode="full_fit")

    assert metadata.metrics == {}
    assert metadata.full_fit_metrics == WEATHER_METRICS
    assert metadata.evaluation_set_hash is None
    assert metadata.metric_domain == "weather_vs_truth"


def test_full_fit_metadata_rejects_training_metrics_as_independent_metrics():
    key = ModelKey("project-a", "line-a", "001", "wind_speed")
    values = model_metadata(key, evaluation_mode="full_fit").to_dict()
    values["metrics"] = dict(WEATHER_METRICS)

    with pytest.raises(ValueError, match="full_fit.*metrics"):
        ModelMetadata.from_dict(values)


def test_full_fit_metadata_rejects_independent_evaluation_hash():
    key = ModelKey("project-a", "line-a", "001", "wind_speed")
    values = model_metadata(key, evaluation_mode="full_fit").to_dict()
    values["evaluation_set_hash"] = "not-an-independent-set"

    with pytest.raises(ValueError, match="full_fit.*evaluation_set_hash"):
        ModelMetadata.from_dict(values)


@pytest.mark.parametrize("field", ["training_params", "dependency_versions"])
def test_required_training_metadata_mappings_cannot_be_empty(field):
    key = ModelKey("project-a", "line-a", "001", "wind_speed")
    values = model_metadata(key).to_dict()
    values[field] = {}

    with pytest.raises(ValueError, match=field):
        ModelMetadata.from_dict(values)


@pytest.mark.parametrize(
    "field",
    [
        "dem_hash",
        "crs_hash",
        "coordinate_hash",
        "conductor_hash",
        "feature_version",
        "correction_config_hash",
    ],
)
def test_compatibility_hashes_must_be_explicit(field):
    values = compatible_hashes().__dict__.copy()
    values[field] = ""

    with pytest.raises(ValueError, match=field):
        ModelCompatibility(**values)


def test_first_full_fit_candidate_is_saved_as_provisional_and_reloads(tmp_path):
    key = ModelKey("project-a", "line-a", "001", "wind_speed")
    registry = ModelRegistry(tmp_path)

    decision = registry.promote(
        model_candidate(key, evaluation_mode="full_fit")
    )

    assert decision.promoted is True
    assert decision.reason == "promoted_provisional"
    assert decision.metadata is not None
    assert decision.metadata.status == "active_provisional"
    assert registry.path_for(key).is_file()
    assert registry.metadata_path_for(key).is_file()
    assert registry.manifest_path_for(key).is_file()

    loaded = ModelRegistry(tmp_path).load(
        key, expected_compatibility=compatible_hashes()
    )
    assert loaded.fallback_reason == ""
    assert loaded.bundle is not None
    assert loaded.metadata is not None
    assert loaded.metadata.status == "active_provisional"
    assert loaded.metadata.metrics == {}
    assert loaded.metadata.full_fit_metrics == WEATHER_METRICS
    assert loaded.bundle.cadence_minutes == 30.0
    assert loaded.bundle.model.predict(np.zeros((2, 2))).tolist() == [0.5, 0.5]


@pytest.mark.parametrize(
    "evaluation_mode", ["temporal_holdout", "rolling_validation"]
)
def test_independent_candidate_promotes_from_full_fit_provisional(
    tmp_path, evaluation_mode
):
    key = ModelKey("project-a", "line-a", "001", "wind_speed")
    registry = ModelRegistry(tmp_path, min_mae_improvement=0.2)
    registry.promote(
        model_candidate(
            key,
            model_version="version-1",
            evaluation_mode="full_fit",
            corrected_mae=1.0,
        )
    )

    decision = registry.promote(
        model_candidate(
            key,
            model_version="version-2",
            evaluation_mode=evaluation_mode,
            corrected_mae=1.7,
            evaluation_set_hash="frozen-evaluation-a",
            model_value=0.9,
        )
    )

    assert decision.promoted is True
    assert decision.reason == "promoted_from_provisional"
    assert decision.metadata.status == "active"
    loaded = registry.load(
        key, expected_compatibility=compatible_hashes()
    )
    assert loaded.fallback_reason == ""
    assert loaded.metadata.model_version == "version-2"
    assert loaded.metadata.status == "active"
    assert loaded.bundle.model.predict(np.zeros((1, 2))).tolist() == [0.9]


def test_provisional_replacement_must_beat_physical_by_threshold(tmp_path):
    key = ModelKey("project-a", "line-a", "001", "wind_speed")
    registry = ModelRegistry(tmp_path, min_mae_improvement=0.5)
    registry.promote(
        model_candidate(
            key,
            model_version="version-1",
            evaluation_mode="full_fit",
        )
    )

    decision = registry.promote(
        model_candidate(
            key,
            model_version="version-2",
            corrected_mae=1.5,
            evaluation_set_hash="frozen-evaluation-a",
        )
    )

    assert decision.promoted is False
    assert decision.reason == "insufficient_mae_improvement"


def test_provisional_replacement_rejects_model_version_conflict(tmp_path):
    key = ModelKey("project-a", "line-a", "001", "wind_speed")
    registry = ModelRegistry(tmp_path)
    registry.promote(
        model_candidate(
            key,
            model_version="version-1",
            evaluation_mode="full_fit",
        )
    )

    decision = registry.promote(
        model_candidate(
            key,
            model_version="version-1",
            corrected_mae=0.5,
            evaluation_set_hash="frozen-evaluation-a",
        )
    )

    assert decision.promoted is False
    assert decision.reason == "model_version_conflict"


def test_persisted_metadata_contains_required_scope_hashes_and_checksum(tmp_path):
    key = ModelKey("project-a", "line-a", "001", "ambient_temp")
    registry = ModelRegistry(tmp_path)
    registry.promote(model_candidate(key, evaluation_mode="full_fit"))

    payload = json.loads(registry.metadata_path_for(key).read_text("utf-8"))

    assert payload["project_id"] == "project-a"
    assert payload["line_id"] == "line-a"
    assert payload["tower_id"] == "001"
    assert payload["target"] == "ambient_temp"
    assert payload["model_version"] == "version-1"
    assert payload["feature_columns"] == ["ambient_temp_local", "lag_1"]
    assert payload["evaluation_mode"] == "full_fit"
    assert payload["metrics"] == {}
    assert payload["full_fit_metrics"] == WEATHER_METRICS
    assert payload["status"] == "active_provisional"
    assert len(payload["checksum"]) == 64
    assert payload["dem_hash"] == "dem-a"
    assert payload["correction_config_hash"] == "correction-a"


def test_full_fit_cannot_replace_existing_champion(tmp_path):
    key = ModelKey("project-a", "line-a", "001", "wind_speed")
    registry = ModelRegistry(tmp_path)
    first = registry.promote(
        model_candidate(
            key,
            model_version="version-1",
            evaluation_mode="temporal_holdout",
            corrected_mae=1.0,
        )
    )
    assert first.promoted is True

    decision = registry.promote(
        model_candidate(
            key,
            model_version="version-2",
            evaluation_mode="full_fit",
            corrected_mae=0.1,
        )
    )

    assert decision.promoted is False
    assert decision.reason == "full_fit_cannot_replace_champion"
    loaded = registry.load(
        key, expected_compatibility=compatible_hashes()
    )
    assert loaded.metadata.model_version == "version-1"


def test_candidate_on_different_frozen_evaluation_set_is_rejected(tmp_path):
    key = ModelKey("project-a", "line-a", "001", "wind_speed")
    registry = ModelRegistry(tmp_path)
    registry.promote(
        model_candidate(
            key,
            model_version="version-1",
            corrected_mae=1.0,
            evaluation_set_hash="evaluation-a",
        )
    )

    decision = registry.promote(
        model_candidate(
            key,
            model_version="version-2",
            corrected_mae=0.1,
            evaluation_set_hash="evaluation-b",
        )
    )

    assert decision.promoted is False
    assert decision.reason == "evaluation_set_mismatch"


def test_candidate_without_independent_evaluation_hash_is_rejected(tmp_path):
    key = ModelKey("project-a", "line-a", "001", "wind_speed")
    registry = ModelRegistry(tmp_path)
    registry.promote(model_candidate(key, model_version="version-1"))

    decision = registry.promote(
        model_candidate(
            key,
            model_version="version-2",
            corrected_mae=0.1,
            evaluation_set_hash=None,
        )
    )

    assert decision.promoted is False
    assert decision.reason == "missing_evaluation_set_hash"


def test_first_independently_evaluated_candidate_requires_evaluation_hash(
    tmp_path,
):
    key = ModelKey("project-a", "line-a", "001", "wind_speed")
    registry = ModelRegistry(tmp_path)

    decision = registry.promote(
        model_candidate(
            key,
            model_version="version-1",
            corrected_mae=0.1,
            evaluation_set_hash=None,
        )
    )

    assert decision.promoted is False
    assert decision.reason == "missing_evaluation_set_hash"
    assert not registry.path_for(key).exists()


def test_candidate_must_exceed_configured_mae_improvement_threshold(tmp_path):
    key = ModelKey("project-a", "line-a", "001", "wind_speed")
    registry = ModelRegistry(tmp_path, min_mae_improvement=0.2)
    registry.promote(
        model_candidate(key, model_version="version-1", corrected_mae=1.0)
    )

    decision = registry.promote(
        model_candidate(key, model_version="version-2", corrected_mae=0.8)
    )

    assert decision.promoted is False
    assert decision.reason == "insufficient_mae_improvement"


def test_candidate_with_same_evaluation_set_and_enough_improvement_is_active(
    tmp_path,
):
    key = ModelKey("project-a", "line-a", "001", "wind_speed")
    registry = ModelRegistry(tmp_path, min_mae_improvement=0.2)
    registry.promote(
        model_candidate(key, model_version="version-1", corrected_mae=1.0)
    )

    decision = registry.promote(
        model_candidate(key, model_version="version-2", corrected_mae=0.7)
    )

    assert decision.promoted is True
    assert decision.reason == "promoted"
    assert decision.metadata.status == "active"
    loaded = registry.load(
        key, expected_compatibility=compatible_hashes()
    )
    assert loaded.metadata.model_version == "version-2"
    assert loaded.metadata.metrics["corrected_mae"] == 0.7


def test_successive_promotions_keep_only_active_and_one_history_generation(
    tmp_path,
):
    key = ModelKey("project-a", "line-a", "001", "wind_speed")
    registry = ModelRegistry(tmp_path)
    corrected_values = [1.0, 0.8, 0.6, 0.4, 0.2]

    for index, corrected_mae in enumerate(corrected_values, start=1):
        decision = registry.promote(
            model_candidate(
                key,
                model_version=f"version-{index}",
                corrected_mae=corrected_mae,
            )
        )
        assert decision.promoted is True

    generation_root = (
        registry.path_for(key).parent.parent / ".wind_speed.generations"
    )
    generations = [path for path in generation_root.iterdir() if path.is_dir()]
    versions = {
        json.loads((path / "metadata.json").read_text("utf-8"))["model_version"]
        for path in generations
    }
    assert len(generations) == 2
    assert versions == {"version-4", "version-5"}
    assert registry.path_for(key).parent.resolve() in generations


def test_configurable_generation_retention_and_validation(tmp_path):
    key = ModelKey("project-a", "line-a", "001", "wind_speed")
    registry = ModelRegistry(tmp_path / "models", max_generations=3)
    for index, corrected_mae in enumerate([1.0, 0.8, 0.6, 0.4], start=1):
        registry.promote(
            model_candidate(
                key,
                model_version=f"version-{index}",
                corrected_mae=corrected_mae,
            )
        )
    generation_root = (
        registry.path_for(key).parent.parent / ".wind_speed.generations"
    )
    assert len(list(generation_root.iterdir())) == 3

    for invalid in (True, 0, -1, 1.5):
        with pytest.raises(ValueError, match="max_generations"):
            ModelRegistry(tmp_path / f"invalid-{invalid}", max_generations=invalid)


def test_successful_publish_cleans_interrupted_and_symlink_generations_safely(
    tmp_path,
):
    key = ModelKey("project-a", "line-a", "001", "wind_speed")
    registry = ModelRegistry(tmp_path / "models")
    registry.promote(
        model_candidate(key, model_version="version-1", corrected_mae=1.0)
    )
    generation_root = (
        registry.path_for(key).parent.parent / ".wind_speed.generations"
    )
    interrupted = generation_root / "interrupted-generation"
    interrupted.mkdir()
    (interrupted / ".model.tmp").write_bytes(b"partial")
    outside = tmp_path / "outside-history"
    outside.mkdir()
    sentinel = outside / "must-survive.txt"
    sentinel.write_text("safe", encoding="utf-8")
    malicious_link = generation_root / "linked-generation"
    malicious_link.symlink_to(outside, target_is_directory=True)

    decision = registry.promote(
        model_candidate(key, model_version="version-2", corrected_mae=0.5)
    )

    assert decision.promoted is True
    assert not interrupted.exists()
    assert not malicious_link.exists()
    assert sentinel.read_text("utf-8") == "safe"
    assert len(list(generation_root.iterdir())) == 2


def test_publish_fsyncs_generation_directories_before_and_after_pointer(
    tmp_path, monkeypatch
):
    calls = []

    def record_directory_fsync(path):
        calls.append(Path(path))
        return True

    monkeypatch.setattr(
        model_registry,
        "_fsync_directory",
        record_directory_fsync,
        raising=False,
    )
    key = ModelKey("project-a", "line-a", "001", "wind_speed")
    registry = ModelRegistry(tmp_path)

    decision = registry.promote(
        model_candidate(key, evaluation_mode="full_fit")
    )

    assert decision.promoted is True
    target_dir = registry.path_for(key).parent
    active_generation = target_dir.resolve(strict=True)
    generation_root = active_generation.parent
    target_parent = target_dir.parent
    assert calls.index(active_generation) < calls.index(generation_root)
    assert calls.index(generation_root) < calls.index(target_parent)


def test_post_commit_parent_fsync_failure_keeps_new_active_generation(
    tmp_path, monkeypatch
):
    key = ModelKey("project-a", "line-a", "001", "wind_speed")
    registry = ModelRegistry(tmp_path)
    registry.promote(
        model_candidate(key, model_version="version-1", corrected_mae=1.0)
    )
    target_parent = registry.path_for(key).parent.parent

    def fail_target_parent_fsync(path):
        if Path(path) == target_parent:
            raise OSError("simulated post-commit directory fsync failure")
        return True

    monkeypatch.setattr(
        model_registry,
        "_fsync_directory",
        fail_target_parent_fsync,
        raising=False,
    )

    decision = registry.promote(
        model_candidate(
            key,
            model_version="version-2",
            corrected_mae=0.5,
            model_value=0.9,
        )
    )

    assert decision.promoted is True
    loaded = registry.load(
        key, expected_compatibility=compatible_hashes()
    )
    assert loaded.fallback_reason == ""
    assert loaded.metadata.model_version == "version-2"
    assert loaded.bundle.model.predict(np.zeros((1, 2))).tolist() == [0.9]


@pytest.mark.parametrize("failure_point", ["prune", "parent_fsync"])
def test_post_commit_runtime_error_still_reports_new_active_metadata(
    tmp_path, monkeypatch, failure_point
):
    key = ModelKey("project-a", "line-a", "001", "wind_speed")
    registry = ModelRegistry(tmp_path)
    registry.promote(
        model_candidate(key, model_version="version-1", corrected_mae=1.0)
    )
    target_parent = registry.path_for(key).parent.parent

    if failure_point == "prune":
        def fail_prune(*args, **kwargs):
            raise RuntimeError("simulated post-commit prune failure")

        monkeypatch.setattr(registry, "_prune_generations", fail_prune)
    else:
        real_fsync = model_registry._fsync_directory

        def fail_parent_fsync(path):
            if Path(path) == target_parent:
                raise RuntimeError("simulated post-commit fsync failure")
            return real_fsync(path)

        monkeypatch.setattr(model_registry, "_fsync_directory", fail_parent_fsync)

    decision = registry.promote(
        model_candidate(
            key,
            model_version="version-2",
            corrected_mae=0.5,
            model_value=0.9,
        )
    )

    assert decision.promoted is True
    assert decision.metadata.model_version == "version-2"
    loaded = registry.load(
        key, expected_compatibility=compatible_hashes()
    )
    assert loaded.fallback_reason == ""
    assert loaded.metadata.model_version == "version-2"
    assert loaded.bundle.model.predict(np.zeros((1, 2))).tolist() == [0.9]


def test_post_commit_temp_cleanup_runtime_error_does_not_override_success(
    tmp_path, monkeypatch
):
    key = ModelKey("project-a", "line-a", "001", "wind_speed")
    registry = ModelRegistry(tmp_path)
    registry.promote(
        model_candidate(key, model_version="version-1", corrected_mae=1.0)
    )
    real_exists = Path.exists
    real_unlink = Path.unlink

    def force_temp_link_exists(path):
        if path.name.startswith(".wind_speed.") and path.name.endswith(".tmp"):
            return True
        return real_exists(path)

    def fail_temp_link_cleanup(path, *args, **kwargs):
        if path.name.startswith(".wind_speed.") and path.name.endswith(".tmp"):
            raise RuntimeError("simulated post-commit temp cleanup failure")
        return real_unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "exists", force_temp_link_exists)
    monkeypatch.setattr(Path, "unlink", fail_temp_link_cleanup)

    decision = registry.promote(
        model_candidate(key, model_version="version-2", corrected_mae=0.5)
    )

    assert decision.promoted is True
    assert decision.metadata.model_version == "version-2"
    loaded = registry.load(
        key, expected_compatibility=compatible_hashes()
    )
    assert loaded.metadata.model_version == "version-2"


def test_pre_commit_runtime_error_fails_and_keeps_old_champion(
    tmp_path, monkeypatch
):
    key = ModelKey("project-a", "line-a", "001", "wind_speed")
    registry = ModelRegistry(tmp_path)
    registry.promote(
        model_candidate(key, model_version="version-1", corrected_mae=1.0)
    )
    real_fsync = model_registry._fsync_directory

    def fail_candidate_generation_fsync(path):
        if Path(path).name.startswith("version-2-"):
            raise RuntimeError("simulated pre-commit fsync failure")
        return real_fsync(path)

    monkeypatch.setattr(
        model_registry,
        "_fsync_directory",
        fail_candidate_generation_fsync,
    )

    decision = registry.promote(
        model_candidate(key, model_version="version-2", corrected_mae=0.5)
    )

    assert decision.promoted is False
    assert decision.reason == "publish_failed:RuntimeError"
    loaded = registry.load(
        key, expected_compatibility=compatible_hashes()
    )
    assert loaded.metadata.model_version == "version-1"


def test_first_candidate_must_improve_over_physical_baseline(tmp_path):
    key = ModelKey("project-a", "line-a", "001", "wind_speed")
    registry = ModelRegistry(tmp_path)

    decision = registry.promote(
        model_candidate(key, corrected_mae=WEATHER_METRICS["baseline_mae"])
    )

    assert decision.promoted is False
    assert decision.reason == "candidate_not_better_than_physical"
    assert not registry.path_for(key).exists()


@pytest.mark.parametrize(
    "field",
    [
        "dem_hash",
        "crs_hash",
        "coordinate_hash",
        "conductor_hash",
        "feature_version",
        "correction_config_hash",
    ],
)
def test_load_rejects_each_incompatible_runtime_hash(tmp_path, field):
    key = ModelKey("project-a", "line-a", "001", "wind_speed")
    registry = ModelRegistry(tmp_path)
    registry.promote(model_candidate(key, evaluation_mode="full_fit"))
    expected_values = compatible_hashes().__dict__.copy()
    expected_values[field] = f"changed-{field}"

    loaded = registry.load(
        key,
        expected_compatibility=ModelCompatibility(**expected_values),
    )

    assert loaded.bundle is None
    assert loaded.fallback_reason == f"incompatible_{field}"


def test_load_requires_explicit_compatibility_expectation(tmp_path):
    key = ModelKey("project-a", "line-a", "001", "wind_speed")

    with pytest.raises(TypeError, match="expected_compatibility"):
        ModelRegistry(tmp_path).load(key)


def test_corrupt_model_falls_back_only_for_affected_tower(tmp_path):
    first_key = ModelKey("project-a", "line-a", "001", "wind_speed")
    second_key = ModelKey("project-a", "line-a", "002", "wind_speed")
    registry = ModelRegistry(tmp_path)
    registry.promote(model_candidate(first_key, evaluation_mode="full_fit"))
    registry.promote(model_candidate(second_key, evaluation_mode="full_fit"))
    registry.path_for(first_key).write_bytes(b"not a joblib model")

    loaded = registry.load_many(
        [first_key, second_key],
        expected_compatibility={
            first_key: compatible_hashes(),
            second_key: compatible_hashes(),
        },
    )

    assert loaded[first_key].fallback_reason == "corrupt_model"
    assert loaded[first_key].bundle is None
    assert loaded[second_key].fallback_reason == ""
    assert loaded[second_key].bundle is not None


@pytest.mark.parametrize(
    ("artifact", "expected_reason"),
    [
        ("metadata", "corrupt_metadata"),
        ("manifest", "corrupt_manifest"),
    ],
)
def test_corrupt_metadata_or_manifest_is_isolated_per_key(
    tmp_path, artifact, expected_reason
):
    first_key = ModelKey("project-a", "line-a", "001", "ambient_temp")
    second_key = ModelKey("project-a", "line-a", "002", "ambient_temp")
    registry = ModelRegistry(tmp_path)
    registry.promote(model_candidate(first_key, evaluation_mode="full_fit"))
    registry.promote(model_candidate(second_key, evaluation_mode="full_fit"))
    path = (
        registry.metadata_path_for(first_key)
        if artifact == "metadata"
        else registry.manifest_path_for(first_key)
    )
    path.write_text("{broken", encoding="utf-8")

    loaded = registry.load_many(
        [first_key, second_key],
        expected_compatibility={
            first_key: compatible_hashes(),
            second_key: compatible_hashes(),
        },
    )

    assert loaded[first_key].fallback_reason == expected_reason
    assert loaded[second_key].bundle is not None


def test_training_result_hashes_are_stable_across_input_row_order():
    trainer = ResidualTrainer(estimator_factory=MeanResidualModel)

    first = trainer.train_target(training_frame(), target="wind_speed")
    second = trainer.train_target(
        training_frame().sample(frac=1.0, random_state=7),
        target="wind_speed",
    )

    assert len(first.metadata["input_data_hash"]) == 64
    assert len(first.metadata["evaluation_set_hash"]) == 64
    assert first.metadata["input_data_hash"] == second.metadata["input_data_hash"]
    assert (
        first.metadata["evaluation_set_hash"]
        == second.metadata["evaluation_set_hash"]
    )


def test_input_hash_is_stable_for_reversed_same_time_lineage_rows():
    physical = np.array([2.0, 3.0, 12.0, 13.0])
    frame = pd.DataFrame(
        {
            "line_id": ["line-a"] * 4,
            "tower_id": ["001"] * 4,
            "timestamp": pd.to_datetime(
                [
                    "2025-01-01 00:00",
                    "2025-01-01 00:30",
                    "2025-01-01 00:00",
                    "2025-01-01 00:30",
                ],
                utc=True,
            ),
            "source_file_hash_physical": [
                "physical-a",
                "physical-a",
                "physical-b",
                "physical-b",
            ],
            "source_file_hash_truth": [
                "truth-a",
                "truth-a",
                "truth-b",
                "truth-b",
            ],
            "wind_speed_local": physical,
            "wind_speed_truth": physical + np.array([0.0, 1.0, 2.0, 3.0]),
        }
    )
    trainer = ResidualTrainer(estimator_factory=MeanResidualModel)

    forward = trainer.train_target(frame, target="wind_speed")
    reversed_rows = trainer.train_target(
        frame.iloc[::-1], target="wind_speed"
    )

    assert forward.metadata["input_data_hash"] == reversed_rows.metadata[
        "input_data_hash"
    ]
    changed_lineage = frame.copy(deep=True)
    changed_lineage.loc[changed_lineage.index[0], "source_file_hash_truth"] = (
        "truth-changed"
    )
    changed = trainer.train_target(changed_lineage, target="wind_speed")
    assert forward.metadata["input_data_hash"] != changed.metadata[
        "input_data_hash"
    ]


def test_training_hash_changes_when_truth_weather_changes():
    changed = training_frame()
    changed.loc[changed.index[-1], "wind_speed_truth"] += 0.25
    trainer = ResidualTrainer(estimator_factory=MeanResidualModel)

    original = trainer.train_target(training_frame(), target="wind_speed")
    modified = trainer.train_target(changed, target="wind_speed")

    assert original.metadata["input_data_hash"] != modified.metadata[
        "input_data_hash"
    ]
    assert original.metadata["evaluation_set_hash"] != modified.metadata[
        "evaluation_set_hash"
    ]


def test_full_fit_training_has_no_independent_evaluation_hash():
    result = ResidualTrainer(
        estimator_factory=MeanResidualModel
    ).train_target(training_frame().iloc[[0]], target="wind_speed")

    assert len(result.metadata["input_data_hash"]) == 64
    assert result.metadata["evaluation_set_hash"] is None


def test_training_result_records_estimator_parameters_and_dependencies():
    result = ResidualTrainer(
        estimator_factory=MeanResidualModel
    ).train_target(training_frame(), target="wind_speed")

    assert result.metadata["training_params"]["estimator_class"].endswith(
        ".MeanResidualModel"
    )
    assert result.metadata["random_state"] == 42
    assert set(result.metadata["dependency_versions"]) >= {
        "python",
        "numpy",
        "pandas",
        "joblib",
        "xgboost",
    }
    assert all(result.metadata["dependency_versions"].values())


def test_training_result_adapter_preserves_temporal_holdout_contract():
    result = ResidualTrainer(
        estimator_factory=MeanResidualModel
    ).train_target(training_frame(), target="wind_speed")

    candidate = candidate_from_training_result(
        result,
        project_id="project-a",
        model_version="training-run-1",
        compatibility=compatible_hashes(),
    )

    assert candidate.key == ModelKey(
        "project-a", "line-a", "001", "wind_speed"
    )
    assert candidate.bundle is result.bundle
    assert candidate.metadata.metrics == result.metrics
    assert candidate.metadata.full_fit_metrics is None
    assert candidate.metadata.evaluation_set_hash == result.metadata[
        "evaluation_set_hash"
    ]
    assert candidate.metadata.input_data_hash == result.metadata[
        "input_data_hash"
    ]
    assert candidate.metadata.cadence_minutes == result.bundle.cadence_minutes


def test_training_result_adapter_keeps_full_fit_metrics_non_independent():
    frame = training_frame().iloc[[0]].copy()
    frame["wind_speed_truth"] = frame["wind_speed_local"] + 2.0
    result = ResidualTrainer(
        estimator_factory=MeanResidualModel
    ).train_target(frame, target="wind_speed")

    candidate = candidate_from_training_result(
        result,
        project_id="project-a",
        model_version="training-run-1",
        compatibility=compatible_hashes(),
    )

    assert candidate.metadata.evaluation_mode == "full_fit"
    assert candidate.metadata.metrics == {}
    assert candidate.metadata.full_fit_metrics == result.metadata[
        "full_fit_metrics"
    ]
    assert candidate.metadata.evaluation_set_hash is None


def test_publish_failure_preserves_champion_and_leaves_no_temporary_files(
    tmp_path, monkeypatch
):
    key = ModelKey("project-a", "line-a", "001", "wind_speed")
    registry = ModelRegistry(tmp_path)
    registry.promote(
        model_candidate(key, model_version="version-1", corrected_mae=1.0)
    )
    target_dir = registry.path_for(key).parent
    real_replace = model_registry.os.replace

    def fail_final_pointer_replace(source, destination):
        if Path(destination) == target_dir:
            raise OSError("simulated atomic pointer failure")
        return real_replace(source, destination)

    monkeypatch.setattr(model_registry.os, "replace", fail_final_pointer_replace)

    decision = registry.promote(
        model_candidate(key, model_version="version-2", corrected_mae=0.5)
    )

    assert decision.promoted is False
    assert decision.reason == "publish_failed:OSError"
    loaded = registry.load(
        key, expected_compatibility=compatible_hashes()
    )
    assert loaded.fallback_reason == ""
    assert loaded.metadata.model_version == "version-1"
    assert not [path for path in tmp_path.rglob("*") if path.name.endswith(".tmp")]
    generation_dirs = [
        path
        for path in (target_dir.parent / ".wind_speed.generations").iterdir()
        if path.is_dir()
    ]
    assert len(generation_dirs) == 1


def test_concurrent_candidates_recheck_champion_inside_key_lock(tmp_path):
    key = ModelKey("project-a", "line-a", "001", "wind_speed")
    registry = ModelRegistry(tmp_path)
    registry.promote(
        model_candidate(key, model_version="version-1", corrected_mae=1.0)
    )
    candidates = [
        model_candidate(key, model_version="version-2", corrected_mae=0.7),
        model_candidate(key, model_version="version-3", corrected_mae=0.6),
    ]

    with ThreadPoolExecutor(max_workers=2) as executor:
        decisions = list(executor.map(registry.promote, candidates))

    assert any(decision.promoted for decision in decisions)
    loaded = registry.load(
        key, expected_compatibility=compatible_hashes()
    )
    assert loaded.fallback_reason == ""
    assert loaded.metadata.model_version == "version-3"
    assert loaded.metadata.metrics["corrected_mae"] == 0.6
    assert not [path for path in tmp_path.rglob("*") if path.name.endswith(".tmp")]


def test_file_lock_serializes_competing_promotions_across_processes(tmp_path):
    key = ModelKey("project-a", "line-a", "001", "wind_speed")
    registry = ModelRegistry(tmp_path)
    registry.promote(
        model_candidate(
            key,
            model_version="version-1",
            corrected_mae=1.0,
            model_value=0.5,
        )
    )
    context = multiprocessing.get_context("spawn")
    start_event = context.Event()
    result_queue = context.Queue()
    processes = [
        context.Process(
            target=_process_promote_worker,
            args=(tmp_path, "version-2", 0.7, 0.7, start_event, result_queue),
        ),
        context.Process(
            target=_process_promote_worker,
            args=(tmp_path, "version-3", 0.6, 0.9, start_event, result_queue),
        ),
    ]
    for process in processes:
        process.start()
    start_event.set()

    _join_processes(processes)
    results = [result_queue.get(timeout=5) for _ in processes]
    result_queue.close()
    result_queue.join_thread()

    assert any(promoted for _, promoted, _ in results)
    loaded = registry.load(
        key, expected_compatibility=compatible_hashes()
    )
    assert loaded.fallback_reason == ""
    assert loaded.metadata.model_version == "version-3"
    assert loaded.bundle.model.predict(np.zeros((1, 2))).tolist() == [0.9]


def test_concurrent_process_reader_observes_only_complete_old_or_new_bundle(
    tmp_path,
):
    key = ModelKey("project-a", "line-a", "001", "wind_speed")
    registry = ModelRegistry(tmp_path)
    registry.promote(
        model_candidate(
            key,
            model_version="version-1",
            corrected_mae=1.0,
            model_value=0.5,
        )
    )
    context = multiprocessing.get_context("spawn")
    ready_queue = context.Queue()
    result_queue = context.Queue()
    start_event = context.Event()
    stop_event = context.Event()
    reader = context.Process(
        target=_process_reader_worker,
        args=(
            tmp_path,
            ready_queue,
            start_event,
            stop_event,
            result_queue,
        ),
    )
    writer = context.Process(
        target=_process_writer_worker,
        args=(tmp_path, start_event, stop_event, result_queue),
    )
    reader.start()
    writer.start()
    assert ready_queue.get(timeout=10) == "reader-ready"
    start_event.set()

    _join_processes([reader, writer])
    messages = [result_queue.get(timeout=5) for _ in range(2)]
    reader_message = next(message for message in messages if message[0] == "reader")
    writer_message = next(message for message in messages if message[0] == "writer")
    observations = reader_message[1]

    assert writer_message[1:] == (True, "promoted")
    assert observations[0] == ("version-1", 0.5, "")
    assert observations[-1] == ("version-2", 0.9, "")
    assert set(observations) <= {
        ("version-1", 0.5, ""),
        ("version-2", 0.9, ""),
    }
    for queue in (ready_queue, result_queue):
        queue.close()
        queue.join_thread()
