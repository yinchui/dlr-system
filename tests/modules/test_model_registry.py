import json
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


def model_candidate(key, **metadata_options):
    metadata = model_metadata(key, **metadata_options)
    bundle = ModelBundle(
        target_name=key.target,
        feature_columns=list(metadata.feature_columns),
        model=ConstantResidualModel(),
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
