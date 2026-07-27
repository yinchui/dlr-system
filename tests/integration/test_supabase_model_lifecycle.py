import hashlib

import numpy as np
import pandas as pd
import pytest

import modules.supabase_model_registry as supabase_registry_module
from modules.ai_prediction import ModelBundle
from modules.ai_training import ConstantResidualEstimator
from modules.dlr_pipeline import DlrPipeline
from modules.model_registry import (
    ModelAttempt,
    ModelCandidate,
    ModelCompatibility,
    ModelKey,
    ModelMetadata,
    ModelRegistry,
)
from modules.supabase_model_registry import (
    RemoteGeneration,
    SupabaseModelRegistry,
    SupabaseModelStore,
)


BACKEND_ID = "xgboost-residual-v1"
CONTRACT_HASH = "c" * 64


def _compatibility(suffix: str = "a") -> ModelCompatibility:
    return ModelCompatibility(
        dem_hash=f"dem-{suffix}",
        crs_hash=f"crs-{suffix}",
        coordinate_hash=f"coordinates-{suffix}",
        conductor_hash=f"conductor-{suffix}",
        feature_version=f"features-{suffix}",
        correction_config_hash=f"correction-{suffix}",
    )


def _candidate(
    key: ModelKey,
    *,
    model_version: str = "version-1",
    model_value: float = 0.5,
    corrected_mae: float = 1.0,
    input_data_hash: str = "a" * 64,
    evaluation_set_hash: str = "e" * 64,
) -> ModelCandidate:
    physical_column = (
        "wind_speed_local"
        if key.target == "wind_speed"
        else "ambient_temp_local"
    )
    metadata = ModelMetadata(
        key=key,
        model_version=model_version,
        feature_columns=(physical_column, "lag_1"),
        training_params={"max_depth": 3},
        random_seed=42,
        time_start="2026-01-01T00:00:00+08:00",
        time_end="2026-01-02T00:00:00+08:00",
        sample_count=4,
        evaluation_mode="temporal_holdout",
        metrics={
            "baseline_mae": 2.0,
            "baseline_rmse": 2.5,
            "corrected_mae": corrected_mae,
            "corrected_rmse": 1.5,
        },
        full_fit_metrics=None,
        residual_bounds=(-2.0, 2.0),
        input_data_hash=input_data_hash,
        evaluation_set_hash=evaluation_set_hash,
        compatibility=_compatibility(),
        dependency_versions={"python": "3.11", "joblib": "1.5"},
        cadence_minutes=30.0,
        training_contract_hash=CONTRACT_HASH,
        backend_id=BACKEND_ID,
        training_outcome="data_fallback",
    )
    bundle = ModelBundle(
        target_name=key.target,
        feature_columns=list(metadata.feature_columns),
        model=ConstantResidualEstimator(model_value),
        cadence_minutes=metadata.cadence_minutes,
        residual_bounds=metadata.residual_bounds,
        line_id=key.line_id,
        tower_id=key.tower_id,
        metadata={
            "training_contract_hash": metadata.training_contract_hash,
            "backend_id": metadata.backend_id,
            "training_outcome": metadata.training_outcome,
        },
    )
    return ModelCandidate(key=key, bundle=bundle, metadata=metadata)


def _load(registry: SupabaseModelRegistry, key: ModelKey, *, suffix="a"):
    return registry.load(
        key,
        expected_compatibility=_compatibility(suffix),
        expected_training_contract_hash=CONTRACT_HASH,
        expected_backend_id=BACKEND_ID,
    )


def _attempt(
    registry: SupabaseModelRegistry,
    candidate: ModelCandidate,
    *,
    champion: ModelMetadata | None = None,
) -> ModelAttempt:
    return registry.build_attempt(
        candidate.key,
        input_data_hash=candidate.metadata.input_data_hash,
        evaluation_set_hash=candidate.metadata.evaluation_set_hash,
        training_contract_hash=candidate.metadata.training_contract_hash,
        backend_id=candidate.metadata.backend_id,
        feature_version=candidate.metadata.compatibility.feature_version,
        champion=champion,
    )


class _MemoryStore:
    def __init__(self):
        self.heads: dict[ModelKey, RemoteGeneration] = {}
        self.artifacts: dict[str, bytes] = {}
        self.rejections: set[str] = set()
        self.recorded_reasons: list[str] = []
        self.events: list[str] = []
        self.activation_mode = "ok"
        self.conflict_generation: RemoteGeneration | None = None
        self.fail_rejection_write = False

    def current(self, key: ModelKey) -> RemoteGeneration | None:
        self.events.append("current")
        return self.heads.get(key)

    def download(self, generation: RemoteGeneration) -> bytes:
        self.events.append("download")
        artifact = self.artifacts[generation.storage_path]
        if hashlib.sha256(artifact).hexdigest() != generation.model_checksum:
            raise OSError("checksum mismatch")
        return artifact

    def upload(
        self,
        generation_id: str,
        key: ModelKey,
        artifact: bytes,
    ) -> str:
        self.events.append("upload")
        if self.activation_mode == "upload_transport_error":
            raise supabase_registry_module.SupabaseTransportError(
                "storage unavailable"
            )
        if self.activation_mode == "upload_error":
            raise OSError("storage unavailable")
        path = (
            f"{key.project_id}/{key.line_id}/{key.tower_id}/{key.target}/"
            f"{generation_id}/model.joblib"
        )
        self.artifacts[path] = artifact
        return path

    def activate(
        self,
        generation_id: str,
        key: ModelKey,
        metadata: ModelMetadata,
        storage_path: str,
        *,
        expected_generation_id: str | None,
    ) -> bool:
        self.events.append("activate")
        current = self.heads.get(key)
        current_id = current.generation_id if current is not None else None
        if self.activation_mode == "conflict":
            if self.conflict_generation is not None:
                self.heads[key] = self.conflict_generation
            return False
        if current_id != expected_generation_id:
            return False
        if self.activation_mode == "activation_error":
            raise OSError("rpc unavailable")
        if self.activation_mode == "activation_transport_error":
            raise supabase_registry_module.SupabaseTransportError(
                "rpc unavailable"
            )
        generation = RemoteGeneration(
            generation_id=generation_id,
            key=key,
            model_version=metadata.model_version,
            storage_path=storage_path,
            model_checksum=metadata.checksum,
            metadata=metadata,
            status=metadata.status,
            revision=1 if current is None else current.revision + 1,
        )
        self.heads[key] = generation
        if self.activation_mode == "commit_then_timeout":
            raise OSError("rpc timeout")
        return True

    def was_rejected(self, attempt: ModelAttempt) -> bool:
        self.events.append("was_rejected")
        return attempt.fingerprint in self.rejections

    def record_rejection(self, attempt: ModelAttempt, reason: str) -> None:
        self.events.append("record_rejection")
        if self.fail_rejection_write:
            raise OSError("rejection table unavailable")
        self.rejections.add(attempt.fingerprint)
        self.recorded_reasons.append(reason)


class _FirstTransportFailureStore(_MemoryStore):
    def __init__(self):
        super().__init__()
        self.current_calls = 0

    def current(self, key: ModelKey) -> RemoteGeneration | None:
        self.current_calls += 1
        if self.current_calls == 1:
            error_type = getattr(
                supabase_registry_module,
                "SupabaseTransportError",
                OSError,
            )
            raise error_type("Supabase model head query failed")
        return super().current(key)


class _MissingObjectError(RuntimeError):
    status = 404


class _MissingObjectBucket:
    def download(self, _path):
        raise _MissingObjectError("object not found")


class _MissingObjectStorage:
    def from_(self, _bucket):
        return _MissingObjectBucket()


class _MissingObjectClient:
    storage = _MissingObjectStorage()


class _MissingObjectStore(_MemoryStore):
    def __init__(self):
        super().__init__()
        self.missing_key = None
        self.sdk_store = SupabaseModelStore(_MissingObjectClient())

    def download(self, generation: RemoteGeneration) -> bytes:
        if generation.key == self.missing_key:
            self.events.append("download")
            return self.sdk_store.download(generation)
        return super().download(generation)


def test_missing_remote_head_returns_model_not_found(tmp_path):
    registry = SupabaseModelRegistry(_MemoryStore(), cache_dir=tmp_path)
    key = ModelKey("project-a", "line-a", "001", "wind_speed")

    result = _load(registry, key)

    assert result.bundle is None
    assert result.fallback_reason == "model_not_found"


def test_first_candidate_is_uploaded_and_activated_before_success(tmp_path):
    store = _MemoryStore()
    registry = SupabaseModelRegistry(store, cache_dir=tmp_path)
    key = ModelKey("project-a", "line-a", "001", "wind_speed")

    decision = registry.promote(_candidate(key))

    assert decision.promoted is True
    assert store.events.index("upload") < store.events.index("activate")
    assert store.heads[key].metadata == decision.metadata


def test_fresh_registry_downloads_remote_generation_without_retraining(tmp_path):
    store = _MemoryStore()
    key = ModelKey("project-a", "line-a", "001", "wind_speed")
    first = SupabaseModelRegistry(store, cache_dir=tmp_path / "first")
    first.promote(_candidate(key, model_value=0.75))

    fresh = SupabaseModelRegistry(store, cache_dir=tmp_path / "fresh")
    loaded = _load(fresh, key)

    assert loaded.fallback_reason == ""
    assert loaded.metadata == store.heads[key].metadata
    assert loaded.bundle.model.predict(np.zeros((2, 2))).tolist() == [0.75, 0.75]
    assert "download" in store.events


def test_downloaded_generation_revalidates_runtime_compatibility(tmp_path):
    store = _MemoryStore()
    key = ModelKey("project-a", "line-a", "001", "ambient_temp")
    SupabaseModelRegistry(store, cache_dir=tmp_path / "first").promote(
        _candidate(key)
    )

    loaded = _load(
        SupabaseModelRegistry(store, cache_dir=tmp_path / "fresh"),
        key,
        suffix="different",
    )

    assert loaded.bundle is None
    assert loaded.fallback_reason == "incompatible_dem_hash"


def test_corrupt_remote_generation_is_isolated_to_its_key(tmp_path):
    store = _MemoryStore()
    bad_key = ModelKey("project-a", "line-a", "001", "wind_speed")
    good_key = ModelKey("project-a", "line-a", "002", "wind_speed")
    publisher = SupabaseModelRegistry(store, cache_dir=tmp_path / "publisher")
    publisher.promote(_candidate(bad_key))
    publisher.promote(_candidate(good_key, model_value=0.9))
    store.artifacts[store.heads[bad_key].storage_path] = b"corrupt"
    store.events.clear()

    fresh = SupabaseModelRegistry(store, cache_dir=tmp_path / "fresh")
    results = fresh.load_many(
        [bad_key, good_key],
        expected_compatibility={
            bad_key: _compatibility(),
            good_key: _compatibility(),
        },
        expected_training_contract_hash={
            bad_key: CONTRACT_HASH,
            good_key: CONTRACT_HASH,
        },
        expected_backend_id={bad_key: BACKEND_ID, good_key: BACKEND_ID},
    )

    assert results[bad_key].bundle is None
    assert results[bad_key].fallback_reason == "load_failed:OSError"
    assert results[good_key].bundle is not None
    assert store.events.count("current") == 2


def test_transport_failure_aborts_only_the_current_load_many_call(tmp_path):
    store = _FirstTransportFailureStore()
    registry = SupabaseModelRegistry(store, cache_dir=tmp_path)
    keys = [
        ModelKey("project-a", "line-a", tower_id, "wind_speed")
        for tower_id in ("001", "002", "003")
    ]

    results = registry.load_many(
        keys,
        expected_compatibility={key: _compatibility() for key in keys},
        expected_training_contract_hash={
            key: CONTRACT_HASH for key in keys
        },
        expected_backend_id={key: BACKEND_ID for key in keys},
    )

    assert store.current_calls == 1
    assert {
        result.fallback_reason for result in results.values()
    } == {"load_failed:SupabaseTransportError"}

    retried = _load(registry, keys[0])

    assert store.current_calls == 2
    assert retried.fallback_reason == "model_not_found"


def test_pipeline_run_circuit_blocks_follow_up_remote_reads(tmp_path):
    store = _FirstTransportFailureStore()
    registry = SupabaseModelRegistry(store, cache_dir=tmp_path)
    keys = [
        ModelKey("project-a", "line-a", tower_id, "wind_speed")
        for tower_id in ("001", "002", "003")
    ]

    registry.begin_pipeline_run()
    results = registry.load_many(
        keys,
        expected_compatibility={key: _compatibility() for key in keys},
        expected_training_contract_hash={key: CONTRACT_HASH for key in keys},
        expected_backend_id={key: BACKEND_ID for key in keys},
    )
    rejected = registry.was_rejected(_attempt(registry, _candidate(keys[0])))

    assert store.current_calls == 1
    assert registry.model_operations_available() is False
    assert rejected is False
    assert {
        result.fallback_reason for result in results.values()
    } == {"load_failed:SupabaseTransportError"}

    registry.end_pipeline_run()
    retried = _load(registry, keys[0])

    assert store.current_calls == 2
    assert retried.fallback_reason == "model_not_found"


def test_transport_upload_failure_opens_pipeline_run_circuit(tmp_path):
    store = _MemoryStore()
    store.activation_mode = "upload_transport_error"
    registry = SupabaseModelRegistry(store, cache_dir=tmp_path)
    key = ModelKey("project-a", "line-a", "001", "wind_speed")

    registry.begin_pipeline_run()
    decision = registry.promote(_candidate(key))

    assert decision.promoted is False
    assert decision.reason == "remote_upload_failed"
    assert registry.model_operations_available() is False
    assert store.events.count("upload") == 1

    registry.end_pipeline_run()
    assert registry.model_operations_available() is True


def test_uncommitted_activation_transport_failure_opens_run_circuit(tmp_path):
    store = _MemoryStore()
    store.activation_mode = "activation_transport_error"
    registry = SupabaseModelRegistry(store, cache_dir=tmp_path)
    key = ModelKey("project-a", "line-a", "001", "wind_speed")

    registry.begin_pipeline_run()
    decision = registry.promote(_candidate(key))

    assert decision.promoted is False
    assert decision.reason == "remote_activation_failed"
    assert registry.model_operations_available() is False
    assert store.events.count("activate") == 1

    registry.end_pipeline_run()


def test_missing_remote_object_is_isolated_to_its_key(tmp_path):
    store = _MissingObjectStore()
    missing_key = ModelKey("project-a", "line-a", "001", "wind_speed")
    good_key = ModelKey("project-a", "line-a", "002", "wind_speed")
    publisher = SupabaseModelRegistry(store, cache_dir=tmp_path / "publisher")
    publisher.promote(_candidate(missing_key))
    publisher.promote(_candidate(good_key, model_value=0.9))
    store.missing_key = missing_key
    store.events.clear()

    results = SupabaseModelRegistry(
        store, cache_dir=tmp_path / "fresh"
    ).load_many(
        [missing_key, good_key],
        expected_compatibility={
            missing_key: _compatibility(),
            good_key: _compatibility(),
        },
        expected_training_contract_hash={
            missing_key: CONTRACT_HASH,
            good_key: CONTRACT_HASH,
        },
        expected_backend_id={
            missing_key: BACKEND_ID,
            good_key: BACKEND_ID,
        },
    )

    assert results[missing_key].fallback_reason == "load_failed:OSError"
    assert results[good_key].bundle is not None
    assert store.events.count("current") == 2


@pytest.mark.parametrize(
    ("mode", "reason"),
    [
        ("upload_error", "remote_upload_failed"),
        ("activation_error", "remote_activation_failed"),
    ],
)
def test_remote_write_failure_never_exposes_local_candidate(
    tmp_path, mode, reason
):
    store = _MemoryStore()
    store.activation_mode = mode
    key = ModelKey("project-a", "line-a", "001", "wind_speed")
    registry = SupabaseModelRegistry(store, cache_dir=tmp_path)

    decision = registry.promote(_candidate(key))
    candidate_path_exists = registry.path_for(key).exists()
    loaded = _load(registry, key)

    assert decision.promoted is False
    assert decision.reason == reason
    assert candidate_path_exists is False
    assert loaded.bundle is None
    assert loaded.fallback_reason == "model_not_found"


def test_cas_conflict_preserves_and_loads_remote_winner(tmp_path):
    key = ModelKey("project-a", "line-a", "001", "wind_speed")
    winner_store = _MemoryStore()
    SupabaseModelRegistry(
        winner_store, cache_dir=tmp_path / "winner"
    ).promote(_candidate(key, model_version="winner", model_value=0.9))
    winner = winner_store.heads[key]

    store = _MemoryStore()
    store.artifacts[winner.storage_path] = winner_store.artifacts[
        winner.storage_path
    ]
    store.activation_mode = "conflict"
    store.conflict_generation = winner
    registry = SupabaseModelRegistry(store, cache_dir=tmp_path / "loser")

    decision = registry.promote(
        _candidate(key, model_version="loser", model_value=0.1)
    )
    loser_path_exists = registry.path_for(key).exists()
    loaded = _load(registry, key)

    assert decision.promoted is False
    assert decision.reason == "remote_head_conflict"
    assert loser_path_exists is False
    assert loaded.metadata.model_version == "winner"
    assert loaded.bundle.model.predict(np.zeros((1, 2))).tolist() == [0.9]


def test_activation_timeout_reconciles_committed_generation(tmp_path):
    store = _MemoryStore()
    store.activation_mode = "commit_then_timeout"
    key = ModelKey("project-a", "line-a", "001", "wind_speed")
    registry = SupabaseModelRegistry(store, cache_dir=tmp_path)

    decision = registry.promote(_candidate(key))

    assert decision.promoted is True
    assert store.heads[key].metadata == decision.metadata
    assert store.events.count("current") >= 2


def test_remote_activation_recovers_from_local_cache_publish_failure(
    tmp_path,
    monkeypatch,
):
    store = _MemoryStore()
    key = ModelKey("project-a", "line-a", "001", "wind_speed")
    registry = SupabaseModelRegistry(store, cache_dir=tmp_path)
    real_publish = ModelRegistry._publish_locked
    failed_once = False

    def fail_first_local_publish(self, *args, **kwargs):
        nonlocal failed_once
        if self is registry and not failed_once:
            failed_once = True
            raise OSError("local cache unavailable")
        return real_publish(self, *args, **kwargs)

    monkeypatch.setattr(
        ModelRegistry,
        "_publish_locked",
        fail_first_local_publish,
    )

    decision = registry.promote(_candidate(key, model_value=0.65))

    assert decision.promoted is True
    assert store.heads[key].metadata == decision.metadata
    assert registry.path_for(key).exists() is False

    loaded = _load(registry, key)

    assert loaded.fallback_reason == ""
    assert loaded.bundle.model.predict(np.zeros((1, 2))).tolist() == [0.65]
    assert registry.path_for(key).is_file()


def test_deterministic_rejection_survives_fresh_registry(tmp_path):
    store = _MemoryStore()
    key = ModelKey("project-a", "line-a", "001", "wind_speed")
    candidate = _candidate(key, corrected_mae=2.0)
    first = SupabaseModelRegistry(store, cache_dir=tmp_path / "first")
    attempt = _attempt(first, candidate)

    decision = first.promote(candidate, attempt=attempt)
    fresh = SupabaseModelRegistry(store, cache_dir=tmp_path / "fresh")

    assert decision.reason == "candidate_not_better_than_physical"
    assert fresh.was_rejected(attempt) is True
    assert store.recorded_reasons == ["candidate_not_better_than_physical"]


def test_rejection_write_failure_remains_retryable(tmp_path):
    store = _MemoryStore()
    store.fail_rejection_write = True
    key = ModelKey("project-a", "line-a", "001", "wind_speed")
    candidate = _candidate(key, corrected_mae=2.0)
    registry = SupabaseModelRegistry(store, cache_dir=tmp_path)
    attempt = _attempt(registry, candidate)

    decision = registry.promote(candidate, attempt=attempt)

    assert decision.reason == "attempt_record_failed:OSError"
    assert registry.was_rejected(attempt) is False


def test_nondeterministic_and_malformed_rejections_are_not_persisted(tmp_path):
    store = _MemoryStore()
    key = ModelKey("project-a", "line-a", "001", "wind_speed")
    registry = SupabaseModelRegistry(store, cache_dir=tmp_path)
    first = _candidate(key, model_version="same", corrected_mae=1.0)
    assert registry.promote(first).promoted is True
    champion = store.heads[key].metadata
    conflict = _candidate(key, model_version="same", corrected_mae=0.5)
    attempt = _attempt(registry, conflict, champion=champion)

    decision = registry.promote(conflict, attempt=attempt)

    assert decision.reason == "model_version_conflict"
    assert store.recorded_reasons == []
    with pytest.raises(ValueError, match="deterministic"):
        registry._record_attempt_rejection_locked(attempt, "  ")


def test_cached_remote_generation_is_rehydrated_after_failed_local_publish(
    tmp_path,
):
    store = _MemoryStore()
    key = ModelKey("project-a", "line-a", "001", "wind_speed")
    registry = SupabaseModelRegistry(store, cache_dir=tmp_path)
    registry.promote(_candidate(key, model_version="winner", model_value=0.8))
    winner_path = registry.path_for(key).resolve(strict=True)
    store.activation_mode = "activation_error"

    failed = registry.promote(
        _candidate(
            key,
            model_version="loser",
            model_value=0.1,
            corrected_mae=0.5,
        )
    )
    active_path_after_failure = registry.path_for(key).resolve(strict=True)
    loaded = _load(registry, key)

    assert failed.promoted is False
    assert active_path_after_failure == winner_path
    assert loaded.metadata.model_version == "winner"
    assert loaded.bundle.model.predict(np.zeros((1, 2))).tolist() == [0.8]


def test_remote_metadata_checksum_must_survive_local_hydration(tmp_path):
    store = _MemoryStore()
    key = ModelKey("project-a", "line-a", "001", "wind_speed")
    SupabaseModelRegistry(store, cache_dir=tmp_path / "first").promote(
        _candidate(key)
    )
    remote_before = store.heads[key]

    loaded = _load(
        SupabaseModelRegistry(store, cache_dir=tmp_path / "fresh"), key
    )

    assert loaded.metadata.checksum == remote_before.model_checksum
    assert loaded.metadata == remote_before.metadata


def _weather_segment(
    role: str,
    *,
    segment_index: int,
    truth_offset: bool = False,
) -> pd.DataFrame:
    timestamps = pd.to_datetime(
        ["2026-07-23 00:00", "2026-07-23 00:30"]
    ).tz_localize("Asia/Shanghai") + pd.Timedelta(days=2 * segment_index)
    wind = np.array([2.0, 4.0])
    temperature = np.array([30.0, 32.0])
    if truth_offset:
        wind = wind + np.array([0.75, 1.5])
        temperature = temperature + np.array([-1.0, -2.0])
    return pd.DataFrame(
        {
            "tower_id": ["001", "001"],
            "timestamp": timestamps,
            "ambient_temp": temperature,
            "wind_speed": wind,
            "wind_direction": [90.0, 100.0],
            "solar_radiation": [0.0, 10.0],
            "humidity": [30.0, 31.0],
            "elevation": [1000.0, 1000.0],
            "dataset_role": role,
            "source_file_hash": [f"{role}-{segment_index}"] * 2,
        }
    )


def _weather_history(role: str, *, truth_offset: bool = False) -> pd.DataFrame:
    return pd.concat(
        [
            _weather_segment(
                role,
                segment_index=index,
                truth_offset=truth_offset,
            )
            for index in range(3)
        ],
        ignore_index=True,
    )


def _conductor() -> dict[str, float]:
    return {
        "D0": 0.0281,
        "R_low_25": 7.283e-5,
        "R_high_75": 8.688e-5,
        "R_high_200": 1.220e-4,
        "emissivity": 0.8,
        "absorptivity": 0.8,
        "max_allow_temp": 80.0,
        "latitude": 39.9,
        "longitude": 116.4,
        "line_azimuth": 90.0,
    }


def test_real_xgboost_generation_is_reused_by_a_fresh_registry():
    store = _MemoryStore()
    first_registry = SupabaseModelRegistry(store)
    first = DlrPipeline(registry=first_registry).run(
        physical=_weather_history("physical"),
        truth=_weather_history("truth", truth_offset=True),
        project_id="project-a",
        line_id="line-a",
        terrain_lookup={},
        ai_enabled=True,
        conductor=_conductor(),
    )
    store.events.clear()

    fresh_registry = SupabaseModelRegistry(store)
    second = DlrPipeline(registry=fresh_registry).run(
        physical=_weather_segment("physical", segment_index=6),
        truth=None,
        project_id="project-a",
        line_id="line-a",
        terrain_lookup={},
        ai_enabled=True,
        conductor=_conductor(),
    )

    expected_keys = tuple(
        ModelKey("project-a", "line-a", "001", target)
        for target in ("wind_speed", "ambient_temp")
    )
    assert first.model_report.trained_targets == expected_keys
    assert second.model_report.loaded_targets == expected_keys
    assert second.model_report.used_targets == expected_keys
    assert second.model_report.fallbacks == ()
    assert "download" in store.events
    assert "upload" not in store.events
    assert "activate" not in store.events
