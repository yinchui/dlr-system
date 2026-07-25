import json
import multiprocessing
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path

import numpy as np
import pytest

import modules.model_registry as model_registry_module
import utils.audit_log as audit_log
from modules.ai_prediction import ModelBundle
from modules.ai_training import ConstantResidualEstimator
from modules.model_registry import (
    ModelCandidate,
    ModelCompatibility,
    ModelKey,
    ModelMetadata,
    ModelRegistry,
)
from utils.audit_log import AuditEvent, JsonAuditLogger, write_result_atomic


class _AuditSource(Enum):
    TEST = "test"


def _event(index=0, **changes):
    values = {
        "run_id": "run-1",
        "result_id": f"result-{index}",
        "line_id": "line-1",
        "tower_id": "001",
        "stage": "test_stage",
        "input_hash": "input-hash",
        "config_hash": "config-hash",
        "source": "test",
        "fallback_reason": "",
        "details": {"index": index},
    }
    values.update(changes)
    return AuditEvent(**values)


def _process_audit_writer(output_dir, process_index, start_event):
    logger = JsonAuditLogger(output_dir)
    if not start_event.wait(10):
        raise TimeoutError("audit process start event was not released")
    for item_index in range(20):
        persisted = logger.write(
            _event(
                process_index * 100 + item_index,
                result_id=f"process-{process_index}-{item_index}",
            )
        )
        if not persisted:
            raise OSError("audit event was not persisted")


def _compatibility(suffix="a"):
    return ModelCompatibility(
        dem_hash=f"dem-{suffix}",
        crs_hash=f"crs-{suffix}",
        coordinate_hash=f"coordinates-{suffix}",
        conductor_hash=f"conductor-{suffix}",
        feature_version=f"features-{suffix}",
        correction_config_hash=f"correction-{suffix}",
    )


def _candidate(key, *, version="version-1", corrected_mae=1.0):
    metrics = {
        "baseline_mae": 2.0,
        "baseline_rmse": 2.5,
        "corrected_mae": corrected_mae,
        "corrected_rmse": 1.5,
    }
    metadata = ModelMetadata(
        key=key,
        model_version=version,
        feature_columns=("wind_speed_local", "lag_1"),
        training_params={"max_depth": 3},
        random_seed=42,
        time_start="2025-01-01T00:00:00+08:00",
        time_end="2025-01-02T00:00:00+08:00",
        sample_count=4,
        evaluation_mode="temporal_holdout",
        metrics=metrics,
        full_fit_metrics=None,
        residual_bounds=(-2.0, 2.0),
        input_data_hash=f"input-{version}",
        evaluation_set_hash="evaluation-a",
        compatibility=_compatibility(),
        dependency_versions={"python": "3.11", "joblib": "1.5.3"},
        cadence_minutes=30.0,
        training_contract_hash="c" * 64,
        backend_id="xgboost-residual-v1",
        training_outcome="data_fallback",
    )
    bundle = ModelBundle(
        target_name=key.target,
        feature_columns=list(metadata.feature_columns),
        model=ConstantResidualEstimator(0.5),
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


def _load(registry, key, *, compatibility=None):
    return registry.load(
        key,
        expected_compatibility=compatibility or _compatibility(),
        expected_training_contract_hash="c" * 64,
        expected_backend_id="xgboost-residual-v1",
    )


def _payloads(output_dir):
    log_path = next(output_dir.glob("*.jsonl"))
    return [json.loads(line) for line in log_path.read_text().splitlines()]


def test_audit_event_has_required_trace_fields(tmp_path):
    logger = JsonAuditLogger(tmp_path)

    assert logger.write(AuditEvent.example()) is True

    payload = _payloads(tmp_path)[0]
    assert {
        "run_id",
        "result_id",
        "line_id",
        "tower_id",
        "stage",
        "input_hash",
        "config_hash",
        "source",
        "fallback_reason",
    } <= payload.keys()


def test_jsonl_supports_datetime_numpy_scalar_and_enum(tmp_path):
    logger = JsonAuditLogger(tmp_path)
    timestamp = datetime(2025, 1, 2, 3, 4, tzinfo=timezone.utc)

    persisted = logger.write(
        _event(
            timestamp=timestamp,
            details={
                "count": np.int64(7),
                "score": np.float64(1.25),
                "source": _AuditSource.TEST,
                "observed_at": timestamp,
            },
        )
    )

    assert persisted is True
    payload = _payloads(tmp_path)[0]
    assert payload["timestamp"] == timestamp.isoformat()
    assert payload["details"] == {
        "count": 7,
        "score": 1.25,
        "source": "test",
        "observed_at": timestamp.isoformat(),
    }


def test_audit_rejects_uploaded_bytes_without_persisting_them(tmp_path):
    logger = JsonAuditLogger(tmp_path)

    persisted = logger.write(_event(details={"uploaded_content": b"secret"}))

    assert persisted is False
    assert list(tmp_path.glob("*.jsonl")) == []
    assert not any(b"secret" in path.read_bytes() for path in tmp_path.iterdir())


def test_atomic_result_write_never_leaves_partial_target(tmp_path):
    target = write_result_atomic(tmp_path, "result-1", {"ok": True})

    assert json.loads(target.read_text()) == {"ok": True}
    assert list(tmp_path.glob("*.tmp")) == []


def test_atomic_result_write_supports_safe_special_scalars(tmp_path):
    observed_at = datetime(2025, 1, 2, tzinfo=timezone.utc)

    target = write_result_atomic(
        tmp_path,
        "result-1",
        {
            "observed_at": observed_at,
            "count": np.int64(3),
            "source": _AuditSource.TEST,
        },
    )

    assert json.loads(target.read_text()) == {
        "observed_at": observed_at.isoformat(),
        "count": 3,
        "source": "test",
    }


def test_atomic_result_rejects_bytes_and_preserves_existing_target(tmp_path):
    target = write_result_atomic(tmp_path, "result-1", {"version": "original"})
    original = target.read_bytes()

    with pytest.raises(TypeError):
        write_result_atomic(tmp_path, "result-1", {"raw": b"secret"})

    assert target.read_bytes() == original
    assert list(tmp_path.glob("*.tmp")) == []
    assert b"secret" not in target.read_bytes()


def test_atomic_result_replace_failure_cleans_temp_and_partial_target(
    tmp_path, monkeypatch
):
    def fail_replace(source, target):
        raise OSError("replace failed")

    monkeypatch.setattr(audit_log.os, "replace", fail_replace)

    with pytest.raises(OSError, match="replace failed"):
        write_result_atomic(tmp_path, "result-1", {"ok": True})

    assert not (tmp_path / "result-1.json").exists()
    assert list(tmp_path.glob("*.tmp")) == []


@pytest.mark.parametrize("result_id", ["", ".", "..", "../escape", "a/b"])
def test_atomic_result_rejects_unsafe_result_id(tmp_path, result_id):
    with pytest.raises(ValueError):
        write_result_atomic(tmp_path, result_id, {"ok": True})


def test_threaded_jsonl_writers_leave_complete_single_line_events(tmp_path):
    logger = JsonAuditLogger(tmp_path)

    with ThreadPoolExecutor(max_workers=8) as executor:
        persisted = list(executor.map(lambda index: logger.write(_event(index)), range(80)))

    payloads = _payloads(tmp_path)
    assert all(persisted)
    assert len(payloads) == 80
    assert {payload["details"]["index"] for payload in payloads} == set(range(80))


def test_process_jsonl_writers_leave_complete_single_line_events(tmp_path):
    context = multiprocessing.get_context("spawn")
    start_event = context.Event()
    processes = [
        context.Process(
            target=_process_audit_writer,
            args=(tmp_path, process_index, start_event),
        )
        for process_index in range(3)
    ]
    for process in processes:
        process.start()
    start_event.set()
    for process in processes:
        process.join(15)
    alive = [process for process in processes if process.is_alive()]
    for process in alive:
        process.terminate()
        process.join(5)

    assert not alive
    assert all(process.exitcode == 0 for process in processes)
    assert len(_payloads(tmp_path)) == 60


def test_model_registry_audits_training_load_promotion_invalidation_prune_and_fallback(
    tmp_path,
):
    logger = JsonAuditLogger(tmp_path / "logs")
    registry = ModelRegistry(
        tmp_path / "models",
        max_generations=1,
        audit_logger=logger,
        audit_run_id="run-a",
        audit_result_id="result-a",
    )
    key = ModelKey("project-a", "line-a", "001", "wind_speed")

    first = registry.promote(_candidate(key))
    loaded = _load(registry, key)
    second = registry.promote(
        _candidate(key, version="version-2", corrected_mae=0.5)
    )
    invalidated = _load(registry, key, compatibility=_compatibility("changed"))

    assert first.promoted is True
    assert first.audit_persisted is True
    assert loaded.bundle is not None
    assert loaded.audit_persisted is True
    assert second.promoted is True
    assert second.audit_persisted is True
    assert invalidated.bundle is None
    assert invalidated.fallback_reason == "incompatible_dem_hash"
    assert invalidated.audit_persisted is True
    payloads = _payloads(tmp_path / "logs")
    stages = {payload["stage"] for payload in payloads}
    assert {
        "model_training",
        "model_load",
        "model_promotion",
        "model_invalidation",
        "model_prune",
        "model_fallback",
    } <= stages
    assert all(payload["run_id"] == "run-a" for payload in payloads)
    assert all(payload["result_id"] == "result-a" for payload in payloads)
    assert all(payload["line_id"] == "line-a" for payload in payloads)
    assert all(payload["tower_id"] == "001" for payload in payloads)


def test_audit_failure_is_observable_without_changing_registry_decisions(tmp_path):
    class FailingAuditLogger:
        def write(self, event):
            raise OSError("audit storage unavailable")

    registry = ModelRegistry(
        tmp_path / "models",
        max_generations=1,
        audit_logger=FailingAuditLogger(),
        audit_run_id="run-a",
        audit_result_id="result-a",
    )
    key = ModelKey("project-a", "line-a", "001", "wind_speed")

    first = registry.promote(_candidate(key))
    second = registry.promote(
        _candidate(key, version="version-2", corrected_mae=0.5)
    )
    loaded = _load(registry, key)

    assert first.promoted is True
    assert first.reason == "promoted_provisional"
    assert first.audit_persisted is False
    assert second.promoted is True
    assert second.reason == "promoted"
    assert second.audit_persisted is False
    assert loaded.bundle is not None
    assert loaded.metadata.model_version == "version-2"
    assert loaded.fallback_reason == ""
    assert loaded.audit_persisted is False


def test_audit_event_construction_failure_does_not_change_registry_decision(
    tmp_path, monkeypatch
):
    class RaisingAuditEvent:
        def __init__(self, **kwargs):
            raise OSError("audit event unavailable")

    class ConfiguredAuditLogger:
        def write(self, event):
            raise AssertionError("invalid event must never reach the logger")

    monkeypatch.setattr(model_registry_module, "AuditEvent", RaisingAuditEvent)
    registry = ModelRegistry(
        tmp_path / "models",
        audit_logger=ConfiguredAuditLogger(),
        audit_run_id="run-a",
        audit_result_id="result-a",
    )
    key = ModelKey("project-a", "line-a", "001", "wind_speed")

    decision = registry.promote(_candidate(key))

    assert decision.promoted is True
    assert decision.reason == "promoted_provisional"
    assert decision.audit_persisted is False


def test_streamlit_pipeline_enables_runtime_model_auditing():
    app_source = (Path(__file__).parents[2] / "dispatch_app_st.py").read_text(
        encoding="utf-8"
    )

    assert "JsonAuditLogger(AUDIT_LOG_DIR)" in app_source
    assert "ModelRegistry(" in app_source
