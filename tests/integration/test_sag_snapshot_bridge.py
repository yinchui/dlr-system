import copy
import hashlib
from io import BytesIO
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import modules.sag_validation as sag_validation_module
from modules.dlr_pipeline import DlrPipeline
from modules.sag_validation import (
    build_visible_sag_result,
    publish_sag_snapshot,
    run_sag_validation,
)
from tests.fixtures.sag_data import drake_conductor, make_line_data
from utils.audit_log import JsonAuditLogger


class _FixedCurrentThermalAdapter:
    def __init__(self):
        self.raw_currents = None

    def calculate_from_long_frame(self, weather, *, base_params):
        shape = (
            weather["tower_id"].nunique(),
            weather["timestamp"].nunique(),
        )
        self.raw_currents = np.full(shape, 1000.0)
        return {
            "max_currents": self.raw_currents.copy(),
            "corrected_winds": np.ones(shape),
            "local_temps": np.full(shape, 25.0),
        }


def _minimal_pipeline_weather():
    timestamps = pd.to_datetime(
        [
            "2026-07-23 00:00",
            "2026-07-23 00:30",
            "2026-07-23 00:00",
            "2026-07-23 00:30",
        ]
    ).tz_localize("Asia/Shanghai")
    return pd.DataFrame(
        {
            "tower_id": ["001", "001", "002", "002"],
            "timestamp": timestamps,
            "ambient_temp": [25.0, 26.0, 24.0, 25.0],
            "wind_speed": [2.0, 2.5, 3.0, 3.5],
            "wind_direction": [90.0, 100.0, 110.0, 120.0],
            "solar_radiation": [0.0, 10.0, 0.0, 20.0],
            "humidity": [30.0, 31.0, 32.0, 33.0],
            "elevation": [1000.0, 1000.0, 1010.0, 1010.0],
            "dataset_role": ["physical"] * 4,
            "source_file_hash": ["sag-bridge-weather"] * 4,
        }
    )


def _angle_only_frame():
    return pd.DataFrame({"倾角": [1.0, 1.1]})


def _empty_xlsx_bytes():
    buffer = BytesIO()
    pd.DataFrame(columns=["倾角"]).to_excel(buffer, index=False)
    return buffer.getvalue()


def test_sag_snapshot_receives_only_factored_pipeline_ratings(tmp_path):
    adapter = _FixedCurrentThermalAdapter()
    conductor = drake_conductor()
    result = DlrPipeline(
        model_root=tmp_path,
        thermal_adapter=adapter,
    ).run(
        physical=_minimal_pipeline_weather(),
        project_id="project-a",
        line_id="line-a",
        terrain_lookup={},
        ai_enabled=False,
        conductor=conductor,
    )
    legacy = result.to_legacy_line_data()
    snapshot = publish_sag_snapshot(
        {},
        legacy,
        conductor,
        source_run_id="dlr-factored-run",
        line_id="line-a",
    )
    raw = np.full((2, 2), 1000.0)
    expected = np.full((2, 2), 800.0)

    np.testing.assert_array_equal(adapter.raw_currents, raw)
    np.testing.assert_array_equal(snapshot.original_currents, expected)

    legacy["max_currents"][:] = 0.0
    np.testing.assert_array_equal(snapshot.original_currents, expected)


def test_completed_dlr_publishes_deep_copied_sag_snapshot(tmp_path):
    line_data = make_line_data()
    before = copy.deepcopy(line_data)
    state = {}

    snapshot = publish_sag_snapshot(
        state,
        line_data,
        drake_conductor(),
        tower_coords=line_data["tower_coords"],
        source_run_id="dlr-run-complete",
        line_id="line-a",
    )
    result = run_sag_validation(
        snapshot,
        _angle_only_frame(),
        selected_tower_id="001",
        output_dir=tmp_path / "results",
        audit_logger=JsonAuditLogger(tmp_path / "logs"),
    )

    np.testing.assert_array_equal(
        line_data["max_currents"], before["max_currents"]
    )
    assert line_data["tower_coords"] == before["tower_coords"]
    assert snapshot.source_run_id == "dlr-run-complete"
    assert snapshot.line_id == "line-a"
    assert state["sag_validation_snapshot"] is snapshot
    assert result.result_path is not None and result.result_path.exists()
    assert result.audit_persisted is True


def test_failed_snapshot_build_preserves_previous_complete_snapshot():
    previous = publish_sag_snapshot(
        {},
        make_line_data(),
        drake_conductor(),
        source_run_id="previous-run",
    )
    state = {"sag_validation_snapshot": previous}
    invalid = make_line_data()
    invalid["max_currents"] = np.ones((1, 1))

    with pytest.raises(ValueError, match="max_currents"):
        publish_sag_snapshot(
            state,
            invalid,
            drake_conductor(),
            source_run_id="failed-run",
        )

    assert state["sag_validation_snapshot"] is previous


def test_backend_payload_keeps_sources_but_never_uploaded_bytes(tmp_path):
    input_hash = "a" * 64
    snapshot = publish_sag_snapshot(
        {},
        make_line_data(),
        drake_conductor(),
        source_run_id="dlr-run-1",
        line_id="line-a",
    )

    result = run_sag_validation(
        snapshot,
        _angle_only_frame(),
        selected_tower_id="001",
        output_dir=tmp_path,
        audit_logger=JsonAuditLogger(tmp_path / "logs"),
        input_hash=input_hash,
    )

    payload = json.loads(result.result_path.read_text())
    assert payload["result_id"] == result.result_id
    assert payload["input_hash"] == input_hash
    assert payload["source_run_id"] == "dlr-run-1"
    assert payload["rows"][0]["parameter_sources"]
    persisted = result.result_path.read_text(encoding="utf-8").lower()
    assert '"raw_bytes"' not in persisted
    assert '"content"' not in persisted


def test_visible_projection_hides_parameter_sources_and_assumptions(tmp_path):
    snapshot = publish_sag_snapshot(
        {}, make_line_data(), drake_conductor(), source_run_id="dlr-run-1"
    )
    result = run_sag_validation(
        snapshot,
        _angle_only_frame(),
        selected_tower_id="001",
        output_dir=tmp_path,
    )

    visible = build_visible_sag_result(result)

    hidden_words = " ".join(map(str, visible.columns)).lower()
    assert "parameter_sources" not in visible.columns
    assert "source" not in hidden_words
    assert "default" not in hidden_words
    assert "assumption" not in hidden_words
    assert list(visible.columns) == [
        "杆塔",
        "时间",
        "状态",
        "反推温度(°C)",
        "理论温度(°C)",
        "温差(°C)",
        "校验电流(A)",
        "最终电流(A)",
    ]


def test_audit_failure_does_not_change_sag_numbers(tmp_path):
    class FailingLogger:
        def write(self, event):
            raise OSError("audit unavailable")

    snapshot = publish_sag_snapshot(
        {}, make_line_data(), drake_conductor(), source_run_id="dlr-run-1"
    )
    baseline = run_sag_validation(
        snapshot,
        _angle_only_frame(),
        selected_tower_id="001",
        output_dir=tmp_path / "baseline",
    )
    degraded = run_sag_validation(
        snapshot,
        _angle_only_frame(),
        selected_tower_id="001",
        output_dir=tmp_path / "degraded",
        audit_logger=FailingLogger(),
    )

    assert degraded.audit_persisted is False
    assert [row.final_current_a for row in degraded.rows] == [
        row.final_current_a for row in baseline.rows
    ]


def test_inclination_normalization_failure_is_audited_and_re_raised(tmp_path):
    snapshot = publish_sag_snapshot(
        {}, make_line_data(), drake_conductor(), source_run_id="dlr-run-1"
    )
    logger = JsonAuditLogger(tmp_path / "logs")

    with pytest.raises(ValueError, match="倾角列"):
        run_sag_validation(
            snapshot,
            pd.DataFrame({"无关列": [1.0]}),
            output_dir=tmp_path / "results",
            audit_logger=logger,
            input_hash="b" * 64,
        )

    event = json.loads(logger.log_path.read_text(encoding="utf-8"))
    assert event["stage"] == "sag_validation"
    assert event["error_code"] == "inclination_normalization_failed:ValueError"
    assert event["fallback_reason"] == event["error_code"]
    assert event["details"]["result_persisted"] is False


def test_upload_parse_failure_is_audited_without_raw_bytes(tmp_path):
    class UploadedInclination:
        name = "angles.txt"

        @staticmethod
        def getvalue():
            return b"private inclination bytes"

    snapshot = publish_sag_snapshot(
        {}, make_line_data(), drake_conductor(), source_run_id="dlr-run-1"
    )
    logger = JsonAuditLogger(tmp_path / "logs")

    with pytest.raises(ValueError, match="CSV 或 XLSX"):
        sag_validation_module.run_sag_validation_upload(
            snapshot,
            UploadedInclination(),
            output_dir=tmp_path / "results",
            audit_logger=logger,
        )

    event = json.loads(logger.log_path.read_text(encoding="utf-8"))
    assert event["stage"] == "sag_input_parse"
    assert event["error_code"] == "input_parse_failed:ValueError"
    assert event["input_hash"] == hashlib.sha256(
        UploadedInclination.getvalue()
    ).hexdigest()
    assert b"private inclination bytes" not in logger.log_path.read_bytes()


@pytest.mark.parametrize(
    ("name", "content"),
    [
        ("empty.csv", "倾角\n".encode()),
        ("empty.xlsx", _empty_xlsx_bytes()),
    ],
)
def test_empty_inclination_upload_is_rejected_and_audited(
    tmp_path, name, content
):
    class EmptyUpload:
        def __init__(self):
            self.name = name

        @staticmethod
        def getvalue():
            return content

    snapshot = publish_sag_snapshot(
        {}, make_line_data(), drake_conductor(), source_run_id="dlr-run-1"
    )
    logger = JsonAuditLogger(tmp_path / "logs")

    with pytest.raises(ValueError, match="至少包含一行"):
        sag_validation_module.run_sag_validation_upload(
            snapshot,
            EmptyUpload(),
            output_dir=tmp_path / "results",
            audit_logger=logger,
        )

    event = json.loads(logger.log_path.read_text(encoding="utf-8"))
    assert event["error_code"] == "inclination_normalization_failed:ValueError"
    assert event["details"]["sample_count"] == 0
    assert list((tmp_path / "results").glob("*.json")) == []


def test_audit_summarizes_row_errors_and_parameter_sources(tmp_path):
    snapshot = publish_sag_snapshot(
        {}, make_line_data(), drake_conductor(), source_run_id="dlr-run-1"
    )
    logger = JsonAuditLogger(tmp_path / "logs")

    run_sag_validation(
        snapshot,
        pd.DataFrame({"倾角": [0.0, 1.0]}),
        selected_tower_id="001",
        output_dir=tmp_path / "results",
        audit_logger=logger,
    )

    event = json.loads(logger.log_path.read_text(encoding="utf-8"))
    assert event["error_code"] == "invalid_angle"
    assert event["details"]["error_code_counts"] == {"invalid_angle": 1}
    assert event["details"]["parameter_source_counts"]["area_m2"] == {
        "measured": 1
    }


def test_result_persistence_failure_is_audited_and_observable(
    tmp_path, monkeypatch
):
    def fail_write(*args, **kwargs):
        raise OSError("storage unavailable")

    monkeypatch.setattr(sag_validation_module, "write_result_atomic", fail_write)
    snapshot = publish_sag_snapshot(
        {}, make_line_data(), drake_conductor(), source_run_id="dlr-run-1"
    )
    logger = JsonAuditLogger(tmp_path / "logs")

    result = run_sag_validation(
        snapshot,
        _angle_only_frame(),
        selected_tower_id="001",
        output_dir=tmp_path / "results",
        audit_logger=logger,
    )

    assert result.result_path is None
    event = json.loads(logger.log_path.read_text(encoding="utf-8"))
    assert event["error_code"] == "result_persist_failed:OSError"
    assert event["details"]["result_persisted"] is False


def test_audit_event_construction_failure_does_not_change_sag_numbers(
    tmp_path, monkeypatch
):
    snapshot = publish_sag_snapshot(
        {}, make_line_data(), drake_conductor(), source_run_id="dlr-run-1"
    )
    baseline = run_sag_validation(
        snapshot,
        _angle_only_frame(),
        selected_tower_id="001",
        output_dir=tmp_path / "baseline",
    )

    def fail_event(*args, **kwargs):
        raise RuntimeError("audit event unavailable")

    monkeypatch.setattr(sag_validation_module, "AuditEvent", fail_event)
    degraded = run_sag_validation(
        snapshot,
        _angle_only_frame(),
        selected_tower_id="001",
        output_dir=tmp_path / "degraded",
        audit_logger=JsonAuditLogger(tmp_path / "logs"),
    )

    assert degraded.audit_persisted is False
    assert degraded.audit_events == ()
    assert [row.final_current_a for row in degraded.rows] == [
        row.final_current_a for row in baseline.rows
    ]


def test_audit_event_failure_never_masks_normalization_error(tmp_path, monkeypatch):
    snapshot = publish_sag_snapshot(
        {}, make_line_data(), drake_conductor(), source_run_id="dlr-run-1"
    )

    def fail_event(*args, **kwargs):
        raise RuntimeError("audit event unavailable")

    monkeypatch.setattr(sag_validation_module, "AuditEvent", fail_event)

    with pytest.raises(ValueError, match="倾角列"):
        run_sag_validation(
            snapshot,
            pd.DataFrame({"无关列": [1.0]}),
            output_dir=tmp_path / "results",
            audit_logger=JsonAuditLogger(tmp_path / "logs"),
        )


def test_main_page_publishes_snapshot_only_after_complete_dlr_result():
    source = Path("dispatch_app_st.py").read_text(encoding="utf-8")
    button_start = source.index("if btn_generate and weather_files:")
    button_end = source.index("    # 结果展示", button_start)
    block = source[button_start:button_end]

    assert "publish_sag_snapshot(\n                st.session_state" in block
    assert "source_run_id=dlr_run_id" in block
    assert "line_id=line_identity.line_id" in block
