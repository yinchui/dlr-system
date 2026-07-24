import numpy as np
import pandas as pd
import pytest
from pathlib import Path

from modules.ai_training import ResidualTrainer
from modules.dlr_pipeline import DlrPipeline, LongFrameThermalAdapter
from modules.model_registry import ModelKey
from modules.weather_correction import CorrectionOptions


def _weather(role: str, *, truth_offset: bool = False) -> pd.DataFrame:
    timestamps = pd.to_datetime(
        [
            "2026-07-23 00:00",
            "2026-07-23 00:30",
            "2026-07-23 00:00",
            "2026-07-23 00:30",
        ]
    ).tz_localize("Asia/Shanghai")
    wind = np.array([2.0, 2.5, 3.0, 3.5])
    temp = np.array([30.0, 31.0, 28.0, 29.0])
    if truth_offset:
        wind = wind + 1.0
        temp = temp - 2.0
    return pd.DataFrame(
        {
            "tower_id": ["001", "001", "002", "002"],
            "timestamp": timestamps,
            "ambient_temp": temp,
            "wind_speed": wind,
            "wind_direction": [90.0, 100.0, 110.0, 120.0],
            "solar_radiation": [0.0, 10.0, 0.0, 20.0],
            "humidity": [30.0, 31.0, 32.0, 33.0],
            "elevation": [1000.0, 1000.0, 1200.0, 1200.0],
            "dataset_role": role,
            "source_file_hash": [f"{role}-source"] * 4,
        }
    )


def _conductor() -> dict:
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


def test_pipeline_trains_missing_models_then_reuses_them(tmp_path):
    pipeline = DlrPipeline(model_root=tmp_path)
    terrain = {
        "001": {"elevation": 1000.0, "slope": 0.0, "aspect": 0.0},
        "002": {"elevation": 1200.0, "slope": 0.0, "aspect": 0.0},
    }

    first = pipeline.run(
        physical=_weather("physical"),
        truth=_weather("truth", truth_offset=True),
        project_id="project-a",
        line_id="line-a",
        interval_minutes=30,
        terrain_lookup=terrain,
        ai_enabled=True,
        conductor=_conductor(),
        truth_tolerance="5min",
    )
    second = pipeline.run(
        physical=_weather("physical"),
        truth=None,
        project_id="project-a",
        line_id="line-a",
        interval_minutes=30,
        terrain_lookup=terrain,
        ai_enabled=True,
        conductor=_conductor(),
    )

    assert first.model_report.trained_targets
    assert second.model_report.loaded_targets == first.model_report.trained_targets
    assert first.model_report.active_model_count == 4
    np.testing.assert_allclose(first.max_currents, second.max_currents)


class _SpyThermalAdapter:
    def __init__(self):
        self.last_conductor = None
        self.last_weather_columns = None

    def calculate_from_long_frame(self, weather, *, base_params):
        self.last_conductor = dict(base_params)
        self.last_weather_columns = tuple(weather.columns)
        tower_count = weather["tower_id"].nunique()
        time_count = weather["timestamp"].nunique()
        return {
            "max_currents": np.full((tower_count, time_count), 1000.0),
            "corrected_winds": np.ones((tower_count, time_count)),
            "local_temps": np.full((tower_count, time_count), 25.0),
        }


def test_pipeline_passes_selected_conductor_and_never_truth_to_dlr(tmp_path):
    spy = _SpyThermalAdapter()
    conductor = _conductor() | {"D0": 0.0338}
    result = DlrPipeline(model_root=tmp_path, thermal_adapter=spy).run(
        physical=_weather("physical"),
        truth=_weather("truth", truth_offset=True),
        project_id="project-a",
        line_id="line-a",
        interval_minutes=30,
        terrain_lookup={},
        ai_enabled=False,
        conductor=conductor,
        truth_tolerance="5min",
    )

    assert spy.last_conductor["D0"] == conductor["D0"]
    assert all("truth" not in column.lower() for column in spy.last_weather_columns)
    assert result.max_currents.shape[0] == 2


def test_pipeline_uses_only_the_common_tower_timestamps(tmp_path):
    physical = _weather("physical").iloc[[0, 1, 2, 3]].copy()
    extra_early = physical.iloc[[0]].copy()
    extra_early["timestamp"] = pd.Timestamp(
        "2026-07-22 23:30", tz="Asia/Shanghai"
    )
    extra_late = physical.iloc[[2]].copy()
    extra_late["timestamp"] = pd.Timestamp(
        "2026-07-23 01:00", tz="Asia/Shanghai"
    )
    physical = pd.concat(
        [extra_early, physical, extra_late], ignore_index=True
    ).sort_values(["tower_id", "timestamp"], ignore_index=True)
    physical["timestamp"] = pd.to_datetime(
        physical["timestamp"], utc=True
    ).dt.tz_convert("Asia/Shanghai")

    result = DlrPipeline(model_root=tmp_path).run(
        physical=physical,
        project_id="project-a",
        line_id="line-a",
        interval_minutes=30,
        terrain_lookup={},
        ai_enabled=False,
        conductor=_conductor(),
    )

    expected = pd.to_datetime(
        ["2026-07-23 00:00", "2026-07-23 00:30"]
    ).tz_localize("Asia/Shanghai")
    assert pd.DatetimeIndex(result.final_weather["timestamp"].unique()).equals(
        expected
    )
    assert result.max_currents.shape == (2, 2)


def test_pipeline_rejects_same_source_truth_but_still_calculates_dlr(tmp_path):
    physical = _weather("physical")
    truth = _weather("truth", truth_offset=True)
    truth["source_file_hash"] = physical["source_file_hash"]

    result = DlrPipeline(model_root=tmp_path).run(
        physical=physical,
        truth=truth,
        project_id="project-a",
        line_id="line-a",
        terrain_lookup={},
        ai_enabled=True,
        conductor=_conductor(),
    )

    assert result.max_currents.shape == (2, 2)
    assert not result.model_report.trained_targets
    assert any(
        fallback.reason == "truth_rejected_same_source_hash"
        for fallback in result.model_report.fallbacks
    )


def test_pipeline_has_no_union_fill_when_towers_do_not_overlap(tmp_path):
    physical = _weather("physical")
    physical.loc[physical["tower_id"] == "002", "timestamp"] += pd.Timedelta(
        days=1
    )

    with pytest.raises(ValueError, match="没有共同时间戳"):
        DlrPipeline(model_root=tmp_path).run(
            physical=physical,
            project_id="project-a",
            line_id="line-a",
            terrain_lookup={},
            ai_enabled=False,
            conductor=_conductor(),
        )


def test_legacy_projection_keeps_tower_by_common_time_matrices(tmp_path):
    result = DlrPipeline(model_root=tmp_path).run(
        physical=_weather("physical"),
        project_id="project-a",
        line_id="line-a",
        terrain_lookup={},
        ai_enabled=False,
        conductor=_conductor(),
    )

    legacy = result.to_legacy_line_data()

    assert {
        "positions",
        "times",
        "datetimes",
        "elevations",
        "solar",
        "temps",
        "winds",
        "angles",
        "terrain_data",
        "correction_details",
        "max_currents",
        "corrected_winds",
        "local_temps",
    } <= legacy.keys()
    for key in (
        "solar",
        "temps",
        "winds",
        "angles",
        "max_currents",
        "corrected_winds",
        "local_temps",
    ):
        assert np.asarray(legacy[key]).shape == (2, 2)


class _FailingTransientAdapter(_SpyThermalAdapter):
    def calculate_transient_from_long_frame(
        self, weather, *, base_params, request, steady_result
    ):
        raise RuntimeError("transient failed")


def test_transient_failure_falls_back_to_same_steady_result(tmp_path):
    adapter = _FailingTransientAdapter()
    result = DlrPipeline(model_root=tmp_path, thermal_adapter=adapter).run(
        physical=_weather("physical"),
        project_id="project-a",
        line_id="line-a",
        terrain_lookup={},
        ai_enabled=False,
        conductor=_conductor(),
        transient_request={"window_minutes": 15},
    )

    np.testing.assert_array_equal(
        result.thermal_result["transient_result"]["max_currents"],
        result.max_currents,
    )
    assert result.transient_fallbacks == ("transient_failed:RuntimeError",)


def test_all_weather_stages_keep_the_full_project_line_tower_key(tmp_path):
    result = DlrPipeline(model_root=tmp_path).run(
        physical=_weather("physical"),
        project_id="project-a",
        line_id="line-a",
        terrain_lookup={},
        ai_enabled=False,
        conductor=_conductor(),
    )

    for frame in (
        result.physical_weather,
        result.terrain_corrected_weather,
        result.final_weather,
        result.comparison_weather,
    ):
        assert {"project_id", "line_id", "tower_id", "timestamp"} <= set(
            frame.columns
        )
        assert set(frame["project_id"]) == {"project-a"}
        assert set(frame["line_id"]) == {"line-a"}


class _SelectiveFailTrainer:
    def __init__(self):
        self.delegate = ResidualTrainer()

    def train_target(self, frame, target, **kwargs):
        if str(frame["tower_id"].iloc[0]) == "001" and target == "wind_speed":
            raise RuntimeError("tower target failed")
        return self.delegate.train_target(frame, target, **kwargs)


def test_one_tower_target_training_failure_does_not_disable_other_models(tmp_path):
    result = DlrPipeline(
        model_root=tmp_path, trainer=_SelectiveFailTrainer()
    ).run(
        physical=_weather("physical"),
        truth=_weather("truth", truth_offset=True),
        project_id="project-a",
        line_id="line-a",
        terrain_lookup={},
        ai_enabled=True,
        conductor=_conductor(),
    )

    failed_key = ModelKey("project-a", "line-a", "001", "wind_speed")
    assert failed_key not in result.model_report.trained_targets
    assert result.model_report.active_model_count == 3
    tower_one = result.comparison_weather.loc[
        result.comparison_weather["tower_id"] == "001"
    ]
    assert not tower_one["wind_speed_used_ai"].any()
    assert tower_one["ambient_temp_used_ai"].all()
    assert any(
        fallback.key == failed_key
        and fallback.reason == "training_failed:RuntimeError"
        for fallback in result.model_report.fallbacks
    )


def test_conductor_change_invalidates_saved_weather_models(tmp_path):
    pipeline = DlrPipeline(model_root=tmp_path)
    pipeline.run(
        physical=_weather("physical"),
        truth=_weather("truth", truth_offset=True),
        project_id="project-a",
        line_id="line-a",
        terrain_lookup={},
        ai_enabled=True,
        conductor=_conductor(),
    )
    changed_conductor = _conductor() | {"D0": 0.0338}

    changed = pipeline.run(
        physical=_weather("physical"),
        project_id="project-a",
        line_id="line-a",
        terrain_lookup={},
        ai_enabled=True,
        conductor=changed_conductor,
    )

    assert not changed.model_report.loaded_targets
    assert changed.model_report.active_model_count == 0
    assert sum(
        fallback.reason == "incompatible_conductor_hash"
        for fallback in changed.model_report.fallbacks
    ) == 4


def test_pipeline_joins_terrain_metadata_before_single_correction(tmp_path):
    terrain = {
        "001": {
            "elevation": 1500.0,
            "slope": 20.0,
            "aspect": 180.0,
            "source": "dem",
            "reason": None,
        },
        "002": {
            "elevation": 1800.0,
            "slope": 5.0,
            "aspect": 90.0,
            "source": "default",
            "reason": "nodata",
        },
    }
    result = DlrPipeline(model_root=tmp_path).run(
        physical=_weather("physical"),
        project_id="project-a",
        line_id="line-a",
        terrain_lookup=terrain,
        correction_options=CorrectionOptions(
            enable_vertical=False,
            enable_terrain=True,
            enable_desert=False,
            enable_wind_direction=True,
        ),
        ai_enabled=False,
        conductor=_conductor(),
    )

    tower_one = result.terrain_corrected_weather.loc[
        result.terrain_corrected_weather["tower_id"] == "001"
    ]
    tower_two = result.terrain_corrected_weather.loc[
        result.terrain_corrected_weather["tower_id"] == "002"
    ]
    assert tower_one["elevation"].eq(1500.0).all()
    assert tower_one["source"].eq("dem").all()
    assert tower_two["reason"].eq("nodata").all()
    assert result.terrain_corrected_weather["correction_stage"].eq(
        "terrain_corrected"
    ).all()
    assert result.final_weather["correction_stage"].eq("final").all()


def test_final_weather_is_a_truth_free_thermal_whitelist(tmp_path):
    truth = _weather("truth", truth_offset=True)
    truth["private_truth_note"] = "must not leak"
    result = DlrPipeline(model_root=tmp_path).run(
        physical=_weather("physical"),
        truth=truth,
        project_id="project-a",
        line_id="line-a",
        terrain_lookup={},
        ai_enabled=False,
        conductor=_conductor(),
    )

    assert tuple(result.final_weather.columns) == (
        "project_id",
        "line_id",
        "tower_id",
        "timestamp",
        "ambient_temp",
        "wind_speed",
        "wind_direction",
        "wind_angle_deg",
        "solar_radiation",
        "humidity",
        "elevation",
        "slope",
        "aspect",
        "source",
        "reason",
        "correction_stage",
    )
    assert all("truth" not in column.lower() for column in result.final_weather)
    assert "wind_speed_truth" in result.comparison_weather
    assert "ambient_temp_truth" in result.comparison_weather


class _RecordingLineAnalyzer:
    def __init__(self):
        self.calls = []
        self.transient_calls = []

    def calculate_max_current_for_points(self, **kwargs):
        self.calls.append(kwargs)
        return {
            "max_currents": np.asarray(kwargs["solar"], dtype=float)[None, :],
            "corrected_winds": np.asarray(kwargs["winds"], dtype=float),
            "local_temps": np.asarray(kwargs["temps"], dtype=float),
        }

    def find_max_current_for_window(
        self, env_params, base_static, params, dt_hours, start_hour=0, end_hour=2
    ):
        self.transient_calls.append(
            {
                "env_params": env_params,
                "base_static": base_static,
                "params": params,
                "dt_hours": dt_hours,
                "start_hour": start_hour,
                "end_hour": end_hour,
            }
        )
        return float(base_static) + len(self.transient_calls)


def test_long_frame_adapter_preserves_each_tower_solar_and_conductor():
    analyzer = _RecordingLineAnalyzer()
    frame = pd.DataFrame(
        {
            "tower_id": ["001", "001", "002", "002"],
            "timestamp": pd.to_datetime(
                [
                    "2026-07-23 00:00",
                    "2026-07-23 00:30",
                    "2026-07-23 00:00",
                    "2026-07-23 00:30",
                ]
            ).tz_localize("Asia/Shanghai"),
            "ambient_temp": [30.0, 31.0, 28.0, 29.0],
            "wind_speed": [2.0, 2.5, 3.0, 3.5],
            "wind_angle_deg": [90.0, 80.0, 70.0, 60.0],
            "solar_radiation": [100.0, 200.0, 700.0, 800.0],
            "elevation": [1000.0, 1000.0, 1200.0, 1200.0],
        }
    )
    conductor = _conductor() | {"D0": 0.0338}

    result = LongFrameThermalAdapter(analyzer).calculate_from_long_frame(
        frame, base_params=conductor
    )

    assert len(analyzer.calls) == 2
    np.testing.assert_array_equal(analyzer.calls[0]["solar"], [100.0, 200.0])
    np.testing.assert_array_equal(analyzer.calls[1]["solar"], [700.0, 800.0])
    assert all(call["base_params"]["D0"] == 0.0338 for call in analyzer.calls)
    np.testing.assert_array_equal(
        result["max_currents"], [[100.0, 200.0], [700.0, 800.0]]
    )


def test_long_frame_adapter_calculates_transient_with_each_tower_weather():
    analyzer = _RecordingLineAnalyzer()
    adapter = LongFrameThermalAdapter(analyzer)
    frame = pd.DataFrame(
        {
            "tower_id": ["001", "001", "002", "002"],
            "timestamp": pd.to_datetime(
                [
                    "2026-07-23 00:00",
                    "2026-07-23 00:30",
                    "2026-07-23 00:00",
                    "2026-07-23 00:30",
                ]
            ).tz_localize("Asia/Shanghai"),
            "ambient_temp": [30.0, 31.0, 28.0, 29.0],
            "wind_speed": [2.0, 2.5, 3.0, 3.5],
            "wind_angle_deg": [90.0, 80.0, 70.0, 60.0],
            "solar_radiation": [100.0, 200.0, 700.0, 800.0],
            "elevation": [1000.0, 1000.0, 1200.0, 1200.0],
        }
    )
    steady = adapter.calculate_from_long_frame(frame, base_params=_conductor())

    transient = adapter.calculate_transient_from_long_frame(
        frame,
        base_params=_conductor(),
        request={"window_minutes": 15},
        steady_result=steady,
    )

    assert len(analyzer.transient_calls) == 2
    np.testing.assert_array_equal(
        analyzer.transient_calls[0]["env_params"]["solar"], [100.0, 200.0]
    )
    np.testing.assert_array_equal(
        analyzer.transient_calls[1]["env_params"]["solar"], [700.0, 800.0]
    )
    assert analyzer.transient_calls[0]["params"]["D0"] == _conductor()["D0"]
    assert transient["max_currents"].shape == steady["max_currents"].shape


def test_corrupt_model_falls_back_only_for_its_tower_target(tmp_path):
    pipeline = DlrPipeline(model_root=tmp_path)
    first = pipeline.run(
        physical=_weather("physical"),
        truth=_weather("truth", truth_offset=True),
        project_id="project-a",
        line_id="line-a",
        terrain_lookup={},
        ai_enabled=True,
        conductor=_conductor(),
    )
    damaged = ModelKey("project-a", "line-a", "001", "wind_speed")
    assert damaged in first.model_report.trained_targets
    pipeline.registry.path_for(damaged).write_bytes(b"not-a-model")

    second = pipeline.run(
        physical=_weather("physical"),
        project_id="project-a",
        line_id="line-a",
        terrain_lookup={},
        ai_enabled=True,
        conductor=_conductor(),
    )

    assert damaged not in second.model_report.loaded_targets
    assert second.model_report.active_model_count == 3
    assert any(
        fallback.key == damaged and fallback.reason == "corrupt_model"
        for fallback in second.model_report.fallbacks
    )
    tower_one = second.comparison_weather.loc[
        second.comparison_weather["tower_id"] == "001"
    ]
    assert not tower_one["wind_speed_used_ai"].any()
    assert tower_one["ambient_temp_used_ai"].all()


def _page_source() -> str:
    return (Path(__file__).parents[2] / "dispatch_app_st.py").read_text(
        encoding="utf-8"
    )


def test_page_adds_truth_upload_and_normalizes_each_role_once():
    source = _page_source()

    assert '"上传真实气象数据"' in source
    truth_upload_start = source.index('"上传真实气象数据"')
    truth_upload_end = source.index(")", truth_upload_start)
    assert "accept_multiple_files=True" in source[
        truth_upload_start:truth_upload_end
    ]
    assert source.count("normalize_uploaded_weather_files(") == 2
    assert 'role="physical"' in source
    assert 'role="truth"' in source
    assert "physical_weather_snapshot" in source
    assert "truth_weather_snapshot" in source


def test_page_main_button_is_a_thin_pipeline_adapter():
    source = _page_source()
    button_start = source.index("if btn_generate and weather_files:")
    button_end = source.index("# 结果展示", button_start)
    button_block = source[button_start:button_end]

    assert "DlrPipeline(" in button_block
    assert ".run(" in button_block
    assert "conductor=st.session_state.conductor_params" in button_block
    assert "result.to_legacy_line_data()" in button_block
    assert "load_weather_data_from_files(" not in button_block
    assert "process_weather_data(" not in button_block
    assert "convert_to_analysis_format(" not in button_block
    assert "apply_weather_corrections(" not in button_block
    assert "calculate_max_current_for_points(" not in button_block


def test_page_reports_weather_error_and_has_no_random_dlr_ai_demo():
    source = _page_source()
    ai_start = source.index("# ---- AI预测部分 ----")
    ai_block = source[ai_start:]

    assert "IEEE 738-2023" in source
    assert "IEEE 738-2013" not in source
    assert "np.random.seed" not in ai_block
    assert '"风速 MAE"' in ai_block
    assert '"温度 MAE"' in ai_block
    assert 'metric("已启用模型数"' in ai_block
    assert "wind_speed_truth" in ai_block
    assert "ambient_temp_truth" in ai_block
