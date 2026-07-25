import joblib
import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from modules.dlr_pipeline import DlrPipeline
from modules.model_registry import ModelKey


def _weather_segment(
    role: str,
    *,
    segment_index: int,
    truth_offset: bool = False,
) -> pd.DataFrame:
    timestamps = pd.to_datetime(
        [
            "2026-07-23 00:00",
            "2026-07-23 00:30",
            "2026-07-23 00:00",
            "2026-07-23 00:30",
        ]
    ).tz_localize("Asia/Shanghai") + pd.Timedelta(days=2 * segment_index)
    wind = np.array([2.0, 4.0, 3.0, 5.0])
    temperature = np.array([30.0, 32.0, 28.0, 31.0])
    if truth_offset:
        wind = wind + np.array([0.75, 1.5, -1.5, -2.25])
        temperature = temperature + np.array([-1.0, -2.0, 1.5, 2.5])
    return pd.DataFrame(
        {
            "tower_id": ["001", "001", "002", "002"],
            "timestamp": timestamps,
            "ambient_temp": temperature,
            "wind_speed": wind,
            "wind_direction": [90.0, 100.0, 110.0, 120.0],
            "solar_radiation": [0.0, 10.0, 0.0, 20.0],
            "humidity": [30.0, 31.0, 32.0, 33.0],
            "elevation": [1000.0, 1000.0, 1200.0, 1200.0],
            "dataset_role": role,
            "source_file_hash": [f"{role}-source-{segment_index}"] * 4,
        }
    )


def _weather_with_independent_segments(
    role: str,
    *,
    truth_offset: bool = False,
) -> pd.DataFrame:
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
    ).sort_values(["tower_id", "timestamp"], ignore_index=True)


def _later_weather(role: str) -> pd.DataFrame:
    return _weather_segment(role, segment_index=6).sort_values(
        ["tower_id", "timestamp"], ignore_index=True
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


def test_real_xgboost_trains_persists_and_reuses_per_tower_models(tmp_path):
    expected_keys = tuple(
        ModelKey("project-a", "line-a", tower_id, target)
        for tower_id in ("001", "002")
        for target in ("wind_speed", "ambient_temp")
    )
    first_pipeline = DlrPipeline(model_root=tmp_path)
    first = first_pipeline.run(
        physical=_weather_with_independent_segments("physical"),
        truth=_weather_with_independent_segments(
            "truth", truth_offset=True
        ),
        project_id="project-a",
        line_id="line-a",
        terrain_lookup={},
        ai_enabled=True,
        conductor=_conductor(),
    )
    second = DlrPipeline(model_root=tmp_path).run(
        physical=_later_weather("physical"),
        truth=None,
        project_id="project-a",
        line_id="line-a",
        terrain_lookup={},
        ai_enabled=True,
        conductor=_conductor(),
    )

    assert first.model_report.trained_targets == expected_keys
    assert first.model_report.used_targets == expected_keys
    assert first.model_report.fallbacks == ()
    assert second.model_report.loaded_targets == expected_keys
    assert second.model_report.used_targets == expected_keys
    assert second.model_report.fallbacks == ()
    assert first.max_currents.size > 0
    assert second.max_currents.size > 0
    assert np.isfinite(first.max_currents).all()
    assert np.isfinite(second.max_currents).all()
    expected_directions = {
        ("001", "wind_speed"): 1.0,
        ("002", "wind_speed"): -1.0,
        ("001", "ambient_temp"): -1.0,
        ("002", "ambient_temp"): 1.0,
    }
    for result in (first, second):
        comparison = result.comparison_weather
        for (tower_id, target), direction in expected_directions.items():
            tower = comparison.loc[
                comparison["tower_id"].astype(str) == tower_id
            ]
            correction = (
                tower[f"{target}_ai"] - tower[f"{target}_physical"]
            )
            assert tower[f"{target}_used_ai"].all()
            assert direction * correction.mean() > 0.25
    registry = first_pipeline.registry
    model_paths = [registry.path_for(key) for key in expected_keys]
    manifest_paths = [registry.manifest_path_for(key) for key in expected_keys]
    assert len(model_paths) == 4
    assert len(manifest_paths) == 4
    assert all(path.is_file() for path in model_paths)
    assert all(path.is_file() for path in manifest_paths)
    assert all(
        type(joblib.load(path).model) is XGBRegressor for path in model_paths
    )
