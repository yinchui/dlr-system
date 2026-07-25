import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from modules.dlr_pipeline import DlrPipeline, derive_line_identity
from modules.model_registry import ModelKey
from modules.weather_correction import CorrectionOptions


def _weather_segment(role, *, segment_index, truth_offset=False):
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
        wind += np.array([0.75, 1.5, -1.5, -2.25])
        temperature += np.array([-1.0, -2.0, 1.5, 2.5])
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
            "longitude": [120.6982, 120.6982, 120.6983, 120.6983],
            "latitude": [49.2871, 49.2871, 49.2872, 49.2872],
            "dataset_role": role,
            "source_file_hash": [f"{role}-{segment_index}"] * 4,
        }
    )


def _weather(role, *, truth_offset=False):
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


def _conductor():
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
        "materials": [
            {"type": "aluminum", "density": 1.116},
            {"type": "steel", "density": 0.5126},
        ],
    }


def _correction_options(**overrides):
    values = {
        "enable_vertical": False,
        "enable_terrain": False,
        "enable_desert": False,
        "enable_wind_direction": False,
    }
    values.update(overrides)
    return CorrectionOptions(**values)


def _run_without_ai(
    tmp_path,
    *,
    physical=None,
    project_id="project-a",
    line_id="line-a",
    conductor=None,
    correction_options=None,
    transient_request=None,
):
    return DlrPipeline(model_root=tmp_path).run(
        physical=_weather("physical") if physical is None else physical,
        project_id=project_id,
        line_id=line_id,
        terrain_lookup={},
        correction_options=correction_options or _correction_options(),
        ai_enabled=False,
        conductor=_conductor() if conductor is None else conductor,
        transient_request=transient_request,
    )


def test_weather_truth_training_to_dlr_is_repeatable(tmp_path):
    model_root = tmp_path / "models"
    physical = _weather("physical")
    line_identity = derive_line_identity(physical, tower_coords={})
    assert line_identity.persistence_allowed is True
    assert line_identity.reason == "complete_coordinates"
    run_kwargs = {
        "physical": physical,
        "project_id": "project-a",
        "line_id": line_identity.line_id,
        "model_persistence_allowed": line_identity.persistence_allowed,
        "coordinate_context": {},
        "terrain_lookup": {},
        "ai_enabled": True,
        "conductor": _conductor(),
    }
    expected_keys = tuple(
        ModelKey("project-a", line_identity.line_id, tower_id, target)
        for tower_id in ("001", "002")
        for target in ("wind_speed", "ambient_temp")
    )

    first = DlrPipeline(model_root=model_root).run(
        **run_kwargs,
        truth=_weather("truth", truth_offset=True),
    )
    second = DlrPipeline(model_root=model_root).run(
        **run_kwargs,
        truth=None,
    )

    assert first.model_report.trained_targets == expected_keys
    assert second.model_report.loaded_targets == expected_keys
    assert first.model_report.used_targets == expected_keys
    assert second.model_report.used_targets == expected_keys
    np.testing.assert_allclose(first.max_currents, second.max_currents)
    assert first.input_hash == second.input_hash
    assert len(first.input_hash) == 64
    comparison = first.comparison_weather
    for target in ("wind_speed", "ambient_temp"):
        predicted = comparison[f"{target}_ai"].to_numpy(dtype=float)
        truth = comparison[f"{target}_truth"].to_numpy(dtype=float)
        valid = np.isfinite(predicted) & np.isfinite(truth)
        errors = predicted[valid] - truth[valid]
        assert getattr(first.weather_metrics, f"{target}_mae") == pytest.approx(
            np.mean(np.abs(errors))
        )
        assert getattr(first.weather_metrics, f"{target}_rmse") == pytest.approx(
            np.sqrt(np.mean(np.square(errors)))
        )
    assert all(value is None for value in second.weather_metrics.__dict__.values())


def test_dlr_input_hash_ignores_runtime_context_and_row_order(tmp_path):
    physical = _weather("physical")
    conductor = _conductor()
    baseline = _run_without_ai(
        tmp_path / "baseline",
        physical=physical,
        conductor=conductor,
        correction_options=_correction_options(roughness_alpha=0.1),
    )

    equivalent_conductor = _conductor()
    equivalent_conductor["materials"] = [
        {"type": "aluminum", "density": 9.0}
    ]
    equivalent_conductor["runtime_marker"] = object()
    equivalent = _run_without_ai(
        tmp_path / "equivalent",
        physical=physical.sample(frac=1.0, random_state=42),
        project_id="another-project",
        line_id="another-line",
        conductor=equivalent_conductor,
        correction_options=_correction_options(roughness_alpha=0.4),
    )

    assert baseline.input_hash == equivalent.input_hash


def test_dlr_input_hash_normalizes_equivalent_numeric_types(tmp_path):
    float_conductor = _conductor()
    integer_conductor = _conductor()
    integer_conductor["max_allow_temp"] = 80

    float_result = _run_without_ai(
        tmp_path / "float",
        conductor=float_conductor,
    )
    integer_result = _run_without_ai(
        tmp_path / "integer",
        conductor=integer_conductor,
    )

    np.testing.assert_allclose(float_result.max_currents, integer_result.max_currents)
    assert float_result.input_hash == integer_result.input_hash


def test_dlr_input_hash_ignores_non_effective_tower_elevation_samples(tmp_path):
    physical = _weather("physical")
    baseline = _run_without_ai(
        tmp_path / "baseline",
        physical=physical,
    )
    changed = physical.copy(deep=True)
    second_tower_sample = changed.index[
        changed["tower_id"].eq("001")
    ][1]
    changed.loc[second_tower_sample, "elevation"] += 500.0

    equivalent = _run_without_ai(
        tmp_path / "equivalent",
        physical=changed,
    )

    np.testing.assert_allclose(baseline.max_currents, equivalent.max_currents)
    assert baseline.input_hash == equivalent.input_hash


@pytest.mark.parametrize(
    "field",
    (
        "tower_id",
        "timestamp",
        "ambient_temp",
        "wind_speed",
        "wind_direction",
        "solar_radiation",
        "elevation",
    ),
)
def test_dlr_input_hash_changes_for_each_final_weather_input(tmp_path, field):
    options = _correction_options(enable_wind_direction=True)
    physical = _weather("physical")
    baseline = _run_without_ai(
        tmp_path / "baseline",
        physical=physical,
        correction_options=options,
    )
    changed = physical.copy(deep=True)
    if field == "tower_id":
        changed.loc[changed["tower_id"] == "001", "tower_id"] = "003"
    elif field == "timestamp":
        changed["timestamp"] = changed["timestamp"] + pd.Timedelta(minutes=1)
    else:
        changed.loc[changed.index[0], field] += 0.25

    modified = _run_without_ai(
        tmp_path / "modified",
        physical=changed,
        correction_options=options,
    )

    assert baseline.input_hash != modified.input_hash


@pytest.mark.parametrize(
    "field",
    (
        "D0",
        "R_low_25",
        "R_high_75",
        "R_high_200",
        "emissivity",
        "absorptivity",
        "max_allow_temp",
    ),
)
def test_dlr_input_hash_changes_for_each_steady_conductor_input(tmp_path, field):
    baseline = _run_without_ai(tmp_path / "baseline")
    conductor = _conductor()
    conductor[field] *= 1.01

    modified = _run_without_ai(
        tmp_path / "modified",
        conductor=conductor,
    )

    assert baseline.input_hash != modified.input_hash


def test_dlr_input_hash_normalizes_equivalent_transient_windows(tmp_path):
    by_end_hour = _run_without_ai(
        tmp_path / "end-hour",
        transient_request={"start_hour": 0.0, "end_hour": 1.0},
    )
    by_duration = _run_without_ai(
        tmp_path / "duration",
        transient_request={
            "start_hour": 0,
            "window_minutes": 60,
            "ignored_metadata": "not-a-thermal-input",
        },
    )
    longer_window = _run_without_ai(
        tmp_path / "longer",
        transient_request={"start_hour": 0.0, "end_hour": 1.5},
    )

    assert not by_end_hour.transient_fallbacks
    assert not by_duration.transient_fallbacks
    assert by_end_hour.input_hash == by_duration.input_hash
    assert by_end_hour.input_hash != longer_window.input_hash


def test_dlr_input_hash_normalizes_clipped_windows_and_negative_zero(tmp_path):
    clipped = _run_without_ai(
        tmp_path / "clipped",
        transient_request={"start_hour": -1.0, "end_hour": 1.0},
    )
    zero = _run_without_ai(
        tmp_path / "zero",
        transient_request={"start_hour": 0.0, "end_hour": 1.0},
    )
    negative_zero = _run_without_ai(
        tmp_path / "negative-zero",
        transient_request={"start_hour": -0.0, "end_hour": 1.0},
    )

    assert not clipped.transient_fallbacks
    assert not zero.transient_fallbacks
    assert not negative_zero.transient_fallbacks
    np.testing.assert_allclose(
        clipped.thermal_result["transient_result"]["max_currents"],
        zero.thermal_result["transient_result"]["max_currents"],
    )
    np.testing.assert_allclose(
        negative_zero.thermal_result["transient_result"]["max_currents"],
        zero.thermal_result["transient_result"]["max_currents"],
    )
    assert clipped.input_hash == zero.input_hash == negative_zero.input_hash


def test_dlr_input_hash_normalizes_equivalent_explicit_initial_temperature(
    tmp_path,
):
    physical = _weather("physical")
    first_samples = physical.groupby("tower_id", sort=False).head(1).index
    physical.loc[first_samples, "ambient_temp"] = 25.0
    request = {"start_hour": 0.0, "end_hour": 1.0}
    implicit = _run_without_ai(
        tmp_path / "implicit",
        physical=physical,
        transient_request=request,
    )
    explicit_conductor = _conductor() | {"T_s": 25.0}
    explicit = _run_without_ai(
        tmp_path / "explicit",
        physical=physical,
        conductor=explicit_conductor,
        transient_request=request,
    )

    np.testing.assert_allclose(
        implicit.thermal_result["transient_result"]["max_currents"],
        explicit.thermal_result["transient_result"]["max_currents"],
    )
    assert implicit.input_hash == explicit.input_hash


def test_dlr_input_hash_is_stable_across_processes(tmp_path):
    conductor = _conductor()
    conductor["runtime_marker"] = object()
    local_hash = _run_without_ai(
        tmp_path / "local",
        conductor=conductor,
    ).input_hash
    code = f"""
import runpy
from modules.dlr_pipeline import DlrPipeline

helpers = runpy.run_path("tests/integration/test_weather_ai_dlr_e2e.py")
conductor = helpers["_conductor"]()
conductor["runtime_marker"] = object()
result = DlrPipeline(model_root={str(tmp_path / 'child')!r}).run(
    physical=helpers["_weather"]("physical"),
    project_id="project-a",
    line_id="line-a",
    terrain_lookup={{}},
    correction_options=helpers["_correction_options"](),
    ai_enabled=False,
    conductor=conductor,
)
print("INPUT_HASH=" + result.input_hash)
"""
    completed = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
    )
    child_hash = next(
        line.removeprefix("INPUT_HASH=")
        for line in completed.stdout.splitlines()
        if line.startswith("INPUT_HASH=")
    )

    assert child_hash == local_hash
