import copy
import math

import numpy as np
import pytest

from modules.thermal_engine import LineAnalyzer, ThermalCalculator


HUGE = 10 ** 1000


def drake_conductor():
    return {
        "D0": 0.02814,
        "R_low_25": 7.283e-5,
        "R_high_75": 8.688e-5,
        "R_high_200": 1.220e-4,
        "emissivity": 0.8,
        "absorptivity": 0.8,
        "latitude": 30.0,
        "line_azimuth": 90.0,
        "day_of_year": 161,
        "materials": [
            {"type": "aluminum", "mass": 1.116},
            {"type": "steel", "mass": 0.5126},
        ],
    }


def jl_630_conductor():
    return {
        "D0": 0.0338,
        "R_low_25": 4.680e-5,
        "R_high_75": 5.830e-5,
        "R_high_200": 8.740e-5,
        "emissivity": 0.8,
        "absorptivity": 0.8,
        "latitude": 30.0,
        "line_azimuth": 90.0,
        "day_of_year": 161,
        "materials": [
            {"type": "aluminum", "mass": 1.701},
            {"type": "steel", "mass": 0.350},
        ],
    }


def drake_thermal_params():
    params = drake_conductor()
    params.update(
        {
            "T_a": 40.0,
            "wind_speed": 0.61,
            "wind_angle": 90.0,
            "elevation": 0.0,
            "time": 11.0,
        }
    )
    return params


def weather_matrix(observation_points=None):
    return {
        "observation_points": np.array(
            ["tower-a", "tower-b"]
            if observation_points is None
            else observation_points
        ),
        "elevations": np.array([0.0, 0.0]),
        "temps": np.array([[40.0, 20.0], [20.0, 40.0]]),
        "winds": np.array([[0.1, 8.0], [8.0, 0.1]]),
        "angles": np.full((2, 2), 90.0),
        "solar": np.array([0.0, 0.0]),
        "times": np.array([10.0, 11.0]),
        "max_temp": 100.0,
    }


@pytest.fixture
def analyzer():
    return LineAnalyzer(ThermalCalculator())


def test_line_analyzer_requires_explicit_selected_conductor(analyzer):
    with pytest.raises(ValueError, match="base_params"):
        analyzer.calculate_max_current_for_points(**weather_matrix())


@pytest.mark.parametrize(
    "missing_field",
    [
        "D0",
        "R_low_25",
        "R_high_75",
        "R_high_200",
        "emissivity",
        "absorptivity",
    ],
)
def test_line_analyzer_rejects_incomplete_selected_conductor(
    analyzer, missing_field
):
    conductor = drake_conductor()
    conductor.pop(missing_field)

    with pytest.raises(ValueError, match=missing_field):
        analyzer.calculate_max_current_for_points(
            **weather_matrix(), base_params=conductor
        )


def test_line_analyzer_uses_selected_conductor(analyzer):
    first = analyzer.calculate_max_current_for_points(
        **weather_matrix(), base_params=drake_conductor()
    )
    second = analyzer.calculate_max_current_for_points(
        **weather_matrix(), base_params=jl_630_conductor()
    )

    assert not np.allclose(first["max_currents"], second["max_currents"])


def test_line_analyzer_rejects_terrain_reapplication(analyzer):
    with pytest.raises(ValueError, match="上游"):
        analyzer.calculate_max_current_for_points(
            **weather_matrix(),
            base_params=drake_conductor(),
            terrain_data={0: {"slope": 10.0}},
        )


def test_line_analyzer_returns_upstream_weather_copies_and_each_time_bottleneck(
    analyzer,
):
    weather = weather_matrix()
    winds_before = weather["winds"].copy()
    temps_before = weather["temps"].copy()
    conductor = drake_conductor()
    conductor_before = copy.deepcopy(conductor)

    result = analyzer.calculate_max_current_for_points(
        **weather, base_params=conductor
    )

    np.testing.assert_array_equal(result["corrected_winds"], winds_before)
    np.testing.assert_array_equal(result["local_temps"], temps_before)
    assert not np.shares_memory(result["corrected_winds"], weather["winds"])
    assert not np.shares_memory(result["local_temps"], weather["temps"])
    assert result["bottleneck_tower_ids"].tolist() == ["tower-a", "tower-b"]
    assert np.argmin(result["max_currents"], axis=0).tolist() == [0, 1]
    np.testing.assert_array_equal(weather["winds"], winds_before)
    np.testing.assert_array_equal(weather["temps"], temps_before)
    assert conductor == conductor_before


def test_numeric_observation_coordinates_use_stable_position_identifiers(analyzer):
    result = analyzer.calculate_max_current_for_points(
        **weather_matrix(observation_points=[0.36, 1.42]),
        base_params=drake_conductor(),
    )

    assert result["bottleneck_tower_ids"].tolist() == [
        "position_0",
        "position_1",
    ]


def test_mixed_observation_points_preserve_numeric_position_identifiers(analyzer):
    weather = weather_matrix()
    weather["observation_points"] = ["tower-A", 1.42]

    result = analyzer.calculate_max_current_for_points(
        **weather,
        base_params=drake_conductor(),
    )

    assert result["bottleneck_tower_ids"].tolist() == [
        "tower-A",
        "position_1",
    ]


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("elevations", np.array([0.0])),
        ("temps", np.ones((1, 2))),
        ("winds", np.ones((2, 1))),
        ("angles", np.ones((2, 3))),
        ("solar", np.ones(3)),
        ("times", np.ones((1, 2))),
    ],
)
def test_line_analyzer_rejects_weather_shape_mismatches(analyzer, key, value):
    weather = weather_matrix()
    weather[key] = value

    with pytest.raises(ValueError, match=key):
        analyzer.calculate_max_current_for_points(
            **weather, base_params=drake_conductor()
        )


@pytest.mark.parametrize(
    ("key", "index"),
    [
        ("elevations", (0,)),
        ("temps", (0, 0)),
        ("winds", (0, 0)),
        ("angles", (0, 0)),
        ("solar", (0,)),
        ("times", (0,)),
    ],
)
def test_line_analyzer_rejects_nonfinite_weather(analyzer, key, index):
    weather = weather_matrix()
    weather[key] = weather[key].copy()
    weather[key][index] = math.nan

    with pytest.raises(ValueError, match=key):
        analyzer.calculate_max_current_for_points(
            **weather, base_params=drake_conductor()
        )


def test_line_analyzer_rejects_overflowing_weather_values(analyzer):
    weather = weather_matrix()
    weather["elevations"] = [HUGE, HUGE]

    with pytest.raises(ValueError, match="elevations"):
        analyzer.calculate_max_current_for_points(
            **weather, base_params=drake_conductor()
        )


def test_line_analyzer_rejects_scalar_time_axis(analyzer):
    weather = weather_matrix()
    weather["times"] = 10.0

    with pytest.raises(ValueError, match="times"):
        analyzer.calculate_max_current_for_points(
            **weather, base_params=drake_conductor()
        )


def test_line_analyzer_rejects_boolean_weather_values(analyzer):
    weather = weather_matrix()
    weather["winds"] = np.ones((2, 2), dtype=bool)

    with pytest.raises(ValueError, match="winds"):
        analyzer.calculate_max_current_for_points(
            **weather, base_params=drake_conductor()
        )


def test_line_analyzer_broadcasts_scalar_solar_radiation(analyzer):
    scalar_weather = weather_matrix()
    scalar_weather["solar"] = 50.0
    array_weather = weather_matrix()
    array_weather["solar"] = np.full(2, 50.0)

    scalar_result = analyzer.calculate_max_current_for_points(
        **scalar_weather, base_params=drake_conductor()
    )
    array_result = analyzer.calculate_max_current_for_points(
        **array_weather, base_params=drake_conductor()
    )

    np.testing.assert_allclose(
        scalar_result["max_currents"], array_result["max_currents"]
    )


@pytest.mark.parametrize("solar", [True, np.bool_(True), math.nan])
def test_line_analyzer_rejects_invalid_scalar_solar(analyzer, solar):
    weather = weather_matrix()
    weather["solar"] = solar

    with pytest.raises(ValueError, match="solar"):
        analyzer.calculate_max_current_for_points(
            **weather, base_params=drake_conductor()
        )


class RecordingThermalCalculator(ThermalCalculator):
    def __init__(self):
        super().__init__()
        self.transient_calls = []

    def calculate_transient_temperature(
        self, params, time_steps, initial_temp, current_profile
    ):
        self.transient_calls.append(
            {
                "params": dict(params),
                "time_steps": list(time_steps),
                "initial_temp": initial_temp,
                "current_profile": list(current_profile),
            }
        )
        return super().calculate_transient_temperature(
            params, time_steps, initial_temp, current_profile
        )


class FailingIntervalThermalCalculator(ThermalCalculator):
    def __init__(self):
        super().__init__()
        self.nonempty_transient_calls = 0

    def calculate_transient_temperature(
        self, params, time_steps, initial_temp, current_profile
    ):
        if time_steps:
            self.nonempty_transient_calls += 1
            raise RuntimeError("interval integration must not start")
        return super().calculate_transient_temperature(
            params, time_steps, initial_temp, current_profile
        )


def test_dynamic_temperature_uses_n_samples_as_n_minus_one_intervals():
    calculator = RecordingThermalCalculator()
    analyzer = LineAnalyzer(calculator)
    current_profile = [1200.0, 1200.0, 5000.0]
    params = {**drake_thermal_params(), "T_s": 100.0}

    temperatures, mirrored = analyzer.calculate_dynamic_temperature(
        env_params={
            "temp": np.array([40.0, 35.0, 30.0]),
            "wind": np.array([0.61, 3.0, 5.0]),
            "angle": np.array([90.0, 45.0, 0.0]),
            "solar": np.array([0.0, 100.0, 200.0]),
            "elevation": np.array([0.0, 100.0, 200.0]),
        },
        params=params,
        current_profile=current_profile,
        dt_hours=10.0 / 3600.0,
    )

    assert [call["time_steps"] for call in calculator.transient_calls] == [
        [10.0],
        [10.0],
    ]
    assert [call["current_profile"] for call in calculator.transient_calls] == [
        [1200.0],
        [1200.0],
    ]
    assert calculator.transient_calls[0]["initial_temp"] == 100.0
    assert {
        key: calculator.transient_calls[0]["params"][key]
        for key in ("T_a", "wind_speed", "wind_angle", "solar_radiation", "elevation")
    } == {
        "T_a": 40.0,
        "wind_speed": 0.61,
        "wind_angle": 90.0,
        "solar_radiation": 0.0,
        "elevation": 0.0,
    }
    assert {
        key: calculator.transient_calls[1]["params"][key]
        for key in ("T_a", "wind_speed", "wind_angle", "solar_radiation", "elevation")
    } == {
        "T_a": 35.0,
        "wind_speed": 3.0,
        "wind_angle": 45.0,
        "solar_radiation": 100.0,
        "elevation": 100.0,
    }
    assert len(temperatures) == len(current_profile)
    np.testing.assert_array_equal(mirrored, temperatures)


def test_dynamic_temperature_uses_later_interval_weather(analyzer):
    params = {
        **drake_thermal_params(),
        "T_s": 100.0,
        "solar_radiation": 0.0,
    }
    common = {
        "temp": np.array([40.0, 40.0, 40.0]),
        "solar": np.zeros(3),
    }

    calm, _ = analyzer.calculate_dynamic_temperature(
        env_params={**common, "wind": np.array([0.1, 0.1, 0.1])},
        params=params,
        current_profile=[1200.0, 1200.0, 1200.0],
        dt_hours=10.0 / 3600.0,
    )
    windy, _ = analyzer.calculate_dynamic_temperature(
        env_params={**common, "wind": np.array([0.1, 8.0, 8.0])},
        params=params,
        current_profile=[1200.0, 1200.0, 1200.0],
        dt_hours=10.0 / 3600.0,
    )

    assert windy[-1] < calm[-1]


def test_dynamic_temperature_handles_ambient_crossing_conductor_temperature(analyzer):
    interval_hours = 10.0 / 3600.0
    params = {
        **drake_thermal_params(),
        "T_s": 40.0,
        "solar_radiation": 0.0,
    }

    temperatures, _ = analyzer.calculate_dynamic_temperature(
        env_params={
            "times": np.array([0.0, interval_hours, 2.0 * interval_hours]),
            "temp": np.array([30.0, 50.0, 50.0]),
            "solar": np.zeros(3),
        },
        params=params,
        current_profile=[0.0, 0.0, 0.0],
        dt_hours=interval_hours,
    )

    assert temperatures[1] < temperatures[0]
    assert temperatures[2] > temperatures[1]
    assert np.all(np.isfinite(temperatures))


def test_dynamic_temperature_rejects_empty_samples(analyzer):
    with pytest.raises(ValueError, match="current_profile"):
        analyzer.calculate_dynamic_temperature(
            env_params={"temp": np.array([])},
            params=drake_thermal_params(),
            current_profile=[],
            dt_hours=1.0,
        )


@pytest.mark.parametrize("dt_hours", [0.0, -1.0, math.nan, math.inf, True])
def test_dynamic_temperature_rejects_invalid_interval(analyzer, dt_hours):
    with pytest.raises(ValueError, match="dt_hours"):
        analyzer.calculate_dynamic_temperature(
            env_params={"temp": np.array([100.0])},
            params=drake_thermal_params(),
            current_profile=[1025.0],
            dt_hours=dt_hours,
        )


def test_dynamic_temperature_rejects_times_that_conflict_with_interval(analyzer):
    ten_seconds = 10.0 / 3600.0

    with pytest.raises(ValueError, match="dt_hours"):
        analyzer.calculate_dynamic_temperature(
            env_params={
                "times": np.array([0.0, ten_seconds, 2.0 * ten_seconds]),
                "temp": np.array([40.0, 40.0, 40.0]),
            },
            params={**drake_thermal_params(), "T_s": 100.0},
            current_profile=[1025.0, 1025.0, 1025.0],
            dt_hours=5.0 / 3600.0,
        )


def test_dynamic_temperature_rejects_nonincreasing_times(analyzer):
    with pytest.raises(ValueError, match="times"):
        analyzer.calculate_dynamic_temperature(
            env_params={
                "times": np.array([0.0, 0.0, 1.0]),
                "temp": np.array([40.0, 40.0, 40.0]),
            },
            params={**drake_thermal_params(), "T_s": 100.0},
            current_profile=[1025.0, 1025.0, 1025.0],
            dt_hours=0.5,
        )


def test_dynamic_temperature_rejects_boolean_current_samples(analyzer):
    with pytest.raises(ValueError, match="current_profile"):
        analyzer.calculate_dynamic_temperature(
            env_params={"temp": np.array([100.0])},
            params=drake_thermal_params(),
            current_profile=[True],
            dt_hours=1.0,
        )


def test_dynamic_temperature_rejects_overflowing_current_samples(analyzer):
    with pytest.raises(ValueError, match="current_profile"):
        analyzer.calculate_dynamic_temperature(
            env_params={"temp": np.array([100.0])},
            params=drake_thermal_params(),
            current_profile=[HUGE],
            dt_hours=1.0,
        )


def test_dynamic_temperature_enforces_total_substep_budget_before_integration():
    calculator = FailingIntervalThermalCalculator()
    analyzer = LineAnalyzer(calculator)
    interval_seconds = 5_000_010.0

    with pytest.raises(ValueError, match="substep"):
        analyzer.calculate_dynamic_temperature(
            env_params={"temp": np.array([40.0, 40.0, 40.0])},
            params=drake_thermal_params(),
            current_profile=[1200.0, 1200.0, 1200.0],
            dt_hours=interval_seconds / 3600.0,
        )

    assert calculator.nonempty_transient_calls == 0


@pytest.mark.parametrize("missing_field", ["R_low_25", "materials"])
def test_dynamic_temperature_requires_complete_conductor(analyzer, missing_field):
    params = drake_thermal_params()
    params.pop(missing_field)

    with pytest.raises(ValueError, match=missing_field):
        analyzer.calculate_dynamic_temperature(
            env_params={"temp": np.array([40.0])},
            params=params,
            current_profile=[500.0],
            dt_hours=1.0,
        )


def test_find_max_current_for_window_rejects_nonincreasing_times(analyzer):
    params = {**drake_thermal_params(), "max_allow_temp": 125.0}

    with pytest.raises(ValueError, match="times"):
        analyzer.find_max_current_for_window(
            env_params={
                "times": np.array([0.0, 1.0, 1.0]),
                "temp": np.array([100.0, 100.0, 100.0]),
            },
            base_static=1025.0,
            params=params,
            dt_hours=1.0,
        )


def test_find_max_current_for_window_rejects_overflowing_weather(analyzer):
    with pytest.raises(ValueError, match="temp"):
        analyzer.find_max_current_for_window(
            env_params={
                "times": [0.0, 1.0],
                "temp": [HUGE, HUGE],
            },
            base_static=100.0,
            params={**drake_thermal_params(), "max_allow_temp": 125.0},
            dt_hours=1.0,
        )


def test_find_max_current_for_window_rejects_mismatched_weather_lengths(analyzer):
    params = {**drake_thermal_params(), "max_allow_temp": 125.0}

    with pytest.raises(ValueError, match="temp"):
        analyzer.find_max_current_for_window(
            env_params={
                "times": np.array([0.0, 1.0, 2.0]),
                "temp": np.array([100.0]),
            },
            base_static=1025.0,
            params=params,
            dt_hours=1.0,
        )


def test_find_max_current_for_window_rejects_dt_mismatch(analyzer):
    params = {**drake_thermal_params(), "max_allow_temp": 125.0}
    ten_seconds = 10.0 / 3600.0

    with pytest.raises(ValueError, match="dt_hours"):
        analyzer.find_max_current_for_window(
            env_params={
                "times": np.array([0.0, ten_seconds, 2.0 * ten_seconds]),
                "temp": np.array([40.0, 40.0, 40.0]),
            },
            base_static=1025.0,
            params=params,
            dt_hours=5.0 / 3600.0,
        )


@pytest.mark.parametrize("base_static", [math.nan, np.bool_(True), -1.0])
def test_empty_window_still_rejects_invalid_base_current(analyzer, base_static):
    env_params, interval_hours = _short_window_environment()

    with pytest.raises(ValueError, match="base_static"):
        analyzer.find_max_current_for_window(
            env_params=env_params,
            base_static=base_static,
            params=_short_window_params(),
            dt_hours=interval_hours,
            start_hour=10.0,
            end_hour=12.0,
        )


@pytest.mark.parametrize(
    ("field", "invalid_value"),
    [
        ("max_allow_temp", math.nan),
        ("max_allow_temp", -273.15),
        ("wind_speed", math.nan),
        ("D0", 0.0),
        ("materials", []),
    ],
)
def test_empty_window_still_rejects_invalid_thermal_params(
    analyzer, field, invalid_value
):
    env_params, interval_hours = _short_window_environment()
    params = _short_window_params()
    params[field] = invalid_value

    with pytest.raises(ValueError, match=field):
        analyzer.find_max_current_for_window(
            env_params=env_params,
            base_static=100.0,
            params=params,
            dt_hours=interval_hours,
            start_hour=10.0,
            end_hour=12.0,
        )


def test_empty_window_still_rejects_invalid_solar_time(analyzer):
    params = drake_thermal_params()
    params["max_allow_temp"] = 125.0

    with pytest.raises(ValueError, match="time"):
        analyzer.find_max_current_for_window(
            env_params={
                "times": np.array([25.0, 26.0, 27.0]),
                "temp": np.array([40.0, 40.0, 40.0]),
            },
            base_static=100.0,
            params=params,
            dt_hours=1.0,
            start_hour=100.0,
            end_hour=102.0,
        )


def test_valid_window_without_data_overlap_returns_zero(analyzer):
    env_params, interval_hours = _short_window_environment()

    result = analyzer.find_max_current_for_window(
        env_params=env_params,
        base_static=100.0,
        params=_short_window_params(),
        dt_hours=interval_hours,
        start_hour=10.0,
        end_hour=12.0,
    )

    assert result == 0.0


@pytest.mark.parametrize("missing_field", ["R_low_25", "materials"])
def test_find_max_current_for_window_requires_complete_conductor(
    analyzer, missing_field
):
    env_params, interval_hours = _short_window_environment()
    params = _short_window_params()
    params.pop(missing_field)

    with pytest.raises(ValueError, match=missing_field):
        analyzer.find_max_current_for_window(
            env_params=env_params,
            base_static=100.0,
            params=params,
            dt_hours=interval_hours,
            start_hour=10.0,
            end_hour=12.0,
        )


def test_find_max_current_for_window_uses_each_interval_weather(analyzer):
    params = {
        **drake_thermal_params(),
        "T_s": 100.0,
        "solar_radiation": 0.0,
        "max_allow_temp": 125.0,
    }
    interval_hours = 7.5 / 60.0
    common = {
        "times": np.array([0.0, interval_hours, 2.0 * interval_hours]),
        "temp": np.array([40.0, 40.0, 40.0]),
        "solar": np.zeros(3),
    }

    calm = analyzer.find_max_current_for_window(
        env_params={**common, "wind": np.array([0.1, 0.1, 0.1])},
        base_static=1025.0,
        params=params,
        dt_hours=interval_hours,
    )
    windy = analyzer.find_max_current_for_window(
        env_params={**common, "wind": np.array([0.1, 8.0, 8.0])},
        base_static=1025.0,
        params=params,
        dt_hours=interval_hours,
    )

    assert windy > calm


def _short_window_environment():
    interval_hours = 10.0 / 3600.0
    return (
        {
            "times": np.array([0.0, interval_hours, 2.0 * interval_hours]),
            "temp": np.array([100.0, 100.0, 100.0]),
            "wind": np.array([0.61, 0.61, 0.61]),
            "solar": np.zeros(3),
        },
        interval_hours,
    )


def _short_window_params():
    return {
        **drake_thermal_params(),
        "T_s": 100.0,
        "solar_radiation": 0.0,
        "max_allow_temp": 100.2,
    }


def test_find_max_current_for_window_checks_non_grid_end_boundary(analyzer):
    env_params, interval_hours = _short_window_environment()
    params = _short_window_params()
    env_before = {key: values.copy() for key, values in env_params.items()}
    params_before = copy.deepcopy(params)

    result = analyzer.find_max_current_for_window(
        env_params=env_params,
        base_static=100.0,
        params=params,
        dt_hours=interval_hours,
        start_hour=0.0,
        end_hour=15.0 / 3600.0,
    )
    temperatures = analyzer.calculator.calculate_transient_temperature(
        params=params,
        time_steps=[10.0, 5.0],
        initial_temp=100.0,
        current_profile=[result, result],
    )

    assert result == pytest.approx(432.42, abs=0.05)
    assert max(temperatures) <= 100.2 + 1e-6
    for key, before in env_before.items():
        np.testing.assert_array_equal(env_params[key], before)
    assert params == params_before


def test_find_max_current_for_window_handles_window_inside_one_interval(analyzer):
    env_params, interval_hours = _short_window_environment()

    result = analyzer.find_max_current_for_window(
        env_params=env_params,
        base_static=100.0,
        params=_short_window_params(),
        dt_hours=interval_hours,
        start_hour=2.0 / 3600.0,
        end_hour=8.0 / 3600.0,
    )

    assert result == pytest.approx(591.22, abs=0.05)


def test_find_max_current_for_window_uses_left_weather_for_partial_intervals():
    calculator = RecordingThermalCalculator()
    analyzer = LineAnalyzer(calculator)
    interval_hours = 10.0 / 3600.0
    params = {
        **drake_thermal_params(),
        "T_s": 100.0,
        "solar_radiation": 0.0,
        "max_allow_temp": 125.0,
    }

    analyzer.find_max_current_for_window(
        env_params={
            "times": np.array([0.0, interval_hours, 2.0 * interval_hours]),
            "temp": np.array([40.0, 40.0, 40.0]),
            "wind": np.array([0.1, 8.0, 8.0]),
            "solar": np.zeros(3),
        },
        base_static=100.0,
        params=params,
        dt_hours=interval_hours,
        start_hour=5.0 / 3600.0,
        end_hour=15.0 / 3600.0,
    )

    integration_calls = [
        call for call in calculator.transient_calls if call["time_steps"]
    ]
    np.testing.assert_allclose(
        [call["time_steps"][0] for call in integration_calls[:3]],
        [5.0, 5.0, 5.0],
        atol=1e-12,
    )
    assert [call["params"]["wind_speed"] for call in integration_calls[:3]] == [
        0.1,
        0.1,
        8.0,
    ]


def test_find_max_current_for_window_preserves_pre_window_heat_history(analyzer):
    env_params, interval_hours = _short_window_environment()
    params = _short_window_params()

    full_window = analyzer.find_max_current_for_window(
        env_params=env_params,
        base_static=100.0,
        params=params,
        dt_hours=interval_hours,
        start_hour=0.0,
        end_hour=20.0 / 3600.0,
    )
    later_window = analyzer.find_max_current_for_window(
        env_params=env_params,
        base_static=100.0,
        params=params,
        dt_hours=interval_hours,
        start_hour=10.0 / 3600.0,
        end_hour=20.0 / 3600.0,
    )

    assert later_window == pytest.approx(full_window, abs=1e-12)


def test_window_augmented_axis_enforces_total_substep_budget_before_integration():
    calculator = FailingIntervalThermalCalculator()
    analyzer = LineAnalyzer(calculator)
    interval_seconds = 5_000_000.0
    interval_hours = interval_seconds / 3600.0

    with pytest.raises(ValueError, match="substep"):
        analyzer.find_max_current_for_window(
            env_params={
                "times": np.array([0.0, interval_hours, 2.0 * interval_hours]),
                "temp": np.array([40.0, 40.0, 40.0]),
                "solar": np.zeros(3),
            },
            base_static=100.0,
            params={**drake_thermal_params(), "max_allow_temp": 125.0},
            dt_hours=interval_hours,
            start_hour=2_500_001.0 / 3600.0,
            end_hour=2.0 * interval_hours,
        )

    assert calculator.nonempty_transient_calls == 0


def test_find_max_current_for_window_searches_below_unsafe_base(analyzer):
    env_params, interval_hours = _short_window_environment()

    result = analyzer.find_max_current_for_window(
        env_params=env_params,
        base_static=3000.0,
        params=_short_window_params(),
        dt_hours=interval_hours,
    )

    temperatures, _ = analyzer.calculate_dynamic_temperature(
        env_params=env_params,
        params=_short_window_params(),
        current_profile=[result, result, result],
        dt_hours=interval_hours,
    )
    assert 0.0 < result < 3000.0
    assert max(temperatures) <= 100.2 + 1e-6


def test_find_max_current_for_window_expands_safe_initial_high(analyzer):
    env_params, interval_hours = _short_window_environment()

    result = analyzer.find_max_current_for_window(
        env_params=env_params,
        base_static=100.0,
        params=_short_window_params(),
        dt_hours=interval_hours,
    )

    assert result > 300.0


def test_find_max_current_for_window_fails_when_zero_current_is_unsafe(analyzer):
    env_params, interval_hours = _short_window_environment()
    params = {**_short_window_params(), "max_allow_temp": 99.0}

    with pytest.raises(ValueError, match="可行"):
        analyzer.find_max_current_for_window(
            env_params=env_params,
            base_static=100.0,
            params=params,
            dt_hours=interval_hours,
        )


def test_window_zero_current_check_handles_ambient_warming(analyzer):
    interval_hours = 10.0 / 3600.0
    params = {
        **drake_thermal_params(),
        "T_s": 40.0,
        "solar_radiation": 0.0,
        "max_allow_temp": 40.1,
    }

    result = analyzer.find_max_current_for_window(
        env_params={
            "times": np.array([0.0, interval_hours, 2.0 * interval_hours]),
            "temp": np.array([30.0, 50.0, 50.0]),
            "solar": np.zeros(3),
        },
        base_static=100.0,
        params=params,
        dt_hours=interval_hours,
    )

    assert math.isfinite(result)
    assert result >= 0.0


def test_time_to_max_temperature_returns_verified_safe_left_without_mutation(
    analyzer,
):
    params = drake_thermal_params()
    before = copy.deepcopy(params)

    elapsed = analyzer.calculate_time_to_max_temp(
        params=params,
        current=1200.0,
        max_temp=100.2,
        initial_temp=100.0,
        time_step=25.0,
    )
    temperature_at_return = analyzer.calculator.calculate_transient_temperature(
        params=params,
        time_steps=[elapsed],
        initial_temp=100.0,
        current_profile=[1200.0],
    )[-1]

    assert 0.0 < elapsed < 10.0
    assert temperature_at_return <= 100.2
    assert params == before


@pytest.mark.parametrize("time_step", [10.0, 1.0, 0.1])
def test_time_to_max_temperature_never_returns_over_temperature_time(
    analyzer, time_step
):
    params = drake_thermal_params()

    elapsed = analyzer.calculate_time_to_max_temp(
        params=params,
        current=1200.0,
        max_temp=100.2,
        initial_temp=100.0,
        time_step=time_step,
    )
    temperature_at_return = analyzer.calculator.calculate_transient_temperature(
        params=params,
        time_steps=[elapsed],
        initial_temp=100.0,
        current_profile=[1200.0],
    )[-1]

    assert temperature_at_return <= 100.2


def test_time_to_max_temperature_reaches_target_after_legacy_two_hour_limit(
    analyzer,
):
    params = drake_thermal_params()
    target_temp = 100.0
    target_rating = analyzer.calculator.calculate_steady_state_current(
        {**params, "T_s": target_temp, "T_avg": target_temp}
    )

    elapsed = analyzer.calculate_time_to_max_temp(
        params=params,
        current=target_rating * 1.00001,
        max_temp=target_temp,
        initial_temp=40.0,
        time_step=10.0,
    )

    assert math.isfinite(elapsed)
    assert elapsed > 7200.0


def test_time_to_max_temperature_uses_target_heat_flow_for_unreachable_case():
    calculator = RecordingThermalCalculator()
    analyzer = LineAnalyzer(calculator)
    params = {**drake_thermal_params(), "solar_radiation": 0.0}
    target_temp = 100.0
    target_rating = calculator.calculate_steady_state_current(
        {**params, "T_s": target_temp, "T_avg": target_temp}
    )

    elapsed = analyzer.calculate_time_to_max_temp(
        params=params,
        current=target_rating,
        max_temp=target_temp,
        initial_temp=40.0,
        time_step=10.0,
    )

    assert elapsed == math.inf
    assert not any(call["time_steps"] for call in calculator.transient_calls)


def test_time_to_max_temperature_keeps_strictly_positive_target_heat_reachable(
    analyzer,
):
    params = drake_thermal_params()
    target_temp = 100.0
    target_balance = analyzer.calculator.calculate_heat_balance(
        {**params, "T_s": target_temp, "T_avg": target_temp}
    )
    current = math.nextafter(target_balance.current_a, math.inf)
    initial_temp = math.nextafter(target_temp, -math.inf)
    target_heat_flow = (
        target_balance.resistance * current ** 2
        + target_balance.q_solar
        - target_balance.q_convection
        - target_balance.q_radiation
    )

    elapsed = analyzer.calculate_time_to_max_temp(
        params=params,
        current=current,
        max_temp=target_temp,
        initial_temp=initial_temp,
        time_step=10.0,
    )

    assert target_heat_flow > 0.0
    assert math.isfinite(elapsed)
    assert elapsed == 0.0


def test_time_to_max_temperature_rejects_too_small_search_step(analyzer):
    with pytest.raises(ValueError, match="time_step"):
        analyzer.calculate_time_to_max_temp(
            params=drake_thermal_params(),
            current=0.0,
            max_temp=80.0,
            initial_temp=100.0,
            time_step=0.0001,
        )


def test_time_to_max_temperature_reports_iteration_budget_exhaustion(
    analyzer, monkeypatch
):
    monkeypatch.setattr(
        analyzer, "_MAX_TIME_TO_MAX_ITERATIONS", 1, raising=False
    )

    with pytest.raises(ValueError, match="iteration"):
        analyzer.calculate_time_to_max_temp(
            params=drake_thermal_params(),
            current=2000.0,
            max_temp=100.0,
            initial_temp=40.0,
            time_step=1.0,
        )


def test_time_to_max_temperature_rejects_target_at_absolute_zero(analyzer):
    with pytest.raises(ValueError, match="max_temp"):
        analyzer.calculate_time_to_max_temp(
            params=drake_thermal_params(),
            current=0.0,
            max_temp=-273.15,
            initial_temp=20.0,
        )


@pytest.mark.parametrize("missing_field", ["R_low_25", "materials"])
def test_time_to_max_temperature_requires_complete_conductor(
    analyzer, missing_field
):
    params = drake_thermal_params()
    params.pop(missing_field)

    with pytest.raises(ValueError, match=missing_field):
        analyzer.calculate_time_to_max_temp(
            params=params,
            current=0.0,
            max_temp=80.0,
            initial_temp=100.0,
            time_step=10.0,
        )


@pytest.mark.parametrize(
    ("field", "invalid_value"),
    [
        ("wind_speed", math.nan),
        ("time", math.nan),
        ("D0", 0.0),
        ("materials", []),
    ],
)
def test_time_to_max_temperature_validates_params_before_early_return(
    analyzer, field, invalid_value
):
    params = drake_thermal_params()
    params[field] = invalid_value

    with pytest.raises(ValueError, match=field):
        analyzer.calculate_time_to_max_temp(
            params=params,
            current=0.0,
            max_temp=80.0,
            initial_temp=100.0,
            time_step=10.0,
        )
