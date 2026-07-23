import copy
import math

import numpy as np
import pytest

from modules.thermal_engine import LineAnalyzer, ThermalCalculator


def test_calculate_max_current_for_points_returns_expected_shapes():
    calculator = ThermalCalculator()
    analyzer = LineAnalyzer(calculator)
    result = analyzer.calculate_max_current_for_points(
        observation_points=np.array([0.36]),
        elevations=np.array([1100.0]),
        temps=np.array([[20.0, 21.0]]),
        winds=np.array([[3.0, 3.2]]),
        angles=np.array([[90.0, 95.0]]),
        solar=np.array([0.0, 50.0]),
        times=np.array([0.0, 1.0]),
        max_temp=80.0,
        base_params=_temperature_params(),
        terrain_data=None,
    )
    assert result["max_currents"].shape == (1, 2)
    assert result["corrected_winds"][0, 0] > 0


def test_find_max_current_for_window_uses_one_interval_per_time_gap():
    calculator = ThermalCalculator()
    analyzer = LineAnalyzer(calculator)
    params = _temperature_params()
    params["max_allow_temp"] = 100.0
    before = copy.deepcopy(params)
    env_params = {
        "times": np.array([0.0, 1.0, 2.0]),
        "temp": np.array([40.0, 40.0, 40.0]),
    }

    result = analyzer.find_max_current_for_window(
        env_params, base_static=500.0, params=params, dt_hours=1.0
    )

    assert 500.0 <= result <= 1500.0
    assert params == before


def _temperature_params():
    return {
        "D0": 0.02814,
        "R_low_25": 7.283e-5,
        "R_high_75": 8.688e-5,
        "R_high_200": 1.220e-4,
        "emissivity": 0.8,
        "absorptivity": 0.8,
        "T_a": 40.0,
        "T_s": 80.0,
        "T_avg": 80.0,
        "wind_speed": 0.61,
        "wind_angle": 90.0,
        "elevation": 0.0,
        "solar_radiation": 0.0,
        "slope": 20.0,
        "aspect": 270.0,
        "materials": [
            {"type": "aluminum", "mass": 1.116},
            {"type": "steel", "mass": 0.5126},
        ],
    }


def test_temperature_calculations_preserve_callers_nested_input():
    calculator = ThermalCalculator()
    params = _temperature_params()
    before = copy.deepcopy(params)
    time_steps = [1.0]
    current_profile = [500.0]
    time_steps_before = list(time_steps)
    current_profile_before = list(current_profile)

    calculator.calculate_steady_state_temperature(params, current=500.0)
    calculator.calculate_transient_temperature(
        params,
        time_steps=time_steps,
        initial_temp=60.0,
        current_profile=current_profile,
    )

    assert params == before
    assert time_steps == time_steps_before
    assert current_profile == current_profile_before


@pytest.mark.parametrize("current", [-1.0, math.nan, math.inf, -math.inf])
def test_steady_state_temperature_rejects_invalid_current(current):
    with pytest.raises(ValueError, match="current"):
        ThermalCalculator().calculate_steady_state_temperature(
            _temperature_params(), current=current
        )


@pytest.mark.parametrize("max_iter", [0, -1, 1.5, True])
def test_steady_state_temperature_rejects_nonpositive_or_noninteger_max_iter(max_iter):
    with pytest.raises(ValueError, match="max_iter"):
        ThermalCalculator().calculate_steady_state_temperature(
            _temperature_params(), current=500.0, max_iter=max_iter
        )


@pytest.mark.parametrize("tol", [0.0, -1.0, math.nan, math.inf, -math.inf])
def test_steady_state_temperature_rejects_invalid_tolerance(tol):
    with pytest.raises(ValueError, match="tol"):
        ThermalCalculator().calculate_steady_state_temperature(
            _temperature_params(), current=500.0, tol=tol
        )


@pytest.mark.parametrize(
    "initial_temp", [-273.15, -274.0, math.nan, math.inf, -math.inf]
)
def test_transient_temperature_rejects_invalid_initial_temperature(initial_temp):
    with pytest.raises(ValueError, match="initial_temp"):
        ThermalCalculator().calculate_transient_temperature(
            _temperature_params(), [1.0], initial_temp, [500.0]
        )


@pytest.mark.parametrize(
    ("time_steps", "current_profile"),
    [([1.0], []), ([], [500.0]), ([1.0], [500.0, 500.0])],
)
def test_transient_temperature_requires_matching_profile_lengths(
    time_steps, current_profile
):
    with pytest.raises(ValueError, match="same length"):
        ThermalCalculator().calculate_transient_temperature(
            _temperature_params(), time_steps, 60.0, current_profile
        )


@pytest.mark.parametrize("time_step", [0.0, -1.0, math.nan, math.inf, -math.inf, True])
def test_transient_temperature_rejects_invalid_time_steps(time_step):
    with pytest.raises(ValueError, match="time_steps"):
        ThermalCalculator().calculate_transient_temperature(
            _temperature_params(), [time_step], 60.0, [500.0]
        )


@pytest.mark.parametrize("current", [-1.0, math.nan, math.inf, -math.inf, True])
def test_transient_temperature_rejects_invalid_current_profile(current):
    with pytest.raises(ValueError, match="current_profile"):
        ThermalCalculator().calculate_transient_temperature(
            _temperature_params(), [1.0], 60.0, [current]
        )


def test_heat_capacity_keeps_ieee_material_specific_heat_values():
    params = _temperature_params()

    result = ThermalCalculator().calculate_heat_capacity(params)

    assert result == 1.116 * 955 + 0.5126 * 476


def test_thermal_calculator_does_not_expose_upstream_weather_corrections():
    calculator = ThermalCalculator()

    for name in (
        "apply_micro_climate_corrections",
        "ALPHA_ROUGHNESS",
        "REF_HEIGHT_GRID",
        "LINE_AVG_HEIGHT",
    ):
        assert not hasattr(calculator, name)
