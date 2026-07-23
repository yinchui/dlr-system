import copy

import numpy as np

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
        terrain_data={0: {"slope": 12.0, "aspect": 270.0, "elevation": 1100.0}},
    )
    assert result["max_currents"].shape == (1, 2)
    assert result["corrected_winds"][0, 0] > 0


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

    calculator.calculate_steady_state_temperature(params, current=500.0, max_iter=2)
    calculator.calculate_transient_temperature(
        params, time_steps=[1.0], initial_temp=60.0, current_profile=[500.0]
    )

    assert params == before


def test_heat_capacity_keeps_ieee_material_specific_heat_values():
    params = _temperature_params()

    result = ThermalCalculator().calculate_heat_capacity(params)

    assert result == 1.116 * 955 + 0.5126 * 476
