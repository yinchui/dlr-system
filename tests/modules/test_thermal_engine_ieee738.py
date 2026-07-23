"""Official IEEE 738 Drake steady-state regression reference."""

import copy
import importlib.util
import math
from pathlib import Path

import pytest

from modules.thermal_engine import ThermalCalculator as EngineThermalCalculator
from thermal_functions import ThermalCalculator as CompatibilityThermalCalculator


_FIXTURE_PATH = Path(__file__).parents[1] / "fixtures" / "ieee738_reference.py"
_fixture_spec = importlib.util.spec_from_file_location("ieee738_reference", _FIXTURE_PATH)
assert _fixture_spec is not None and _fixture_spec.loader is not None
_fixture_module = importlib.util.module_from_spec(_fixture_spec)
_fixture_spec.loader.exec_module(_fixture_module)
DRAKE_STEADY_PARAMS = _fixture_module.DRAKE_STEADY_PARAMS


def test_ieee_738_drake_steady_reference():
    """IEEE Std 738-2023 4.6.1 reproduces the published Drake heat balance."""
    result = EngineThermalCalculator().calculate_heat_balance(DRAKE_STEADY_PARAMS)

    assert result.q_convection_natural == pytest.approx(42.42, abs=0.15)
    assert result.q_convection_low_re == pytest.approx(82.10, abs=0.20)
    assert result.q_convection_high_re == pytest.approx(77.06, abs=0.20)
    assert result.q_convection == pytest.approx(82.10, abs=0.20)
    assert result.q_radiation == pytest.approx(39.11, abs=0.15)
    assert result.q_solar == pytest.approx(22.45, abs=0.25)
    assert result.resistance == pytest.approx(9.391e-5, rel=2e-3)
    assert result.current_a == pytest.approx(1025.0, abs=2.0)


def test_zero_wind_uses_natural_convection_only():
    params = {**DRAKE_STEADY_PARAMS, "wind_speed": 0.0}

    result = EngineThermalCalculator().calculate_heat_balance(params)

    assert result.q_convection_natural > 0.0
    assert result.q_convection_low_re == 0.0
    assert result.q_convection_high_re == 0.0
    assert result.q_convection == result.q_convection_natural


@pytest.mark.parametrize(
    ("angle", "expected_factor"),
    [(0.0, 0.388), (90.0, 1.0), (180.0, 0.388), (-90.0, 1.0), (270.0, 1.0)],
)
def test_wind_angle_is_normalized_to_conductor_axis(angle, expected_factor):
    assert EngineThermalCalculator.wind_angle_factor(angle) == pytest.approx(
        expected_factor, abs=1e-12
    )


@pytest.mark.parametrize("angle", [math.nan, math.inf, -math.inf, True, False])
def test_wind_angle_factor_rejects_nonfinite_and_boolean_values(angle):
    with pytest.raises(ValueError, match="wind_angle"):
        EngineThermalCalculator.wind_angle_factor(angle)


def test_convection_uses_public_wind_angle_factor():
    class HalfAngleFactorCalculator(EngineThermalCalculator):
        @staticmethod
        def wind_angle_factor(angle):
            return 0.5

    perpendicular = EngineThermalCalculator().calculate_heat_balance(
        {**DRAKE_STEADY_PARAMS, "wind_speed": 5.0, "wind_angle": 90.0}
    )
    result = HalfAngleFactorCalculator().calculate_heat_balance(
        {**DRAKE_STEADY_PARAMS, "wind_speed": 5.0, "wind_angle": 12.0}
    )

    assert result.q_convection_low_re / perpendicular.q_convection_low_re == pytest.approx(
        0.5, abs=1e-12
    )
    assert result.q_convection_high_re / perpendicular.q_convection_high_re == pytest.approx(
        0.5, abs=1e-12
    )


def test_heat_balance_and_current_preserve_input_and_ignore_terrain_keys():
    calculator = EngineThermalCalculator()
    plain = copy.deepcopy(DRAKE_STEADY_PARAMS)
    terrain = {**copy.deepcopy(plain), "slope": 35.0, "aspect": 270.0}
    terrain_before = copy.deepcopy(terrain)
    plain_before = copy.deepcopy(plain)

    plain_result = calculator.calculate_heat_balance(plain)
    terrain_result = calculator.calculate_heat_balance(terrain)
    plain_current = calculator.calculate_steady_state_current(plain)
    terrain_current = calculator.calculate_steady_state_current(terrain)

    assert plain == plain_before
    assert terrain == terrain_before
    assert terrain_result == plain_result
    assert terrain_current == plain_current


def test_measured_solar_is_local_effective_radiation_without_clear_sky_or_albedo():
    params = copy.deepcopy(DRAKE_STEADY_PARAMS)
    for key in ("latitude", "line_azimuth", "day_of_year", "time"):
        params.pop(key)
    params.update({"solar_radiation": 500.0, "GROUND_ALBEDO": 0.95})

    result = EngineThermalCalculator().calculate_heat_balance(params)

    assert result.q_solar == pytest.approx(
        params["absorptivity"] * params["solar_radiation"] * params["D0"]
    )


def test_clear_sky_solar_matches_drake_reference_without_measured_radiation():
    result = EngineThermalCalculator().calculate_heat_balance(DRAKE_STEADY_PARAMS)

    assert result.q_solar == pytest.approx(22.45, abs=0.25)


def test_solar_azimuth_at_equinox_equator_preserves_east_west_direction():
    calculator = EngineThermalCalculator()
    params = {
        "latitude": 0.0,
        "day_of_year": 81,
        "time": 9.0,
        "line_azimuth": 0.0,
        "elevation": 0.0,
        "D0": 0.02814,
        "absorptivity": 0.8,
    }

    morning_azimuth = calculator.calculate_solar_azimuth(params)
    afternoon_azimuth = calculator.calculate_solar_azimuth({**params, "time": 15.0})
    solar_altitude = calculator.calculate_solar_altitude(params)
    perpendicular_gain = (
        params["absorptivity"]
        * calculator.calculate_elevation_corrected_radiation(
            params, calculator.calculate_solar_radiation(params, solar_altitude)
        )
        * params["D0"]
    )

    assert morning_azimuth == pytest.approx(90.0, abs=1e-10)
    assert afternoon_azimuth == pytest.approx(270.0, abs=1e-10)
    assert calculator.calculate_solar_gain(params) == pytest.approx(perpendicular_gain)


@pytest.mark.parametrize(
    ("time", "expected_azimuth"),
    [(11.0, 113.98441942855138), (13.0, 246.01558057144862)],
)
def test_solar_azimuth_preserves_normal_morning_and_afternoon_quadrants(
    time, expected_azimuth
):
    params = {**DRAKE_STEADY_PARAMS, "time": time}

    assert EngineThermalCalculator().calculate_solar_azimuth(params) == pytest.approx(
        expected_azimuth, abs=1e-10
    )


def test_resistance_at_100_c_uses_the_drake_25_to_75_segment():
    calculator = EngineThermalCalculator()
    at_100 = calculator.calculate_resistance({**DRAKE_STEADY_PARAMS, "T_avg": 100.0})
    below = calculator.calculate_resistance({**DRAKE_STEADY_PARAMS, "T_avg": 100.0 - 1e-6})

    assert at_100 == pytest.approx(9.391e-5, rel=1e-4)
    assert below == pytest.approx(at_100, abs=1e-10)


def test_resistance_above_100_c_uses_ieee_25_to_200_endpoints_directly():
    params = {
        "T_s": 150.0,
        "T_avg": 150.0,
        "R_low_25": 1.0,
        "R_high_75": 2.0,
        "R_high_200": 10.0,
    }

    assert EngineThermalCalculator().calculate_resistance(params) == pytest.approx(
        7.428571428571429
    )


def test_compatibility_import_and_legacy_entries_share_heat_balance_values():
    assert CompatibilityThermalCalculator is EngineThermalCalculator
    calculator = CompatibilityThermalCalculator()
    result = calculator.calculate_heat_balance(DRAKE_STEADY_PARAMS)

    assert calculator.calculate_convection(DRAKE_STEADY_PARAMS) == result.q_convection
    assert calculator.calculate_radiation(DRAKE_STEADY_PARAMS) == result.q_radiation
    assert calculator.calculate_solar_gain(DRAKE_STEADY_PARAMS) == result.q_solar
    assert calculator.calculate_resistance(DRAKE_STEADY_PARAMS) == result.resistance
    assert calculator.calculate_steady_state_current(DRAKE_STEADY_PARAMS) == result.current_a


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("D0", 0.0),
        ("D0", math.nan),
        ("wind_speed", -0.1),
        ("wind_speed", math.inf),
        ("T_s", math.nan),
        ("T_a", -273.15),
        ("solar_radiation", -0.1),
        ("solar_radiation", math.nan),
        ("emissivity", -0.1),
        ("emissivity", 1.1),
        ("absorptivity", math.inf),
    ],
)
def test_invalid_physical_inputs_are_rejected(key, value):
    params = {**DRAKE_STEADY_PARAMS, key: value}

    with pytest.raises(ValueError, match=key):
        EngineThermalCalculator().calculate_heat_balance(params)


def test_conductor_temperature_below_ambient_is_rejected():
    with pytest.raises(ValueError, match="T_s"):
        EngineThermalCalculator().calculate_heat_balance(
            {**DRAKE_STEADY_PARAMS, "T_s": 39.0, "T_avg": 39.0}
        )


def test_equal_conductor_and_ambient_temperature_is_finite():
    params = {**DRAKE_STEADY_PARAMS, "T_s": 40.0, "T_avg": 40.0}

    result = EngineThermalCalculator().calculate_heat_balance(params)

    assert result.q_convection_natural == 0.0
    assert result.q_convection_low_re == 0.0
    assert result.q_convection_high_re == 0.0
    assert result.q_convection == 0.0
    assert result.q_radiation == pytest.approx(0.0)
    assert result.current_a == 0.0
    assert all(math.isfinite(value) for value in vars(result).values())


def test_steady_state_temperature_expands_verified_bracket_and_rejects_no_root():
    calculator = EngineThermalCalculator()
    params = copy.deepcopy(DRAKE_STEADY_PARAMS)
    before = copy.deepcopy(params)

    temperature = calculator.calculate_steady_state_temperature(
        params, current=3000.0, max_iter=100, tol=1e-6
    )
    balance = calculator.calculate_heat_balance(
        {**params, "T_s": temperature, "T_avg": temperature}
    )
    residual = (
        3000.0 ** 2 * balance.resistance
        + balance.q_solar
        - balance.q_convection
        - balance.q_radiation
    )

    assert temperature == pytest.approx(475.232159, abs=1e-6)
    assert residual == pytest.approx(0.0, abs=1e-6)
    with pytest.raises(ValueError, match="physical temperature range"):
        calculator.calculate_steady_state_temperature(params, current=1_000_000.0)
    assert params == before


def test_steady_state_temperature_accepts_verified_low_endpoint_immediately():
    calculator = EngineThermalCalculator()
    params = {
        **copy.deepcopy(DRAKE_STEADY_PARAMS),
        "T_s": DRAKE_STEADY_PARAMS["T_a"],
        "T_avg": DRAKE_STEADY_PARAMS["T_a"],
        "solar_radiation": 0.0,
    }
    before = copy.deepcopy(params)
    current = math.sqrt(0.5e-6 / calculator.calculate_resistance(params))

    temperature = calculator.calculate_steady_state_temperature(
        params, current=current, max_iter=1, tol=1e-12
    )

    assert temperature == params["T_a"]
    assert params == before


def test_steady_state_temperature_accepts_verified_high_endpoint_immediately():
    calculator = EngineThermalCalculator()
    params = {
        **copy.deepcopy(DRAKE_STEADY_PARAMS),
        "T_s": 200.0,
        "T_avg": 200.0,
    }
    before = copy.deepcopy(params)
    current = calculator.calculate_steady_state_current(params)

    temperature = calculator.calculate_steady_state_temperature(
        params, current=current, max_iter=1, tol=1e-12
    )

    assert temperature == 200.0
    assert params == before


def test_steady_state_temperature_rejects_discontinuous_resistance_pseudo_root():
    params = copy.deepcopy(DRAKE_STEADY_PARAMS)
    before = copy.deepcopy(params)

    with pytest.raises(ValueError):
        EngineThermalCalculator().calculate_steady_state_temperature(
            params,
            current=1025.2248468542948,
            max_iter=200,
            tol=1e-12,
        )

    assert params == before


def test_steady_state_temperature_keeps_normal_drake_solution():
    params = copy.deepcopy(DRAKE_STEADY_PARAMS)
    before = copy.deepcopy(params)
    calculator = EngineThermalCalculator()

    temperature = calculator.calculate_steady_state_temperature(params, current=1000.0)
    balance = calculator.calculate_heat_balance(
        {**params, "T_s": temperature, "T_avg": temperature}
    )
    residual = (
        1000.0 ** 2 * balance.resistance
        + balance.q_solar
        - balance.q_convection
        - balance.q_radiation
    )

    assert temperature == pytest.approx(97.5, abs=0.1)
    assert residual == pytest.approx(0.0, abs=1e-6)
    assert params == before


def test_steady_state_temperature_keeps_true_drake_rated_temperature_root():
    params = copy.deepcopy(DRAKE_STEADY_PARAMS)
    before = copy.deepcopy(params)
    calculator = EngineThermalCalculator()
    current = calculator.calculate_steady_state_current(params)

    temperature = calculator.calculate_steady_state_temperature(
        params, current=current, max_iter=3, tol=1e-12
    )

    assert temperature == pytest.approx(100.0, abs=1e-9)
    assert params == before
