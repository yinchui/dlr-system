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


def test_resistance_at_100_c_matches_drake_and_is_continuous():
    calculator = EngineThermalCalculator()
    at_100 = calculator.calculate_resistance({**DRAKE_STEADY_PARAMS, "T_avg": 100.0})
    below = calculator.calculate_resistance({**DRAKE_STEADY_PARAMS, "T_avg": 100.0 - 1e-6})
    above = calculator.calculate_resistance({**DRAKE_STEADY_PARAMS, "T_avg": 100.0 + 1e-6})

    assert at_100 == pytest.approx(9.391e-5, rel=1e-4)
    assert below == pytest.approx(at_100, abs=1e-10)
    assert above == pytest.approx(at_100, abs=1e-10)


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
