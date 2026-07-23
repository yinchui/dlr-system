import copy
import math

import numpy as np
import pytest

from modules.thermal_engine import ThermalCalculator


def drake_transient_params():
    return {
        "D0": 0.02814,
        "R_low_25": 7.283e-5,
        "R_high_75": 8.688e-5,
        "R_high_200": 1.220e-4,
        "emissivity": 0.8,
        "absorptivity": 0.8,
        "T_a": 40.0,
        "T_s": 100.0,
        "T_avg": 100.0,
        "wind_speed": 0.61,
        "wind_angle": 90.0,
        "elevation": 0.0,
        "latitude": 30.0,
        "line_azimuth": 90.0,
        "day_of_year": 161,
        "time": 11.0,
        "materials": [
            {"type": "aluminum", "mass": 1.116},
            {"type": "steel", "mass": 0.5126},
        ],
    }


def test_drake_step_response_uses_ten_second_steps():
    temperatures = ThermalCalculator().calculate_transient_temperature(
        params=drake_transient_params(),
        time_steps=[10.0, 10.0],
        initial_temp=100.0,
        current_profile=[1200.0, 1200.0],
    )

    assert temperatures[1] - temperatures[0] == pytest.approx(0.28, abs=0.03)
    assert temperatures[2] - temperatures[1] == pytest.approx(0.27, abs=0.03)


def test_external_interval_is_split_into_ten_second_steps_and_remainder():
    calculator = ThermalCalculator()
    params = drake_transient_params()

    external = calculator.calculate_transient_temperature(
        params=params,
        time_steps=[25.0],
        initial_temp=100.0,
        current_profile=[1200.0],
    )
    reference = calculator.calculate_transient_temperature(
        params=params,
        time_steps=[10.0, 10.0, 5.0],
        initial_temp=100.0,
        current_profile=[1200.0, 1200.0, 1200.0],
    )

    assert len(external) == 2
    assert external[-1] == pytest.approx(reference[-1], abs=1e-12)


def test_drake_fifteen_minute_transient_rating_reaches_125_c():
    temperatures = ThermalCalculator().calculate_transient_temperature(
        params=drake_transient_params(),
        time_steps=[15.0 * 60.0],
        initial_temp=100.0,
        current_profile=[1312.0],
    )

    assert temperatures == pytest.approx([100.0, 125.0], abs=0.2)


def test_transient_supports_heat_gain_when_ambient_exceeds_conductor():
    params = {
        **drake_transient_params(),
        "T_a": 50.0,
        "T_s": 40.0,
        "T_avg": 40.0,
        "solar_radiation": 0.0,
    }

    temperatures = ThermalCalculator().calculate_transient_temperature(
        params=params,
        time_steps=[10.0],
        initial_temp=40.0,
        current_profile=[0.0],
    )

    assert 40.0 < temperatures[-1] < 50.0


def test_transient_uses_explicit_material_heat_capacity_and_preserves_input():
    calculator = ThermalCalculator()
    params = drake_transient_params()
    before = copy.deepcopy(params)

    heat_capacity = calculator.calculate_heat_capacity(params)
    calculator.calculate_transient_temperature(
        params=params,
        time_steps=[10.0],
        initial_temp=100.0,
        current_profile=[1200.0],
    )

    assert heat_capacity == pytest.approx(1310.0, abs=0.5)
    assert params == before


@pytest.mark.parametrize(
    ("materials", "message"),
    [
        (None, "materials"),
        ([], "materials"),
        ([{"type": "unknown", "mass": 1.0}], "type"),
        ([{"type": "aluminum", "mass": 0.0}], "mass"),
        ([{"type": "aluminum", "mass": -1.0}], "mass"),
        ([{"type": "aluminum", "mass": math.nan}], "mass"),
    ],
)
def test_heat_capacity_rejects_missing_or_invalid_explicit_materials(
    materials, message
):
    params = drake_transient_params()
    if materials is None:
        params.pop("materials")
    else:
        params["materials"] = materials

    with pytest.raises(ValueError, match=message):
        ThermalCalculator().calculate_heat_capacity(params)


def test_empty_transient_series_still_validates_weather():
    params = drake_transient_params()
    params["wind_speed"] = math.nan

    with pytest.raises(ValueError, match="wind_speed"):
        ThermalCalculator().calculate_transient_temperature(
            params=params,
            time_steps=[],
            initial_temp=100.0,
            current_profile=[],
        )


@pytest.mark.parametrize(
    ("time_steps", "current_profile", "message"),
    [
        (np.array([np.bool_(True)]), [1200.0], "time_steps"),
        ([10.0], np.array([np.bool_(True)]), "current_profile"),
    ],
)
def test_transient_rejects_numpy_boolean_steps_and_currents(
    time_steps, current_profile, message
):
    with pytest.raises(ValueError, match=message):
        ThermalCalculator().calculate_transient_temperature(
            params=drake_transient_params(),
            time_steps=time_steps,
            initial_temp=100.0,
            current_profile=current_profile,
        )


@pytest.mark.parametrize("field", ["time_steps", "current_profile"])
@pytest.mark.parametrize(
    "invalid_value",
    [
        10.0,
        None,
        np.array([[10.0]]),
        "5",
        ["5"],
        [10 ** 1000],
        np.bool_(True),
    ],
)
def test_transient_rejects_non_numeric_or_non_vector_profiles(
    field, invalid_value
):
    arguments = {
        "time_steps": [10.0],
        "current_profile": [1200.0],
    }
    arguments[field] = invalid_value

    with pytest.raises(ValueError, match=field):
        ThermalCalculator().calculate_transient_temperature(
            params=drake_transient_params(),
            time_steps=arguments["time_steps"],
            initial_temp=100.0,
            current_profile=arguments["current_profile"],
        )
