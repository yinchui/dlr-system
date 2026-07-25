from dataclasses import FrozenInstanceError

import numpy as np
import pandas as pd
import pytest

from modules.sag_validation import (
    ParameterSource,
    build_sag_snapshot,
    normalize_inclination_dataframe,
    resolve_sag_parameters,
)
from tests.fixtures.sag_data import (
    drake_conductor,
    make_inclination_row,
    make_line_data,
)


def _snapshot(*, times=2, include_operating_current=False):
    return build_sag_snapshot(
        make_line_data(
            times=times,
            include_operating_current=include_operating_current,
        ),
        drake_conductor(),
    )


def test_angle_only_input_uses_selected_tower_and_snapshot_times():
    result = normalize_inclination_dataframe(
        pd.DataFrame({"倾角": [1.0, 1.1]}),
        selected_tower_id="001",
        snapshot=_snapshot(times=2),
    )

    assert result["tower_id"].tolist() == ["001", "001"]
    assert result["timestamp"].notna().all()
    assert result["timestamp"].tolist() == list(_snapshot(times=2).timestamps)
    assert result["angle_deg"].tolist() == [1.0, 1.1]


def test_explicit_tower_and_timestamp_are_preserved():
    expected_time = pd.Timestamp("2026-07-23 00:30", tz="Asia/Shanghai")
    result = normalize_inclination_dataframe(
        pd.DataFrame(
            {
                "杆塔": ["002号"],
                "时间": ["2026-07-23 00:30+08:00"],
                "倾角": [1.2],
            }
        ),
        selected_tower_id="001",
        snapshot=_snapshot(),
    )

    assert result.loc[0, "tower_id"] == "002"
    assert result.loc[0, "timestamp"] == expected_time


def test_missing_timestamp_with_different_length_uses_stable_sample_index():
    result = normalize_inclination_dataframe(
        pd.DataFrame({"倾角": [1.0, 1.1, 1.2]}),
        selected_tower_id="001",
        snapshot=_snapshot(times=2),
    )

    assert result["sample_index"].tolist() == [0, 1, 2]
    assert result["timestamp"].tolist() == [0, 1, 2]


def test_inclination_column_is_required():
    with pytest.raises(ValueError, match="倾角"):
        normalize_inclination_dataframe(
            pd.DataFrame({"时间": ["2026-07-23 00:00"]}),
            selected_tower_id="001",
            snapshot=_snapshot(),
        )


def test_parameter_priority_is_measured_then_derived_then_default():
    params = resolve_sag_parameters(
        inclination_row=make_inclination_row(),
        snapshot=_snapshot(),
        conductor=drake_conductor(),
    )

    assert params.span_m.source is ParameterSource.DERIVED
    assert params.area_m2.source is ParameterSource.MEASURED
    assert params.reference_tension_n.source is ParameterSource.DEFAULT
    assert set(value.source.value for value in params.values()) <= {
        "measured",
        "derived",
        "default",
    }


def test_measured_row_value_overrides_derived_span():
    params = resolve_sag_parameters(
        inclination_row=make_inclination_row(span_m=275.0),
        snapshot=_snapshot(),
        conductor=drake_conductor(),
    )

    assert params.span_m.value == 275.0
    assert params.span_m.source is ParameterSource.MEASURED


def test_unit_weight_is_derived_from_conductor_mass_and_gravity():
    params = resolve_sag_parameters(
        inclination_row=make_inclination_row(),
        snapshot=_snapshot(),
        conductor=drake_conductor(),
    )

    assert params.unit_weight_n_m.value == pytest.approx(1.6286 * 9.80665)
    assert params.unit_weight_n_m.source is ParameterSource.DERIVED


def test_unit_weight_uses_material_mass_components_when_total_is_missing():
    conductor = drake_conductor()
    conductor.pop("mass_per_length_kg_m")
    conductor["materials"] = [
        {"type": "aluminum", "density": 1.116},
        {"type": "steel", "density": 0.5126},
    ]

    params = resolve_sag_parameters(
        inclination_row=make_inclination_row(),
        snapshot=build_sag_snapshot(make_line_data(), conductor),
        conductor=conductor,
    )

    assert params.unit_weight_n_m.value == pytest.approx(1.6286 * 9.80665)
    assert "material_components" in params.unit_weight_n_m.detail


def test_negative_reference_temperature_remains_a_measured_value():
    params = resolve_sag_parameters(
        inclination_row=make_inclination_row(reference_temp_c=-10.0),
        snapshot=_snapshot(),
        conductor=drake_conductor(),
    )

    assert params.reference_temp_c.value == -10.0
    assert params.reference_temp_c.source is ParameterSource.MEASURED


def test_local_ambient_temperature_is_never_used_as_theoretical_temperature():
    params = resolve_sag_parameters(
        inclination_row=make_inclination_row(),
        snapshot=_snapshot(),
        conductor=drake_conductor(),
    )

    assert params.ambient_temp_c.value == 25.0
    assert params.theoretical_temp_c.value == 20.0
    assert params.theoretical_temp_c.source is ParameterSource.DEFAULT


def test_theoretical_temperature_requires_operating_current_and_solver():
    calls = []

    def solve_temperature(weather, current):
        calls.append((dict(weather), current))
        return 57.5

    params = resolve_sag_parameters(
        inclination_row=make_inclination_row(),
        snapshot=_snapshot(include_operating_current=True),
        conductor=drake_conductor(),
        temperature_solver=solve_temperature,
    )

    assert calls and calls[0][1] == 600.0
    assert params.theoretical_temp_c.value == 57.5
    assert params.theoretical_temp_c.source is ParameterSource.DERIVED


def test_zero_operating_current_is_still_passed_to_temperature_solver():
    calls = []

    def solve_temperature(weather, current):
        calls.append(current)
        return weather["T_a"]

    params = resolve_sag_parameters(
        inclination_row=make_inclination_row(operating_current_a=0.0),
        snapshot=_snapshot(),
        conductor=drake_conductor(),
        temperature_solver=solve_temperature,
    )

    assert calls == [0.0]
    assert params.operating_current_a.value == 0.0
    assert params.theoretical_temp_c.source is ParameterSource.DERIVED


def test_recalculated_current_uses_only_injected_callable():
    params = resolve_sag_parameters(
        inclination_row=make_inclination_row(),
        snapshot=_snapshot(),
        conductor=drake_conductor(),
        current_recalculator=lambda weather: 875.0,
    )

    assert params.recalculated_current_a.value == 875.0
    assert params.recalculated_current_a.source is ParameterSource.DERIVED


def test_snapshot_does_not_alias_main_arrays_or_nested_mappings():
    line_data = make_line_data()
    conductor = drake_conductor()
    snapshot = build_sag_snapshot(line_data, conductor)

    line_data["max_currents"][0, 0] = -1.0
    line_data["tower_coords"]["001"]["lon"] = -1.0
    conductor["area_m2"] = -1.0

    assert snapshot.original_currents[0][0] >= 0.0
    assert snapshot.coordinates["001"]["lon"] == 120.0
    assert snapshot.conductor_params["area_m2"] > 0.0
    with pytest.raises(TypeError):
        snapshot.coordinates["001"]["lon"] = 0.0
    with pytest.raises(TypeError):
        snapshot.conductor_params["area_m2"] = 0.0
    with pytest.raises(FrozenInstanceError):
        snapshot.source_run_id = "changed"


def test_snapshot_rejects_mismatched_rating_shape():
    line_data = make_line_data()
    line_data["max_currents"] = np.ones((1, 2))

    with pytest.raises(ValueError, match="max_currents"):
        build_sag_snapshot(line_data, drake_conductor())


def test_snapshot_requires_completed_dlr_ratings():
    line_data = make_line_data()
    line_data.pop("max_currents")

    with pytest.raises(ValueError, match="max_currents"):
        build_sag_snapshot(line_data, drake_conductor())
