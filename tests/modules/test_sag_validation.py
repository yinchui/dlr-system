from dataclasses import FrozenInstanceError

import numpy as np
import pandas as pd
import pytest

from modules.sag_validation import (
    ParameterSource,
    SagValidationSnapshot,
    SagState,
    SagValidationConfig,
    SagValidationService,
    adaptive_temperature_threshold,
    build_sag_snapshot,
    compute_derating,
    horizontal_tension,
    infer_mean_temperature,
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


def test_explicit_timestamp_outside_snapshot_is_not_reassigned_by_row_position():
    normalized = normalize_inclination_dataframe(
        pd.DataFrame(
            {
                "杆塔": ["001号"],
                "时间": ["2026-07-24 00:00+08:00"],
                "倾角": [1.0],
            }
        ),
        selected_tower_id="001",
        snapshot=_snapshot(),
    )

    with pytest.raises(ValueError, match="does not match"):
        resolve_sag_parameters(
            inclination_row=normalized.iloc[0],
            snapshot=_snapshot(),
            conductor=drake_conductor(),
        )


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


def test_direct_snapshot_constructor_freezes_mutable_inputs():
    coordinates = {"001": {"lon": 120.0, "lat": 40.0}}
    conductor = {"area_m2": 4.685e-4, "materials": [{"mass": 1.0}]}
    currents = [[1000.0]]
    common_matrix = ((1.0,),)

    snapshot = SagValidationSnapshot(
        source_run_id="run-direct",
        tower_ids=("001",),
        timestamps=(0,),
        coordinates=coordinates,
        conductor_params=conductor,
        original_currents=currents,
        ambient_temperatures=common_matrix,
        wind_speeds=common_matrix,
        wind_angles=common_matrix,
        solar_radiation=common_matrix,
        elevations=common_matrix,
    )
    coordinates["001"]["lon"] = -1.0
    conductor["materials"][0]["mass"] = -1.0
    currents[0][0] = -1.0

    assert snapshot.coordinates["001"]["lon"] == 120.0
    assert snapshot.conductor_params["materials"][0]["mass"] == 1.0
    assert snapshot.original_currents[0][0] == 1000.0
    with pytest.raises(TypeError):
        snapshot.coordinates["001"]["lon"] = 0.0


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


def test_patent_tension_and_temperature_formula():
    tension = horizontal_tension(
        weight_n_m=10.0,
        span_m=100.0,
        angle_deg=45.0,
    )
    mean_temperature = infer_mean_temperature(
        current_tension_n=tension,
        reference_tension_n=1000.0,
        elastic_modulus_pa=1.0e11,
        area_m2=1.0e-4,
        thermal_expansion_per_c=1.0e-5,
        reference_temp_c=20.0,
    )

    assert tension == pytest.approx(500.0)
    assert mean_temperature == pytest.approx(25.0)


def test_checked_current_is_minimum_of_three_candidates():
    result = compute_derating(
        ambient_temp_c=20.0,
        theoretical_temp_c=60.0,
        measured_temp_c=80.0,
        original_current_a=1000.0,
        recalculated_current_a=900.0,
    )

    assert result.factor == pytest.approx((40.0 / 60.0) ** 0.5)
    assert result.checked_current_a == pytest.approx(
        min(1000.0, 900.0, 1000.0 * (40.0 / 60.0) ** 0.5)
    )


def test_adaptive_threshold_increases_with_each_uncertainty_source():
    stable = adaptive_temperature_threshold(
        base_threshold_c=5.0,
        angle_samples=[1.0, 1.0, 1.0],
        wind_samples=[3.0, 3.0, 3.0],
        span_m=100.0,
        historical_errors=[0.0, 0.0, 0.0],
    )

    assert adaptive_temperature_threshold(
        base_threshold_c=5.0,
        angle_samples=[0.5, 1.0, 1.5],
        wind_samples=[3.0, 3.0, 3.0],
        span_m=100.0,
        historical_errors=[0.0, 0.0, 0.0],
    ) > stable
    assert adaptive_temperature_threshold(
        base_threshold_c=5.0,
        angle_samples=[1.0, 1.0, 1.0],
        wind_samples=[1.0, 3.0, 5.0],
        span_m=100.0,
        historical_errors=[0.0, 0.0, 0.0],
    ) > stable
    assert adaptive_temperature_threshold(
        base_threshold_c=5.0,
        angle_samples=[1.0, 1.0, 1.0],
        wind_samples=[3.0, 3.0, 3.0],
        span_m=600.0,
        historical_errors=[0.0, 0.0, 0.0],
    ) > stable
    assert adaptive_temperature_threshold(
        base_threshold_c=5.0,
        angle_samples=[1.0, 1.0, 1.0],
        wind_samples=[3.0, 3.0, 3.0],
        span_m=100.0,
        historical_errors=[0.0, 4.0, 8.0],
    ) > stable


def _state_config():
    return SagValidationConfig(
        base_threshold_c=5.0,
        recovery_ratio=0.6,
        recovery_samples=2,
        recovery_alpha=0.5,
        convergence_tolerance_a=0.1,
    )


def _state_row(
    *,
    tower_id="001",
    timestamp=0,
    angle_deg=45.0,
    theoretical_temp_c=15.0,
):
    return {
        "tower_id": tower_id,
        "timestamp": timestamp,
        "sample_index": int(timestamp) if isinstance(timestamp, int) else 0,
        "angle_deg": angle_deg,
        "span_m": 100.0,
        "unit_weight_n_m": 10.0,
        "reference_tension_n": 1000.0,
        "reference_temp_c": 20.0,
        "elastic_modulus_pa": 1.0e11,
        "area_m2": 1.0e-4,
        "thermal_expansion_per_c": 1.0e-5,
        "ambient_temp_c": 0.0,
        "theoretical_temp_c": theoretical_temp_c,
        "original_current_a": 1000.0,
        "recalculated_current_a": 900.0,
        "wind_speed": 3.0,
    }


def test_trigger_requires_error_strictly_above_threshold():
    result = SagValidationService(config=_state_config()).validate_batch(
        pd.DataFrame([_state_row(theoretical_temp_c=20.0)])
    )

    assert result[0].measured_temp_c == pytest.approx(25.0)
    assert result[0].temperature_error_c == pytest.approx(5.0)
    assert result[0].threshold_c == pytest.approx(5.0)
    assert result[0].state is SagState.NORMAL
    assert result[0].final_current_a == pytest.approx(1000.0)


def test_invalid_sample_does_not_advance_recovery_state():
    rows = [
        _state_row(timestamp=0, theoretical_temp_c=15.0),
        _state_row(timestamp=1, angle_deg=0.0, theoretical_temp_c=23.0),
        _state_row(timestamp=2, theoretical_temp_c=23.0),
        _state_row(timestamp=3, theoretical_temp_c=23.0),
    ]

    results = SagValidationService(config=_state_config()).validate_batch(
        pd.DataFrame(rows)
    )

    assert results[0].state is SagState.RISK
    assert results[1].state is SagState.INVALID
    assert results[1].final_current_a == results[0].final_current_a
    assert results[2].state is not SagState.NORMAL
    assert results[2].state is SagState.RISK
    assert results[3].state is SagState.RECOVERY
    assert results[3].final_current_a > results[0].final_current_a
    assert results[3].final_current_a < results[3].original_current_a


def test_state_is_isolated_per_tower():
    rows = [
        _state_row(tower_id="001", timestamp=0, theoretical_temp_c=15.0),
        _state_row(tower_id="002", timestamp=0, theoretical_temp_c=23.0),
    ]

    results = SagValidationService(config=_state_config()).validate_batch(
        pd.DataFrame(rows)
    )

    assert results[0].state is SagState.RISK
    assert results[1].state is SagState.NORMAL
    assert results[1].final_current_a == pytest.approx(1000.0)


def test_invalid_record_is_isolated_and_all_outputs_remain_finite():
    rows = [
        _state_row(timestamp=0, angle_deg=float("nan")),
        _state_row(timestamp=1, theoretical_temp_c=23.0),
    ]

    results = SagValidationService(config=_state_config()).validate_batch(
        pd.DataFrame(rows)
    )

    assert results[0].state is SagState.INVALID
    assert results[0].error_code
    assert results[1].state is SagState.NORMAL
    for result in results:
        assert np.isfinite(result.final_current_a)
        assert np.isfinite(result.original_current_a)


def test_finite_but_nonphysical_inferred_temperature_is_invalid():
    rows = [
        _state_row(timestamp=0, angle_deg=0.05),
        _state_row(timestamp=1, theoretical_temp_c=23.0),
    ]

    results = SagValidationService(config=_state_config()).validate_batch(
        pd.DataFrame(rows)
    )

    assert results[0].state is SagState.INVALID
    assert results[0].error_code == "nonphysical_temperature"
    assert results[1].state is SagState.NORMAL
