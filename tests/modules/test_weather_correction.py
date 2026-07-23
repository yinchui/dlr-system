import ast
import copy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from modules.weather_correction import CorrectionOptions, WeatherCorrectionService


def weather_frame(**overrides):
    data = {
        "tower_id": ["001"],
        "position": [36],
        "ambient_temp": [20.0],
        "wind_speed": [4.0],
        "wind_direction": [0.0],
        "solar_radiation": [600.0],
        "humidity": [25.0],
    }
    data.update(overrides)
    frame = pd.DataFrame(data)
    frame.attrs["source"] = {"role": "physical"}
    return frame


def matrix_adapter():
    app_path = Path(__file__).parents[2] / "dispatch_app_st.py"
    source = app_path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(app_path))
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "apply_weather_corrections"
    )
    namespace = {
        "copy": copy,
        "np": np,
        "pd": pd,
        "CorrectionOptions": CorrectionOptions,
        "WeatherCorrectionService": WeatherCorrectionService,
    }
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(app_path), "exec"), namespace)
    return namespace["apply_weather_corrections"]


def test_apply_is_pure_and_marks_a_single_correction_stage():
    source = weather_frame()
    source_snapshot = copy.deepcopy(source)
    source_attrs = copy.deepcopy(source.attrs)

    corrected = WeatherCorrectionService().apply(
        source,
        terrain_lookup={"001": {"slope": 0.0, "aspect": 0.0, "elevation": 1000.0}},
        options=CorrectionOptions(
            enable_vertical=False,
            enable_terrain=False,
            enable_desert=False,
            enable_wind_direction=False,
        ),
    )

    pd.testing.assert_frame_equal(source, source_snapshot)
    assert source.attrs == source_attrs
    assert corrected is not source
    assert corrected.attrs == source_attrs
    assert {
        "wind_speed_physical",
        "ambient_temp_physical",
        "solar_radiation_physical",
        "wind_speed_local",
        "ambient_temp_local",
        "solar_radiation_local",
        "wind_angle_deg",
        "correction_stage",
    }.issubset(corrected.columns)
    assert corrected.loc[0, "correction_stage"] == "terrain_corrected"
    assert corrected.loc[0, "wind_speed_local"] == pytest.approx(4.0)

    with pytest.raises(ValueError, match="已经修正"):
        WeatherCorrectionService().apply(corrected, terrain_lookup={}, options=CorrectionOptions())


def test_wind_direction_is_an_ieee_angle_not_a_wind_speed_multiplier():
    corrected = WeatherCorrectionService().apply(
        weather_frame(wind_direction=[0.0], wind_speed=[4.0]),
        terrain_lookup={},
        options=CorrectionOptions(
            enable_vertical=False,
            enable_terrain=False,
            enable_desert=False,
            enable_wind_direction=True,
            line_azimuth_deg=0.0,
        ),
    )

    assert corrected.loc[0, "wind_speed_local"] == pytest.approx(4.0)
    assert corrected.loc[0, "wind_angle_deg"] == pytest.approx(0.0)


def test_disabled_wind_direction_uses_crosswind_angle_without_changing_wind():
    corrected = WeatherCorrectionService().apply(
        weather_frame(wind_direction=[0.0], wind_speed=[4.0]),
        terrain_lookup={},
        options=CorrectionOptions(
            enable_vertical=False,
            enable_terrain=False,
            enable_desert=False,
            enable_wind_direction=False,
            line_azimuth_deg=0.0,
        ),
    )

    assert corrected.loc[0, "wind_speed_local"] == pytest.approx(4.0)
    assert corrected.loc[0, "wind_angle_deg"] == pytest.approx(90.0)


@pytest.mark.parametrize(
    ("source_index", "winds", "temps", "solar"),
    [
        ([0, 0], [4.0, 7.0], [20.0, 30.0], [600.0, 700.0]),
        ([1, 0, 1], [4.0, 7.0, 9.0], [20.0, 30.0, 40.0], [600.0, 700.0, 800.0]),
    ],
)
def test_duplicate_input_indexes_keep_each_rows_physical_values(
    source_index, winds, temps, solar
):
    source = pd.DataFrame(
        {
            "tower_id": [f"00{index + 1}" for index in range(len(source_index))],
            "position": list(range(36, 36 + len(source_index))),
            "ambient_temp": temps,
            "wind_speed": winds,
            "wind_direction": [0.0] * len(source_index),
            "solar_radiation": solar,
        },
        index=source_index,
    )
    source.attrs["source"] = {"role": "physical"}
    snapshot = copy.deepcopy(source)
    source_attrs = copy.deepcopy(source.attrs)

    corrected = WeatherCorrectionService().apply(
        source,
        terrain_lookup={},
        options=CorrectionOptions(
            enable_vertical=False,
            enable_terrain=False,
            enable_desert=False,
            enable_wind_direction=False,
        ),
    )

    pd.testing.assert_frame_equal(source, snapshot)
    assert source.attrs == source_attrs
    assert corrected.index.equals(source.index)
    np.testing.assert_allclose(corrected["wind_speed_local"].to_numpy(), winds)
    np.testing.assert_allclose(corrected["ambient_temp_local"].to_numpy(), temps)
    np.testing.assert_allclose(corrected["solar_radiation_local"].to_numpy(), solar)
    np.testing.assert_allclose(corrected["wind_angle_deg"].to_numpy(), 90.0)


def test_desert_correction_changes_only_solar_radiation():
    corrected = WeatherCorrectionService().apply(
        weather_frame(),
        terrain_lookup={},
        options=CorrectionOptions(
            enable_vertical=False,
            enable_terrain=False,
            enable_desert=True,
            enable_wind_direction=False,
            ground_temp_offset=15.0,
        ),
    )

    assert corrected.loc[0, "wind_speed_local"] == pytest.approx(4.0)
    assert corrected.loc[0, "ambient_temp_local"] == pytest.approx(20.0)
    assert corrected.loc[0, "solar_radiation_local"] > corrected.loc[0, "solar_radiation_physical"]


def test_terrain_lookup_matches_tower_id_position_or_legacy_index_per_row():
    source = weather_frame(
        tower_id=["001", "002", "003"],
        position=[36, 37, 38],
        wind_direction=[180.0, 180.0, 180.0],
        wind_speed=[4.0, 4.0, 4.0],
        ambient_temp=[20.0, 20.0, 20.0],
        solar_radiation=[600.0, 600.0, 600.0],
        humidity=[25.0, 25.0, 25.0],
    )
    corrected = WeatherCorrectionService().apply(
        source,
        terrain_lookup={
            "001": {"slope": 45.0, "aspect": 0.0},
            37: {"slope": 45.0, "aspect": 0.0},
            2: {"slope": 45.0, "aspect": 0.0},
        },
        options=CorrectionOptions(
            enable_vertical=False,
            enable_terrain=True,
            enable_desert=False,
            enable_wind_direction=False,
        ),
    )

    assert corrected["wind_speed_local"].tolist() == pytest.approx([5.2, 5.2, 5.2])


@pytest.mark.parametrize(
    ("column", "value"),
    [
        (column, value)
        for column in (
            "wind_speed",
            "ambient_temp",
            "solar_radiation",
            "wind_direction",
        )
        for value in (np.nan, np.inf)
    ],
)
def test_apply_rejects_nonfinite_physical_observations(column, value):
    source = weather_frame(**{column: [value]})

    with pytest.raises(ValueError, match=rf"{column}.*非法行"):
        WeatherCorrectionService().apply(source, terrain_lookup={}, options=CorrectionOptions())


@pytest.mark.parametrize(
    "column",
    ("wind_speed", "ambient_temp", "solar_radiation", "wind_direction"),
)
def test_apply_rejects_missing_required_physical_columns(column):
    source = weather_frame().drop(columns=column)

    with pytest.raises(ValueError, match=rf"缺少必需气象列.*{column}"):
        WeatherCorrectionService().apply(source, terrain_lookup={}, options=CorrectionOptions())


@pytest.mark.parametrize(
    ("column", "value"),
    [
        ("wind_speed", -0.1),
        ("wind_speed", 75.1),
        ("ambient_temp", -60.1),
        ("ambient_temp", 70.1),
        ("solar_radiation", -0.1),
    ],
)
def test_apply_rejects_out_of_range_physical_observations(column, value):
    source = weather_frame(**{column: [value]})

    with pytest.raises(ValueError, match=rf"{column}.*非法行"):
        WeatherCorrectionService().apply(source, terrain_lookup={}, options=CorrectionOptions())


def test_apply_keeps_zero_physical_observations_valid():
    corrected = WeatherCorrectionService().apply(
        weather_frame(
            ambient_temp=[0.0],
            wind_speed=[0.0],
            wind_direction=[0.0],
            solar_radiation=[0.0],
        ),
        terrain_lookup={},
        options=CorrectionOptions(
            enable_vertical=False,
            enable_terrain=False,
            enable_desert=False,
            enable_wind_direction=False,
        ),
    )

    assert corrected.loc[0, "wind_speed_local"] == pytest.approx(0.0)
    assert corrected.loc[0, "ambient_temp_local"] == pytest.approx(0.0)
    assert corrected.loc[0, "solar_radiation_local"] == pytest.approx(0.0)
    assert corrected.loc[0, "wind_angle_deg"] == pytest.approx(90.0)


def test_invalid_correction_options_keep_valid_observations_finite():
    corrected = WeatherCorrectionService().apply(
        weather_frame(
            ambient_temp=[20.0],
            wind_speed=[4.0],
            wind_direction=[0.0],
            solar_radiation=[600.0],
        ),
        terrain_lookup=None,
        options=CorrectionOptions(
            ref_height_m=0.0,
            line_height_m=np.inf,
            roughness_alpha=np.nan,
            temp_lapse_rate=np.inf,
            ground_albedo=np.inf,
            ground_temp_offset=np.inf,
        ),
    )

    assert np.isfinite(
        corrected.loc[
            0,
            [
                "wind_speed_physical",
                "ambient_temp_physical",
                "solar_radiation_physical",
                "wind_speed_local",
                "ambient_temp_local",
                "solar_radiation_local",
                "wind_angle_deg",
            ],
        ].to_numpy(dtype=float),
    ).all()


def test_legacy_matrix_adapter_is_pure_and_rejects_repeated_correction():
    line_data = {
        "positions": [36, 37],
        "times": np.array([0.0, 1.0]),
        "winds": np.array([[4.0, 5.0], [6.0, 7.0]]),
        "temps": np.array([[20.0, 21.0], [22.0, 23.0]]),
        "solar": np.array([600.0, 700.0]),
        "angles": np.array([[0.0, 20.0], [40.0, 60.0]]),
        "terrain_data": {0: {"slope": 0.0, "aspect": 0.0}},
    }
    snapshot = copy.deepcopy(line_data)
    adapter = matrix_adapter()

    corrected = adapter(
        line_data,
        {
            "vertical": False,
            "terrain": False,
            "desert": False,
            "wind_dir": False,
            "conductor_height": 20.0,
            "anemometer_height": 10.0,
            "roughness_alpha": 0.15,
            "desert_albedo": 0.35,
            "ground_temp_offset": 15.0,
        },
        {"line_azimuth": 0.0},
    )

    assert corrected is not line_data
    assert corrected["winds"] is not line_data["winds"]
    assert corrected["temps"] is not line_data["temps"]
    assert corrected["solar"] is not line_data["solar"]
    assert corrected["angles"] is not line_data["angles"]
    np.testing.assert_allclose(corrected["angles"], [[90.0, 90.0], [90.0, 90.0]])
    for key, value in line_data.items():
        if isinstance(value, np.ndarray):
            np.testing.assert_array_equal(value, snapshot[key])
        else:
            assert value == snapshot[key]

    with pytest.raises(ValueError, match="已经修正"):
        adapter(corrected, {}, {"line_azimuth": 0.0})


def test_legacy_matrix_adapter_propagates_invalid_physical_observations():
    line_data = {
        "positions": [36],
        "times": np.array([0.0]),
        "winds": np.array([[np.nan]]),
        "temps": np.array([[20.0]]),
        "solar": np.array([600.0]),
        "angles": np.array([[0.0]]),
    }

    with pytest.raises(ValueError, match=r"wind_speed.*非法行"):
        matrix_adapter()(line_data, {}, {"line_azimuth": 0.0})


def test_legacy_matrix_adapter_uses_complete_index_terrain_mapping_without_crossing_towers():
    line_data = {
        "positions": [1, 2],
        "times": np.array([0.0]),
        "winds": np.array([[4.0], [4.0]]),
        "temps": np.array([[20.0], [20.0]]),
        "solar": np.array([600.0]),
        "angles": np.array([[180.0], [180.0]]),
        "terrain_data": {
            0: {"slope": 0.0, "aspect": 0.0},
            1: {"slope": 45.0, "aspect": 0.0},
        },
    }

    corrected = matrix_adapter()(
        line_data,
        {"terrain": True},
        {"line_azimuth": 0.0},
    )

    np.testing.assert_allclose(corrected["winds"], [[4.0], [5.2]])


def test_legacy_matrix_adapter_allows_identical_complete_terrain_key_modes():
    line_data = {
        "positions": [0, 1],
        "times": np.array([0.0]),
        "winds": np.array([[4.0], [4.0]]),
        "temps": np.array([[20.0], [20.0]]),
        "solar": np.array([600.0]),
        "angles": np.array([[180.0], [180.0]]),
        "terrain_data": {
            0: {"slope": 0.0, "aspect": 0.0},
            1: {"slope": 45.0, "aspect": 0.0},
        },
    }

    corrected = matrix_adapter()(
        line_data,
        {"terrain": True},
        {"line_azimuth": 0.0},
    )

    np.testing.assert_allclose(corrected["winds"], [[4.0], [5.2]])


def test_legacy_matrix_adapter_uses_complete_canonical_terrain_mapping():
    line_data = {
        "positions": ["001", "002"],
        "times": np.array([0.0]),
        "winds": np.array([[4.0], [4.0]]),
        "temps": np.array([[20.0], [20.0]]),
        "solar": np.array([600.0]),
        "angles": np.array([[180.0], [180.0]]),
        "terrain_data": {
            "001": {"slope": 0.0, "aspect": 0.0},
            "002": {"slope": 45.0, "aspect": 0.0},
        },
    }

    corrected = matrix_adapter()(
        line_data,
        {"terrain": True},
        {"line_azimuth": 0.0},
    )

    np.testing.assert_allclose(corrected["winds"], [[4.0], [5.2]])


def test_legacy_matrix_adapter_rejects_ambiguous_complete_terrain_mapping():
    line_data = {
        "positions": [1, 2],
        "times": np.array([0.0]),
        "winds": np.array([[4.0], [4.0]]),
        "temps": np.array([[20.0], [20.0]]),
        "solar": np.array([600.0]),
        "angles": np.array([[180.0], [180.0]]),
        "terrain_data": {
            0: {"slope": 0.0, "aspect": 0.0},
            1: {"slope": 45.0, "aspect": 0.0},
            "2": {"slope": 0.0, "aspect": 0.0},
        },
    }

    with pytest.raises(ValueError, match="地形键歧义"):
        matrix_adapter()(line_data, {"terrain": True}, {"line_azimuth": 0.0})


def test_legacy_matrix_adapter_rejects_ambiguous_partial_terrain_mapping():
    line_data = {
        "positions": [1, 2],
        "times": np.array([0.0]),
        "winds": np.array([[4.0], [4.0]]),
        "temps": np.array([[20.0], [20.0]]),
        "solar": np.array([600.0]),
        "angles": np.array([[180.0], [180.0]]),
        "terrain_data": {1: {"slope": 45.0, "aspect": 0.0}},
    }

    with pytest.raises(ValueError, match="地形键歧义"):
        matrix_adapter()(line_data, {"terrain": True}, {"line_azimuth": 0.0})


def test_page_keeps_controls_but_delegates_all_weather_math_to_service():
    source = (Path(__file__).parents[2] / "dispatch_app_st.py").read_text(encoding="utf-8")
    correction_start = source.index("# 气象参数修正模块")
    correction_end = source.index("# 标准导线数据库", correction_start)
    correction_block = source[correction_start:correction_end]

    assert "WeatherCorrectionService" in correction_block
    assert "vertical_wind_correction" not in correction_block
    assert "terrain_wind_correction" not in correction_block
    assert "desert_radiation_correction" not in correction_block
    assert "wind_direction_correction" not in correction_block
    assert "np.sin" not in correction_block
    assert "np.exp" not in correction_block
    assert 'st.checkbox("垂直修正（风速高度折算）", value=True)' in source
    assert 'st.checkbox("地形修正（坡度/坡向）", value=True)' in source
    assert 'st.checkbox("沙漠环境修正（辐射增强）", value=True)' in source
    assert 'st.checkbox("风向修正（有效横风分量）", value=True)' in source

    thermal_start = source.index("st.session_state.analyzer.calculate_max_current_for_points(")
    thermal_end = source.index("line_data['max_currents']", thermal_start)
    thermal_call = source[thermal_start:thermal_end]
    assert "terrain_data=None" in thermal_call

    application_start = source.index("# 应用气象修正")
    application_end = source.index("progress_bar.progress(70)", application_start)
    application_block = source[application_start:application_end]
    assert application_block.index("line_data = apply_weather_corrections(") < application_block.index("if any(")
