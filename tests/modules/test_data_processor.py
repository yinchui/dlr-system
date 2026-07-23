import pandas as pd
import pandas.testing as pdt
import pytest

import modules.data_processor as data_processor
from modules.terrain import build_terrain_lookup
from tests.fixtures.sample_data import (
    make_tower_time_weather_dataframe,
    make_weather_dataframe,
)


build_weather_dataset = data_processor.build_weather_dataset
interpolate_analysis_dataset = data_processor.interpolate_analysis_dataset
normalize_weather_input_dataframe = data_processor.normalize_weather_input_dataframe


def test_build_weather_dataset_normalizes_columns_and_timestamps():
    dataset = build_weather_dataset(make_weather_dataframe())
    assert dataset.positions == [36, 372]
    assert len(dataset.timestamps) == 2
    assert dataset.wind_speeds[36][0] == 3.2


def test_normalize_weather_input_dataframe_supports_tower_time_format():
    raw = make_new_weather_dataframe()

    normalized = normalize_weather_input_dataframe(raw)

    assert normalized["position"].tolist() == [14, 15]
    assert normalized["date"].tolist() == ["2026-07-23", "2026-07-23"]
    assert normalized["time_str"].tolist() == ["00:00", "00:30"]
    assert normalized["wind_speed"].tolist() == [1.6, 1.55]
    assert normalized["ambient_temp"].tolist() == [16.96, 16.76]
    assert normalized["solar_radiation"].tolist() == [0, 0]
    assert normalized["elevation"].tolist() == [1000, 1000]


def test_canonical_weather_preserves_tower_id_timezone_and_source_role():
    raw = pd.DataFrame(
        {
            "时间": ["2026-07-23 00:00"],
            "杆塔": ["001号"],
            "风速WS(m/s)": [2.0],
            "风向WD(°)": [359.0],
            "温度TEM(℃)": [20.0],
        }
    )

    result = data_processor.canonicalize_weather_frame(
        raw,
        role="physical",
        timezone="Asia/Shanghai",
        source_hash="abc",
    )

    assert result.frame.loc[0, "tower_id"] == "001"
    assert str(result.frame["timestamp"].dt.tz) == "Asia/Shanghai"
    assert result.frame.loc[0, "dataset_role"] == "physical"
    assert result.frame.loc[0, "source_file_hash"] == "abc"
    assert result.report.input_rows == 1
    assert result.report.valid_rows == 1


def test_canonical_weather_drops_only_invalid_rows_and_reports_them():
    raw = make_weather_dataframe()
    raw.loc[0, "风速"] = -1

    result = data_processor.canonicalize_weather_frame(raw, role="truth")

    assert len(result.frame) == len(raw) - 1
    assert result.report.dropped_rows == 1
    assert result.report.reasons["wind_speed_out_of_range"] == 1


def test_canonical_weather_reports_invalid_keys_required_values_and_duplicate():
    raw = pd.DataFrame(
        {
            "位置": ["001号", "塔A", "003号", "004号", "001号"],
            "日期": ["2026-07-23", "2026-07-23", "bad", "2026-07-23", "2026-07-23"],
            "时刻": ["00:00", "00:15", "00:30", "00:45", "00:00"],
            "环境温度": [20.0, 21.0, 22.0, None, 23.0],
            "风速": [2.0, 2.1, 2.2, 2.3, 2.4],
            "风向": [10.0, 20.0, 30.0, 40.0, 50.0],
        }
    )

    result = data_processor.canonicalize_weather_frame(raw, role="physical")

    assert result.frame[["tower_id", "ambient_temp"]].to_dict("records") == [
        {"tower_id": "001", "ambient_temp": 20.0}
    ]
    assert result.report.input_rows == 5
    assert result.report.valid_rows == 1
    assert result.report.dropped_rows == 4
    assert result.report.duplicate_rows == 1
    assert result.report.reasons == {
        "invalid_tower_id": 1,
        "invalid_timestamp": 1,
        "missing_ambient_temp": 1,
        "duplicate_tower_timestamp": 1,
    }


@pytest.mark.parametrize(
    ("column", "value", "reason"),
    [
        ("风速", 75.1, "wind_speed_out_of_range"),
        ("环境温度", 70.1, "ambient_temp_out_of_range"),
        ("风向", float("inf"), "wind_direction_out_of_range"),
        ("风向", -0.1, "wind_direction_out_of_range"),
        ("相对湿度", 100.1, "humidity_out_of_range"),
        ("太阳辐射强度", -0.1, "solar_radiation_out_of_range"),
        ("经度", float("inf"), "longitude_not_finite"),
        ("纬度", float("nan"), "latitude_not_finite"),
    ],
)
def test_canonical_weather_rejects_invalid_numeric_ranges(column, value, reason):
    raw = make_weather_dataframe().iloc[[0]].copy()
    raw[column] = value

    result = data_processor.canonicalize_weather_frame(raw, role="truth")

    assert result.frame.empty
    assert result.report.reasons[reason] == 1


def test_canonical_weather_normalizes_360_direction_and_safe_defaults():
    raw = pd.DataFrame(
        {
            "时间": ["2026-07-23T00:00:00+00:00"],
            "杆塔": ["0007号"],
            "风速WS(m/s)": [0.0],
            "风向WD(°)": [360.0],
            "温度TEM(℃)": [-60.0],
        }
    )

    result = data_processor.canonicalize_weather_frame(raw, role="physical")

    row = result.frame.iloc[0]
    assert row["tower_id"] == "0007"
    assert row["timestamp"].isoformat() == "2026-07-23T08:00:00+08:00"
    assert row["wind_direction"] == 0.0
    assert row["solar_radiation"] == 0.0
    assert row["humidity"] == 50.0
    assert row["elevation"] == 1000.0


def test_canonical_weather_does_not_modify_input_dataframe():
    raw = make_tower_time_weather_dataframe()
    before = raw.copy(deep=True)

    data_processor.canonicalize_weather_frame(raw, role="truth")

    pdt.assert_frame_equal(raw, before)


def test_canonical_weather_rejects_invalid_role():
    with pytest.raises(ValueError, match="physical.*truth"):
        data_processor.canonicalize_weather_frame(
            make_weather_dataframe(),
            role="training",
        )


def test_normalize_tower_id_requires_a_parseable_number():
    assert data_processor.normalize_tower_id("001号") == "001"
    assert data_processor.normalize_tower_id("塔001号") == "001"
    assert data_processor.normalize_tower_id(36) == "36"
    with pytest.raises(ValueError, match="无法解析杆塔编号"):
        data_processor.normalize_tower_id("塔A")


@pytest.mark.parametrize("value", [36.0, "36.0"])
def test_normalize_tower_id_normalizes_integer_decimal_values(value):
    assert data_processor.normalize_tower_id(value) == "36"


@pytest.mark.parametrize(
    "value",
    [36.5, "36.5", float("nan"), float("inf"), "-inf"],
)
def test_normalize_tower_id_rejects_non_integer_or_non_finite_values(value):
    with pytest.raises(ValueError, match="无法解析杆塔编号"):
        data_processor.normalize_tower_id(value)


def test_canonical_weather_rejects_non_integer_legacy_tower_without_tail_match():
    raw = pd.DataFrame(
        {
            "位置": [36.0, 36.5],
            "日期": ["2026-07-23", "2026-07-23"],
            "时刻": ["00:00", "00:30"],
            "环境温度": [20.0, 20.0],
            "风速": [2.0, 2.0],
            "风向": [90.0, 90.0],
        }
    )

    result = data_processor.canonicalize_weather_frame(raw, role="physical")

    assert result.frame["tower_id"].tolist() == ["36"]
    assert result.report.reasons["invalid_tower_id"] == 1


@pytest.mark.parametrize(
    ("invalid_time", "valid_time"),
    [
        ("2026-03-08 02:30", "2026-03-08 03:30"),
        ("2026-11-01 01:30", "2026-11-01 03:30"),
    ],
)
def test_canonical_weather_drops_dst_invalid_time_but_keeps_valid_row(
    invalid_time,
    valid_time,
):
    raw = pd.DataFrame(
        {
            "时间": [invalid_time, valid_time],
            "杆塔": ["001号", "002号"],
            "风速WS(m/s)": [2.0, 2.0],
            "风向WD(°)": [90.0, 90.0],
            "温度TEM(℃)": [20.0, 20.0],
        }
    )

    result = data_processor.canonicalize_weather_frame(
        raw,
        role="physical",
        timezone="America/New_York",
    )

    assert result.frame["tower_id"].tolist() == ["002"]
    assert result.report.dropped_rows == 1
    assert result.report.reasons["invalid_timestamp"] == 1


def test_canonical_weather_parses_mixed_timezone_offsets_row_by_row():
    raw = pd.DataFrame(
        {
            "时间": [
                "2026-07-23 00:00:00+08:00",
                "2026-07-23 00:00:00+09:00",
            ],
            "杆塔": ["001号", "002号"],
            "风速WS(m/s)": [2.0, 2.0],
            "风向WD(°)": [90.0, 90.0],
            "温度TEM(℃)": [20.0, 20.0],
        }
    )

    result = data_processor.canonicalize_weather_frame(
        raw,
        role="physical",
        timezone="Asia/Shanghai",
    )

    assert result.frame["timestamp"].map(lambda value: value.isoformat()).tolist() == [
        "2026-07-23T00:00:00+08:00",
        "2026-07-22T23:00:00+08:00",
    ]


def test_canonical_weather_drops_new_format_row_when_direction_is_missing():
    raw = pd.DataFrame(
        {
            "时间": ["2026-07-23 00:00"],
            "杆塔": ["001号"],
            "风速WS(m/s)": [2.0],
            "温度TEM(℃)": [20.0],
        }
    )

    result = data_processor.canonicalize_weather_frame(raw, role="physical")

    assert result.frame.empty
    assert result.report.dropped_rows == 1
    assert result.report.reasons["missing_wind_direction"] == 1


def test_interpolate_analysis_dataset_returns_expected_shapes():
    dataset = build_weather_dataset(make_weather_dataframe())
    analysis = interpolate_analysis_dataset(dataset, interval_minutes=30)
    assert analysis["temps"].shape == (2, 3)
    assert analysis["winds"].shape == (2, 3)


def test_build_terrain_lookup_returns_defaults_when_missing():
    terrain = build_terrain_lookup(None, {}, [36, 372])
    assert terrain[0]["slope"] == 0
    assert terrain[1]["elevation"] == 1000


def make_new_weather_dataframe():
    return pd.DataFrame(
        {
            "时间": ["2026-07-23 00:00", "2026-07-23 00:30"],
            "杆塔": ["14号", "15号"],
            "经度": [120.6982, 120.6982],
            "纬度": [49.2871, 49.2871],
            "风速WS(m/s)": [1.6, 1.55],
            "风向WD(°)": [172, 175],
            "阵风GUST(m/s)": [3.78, 3.7],
            "温度TEM(℃)": [16.96, 16.76],
            "相对湿度RHU(%)": [66.1, 67.1],
            "1h降水PRE(mm)": [0, 0],
        }
    )
