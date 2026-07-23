import pandas as pd

from modules.data_processor import (
    build_weather_dataset,
    interpolate_analysis_dataset,
    normalize_weather_input_dataframe,
)
from modules.terrain import build_terrain_lookup
from tests.fixtures.sample_data import make_weather_dataframe


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
