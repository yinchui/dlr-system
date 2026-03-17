from modules.data_processor import build_weather_dataset, interpolate_analysis_dataset
from modules.terrain import build_terrain_lookup
from tests.fixtures.sample_data import make_weather_dataframe


def test_build_weather_dataset_normalizes_columns_and_timestamps():
    dataset = build_weather_dataset(make_weather_dataframe())
    assert dataset.positions == [36, 372]
    assert len(dataset.timestamps) == 2
    assert dataset.wind_speeds[36][0] == 3.2


def test_interpolate_analysis_dataset_returns_expected_shapes():
    dataset = build_weather_dataset(make_weather_dataframe())
    analysis = interpolate_analysis_dataset(dataset, interval_minutes=30)
    assert analysis["temps"].shape == (2, 3)
    assert analysis["winds"].shape == (2, 3)


def test_build_terrain_lookup_returns_defaults_when_missing():
    terrain = build_terrain_lookup(None, {}, [36, 372])
    assert terrain[0]["slope"] == 0
    assert terrain[1]["elevation"] == 1000
