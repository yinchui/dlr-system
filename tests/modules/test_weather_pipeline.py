import importlib

import numpy as np
import pandas as pd
import pytest


TIMEZONE = "Asia/Shanghai"


def weather_pipeline_module():
    return importlib.import_module("modules.weather_pipeline")


def timestamps(*values, timezone=TIMEZONE):
    return pd.DatetimeIndex(values).tz_localize(timezone)


def make_weather_frame(
    tower_ids,
    timestamp_values,
    ambient_temps,
    wind_speeds,
    wind_directions,
    *,
    role="physical",
    source_hash="physical-hash",
    timezone=TIMEZONE,
):
    size = len(tower_ids)
    return pd.DataFrame(
        {
            "tower_id": tower_ids,
            "timestamp": timestamps(*timestamp_values, timezone=timezone),
            "ambient_temp": ambient_temps,
            "wind_speed": wind_speeds,
            "wind_direction": wind_directions,
            "solar_radiation": np.linspace(100.0, 100.0 + size - 1, size),
            "humidity": np.linspace(30.0, 30.0 + size - 1, size),
            "elevation": np.linspace(1000.0, 1000.0 + size - 1, size),
            "dataset_role": [role] * size,
            "source_file_hash": [source_hash] * size,
        }
    )


def make_two_tower_uneven_weather():
    return make_weather_frame(
        ["002", "001", "002", "001"],
        [
            "2026-07-23 00:30",
            "2026-07-23 00:00",
            "2026-07-23 01:30",
            "2026-07-23 01:00",
        ],
        [50.0, 10.0, 60.0, 20.0],
        [12.0, 2.0, 14.0, 4.0],
        [180.0, 350.0, 200.0, 10.0],
    )


def make_direction_wrap_weather(start_direction, end_direction):
    return make_weather_frame(
        ["001", "001"],
        ["2026-07-23 00:00", "2026-07-23 01:00"],
        [20.0, 22.0],
        [2.0, 4.0],
        [start_direction, end_direction],
    )


def empty_weather_frame(*, role, timezone=TIMEZONE):
    frame = pd.DataFrame(
        {
            "tower_id": pd.Series(dtype="object"),
            "timestamp": pd.Series(
                dtype=pd.DatetimeTZDtype(tz=timezone)
            ),
            "ambient_temp": pd.Series(dtype="float64"),
            "wind_speed": pd.Series(dtype="float64"),
            "wind_direction": pd.Series(dtype="float64"),
            "solar_radiation": pd.Series(dtype="float64"),
            "humidity": pd.Series(dtype="float64"),
            "elevation": pd.Series(dtype="float64"),
            "dataset_role": pd.Series(dtype="object"),
            "source_file_hash": pd.Series(dtype="object"),
        }
    )
    frame.attrs["role"] = role
    return frame


def test_resampling_never_uses_another_tower_values():
    weather_pipeline = weather_pipeline_module()
    source = make_two_tower_uneven_weather()

    result = weather_pipeline.resample_weather_by_tower(
        source, interval_minutes=30
    )

    tower_a = result[result["tower_id"] == "001"]
    tower_b = result[result["tower_id"] == "002"]
    assert tower_a["ambient_temp"].max() < 30
    assert tower_b["ambient_temp"].min() > 40
    assert tower_a["timestamp"].tolist() == list(
        timestamps(
            "2026-07-23 00:00",
            "2026-07-23 00:30",
            "2026-07-23 01:00",
        )
    )


def test_wind_direction_uses_circular_interpolation():
    weather_pipeline = weather_pipeline_module()
    source = make_direction_wrap_weather(359.0, 1.0)

    result = weather_pipeline.resample_weather_by_tower(
        source, interval_minutes=30
    )

    middle = result.iloc[1]["wind_direction"]
    assert middle < 5 or middle > 355


def test_resampling_preserves_exact_end_when_span_is_not_divisible():
    weather_pipeline = weather_pipeline_module()
    source = make_weather_frame(
        ["001", "001"],
        ["2026-07-23 00:00", "2026-07-23 00:50"],
        [10.0, 25.0],
        [2.0, 7.0],
        [350.0, 10.0],
    )

    result = weather_pipeline.resample_weather_by_tower(
        source, interval_minutes=30
    )

    assert result["timestamp"].tolist() == list(
        timestamps(
            "2026-07-23 00:00",
            "2026-07-23 00:30",
            "2026-07-23 00:50",
        )
    )
    assert result.iloc[-1]["ambient_temp"] == pytest.approx(25.0)
    assert result.iloc[-1]["wind_speed"] == pytest.approx(7.0)
    assert result.iloc[-1]["wind_direction"] == pytest.approx(10.0)


def test_circular_interpolation_leaves_unbounded_endpoints_missing():
    weather_pipeline = weather_pipeline_module()
    series = pd.Series(
        [np.nan, 350.0, np.nan, 10.0, np.nan],
        index=pd.date_range(
            "2026-07-23", periods=5, freq="30min", tz=TIMEZONE
        ),
        name="wind_direction",
    )

    result = weather_pipeline.circular_interpolate(series)

    assert pd.isna(result.iloc[0])
    assert result.iloc[2] < 5 or result.iloc[2] > 355
    assert pd.isna(result.iloc[-1])
    assert result.index.equals(series.index)
    assert result.name == series.name


def test_resampling_preserves_fields_timezone_and_input():
    weather_pipeline = weather_pipeline_module()
    source = make_direction_wrap_weather(90.0, 120.0)
    source["longitude"] = [100.0, 102.0]
    source["latitude"] = [40.0, 42.0]
    source["measurement_height"] = [10.0, 20.0]
    source["station_label"] = ["station-a", "station-a"]
    source.attrs["origin"] = "canonical"
    original = source.copy(deep=True)
    original_attrs = source.attrs.copy()

    result = weather_pipeline.resample_weather_by_tower(
        source, interval_minutes=30
    )

    pd.testing.assert_frame_equal(source, original)
    assert source.attrs == original_attrs
    assert result.attrs == original_attrs
    assert isinstance(result["timestamp"].dtype, pd.DatetimeTZDtype)
    assert str(result["timestamp"].dt.tz) == TIMEZONE
    assert result.columns.tolist() == source.columns.tolist()
    assert result.loc[1, "longitude"] == pytest.approx(101.0)
    assert result.loc[1, "latitude"] == pytest.approx(41.0)
    assert result.loc[1, "measurement_height"] == pytest.approx(15.0)
    assert result.loc[1, "station_label"] == "station-a"
    assert result.loc[1, "source_file_hash"] == "physical-hash"


def test_resampling_keeps_single_point_tower_unchanged():
    weather_pipeline = weather_pipeline_module()
    source = make_weather_frame(
        ["001"],
        ["2026-07-23 00:15"],
        [20.0],
        [3.0],
        [45.0],
    )

    result = weather_pipeline.resample_weather_by_tower(source)

    pd.testing.assert_frame_equal(result, source)


def test_resampling_empty_frame_preserves_schema_timezone_and_attrs():
    weather_pipeline = weather_pipeline_module()
    source = empty_weather_frame(role="physical")

    result = weather_pipeline.resample_weather_by_tower(source)

    assert result.empty
    assert result.columns.tolist() == source.columns.tolist()
    assert result.dtypes.equals(source.dtypes)
    assert result.attrs == source.attrs
    assert result is not source


@pytest.mark.parametrize("interval_minutes", [0, -1])
def test_resampling_rejects_nonpositive_interval(interval_minutes):
    weather_pipeline = weather_pipeline_module()

    with pytest.raises(ValueError, match="interval_minutes.*正数"):
        weather_pipeline.resample_weather_by_tower(
            make_direction_wrap_weather(0.0, 90.0),
            interval_minutes=interval_minutes,
        )


def test_resampling_rejects_duplicate_tower_timestamp():
    weather_pipeline = weather_pipeline_module()
    source = make_direction_wrap_weather(0.0, 90.0)
    source = pd.concat([source, source.iloc[[0]]], ignore_index=True)

    with pytest.raises(ValueError, match="重复.*tower_id.*timestamp"):
        weather_pipeline.resample_weather_by_tower(source)


def test_resampling_uses_each_towers_own_start_and_end():
    weather_pipeline = weather_pipeline_module()
    source = make_two_tower_uneven_weather()

    result = weather_pipeline.resample_weather_by_tower(
        source, interval_minutes=30
    )

    ranges = result.groupby("tower_id")["timestamp"].agg(["min", "max"])
    assert ranges.loc["001", "min"] == timestamps("2026-07-23 00:00")[0]
    assert ranges.loc["001", "max"] == timestamps("2026-07-23 01:00")[0]
    assert ranges.loc["002", "min"] == timestamps("2026-07-23 00:30")[0]
    assert ranges.loc["002", "max"] == timestamps("2026-07-23 01:30")[0]
    assert result[["tower_id", "timestamp"]].values.tolist() == sorted(
        result[["tower_id", "timestamp"]].values.tolist(),
        key=lambda row: (row[0], row[1]),
    )


def test_resampling_requires_timezone_aware_canonical_timestamps():
    weather_pipeline = weather_pipeline_module()
    source = make_direction_wrap_weather(0.0, 90.0)
    source["timestamp"] = source["timestamp"].dt.tz_localize(None)

    with pytest.raises(ValueError, match="timestamp.*时区"):
        weather_pipeline.resample_weather_by_tower(source)


def test_resampling_requires_canonical_weather_columns():
    weather_pipeline = weather_pipeline_module()
    source = make_direction_wrap_weather(0.0, 90.0).drop(
        columns="ambient_temp"
    )

    with pytest.raises(ValueError, match="ambient_temp"):
        weather_pipeline.resample_weather_by_tower(source)


def test_resampling_accepts_canonical_physical_role():
    weather_pipeline = weather_pipeline_module()
    source = make_direction_wrap_weather(0.0, 90.0)

    result = weather_pipeline.resample_weather_by_tower(
        source, interval_minutes=30
    )

    assert result["dataset_role"].tolist() == [
        "physical",
        "physical",
        "physical",
    ]


def test_resampling_rejects_truth_before_it_can_interpolate_future_values():
    weather_pipeline = weather_pipeline_module()
    truth = make_weather_frame(
        ["001", "001"],
        ["2026-07-23 00:00", "2026-07-23 01:00"],
        [0.0, 60.0],
        [0.0, 60.0],
        [0.0, 0.0],
        role="truth",
        source_hash="truth-hash",
    )

    with pytest.raises(ValueError, match="真实值.*backward alignment"):
        weather_pipeline.resample_weather_by_tower(
            truth, interval_minutes=30
        )


def test_resampling_rejects_mixed_dataset_roles():
    weather_pipeline = weather_pipeline_module()
    source = make_direction_wrap_weather(0.0, 90.0)
    source["dataset_role"] = ["physical", "truth"]

    with pytest.raises(ValueError, match="真实值.*backward alignment"):
        weather_pipeline.resample_weather_by_tower(source)


@pytest.mark.parametrize("invalid_role", ["", "forecast", None])
def test_resampling_rejects_empty_or_unknown_dataset_role(invalid_role):
    weather_pipeline = weather_pipeline_module()
    source = make_direction_wrap_weather(0.0, 90.0)
    source["dataset_role"] = invalid_role

    with pytest.raises(ValueError, match="真实值.*backward alignment"):
        weather_pipeline.resample_weather_by_tower(source)


def test_resampling_requires_dataset_role_column():
    weather_pipeline = weather_pipeline_module()
    source = make_direction_wrap_weather(0.0, 90.0).drop(
        columns="dataset_role"
    )

    with pytest.raises(ValueError, match="dataset_role"):
        weather_pipeline.resample_weather_by_tower(source)


def test_truth_alignment_uses_same_tower_and_no_future_sample():
    weather_pipeline = weather_pipeline_module()
    physical = make_weather_frame(
        ["001"],
        ["2026-07-23 00:10"],
        [20.0],
        [2.0],
        [90.0],
    )
    truth = make_weather_frame(
        ["001", "001", "002"],
        [
            "2026-07-23 00:08",
            "2026-07-23 00:11",
            "2026-07-23 00:09",
        ],
        [19.0, 21.0, 55.0],
        [1.8, 2.2, 20.0],
        [88.0, 92.0, 180.0],
        role="truth",
        source_hash="truth-hash",
    )

    aligned, report = weather_pipeline.align_physical_and_truth(
        physical, truth, tolerance=pd.Timedelta("10min")
    )

    assert aligned.loc[0, "truth_timestamp"] == timestamps(
        "2026-07-23 00:08"
    )[0]
    assert aligned.loc[0, "tower_id"] == "001"
    assert aligned.loc[0, "ambient_temp_truth"] == pytest.approx(19.0)
    assert aligned.loc[0, "ambient_temp_physical"] == pytest.approx(20.0)
    assert aligned.loc[0, "truth_timestamp"] <= aligned.loc[0, "timestamp"]
    assert report.matched_rows == 1


def test_truth_alignment_does_not_match_only_future_truth():
    weather_pipeline = weather_pipeline_module()
    physical = make_weather_frame(
        ["001"], ["2026-07-23 00:10"], [20.0], [2.0], [90.0]
    )
    truth = make_weather_frame(
        ["001"],
        ["2026-07-23 00:11"],
        [19.0],
        [1.8],
        [88.0],
        role="truth",
        source_hash="truth-hash",
    )

    aligned, report = weather_pipeline.align_physical_and_truth(
        physical, truth, tolerance=pd.Timedelta("10min")
    )

    assert pd.isna(aligned.loc[0, "truth_timestamp"])
    assert pd.isna(aligned.loc[0, "wind_speed_truth"])
    assert report.matched_rows == 0
    assert report.unmatched_rows == 1


def test_truth_alignment_never_uses_another_tower():
    weather_pipeline = weather_pipeline_module()
    physical = make_weather_frame(
        ["001"], ["2026-07-23 00:10"], [20.0], [2.0], [90.0]
    )
    truth = make_weather_frame(
        ["002"],
        ["2026-07-23 00:10"],
        [50.0],
        [15.0],
        [180.0],
        role="truth",
        source_hash="truth-hash",
    )

    aligned, report = weather_pipeline.align_physical_and_truth(
        physical, truth, tolerance=pd.Timedelta("10min")
    )

    assert pd.isna(aligned.loc[0, "truth_timestamp"])
    assert report.matched_rows == 0


def test_truth_alignment_honors_tolerance_and_exact_match():
    weather_pipeline = weather_pipeline_module()
    physical = make_weather_frame(
        ["001", "001"],
        ["2026-07-23 00:10", "2026-07-23 00:20"],
        [20.0, 21.0],
        [2.0, 3.0],
        [90.0, 100.0],
    )
    truth = make_weather_frame(
        ["001", "001"],
        ["2026-07-23 00:00", "2026-07-23 00:20"],
        [19.0, 20.0],
        [1.8, 2.8],
        [88.0, 98.0],
        role="truth",
        source_hash="truth-hash",
    )

    aligned, report = weather_pipeline.align_physical_and_truth(
        physical, truth, tolerance=pd.Timedelta("5min")
    )

    assert pd.isna(aligned.loc[0, "truth_timestamp"])
    assert aligned.loc[1, "truth_timestamp"] == aligned.loc[1, "timestamp"]
    assert report.matched_rows == 1
    assert report.unmatched_rows == 1
    assert report.coverage == pytest.approx(0.5)


def test_same_input_hash_is_rejected_for_training_even_off_first_row():
    weather_pipeline = weather_pipeline_module()
    physical = make_weather_frame(
        ["001", "001"],
        ["2026-07-23 00:00", "2026-07-23 00:30"],
        [20.0, 21.0],
        [2.0, 3.0],
        [90.0, 100.0],
    )
    physical["source_file_hash"] = ["physical-only", "shared-hash"]
    truth = make_weather_frame(
        ["001"],
        ["2026-07-23 00:00"],
        [19.0],
        [1.8],
        [88.0],
        role="truth",
        source_hash="shared-hash",
    )

    with pytest.raises(ValueError, match="不能同时作为"):
        weather_pipeline.align_physical_and_truth(
            physical, truth, tolerance=pd.Timedelta("10min")
        )


def test_truth_alignment_handles_empty_physical_with_zero_coverage():
    weather_pipeline = weather_pipeline_module()
    physical = empty_weather_frame(role="physical")
    truth = make_weather_frame(
        ["001"],
        ["2026-07-23 00:00"],
        [19.0],
        [1.8],
        [88.0],
        role="truth",
        source_hash="truth-hash",
    )

    aligned, report = weather_pipeline.align_physical_and_truth(
        physical, truth, tolerance=pd.Timedelta("10min")
    )

    assert aligned.empty
    assert {"tower_id", "timestamp", "truth_timestamp"}.issubset(
        aligned.columns
    )
    assert isinstance(aligned["truth_timestamp"].dtype, pd.DatetimeTZDtype)
    assert report == weather_pipeline.AlignmentReport(0, 1, 0, 0, 0.0)


def test_truth_alignment_handles_empty_truth_and_counts_original_rows():
    weather_pipeline = weather_pipeline_module()
    physical = make_weather_frame(
        ["001", "002"],
        ["2026-07-23 00:00", "2026-07-23 00:00"],
        [20.0, 30.0],
        [2.0, 3.0],
        [90.0, 100.0],
    )
    truth = empty_weather_frame(role="truth")

    aligned, report = weather_pipeline.align_physical_and_truth(
        physical, truth, tolerance=pd.Timedelta("10min")
    )

    assert len(aligned) == 2
    assert aligned["truth_timestamp"].isna().all()
    assert aligned["ambient_temp_truth"].isna().all()
    assert report == weather_pipeline.AlignmentReport(2, 0, 0, 2, 0.0)


def test_truth_alignment_preserves_physical_order_and_both_inputs():
    weather_pipeline = weather_pipeline_module()
    physical = make_weather_frame(
        ["002", "001", "002"],
        [
            "2026-07-23 00:10",
            "2026-07-23 00:05",
            "2026-07-23 00:20",
        ],
        [30.0, 20.0, 31.0],
        [3.0, 2.0, 4.0],
        [100.0, 90.0, 110.0],
    )
    truth = make_weather_frame(
        ["001", "002"],
        ["2026-07-23 00:00", "2026-07-23 00:00"],
        [19.0, 29.0],
        [1.8, 2.8],
        [88.0, 98.0],
        role="truth",
        source_hash="truth-hash",
    )
    physical_original = physical.copy(deep=True)
    truth_original = truth.copy(deep=True)

    aligned, report = weather_pipeline.align_physical_and_truth(
        physical, truth, tolerance=pd.Timedelta("30min")
    )

    pd.testing.assert_frame_equal(physical, physical_original)
    pd.testing.assert_frame_equal(truth, truth_original)
    assert aligned["tower_id"].tolist() == ["002", "001", "002"]
    assert report == weather_pipeline.AlignmentReport(3, 2, 3, 0, 1.0)


def test_truth_alignment_converts_truth_timezone_without_dropping_timezone():
    weather_pipeline = weather_pipeline_module()
    physical = make_weather_frame(
        ["001"], ["2026-07-23 08:00"], [20.0], [2.0], [90.0]
    )
    truth = make_weather_frame(
        ["001"],
        ["2026-07-23 00:00"],
        [19.0],
        [1.8],
        [88.0],
        role="truth",
        source_hash="truth-hash",
        timezone="UTC",
    )

    aligned, report = weather_pipeline.align_physical_and_truth(
        physical, truth, tolerance=pd.Timedelta(0)
    )

    assert report.matched_rows == 1
    assert aligned.loc[0, "truth_timestamp"] == aligned.loc[0, "timestamp"]
    assert str(aligned["truth_timestamp"].dt.tz) == TIMEZONE


@pytest.mark.parametrize("tolerance", [pd.Timedelta("-1min"), pd.NaT])
def test_truth_alignment_rejects_invalid_tolerance(tolerance):
    weather_pipeline = weather_pipeline_module()

    with pytest.raises(ValueError, match="tolerance.*非负"):
        weather_pipeline.align_physical_and_truth(
            empty_weather_frame(role="physical"),
            empty_weather_frame(role="truth"),
            tolerance=tolerance,
        )


def test_truth_alignment_rejects_naive_timestamps():
    weather_pipeline = weather_pipeline_module()
    physical = make_weather_frame(
        ["001"], ["2026-07-23 00:00"], [20.0], [2.0], [90.0]
    )
    truth = make_weather_frame(
        ["001"],
        ["2026-07-23 00:00"],
        [19.0],
        [1.8],
        [88.0],
        role="truth",
        source_hash="truth-hash",
    )
    truth["timestamp"] = truth["timestamp"].dt.tz_localize(None)

    with pytest.raises(ValueError, match="timestamp.*时区"):
        weather_pipeline.align_physical_and_truth(
            physical, truth, tolerance=pd.Timedelta("10min")
        )


def test_truth_values_are_normalized_to_physical_measurement_height():
    weather_pipeline = weather_pipeline_module()
    physical = make_weather_frame(
        ["001"], ["2026-07-23 00:00"], [25.0], [5.0], [90.0]
    )
    truth = make_weather_frame(
        ["001"],
        ["2026-07-23 00:00"],
        [20.0],
        [4.0],
        [88.0],
        role="truth",
        source_hash="truth-hash",
    )
    physical["measurement_height"] = 20.0
    truth["measurement_height"] = 10.0

    aligned, report = weather_pipeline.align_physical_and_truth(
        physical,
        truth,
        tolerance=pd.Timedelta(0),
        roughness_alpha=0.2,
        temp_lapse_rate=0.01,
    )

    assert report.matched_rows == 1
    assert aligned.loc[0, "wind_speed_physical"] == pytest.approx(5.0)
    assert aligned.loc[0, "ambient_temp_physical"] == pytest.approx(25.0)
    assert aligned.loc[0, "wind_speed_truth_raw"] == pytest.approx(4.0)
    assert aligned.loc[0, "ambient_temp_truth_raw"] == pytest.approx(20.0)
    assert aligned.loc[0, "wind_speed_truth"] == pytest.approx(
        4.0 * (20.0 / 10.0) ** 0.2
    )
    assert aligned.loc[0, "ambient_temp_truth"] == pytest.approx(19.9)
    assert aligned.loc[0, "measurement_height_truth_original"] == 10.0
    assert aligned.loc[0, "measurement_height_common"] == 20.0
    assert bool(aligned.loc[0, "height_normalized"])


@pytest.mark.parametrize(
    ("physical_height", "truth_height"),
    [(-1.0, 10.0), (20.0, 0.0), (np.inf, 10.0)],
)
def test_invalid_measurement_height_keeps_truth_values_finite_and_raw(
    physical_height, truth_height
):
    weather_pipeline = weather_pipeline_module()
    physical = make_weather_frame(
        ["001"], ["2026-07-23 00:00"], [25.0], [5.0], [90.0]
    )
    truth = make_weather_frame(
        ["001"],
        ["2026-07-23 00:00"],
        [20.0],
        [4.0],
        [88.0],
        role="truth",
        source_hash="truth-hash",
    )
    physical["measurement_height"] = physical_height
    truth["measurement_height"] = truth_height

    aligned, _ = weather_pipeline.align_physical_and_truth(
        physical, truth, tolerance=pd.Timedelta(0)
    )

    assert aligned.loc[0, "wind_speed_truth"] == pytest.approx(4.0)
    assert aligned.loc[0, "ambient_temp_truth"] == pytest.approx(20.0)
    assert np.isfinite(aligned.loc[0, "wind_speed_truth"])
    assert not bool(aligned.loc[0, "height_normalized"])
    assert pd.isna(aligned.loc[0, "measurement_height_common"])


def test_missing_measurement_height_is_treated_as_already_aligned():
    weather_pipeline = weather_pipeline_module()
    physical = make_weather_frame(
        ["001"], ["2026-07-23 00:00"], [25.0], [5.0], [90.0]
    )
    truth = make_weather_frame(
        ["001"],
        ["2026-07-23 00:00"],
        [20.0],
        [4.0],
        [88.0],
        role="truth",
        source_hash="truth-hash",
    )

    aligned, _ = weather_pipeline.align_physical_and_truth(
        physical, truth, tolerance=pd.Timedelta(0)
    )

    assert aligned.loc[0, "wind_speed_truth"] == pytest.approx(4.0)
    assert aligned.loc[0, "ambient_temp_truth"] == pytest.approx(20.0)
    assert pd.isna(aligned.loc[0, "measurement_height_truth_original"])
    assert pd.isna(aligned.loc[0, "measurement_height_common"])
    assert not bool(aligned.loc[0, "height_normalized"])
