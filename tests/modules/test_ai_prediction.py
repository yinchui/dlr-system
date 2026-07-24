import numpy as np
import pandas as pd
import pytest

import modules.ai_prediction as ai_prediction
from config.config import PROJECT_TIMEZONE
from modules.ai_prediction import FeatureBuilder, ModelBundle, ResidualPredictor


class OffsetModel:
    def predict(self, features):
        return [0.5] * len(features)


class SequenceModel:
    def __init__(self, values):
        self.values = values

    def predict(self, features):
        return self.values[: len(features)]


class ExplodingModel:
    def predict(self, features):
        raise OSError("model unavailable")


class RecordingLagModel:
    def predict(self, features):
        self.lag_values = features["lag_1"].tolist()
        return np.zeros(len(features))


def make_interleaved_two_tower_training_frame():
    return pd.DataFrame(
        {
            "line_id": ["line-a"] * 4,
            "tower_id": ["001", "002", "001", "002"],
            "timestamp": pd.to_datetime(
                [
                    "2025-12-10 00:00",
                    "2025-12-10 00:00",
                    "2025-12-10 00:30",
                    "2025-12-10 00:30",
                ]
            ),
            "source_file_hash": ["dataset-a"] * 4,
            "wind_speed_local": [1.0, 10.0, 2.0, 20.0],
        },
        index=[8, 3, 7, 2],
    )


def test_features_never_lag_across_towers():
    frame = make_interleaved_two_tower_training_frame()
    features = FeatureBuilder().transform(
        frame, physical_col="wind_speed_local"
    )

    first_rows = features.groupby("tower_id", sort=False).head(1)

    assert np.allclose(
        first_rows["lag_1"], first_rows["wind_speed_local"]
    )
    assert features.loc[7, "lag_1"] == 1.0
    assert features.loc[2, "lag_1"] == 10.0


def test_features_restore_input_order_and_do_not_mutate_input():
    frame = make_interleaved_two_tower_training_frame()
    original = frame.copy(deep=True)

    features = FeatureBuilder().transform(
        frame, physical_col="wind_speed_local"
    )

    pd.testing.assert_frame_equal(frame, original)
    assert features.index.tolist() == frame.index.tolist()
    assert features["tower_id"].tolist() == frame["tower_id"].tolist()
    assert all(isinstance(value, str) for value in features["tower_id"])


def test_lag_resets_across_dataset_and_irregular_time_gap():
    frame = pd.DataFrame(
        {
            "line_id": ["line-a"] * 5,
            "tower_id": ["001"] * 5,
            "timestamp": pd.to_datetime(
                [
                    "2025-01-01 00:00",
                    "2025-01-01 00:30",
                    "2025-01-01 04:00",
                    "2025-01-01 05:00",
                    "2025-01-01 05:30",
                ]
            ),
            "source_file_hash": ["a", "a", "a", "b", "b"],
            "wind_speed_local": [1.0, 2.0, 9.0, 20.0, 21.0],
        }
    )

    features = FeatureBuilder().transform(
        frame, physical_col="wind_speed_local"
    )

    assert features["lag_1"].tolist() == [1.0, 1.0, 9.0, 20.0, 20.0]


def test_lag_resets_when_truth_source_hash_changes():
    frame = pd.DataFrame(
        {
            "tower_id": ["001"] * 3,
            "timestamp": pd.to_datetime(
                [
                    "2025-01-01 00:00",
                    "2025-01-01 00:30",
                    "2025-01-01 01:00",
                ]
            ),
            "source_file_hash": ["physical-a"] * 3,
            "source_file_hash_truth": ["truth-a", "truth-a", "truth-b"],
            "wind_speed_local": [1.0, 2.0, 3.0],
        }
    )

    features = FeatureBuilder().transform(
        frame, physical_col="wind_speed_local"
    )

    assert features["lag_1"].tolist() == [1.0, 1.0, 3.0]


def test_explicit_cadence_resets_every_gap_without_guessing_from_group():
    frame = pd.DataFrame(
        {
            "tower_id": ["001"] * 4,
            "timestamp": pd.to_datetime(
                [
                    "2025-01-01 00:00",
                    "2025-01-01 01:00",
                    "2025-01-01 03:00",
                    "2025-01-01 05:00",
                ]
            ),
            "wind_speed_local": [1.0, 2.0, 3.0, 4.0],
        }
    )

    features = FeatureBuilder(cadence_minutes=60).transform(
        frame, physical_col="wind_speed_local"
    )

    assert features["lag_1"].tolist() == [1.0, 1.0, 3.0, 4.0]


def test_predictor_uses_cadence_persisted_in_model_bundle():
    frame = pd.DataFrame(
        {
            "tower_id": ["001", "001"],
            "timestamp": pd.to_datetime(
                ["2025-01-01 00:00", "2025-01-01 01:00"]
            ),
            "wind_speed_local": [3.0, 4.0],
        }
    )
    model = RecordingLagModel()
    bundle = ModelBundle(
        target_name="wind_speed",
        feature_columns=["lag_1"],
        model=model,
        cadence_minutes=60,
    )

    ResidualPredictor({"wind_speed": bundle}).predict(
        frame, target_name="wind_speed", physical_col="wind_speed_local"
    )

    assert model.lag_values == [3.0, 3.0]


def test_model_bundle_keeps_residual_bounds_positional_compatibility():
    bundle = ModelBundle(
        "wind_speed",
        ["wind_speed_local"],
        OffsetModel(),
        None,
        (-5.0, 5.0),
    )

    assert bundle.residual_bounds == (-5.0, 5.0)


def test_feature_columns_are_deterministic_and_cycles_are_finite():
    frame = make_interleaved_two_tower_training_frame()
    frame["wind_direction"] = [0.0, 90.0, 180.0, 270.0]
    frame["solar_radiation_local"] = 600.0
    frame["humidity"] = 20.0
    frame["elevation"] = 1200.0
    frame["slope"] = 5.0
    frame["aspect"] = 180.0
    builder = FeatureBuilder()

    first = builder.transform(frame, physical_col="wind_speed_local")
    second = builder.transform(frame, physical_col="wind_speed_local")

    assert builder.feature_columns("wind_speed_local") == [
        "wind_speed_local",
        "lag_1",
        "hour_sin",
        "hour_cos",
        "day_of_year_sin",
        "day_of_year_cos",
        "wind_direction_sin",
        "wind_direction_cos",
        "solar_radiation_feature",
        "humidity_feature",
        "elevation_feature",
        "slope_feature",
        "aspect_feature",
    ]
    pd.testing.assert_frame_equal(
        first[builder.feature_columns("wind_speed_local")],
        second[builder.feature_columns("wind_speed_local")],
    )
    assert np.isfinite(
        first[builder.feature_columns("wind_speed_local")].to_numpy()
    ).all()


def test_naive_feature_timestamps_are_localized_to_project_timezone():
    frame = pd.DataFrame(
        {
            "tower_id": ["001"],
            "timestamp": [pd.Timestamp("2025-01-01 08:00")],
            "wind_speed_local": [3.0],
        }
    )

    features = FeatureBuilder().transform(
        frame, physical_col="wind_speed_local"
    )

    assert str(features["timestamp"].dt.tz) == PROJECT_TIMEZONE
    assert features.loc[0, "hour"] == 8


def test_equivalent_aware_timestamps_have_identical_project_time_features():
    utc_frame = pd.DataFrame(
        {
            "tower_id": ["001"],
            "timestamp": [pd.Timestamp("2025-01-01 00:00", tz="UTC")],
            "wind_speed_local": [3.0],
        }
    )
    local_frame = utc_frame.copy(deep=True)
    local_frame["timestamp"] = [
        pd.Timestamp("2025-01-01 08:00", tz=PROJECT_TIMEZONE)
    ]

    utc_features = FeatureBuilder().transform(
        utc_frame, physical_col="wind_speed_local"
    )
    local_features = FeatureBuilder().transform(
        local_frame, physical_col="wind_speed_local"
    )

    assert utc_features.loc[0, "timestamp"] == local_features.loc[0, "timestamp"]
    cycle_columns = [
        "hour_sin",
        "hour_cos",
        "day_of_year_sin",
        "day_of_year_cos",
    ]
    assert np.allclose(
        utc_features.loc[0, cycle_columns].to_numpy(dtype=float),
        local_features.loc[0, cycle_columns].to_numpy(dtype=float),
    )


def test_mixed_naive_and_aware_timestamps_normalize_in_one_frame():
    frame = pd.DataFrame(
        {
            "tower_id": ["001", "001"],
            "timestamp": pd.Series(
                [
                    pd.Timestamp("2025-01-01 08:00"),
                    pd.Timestamp("2025-01-01 01:00", tz="UTC"),
                ],
                dtype=object,
            ),
            "wind_speed_local": [3.0, 4.0],
        }
    )

    features = FeatureBuilder(cadence_minutes=60).transform(
        frame, physical_col="wind_speed_local"
    )

    assert str(features["timestamp"].dt.tz) == PROJECT_TIMEZONE
    assert features["hour"].tolist() == [8, 9]
    assert features["lag_1"].tolist() == [3.0, 3.0]


def test_mixed_aware_timezones_normalize_in_one_frame():
    frame = pd.DataFrame(
        {
            "tower_id": ["001", "002"],
            "timestamp": pd.Series(
                [
                    pd.Timestamp("2025-01-01 00:00", tz="UTC"),
                    pd.Timestamp(
                        "2025-01-01 08:00", tz=PROJECT_TIMEZONE
                    ),
                ],
                dtype=object,
            ),
            "wind_speed_local": [3.0, 4.0],
        }
    )

    features = FeatureBuilder().transform(
        frame, physical_col="wind_speed_local"
    )

    assert str(features["timestamp"].dt.tz) == PROJECT_TIMEZONE
    assert features.loc[0, "timestamp"] == features.loc[1, "timestamp"]
    cycle_columns = [
        "hour_sin",
        "hour_cos",
        "day_of_year_sin",
        "day_of_year_cos",
    ]
    assert np.allclose(
        features.loc[0, cycle_columns].to_numpy(dtype=float),
        features.loc[1, cycle_columns].to_numpy(dtype=float),
    )


@pytest.mark.parametrize(
    "timestamp",
    ["2025-11-02 01:30", "2025-03-09 02:30"],
)
def test_ambiguous_or_nonexistent_naive_local_time_is_rejected(
    monkeypatch, timestamp
):
    monkeypatch.setattr(
        ai_prediction, "PROJECT_TIMEZONE", "America/New_York"
    )
    frame = pd.DataFrame(
        {
            "tower_id": ["001"],
            "timestamp": [timestamp],
            "wind_speed_local": [3.0],
        }
    )

    with pytest.raises(ValueError, match="timestamp"):
        FeatureBuilder().transform(frame, physical_col="wind_speed_local")


@pytest.mark.parametrize(
    ("column", "value"),
    [
        ("tower_id", None),
        ("timestamp", pd.NaT),
        ("wind_speed_local", np.inf),
    ],
)
def test_feature_builder_rejects_invalid_identity_time_and_physical_values(
    column, value
):
    frame = make_interleaved_two_tower_training_frame()
    frame.loc[frame.index[0], column] = value

    with pytest.raises(ValueError):
        FeatureBuilder().transform(frame, physical_col="wind_speed_local")


def test_feature_builder_rejects_duplicate_tower_timestamp_in_one_dataset():
    frame = make_interleaved_two_tower_training_frame()
    frame.loc[7, "timestamp"] = frame.loc[8, "timestamp"]

    with pytest.raises(ValueError, match="duplicate|重复"):
        FeatureBuilder().transform(frame, physical_col="wind_speed_local")


def test_predictor_returns_physical_plus_residual_prediction():
    df = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2025-12-10 00:00", "2025-12-10 01:00"]),
            "wind_speed_physical": [3.0, 4.0],
        }
    )
    predictor = ResidualPredictor(
        {
            "wind_speed": ModelBundle(
                target_name="wind_speed",
                feature_columns=["hour_sin", "hour_cos", "wind_speed_physical"],
                model=OffsetModel(),
            )
        }
    )
    predicted = predictor.predict(df, target_name="wind_speed", physical_col="wind_speed_physical")
    assert predicted["wind_speed_residual"].tolist() == [0.5, 0.5]
    assert predicted["wind_speed_final"].tolist() == [3.5, 4.5]


def test_predictor_clips_residual_then_rejects_out_of_bounds_candidates():
    df = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                ["2025-12-10 00:00", "2025-12-10 01:00"]
            ),
            "wind_speed_physical": [74.0, 1.0],
        }
    )
    bundle = ModelBundle(
        target_name="wind_speed",
        feature_columns=["wind_speed_physical"],
        model=SequenceModel([100.0, -100.0]),
        residual_bounds=(-5.0, 5.0),
    )

    predicted = ResidualPredictor({"wind_speed": bundle}).predict(
        df, target_name="wind_speed", physical_col="wind_speed_physical"
    )

    assert predicted["wind_speed_residual"].tolist() == [0.0, 0.0]
    assert predicted["wind_speed_final"].tolist() == [74.0, 1.0]
    assert predicted["used_ai"].tolist() == [False, False]
    assert predicted["fallback_reason"].tolist() == [
        "physical_bounds_exceeded",
        "physical_bounds_exceeded",
    ]


def test_predictor_falls_back_when_final_candidate_exceeds_physical_bounds():
    frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2025-12-10 00:00"]),
            "wind_speed_local": [74.0],
        }
    )
    bundle = ModelBundle(
        target_name="wind_speed",
        feature_columns=["wind_speed_local"],
        model=SequenceModel([10.0]),
        residual_bounds=(-5.0, 5.0),
    )

    predicted = ResidualPredictor({"wind_speed": bundle}).predict(
        frame, target_name="wind_speed", physical_col="wind_speed_local"
    )

    assert predicted.loc[0, "wind_speed_final"] == 74.0
    assert predicted.loc[0, "wind_speed_residual"] == 0.0
    assert predicted.loc[0, "used_ai"] == False
    assert predicted.loc[0, "fallback_reason"] == "physical_bounds_exceeded"


def test_predictor_falls_back_to_physical_for_non_finite_prediction():
    df = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                ["2025-12-10 00:00", "2025-12-10 01:00"]
            ),
            "ambient_temp_local": [30.0, 31.0],
        }
    )
    bundle = ModelBundle(
        target_name="ambient_temp",
        feature_columns=["ambient_temp_local"],
        model=SequenceModel([np.nan, 2.0]),
        residual_bounds=(-3.0, 3.0),
    )

    predicted = ResidualPredictor({"ambient_temp": bundle}).predict(
        df, target_name="ambient_temp", physical_col="ambient_temp_local"
    )

    assert predicted["ambient_temp_final"].tolist() == [30.0, 33.0]
    assert predicted["ambient_temp_residual"].tolist() == [0.0, 2.0]
    assert predicted["used_ai"].tolist() == [False, True]
    assert predicted["fallback_reason"].tolist() == [
        "non_finite_prediction",
        "",
    ]


def test_predictor_model_exception_falls_back_without_propagating():
    frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2025-12-10 00:00"]),
            "wind_speed_local": [3.0],
        }
    )
    bundle = ModelBundle(
        target_name="wind_speed",
        feature_columns=["wind_speed_local"],
        model=ExplodingModel(),
    )

    predicted = ResidualPredictor({"wind_speed": bundle}).predict(
        frame, target_name="wind_speed", physical_col="wind_speed_local"
    )

    assert predicted.loc[0, "wind_speed_final"] == 3.0
    assert predicted.loc[0, "used_ai"] == False
    assert predicted.loc[0, "fallback_reason"] == "prediction_failed:OSError"


def test_predictor_returns_valid_physical_value_after_prediction_fallback():
    frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2025-12-10 00:00"]),
            "wind_speed_local": [70.0],
        }
    )
    bundle = ModelBundle(
        target_name="wind_speed",
        feature_columns=["wind_speed_local"],
        model=SequenceModel([np.nan]),
    )

    predicted = ResidualPredictor({"wind_speed": bundle}).predict(
        frame, target_name="wind_speed", physical_col="wind_speed_local"
    )

    assert predicted.loc[0, "wind_speed_final"] == 70.0


def test_all_fallback_paths_return_the_exact_input_physical_value():
    frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2025-12-10 00:00"]),
            "wind_speed_local": [80.0],
        }
    )
    unavailable = ResidualPredictor().predict(
        frame, target_name="wind_speed", physical_col="wind_speed_local"
    )
    non_finite = ResidualPredictor(
        {
            "wind_speed": ModelBundle(
                target_name="wind_speed",
                feature_columns=["wind_speed_local"],
                model=SequenceModel([np.nan]),
            )
        }
    ).predict(
        frame, target_name="wind_speed", physical_col="wind_speed_local"
    )

    assert unavailable.loc[0, "wind_speed_final"] == 80.0
    assert non_finite.loc[0, "wind_speed_final"] == 80.0
    assert unavailable.loc[0, "used_ai"] == False
    assert non_finite.loc[0, "used_ai"] == False


def test_predictor_rejects_invalid_timestamp_before_model_fallback():
    frame = pd.DataFrame(
        {"timestamp": ["not-a-time"], "wind_speed_local": [3.0]}
    )
    bundle = ModelBundle(
        target_name="wind_speed",
        feature_columns=["wind_speed_local"],
        model=ExplodingModel(),
    )

    with pytest.raises(ValueError, match="timestamp"):
        ResidualPredictor({"wind_speed": bundle}).predict(
            frame, target_name="wind_speed", physical_col="wind_speed_local"
        )


@pytest.mark.parametrize(
    ("target_name", "physical_col"),
    [
        ("wind_speed", "ambient_temp_local"),
        ("ambient_temp", "wind_speed_local"),
        ("dlr", "wind_speed_local"),
    ],
)
def test_predictor_rejects_unknown_or_mismatched_target_physical_pair(
    target_name, physical_col
):
    frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2025-12-10 00:00"]),
            physical_col: [3.0],
        }
    )

    with pytest.raises(ValueError, match="target|physical"):
        ResidualPredictor().predict(
            frame, target_name=target_name, physical_col=physical_col
        )
