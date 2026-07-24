import numpy as np
import pandas as pd
import pytest

import modules.ai_training as ai_training
from modules.ai_prediction import FeatureBuilder, ResidualPredictor
from modules.ai_training import ResidualTrainer


class MeanResidualEstimator:
    def fit(self, features, target):
        self.value = float(np.mean(target))
        return self

    def predict(self, features):
        return np.full(len(features), self.value)


class FixedResidualEstimator:
    def __init__(self, value):
        self.value = value

    def fit(self, features, target):
        return self

    def predict(self, features):
        return np.full(len(features), self.value)


class BrokenEstimator:
    def fit(self, features, target):
        raise RuntimeError("training exploded")

    def predict(self, features):
        return np.zeros(len(features))


class NonFiniteEstimator:
    def fit(self, features, target):
        return self

    def predict(self, features):
        return np.full(len(features), np.nan)


class HoldoutNonFiniteEstimator:
    def __init__(self):
        self.predict_calls = 0

    def fit(self, features, target):
        return self

    def predict(self, features):
        self.predict_calls += 1
        if self.predict_calls == 1:
            return np.zeros(len(features))
        return np.full(len(features), np.nan)


def make_training_frame(
    residuals=(1.0, 1.0, 1.0, 1.0),
    *,
    target="wind_speed",
    line_id="line-a",
    tower_id="001",
    timestamps=None,
):
    if timestamps is None:
        timestamps = pd.to_datetime(
            [
                "2025-01-01 00:00",
                "2025-01-01 00:30",
                "2025-01-02 00:00",
                "2025-01-02 00:30",
            ],
            utc=True,
        )
    residuals = np.asarray(residuals, dtype=float)
    count = len(residuals)
    if target == "wind_speed":
        physical_col = "wind_speed_local"
        truth_col = "wind_speed_truth"
        physical = np.arange(2.0, 2.0 + count)
    else:
        physical_col = "ambient_temp_local"
        truth_col = "ambient_temp_truth"
        physical = np.arange(20.0, 20.0 + count)
    return pd.DataFrame(
        {
            "line_id": [line_id] * count,
            "tower_id": [tower_id] * count,
            "timestamp": timestamps,
            "source_file_hash": ["physical-a"] * count,
            physical_col: physical,
            truth_col: physical + residuals,
            "wind_direction": np.linspace(0.0, 90.0, count),
            "solar_radiation_local": np.full(count, 500.0),
            "humidity": np.full(count, 30.0),
            "elevation": np.full(count, 1100.0),
            "slope": np.full(count, 4.0),
            "aspect": np.full(count, 180.0),
        }
    )


def offset_factory():
    return FixedResidualEstimator(1.0)


def constant_factory():
    return MeanResidualEstimator()


def test_training_metrics_compare_weather_to_truth_not_dlr():
    result = ResidualTrainer(estimator_factory=offset_factory).train_target(
        make_training_frame(), target="wind_speed"
    )

    assert set(result.metrics) == {
        "baseline_mae",
        "baseline_rmse",
        "corrected_mae",
        "corrected_rmse",
    }
    assert result.metrics["corrected_mae"] < result.metrics["baseline_mae"]
    assert result.metrics == {
        "baseline_mae": pytest.approx(1.0),
        "baseline_rmse": pytest.approx(1.0),
        "corrected_mae": pytest.approx(0.0),
        "corrected_rmse": pytest.approx(0.0),
    }
    assert result.metadata["evaluation_mode"] == "temporal_holdout"
    assert result.metadata["metric_domain"] == "weather_vs_truth"
    assert "dlr" not in " ".join(result.metrics).lower()
    assert "mape" not in " ".join(result.metrics).lower()


def test_single_sample_is_trained_without_rejection():
    frame = make_training_frame(
        residuals=(2.0,),
        target="ambient_temp",
        timestamps=pd.to_datetime(["2025-01-01 00:00"], utc=True),
    )

    result = ResidualTrainer(
        estimator_factory=constant_factory
    ).train_target(frame, target="ambient_temp")

    assert result.metadata["evaluation_mode"] == "full_fit"
    assert result.metadata["sample_count"] == 1
    assert result.metadata["independent_evaluation"] is False
    assert result.metrics == {}
    assert set(result.metadata["full_fit_metrics"]) == {
        "baseline_mae",
        "baseline_rmse",
        "corrected_mae",
        "corrected_rmse",
    }


def test_constant_residual_uses_robust_constant_fallback():
    result = ResidualTrainer(
        estimator_factory=lambda: BrokenEstimator()
    ).train_target(make_training_frame(), target="wind_speed")

    assert result.metadata["fallback_reason"] == "constant_residual"
    assert np.allclose(result.bundle.model.predict(np.zeros((3, 1))), 1.0)


def test_temporal_holdout_never_fits_future_block_during_evaluation():
    fitted_targets = []

    class RecordingEstimator(MeanResidualEstimator):
        def fit(self, features, target):
            fitted_targets.append(np.asarray(target, dtype=float).copy())
            return super().fit(features, target)

    frame = make_training_frame(residuals=(1.0, 2.0, 100.0, 101.0))

    result = ResidualTrainer(
        estimator_factory=RecordingEstimator
    ).train_target(frame, target="wind_speed")

    assert result.metadata["evaluation_mode"] == "temporal_holdout"
    assert fitted_targets[0].tolist() == [1.0, 2.0]
    assert fitted_targets[-1].tolist() == [1.0, 2.0, 100.0, 101.0]


def test_estimator_exception_falls_back_without_blocking_training():
    result = ResidualTrainer(
        estimator_factory=lambda: BrokenEstimator()
    ).train_target(
        make_training_frame(residuals=(0.0, 1.0, 2.0, 3.0)),
        target="wind_speed",
    )

    assert "estimator_fit_failed" in result.metadata["fallback_reason"]
    prediction = result.bundle.model.predict(np.zeros((2, 1)))
    assert np.isfinite(prediction).all()


def test_missing_xgboost_falls_back_to_median_residual(monkeypatch):
    def unavailable():
        raise ImportError("xgboost unavailable")

    monkeypatch.setattr(ai_training, "_load_xgb_regressor", unavailable)

    result = ResidualTrainer().train_target(
        make_training_frame(residuals=(0.0, 1.0, 2.0, 3.0)),
        target="wind_speed",
    )

    assert result.metadata["fallback_reason"] == "xgboost_unavailable"
    prediction = result.bundle.model.predict(np.zeros((2, 1)))
    assert prediction.tolist() == [1.5, 1.5]


def test_non_finite_estimator_prediction_falls_back_to_physical():
    result = ResidualTrainer(
        estimator_factory=NonFiniteEstimator
    ).train_target(
        make_training_frame(residuals=(0.0, 1.0, 2.0, 3.0)),
        target="wind_speed",
    )

    assert "non_finite_prediction" in result.metadata["fallback_reason"]
    assert np.isfinite(
        result.bundle.model.predict(np.zeros((2, 1))
    )).all()


def test_non_finite_holdout_prediction_is_recorded_and_uses_baseline():
    result = ResidualTrainer(
        estimator_factory=HoldoutNonFiniteEstimator
    ).train_target(
        make_training_frame(residuals=(0.0, 1.0, 2.0, 3.0)),
        target="wind_speed",
    )

    assert result.metadata["evaluation_fallback_reason"] == (
        "non_finite_prediction"
    )
    assert result.metrics["corrected_mae"] == result.metrics["baseline_mae"]


def test_out_of_bounds_holdout_candidate_uses_physical_baseline_metrics():
    frame = pd.DataFrame(
        {
            "line_id": ["line-a"] * 3,
            "tower_id": ["001"] * 3,
            "timestamp": pd.to_datetime(
                [
                    "2025-01-01 00:00",
                    "2025-01-01 00:30",
                    "2025-01-02 00:00",
                ],
                utc=True,
            ),
            "source_file_hash": ["physical-a"] * 3,
            "wind_speed_local": [10.0, 11.0, 74.0],
            "wind_speed_truth": [14.0, 17.0, 74.0],
        }
    )

    result = ResidualTrainer(
        estimator_factory=lambda: FixedResidualEstimator(5.0)
    ).train_target(frame, target="wind_speed")

    assert result.metadata["evaluation_fallback_reason"] == (
        "physical_bounds_exceeded"
    )
    assert result.metrics["baseline_mae"] == 0.0
    assert result.metrics["corrected_mae"] == 0.0


def test_trained_bundle_clips_residual_and_weather_per_tower():
    frame = make_training_frame(
        residuals=(-2.0, -1.0, 1.0, 2.0),
        timestamps=pd.to_datetime(
            [
                "2025-01-01 00:00",
                "2025-01-01 01:00",
                "2025-01-02 00:00",
                "2025-01-02 01:00",
            ],
            utc=True,
        ),
    )
    result = ResidualTrainer(
        estimator_factory=lambda: FixedResidualEstimator(1000.0)
    ).train_target(frame, target="wind_speed")
    prediction_frame = frame.iloc[[0]].copy()
    prediction_frame["wind_speed_local"] = 74.5

    predicted = ResidualPredictor(
        {"wind_speed": result.bundle}
    ).predict(
        prediction_frame,
        target_name="wind_speed",
        physical_col="wind_speed_local",
    )

    lower, upper = result.bundle.residual_bounds
    assert -2.0 <= lower <= upper <= 2.0
    assert predicted.loc[predicted.index[0], "wind_speed_residual"] <= upper
    assert predicted.loc[predicted.index[0], "wind_speed_residual"] == 0.0
    assert predicted.loc[predicted.index[0], "wind_speed_final"] == 74.5
    assert predicted.loc[predicted.index[0], "used_ai"] == False
    assert predicted.loc[predicted.index[0], "fallback_reason"] == (
        "physical_bounds_exceeded"
    )


def test_train_target_rejects_mixed_lines_or_towers():
    first = make_training_frame()
    second = make_training_frame(line_id="line-b", tower_id="002")

    with pytest.raises(ValueError, match="single|单个"):
        ResidualTrainer(estimator_factory=constant_factory).train_target(
            pd.concat([first, second], ignore_index=True),
            target="wind_speed",
        )


def test_train_many_isolates_line_tower_and_target_models():
    first = make_training_frame()
    second = make_training_frame(line_id="line-b", tower_id="001")
    combined = pd.concat([first, second], ignore_index=True)

    results = ResidualTrainer(
        estimator_factory=constant_factory
    ).train_many(combined, targets=("wind_speed",))

    assert set(results) == {
        ("line-a", "001", "wind_speed"),
        ("line-b", "001", "wind_speed"),
    }
    assert results[("line-a", "001", "wind_speed")].bundle is not results[
        ("line-b", "001", "wind_speed")
    ].bundle


@pytest.mark.parametrize(
    ("column", "value"),
    [
        ("wind_speed_local", np.nan),
        ("wind_speed_truth", np.inf),
        ("line_id", ""),
        ("tower_id", None),
        ("timestamp", pd.NaT),
    ],
)
def test_training_rejects_non_finite_and_invalid_schema(column, value):
    frame = make_training_frame()
    frame.loc[0, column] = value

    with pytest.raises(ValueError):
        ResidualTrainer(estimator_factory=constant_factory).train_target(
            frame, target="wind_speed"
        )


def test_training_rejects_raw_physical_column_override():
    frame = make_training_frame()
    frame["wind_speed_physical"] = frame["wind_speed_local"]

    with pytest.raises(ValueError, match="terrain|corrected"):
        ResidualTrainer(estimator_factory=constant_factory).train_target(
            frame,
            target="wind_speed",
            physical_col="wind_speed_physical",
        )


@pytest.mark.parametrize(
    "truth_col", ["dlr_truth", "ambient_temp_truth", "wind_speed_local"]
)
def test_training_rejects_non_weather_or_mismatched_truth_override(truth_col):
    frame = make_training_frame()
    if truth_col not in frame.columns:
        frame[truth_col] = 1000.0

    with pytest.raises(ValueError, match="truth|weather"):
        ResidualTrainer(estimator_factory=constant_factory).train_target(
            frame,
            target="wind_speed",
            truth_col=truth_col,
        )


def test_training_does_not_mutate_input_frame():
    frame = make_training_frame(residuals=(0.0, 1.0, 2.0, 3.0))
    original = frame.copy(deep=True)

    ResidualTrainer(estimator_factory=constant_factory).train_target(
        frame, target="wind_speed"
    )

    pd.testing.assert_frame_equal(frame, original)


def test_default_estimator_has_deterministic_xgboost_parameters(monkeypatch):
    captured = {}

    class CapturingXGBRegressor:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(
        ai_training, "_load_xgb_regressor", lambda: CapturingXGBRegressor
    )

    ai_training.default_estimator()

    assert captured == {
        "objective": "reg:squarederror",
        "n_estimators": 120,
        "max_depth": 3,
        "learning_rate": 0.05,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "random_state": 42,
        "n_jobs": 1,
    }


def test_training_persists_feature_cadence_in_bundle_and_metadata():
    trainer = ResidualTrainer(
        estimator_factory=constant_factory,
        feature_builder=FeatureBuilder(cadence_minutes=60),
    )

    result = trainer.train_target(make_training_frame(), target="wind_speed")

    assert result.bundle.cadence_minutes == 60.0
    assert result.metadata["cadence_minutes"] == 60.0
