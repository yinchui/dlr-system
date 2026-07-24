import json
import types
from dataclasses import dataclass, replace
from functools import partial
from pathlib import Path

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


class OrderSensitiveEstimator:
    def __init__(self, offset=0.0):
        self.offset = float(offset)

    def fit(self, features, target):
        values = np.asarray(target, dtype=float)
        weights = np.arange(1.0, len(values) + 1.0)
        self.value = float(np.average(values, weights=weights) + self.offset)
        return self

    def predict(self, features):
        return np.full(len(features), self.value)

    def get_params(self, deep=False):
        return {"offset": self.offset}


class ConfiguredEstimatorFactory:
    def __init__(self, offset=0.0, values=None):
        self.offset = float(offset)
        self.values = values
        self.calls = []

    def __call__(self):
        self.calls.append(self.offset)
        return OrderSensitiveEstimator(offset=self.offset)

    def training_contract_descriptor(self):
        return {"offset": self.offset, "values": self.values}


class ConfigurableTrainer:
    def __init__(self, poor):
        self.poor = bool(poor)
        self.delegate = ResidualTrainer(estimator_factory=constant_factory)
        self.feature_builder = self.delegate.feature_builder
        self.calls = []

    def prepare_target(self, frame, target, **kwargs):
        return self.delegate.prepare_target(frame, target, **kwargs)

    def train_prepared(self, preparation):
        self.calls.append((preparation.tower_id, preparation.target))
        result = self.delegate.train_prepared(preparation)
        if not self.poor or not result.metrics:
            return result
        metrics = dict(result.metrics)
        metrics["corrected_mae"] = metrics["baseline_mae"]
        metrics["corrected_rmse"] = metrics["baseline_rmse"]
        return replace(result, metrics=metrics)

    def train_target(self, frame, target, **kwargs):
        preparation = self.prepare_target(frame, target, **kwargs)
        return self.train_prepared(preparation)

    def training_contract_descriptor(self):
        return {
            "poor": self.poor,
            "delegate": self.delegate.training_contract_descriptor(),
        }


class SecretReprObject:
    def __repr__(self):
        return "SecretReprObject(secret-123)"


class SecretParameterEstimator(MeanResidualEstimator):
    def get_params(self, deep=False):
        return {
            "customer_secret": "secret-123",
            "diagnostic_frame": pd.DataFrame(
                {"customer_secret": ["secret-123"]}
            ),
            "artifact_path": Path("/private/secret-123/model.bin"),
            "opaque": SecretReprObject(),
        }


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


def test_training_preparation_matches_training_without_fitting_early():
    estimator_calls = []

    def estimator_factory():
        estimator_calls.append("created")
        return MeanResidualEstimator()

    estimator_factory.training_contract_descriptor = lambda: {
        "type": "recording-mean-residual-factory"
    }
    trainer = ResidualTrainer(estimator_factory=estimator_factory)
    preparation = trainer.prepare_target(
        make_training_frame(residuals=(0.0, 1.0, 2.0, 3.0)),
        target="wind_speed",
    )

    assert estimator_calls == []
    assert preparation.evaluation_mode == "temporal_holdout"
    assert preparation.evaluation_set_hash

    result = trainer.train_prepared(preparation)

    assert estimator_calls
    assert result.metadata["input_data_hash"] == preparation.input_data_hash
    assert result.metadata["evaluation_mode"] == preparation.evaluation_mode
    assert (
        result.metadata["evaluation_set_hash"]
        == preparation.evaluation_set_hash
    )


def test_training_preparation_reports_full_fit_for_one_continuous_block():
    timestamps = pd.date_range(
        "2025-01-01 00:00",
        periods=4,
        freq="30min",
        tz="UTC",
    )
    trainer = ResidualTrainer(
        estimator_factory=lambda: (_ for _ in ()).throw(
            AssertionError("preparation must not construct an estimator")
        )
    )

    preparation = trainer.prepare_target(
        make_training_frame(
            residuals=(0.0, 1.0, 2.0, 3.0),
            timestamps=timestamps,
        ),
        target="wind_speed",
    )

    assert preparation.evaluation_mode == "full_fit"
    assert preparation.evaluation_set_hash is None


def test_training_is_canonical_for_reversed_input_rows():
    frame = make_training_frame(residuals=(0.0, 4.0, -2.0, 3.0))
    trainer = ResidualTrainer(
        estimator_factory=partial(OrderSensitiveEstimator, offset=0.25)
    )

    forward_preparation = trainer.prepare_target(frame, target="wind_speed")
    reversed_preparation = trainer.prepare_target(
        frame.iloc[::-1], target="wind_speed"
    )
    forward = trainer.train_prepared(forward_preparation)
    reversed_rows = trainer.train_prepared(reversed_preparation)

    assert forward.metadata["input_data_hash"] == reversed_rows.metadata[
        "input_data_hash"
    ]
    assert forward.metadata["evaluation_set_hash"] == reversed_rows.metadata[
        "evaluation_set_hash"
    ]
    pd.testing.assert_frame_equal(
        forward_preparation.working,
        reversed_preparation.working,
    )
    assert forward.metrics == reversed_rows.metrics
    assert forward.bundle.model.value == pytest.approx(
        reversed_rows.bundle.model.value
    )


def test_training_contract_changes_with_estimator_parameters():
    frame = make_training_frame(residuals=(0.0, 1.0, 2.0, 3.0))
    first = ResidualTrainer(
        estimator_factory=partial(OrderSensitiveEstimator, offset=0.0)
    ).prepare_target(frame, target="wind_speed")
    changed = ResidualTrainer(
        estimator_factory=partial(OrderSensitiveEstimator, offset=1.0)
    ).prepare_target(frame, target="wind_speed")

    assert first.training_contract_hash != changed.training_contract_hash


def test_training_contract_distinguishes_callable_instance_configuration():
    frame = make_training_frame(residuals=(0.0, 1.0, 2.0, 3.0))

    zero = ResidualTrainer(
        estimator_factory=ConfiguredEstimatorFactory(offset=0.0)
    ).prepare_target(frame, target="wind_speed")
    five = ResidualTrainer(
        estimator_factory=ConfiguredEstimatorFactory(offset=5.0)
    ).prepare_target(frame, target="wind_speed")

    assert zero.training_contract_hash != five.training_contract_hash


def test_training_contract_distinguishes_dynamic_lambda_bytecode_constants():
    frame = make_training_frame(residuals=(0.0, 1.0, 2.0, 3.0))
    namespace = {"FixedResidualEstimator": FixedResidualEstimator}
    zero_factory = eval(
        "lambda: FixedResidualEstimator(0.0)", namespace  # noqa: S307
    )
    five_factory = eval(
        "lambda: FixedResidualEstimator(5.0)", namespace  # noqa: S307
    )

    zero = ResidualTrainer(estimator_factory=zero_factory).prepare_target(
        frame, target="wind_speed"
    )
    five = ResidualTrainer(estimator_factory=five_factory).prepare_target(
        frame, target="wind_speed"
    )

    assert zero.training_contract_hash != five.training_contract_hash


def test_training_contract_rejects_opaque_public_class_configuration():
    class OpaqueConfiguredEstimator:
        strategy = object()

        def fit(self, features, target):
            return self

        def predict(self, features):
            return np.zeros(len(features))

    with pytest.raises(
        ai_training.TrainingContractError,
        match="describe|contract|configuration",
    ):
        ResidualTrainer(estimator_factory=OpaqueConfiguredEstimator)


def test_train_prepared_rejects_referenced_global_callable_changed_after_init():
    namespace = {"Estimator": MeanResidualEstimator}
    estimator_factory = eval("lambda: Estimator()", namespace)  # noqa: S307
    trainer = ResidualTrainer(estimator_factory=estimator_factory)
    preparation = trainer.prepare_target(
        make_training_frame(residuals=(0.0, 1.0, 2.0, 3.0)),
        target="wind_speed",
    )

    namespace["Estimator"] = NonFiniteEstimator

    with pytest.raises(
        ai_training.TrainingContractError,
        match="factory.*contract|contract.*changed",
    ):
        trainer.train_prepared(preparation)


def test_training_contract_tracks_globals_used_by_nested_code_objects():
    source = (
        "lambda: [FixedResidualEstimator(OFFSET) for _ in (0,)][0]"
    )
    zero_factory = eval(  # noqa: S307
        source,
        {"FixedResidualEstimator": FixedResidualEstimator, "OFFSET": 0.0},
    )
    five_factory = eval(  # noqa: S307
        source,
        {"FixedResidualEstimator": FixedResidualEstimator, "OFFSET": 5.0},
    )
    frame = make_training_frame(residuals=(0.0, 1.0, 2.0, 3.0))

    zero = ResidualTrainer(estimator_factory=zero_factory).prepare_target(
        frame, target="wind_speed"
    )
    five = ResidualTrainer(estimator_factory=five_factory).prepare_target(
        frame, target="wind_speed"
    )

    assert zero.training_contract_hash != five.training_contract_hash


def test_training_contract_tracks_globals_used_by_referenced_helper():
    source = """
def helper_offset():
    return OFFSET

def estimator_factory():
    return FixedResidualEstimator(helper_offset())
"""

    def make_factory(offset):
        namespace = {
            "__name__": __name__,
            "FixedResidualEstimator": FixedResidualEstimator,
            "OFFSET": float(offset),
        }
        exec(source, namespace)  # noqa: S102
        return namespace["estimator_factory"]

    frame = make_training_frame(residuals=(0.0, 1.0, 2.0, 3.0))
    zero = ResidualTrainer(estimator_factory=make_factory(0.0)).prepare_target(
        frame, target="wind_speed"
    )
    five = ResidualTrainer(estimator_factory=make_factory(5.0)).prepare_target(
        frame, target="wind_speed"
    )

    assert zero.training_contract_hash != five.training_contract_hash


def test_training_contract_tracks_referenced_module_attribute_names():
    estimator_module = types.ModuleType("contract_test_estimators")

    class ZeroEstimator(FixedResidualEstimator):
        def __init__(self):
            super().__init__(0.0)

    class FiveEstimator(FixedResidualEstimator):
        def __init__(self):
            super().__init__(5.0)

    estimator_module.Zero = ZeroEstimator
    estimator_module.Five = FiveEstimator
    zero_factory = eval(  # noqa: S307
        "lambda: estimator_module.Zero()", {"estimator_module": estimator_module}
    )
    five_factory = eval(  # noqa: S307
        "lambda: estimator_module.Five()", {"estimator_module": estimator_module}
    )
    frame = make_training_frame(residuals=(0.0, 1.0, 2.0, 3.0))

    zero = ResidualTrainer(estimator_factory=zero_factory).prepare_target(
        frame, target="wind_speed"
    )
    five = ResidualTrainer(estimator_factory=five_factory).prepare_target(
        frame, target="wind_speed"
    )

    assert zero.training_contract_hash != five.training_contract_hash


def test_train_prepared_rejects_referenced_class_configuration_change():
    class PublicConfiguredEstimator(MeanResidualEstimator):
        offset = 0.0

        def fit(self, features, target):
            self.value = float(np.mean(target) + type(self).offset)
            return self

    estimator_factory = eval(  # noqa: S307
        "lambda: PublicConfiguredEstimator()",
        {"PublicConfiguredEstimator": PublicConfiguredEstimator},
    )

    trainer = ResidualTrainer(estimator_factory=estimator_factory)
    preparation = trainer.prepare_target(
        make_training_frame(residuals=(0.0, 1.0, 2.0, 3.0)),
        target="wind_speed",
    )
    PublicConfiguredEstimator.offset = 7.0

    with pytest.raises(ai_training.TrainingContractError, match="contract.*changed"):
        trainer.train_prepared(preparation)


def test_train_prepared_rejects_inherited_class_configuration_change():
    class BaseConfiguredEstimator:
        offset = 0.0

        def fit(self, features, target):
            self.value = float(np.mean(target) + type(self).offset)
            return self

        def predict(self, features):
            return np.full(len(features), self.value)

    class InheritedConfiguredEstimator(BaseConfiguredEstimator):
        def __init__(self):
            self.value = 0.0

    trainer = ResidualTrainer(estimator_factory=InheritedConfiguredEstimator)
    preparation = trainer.prepare_target(
        make_training_frame(residuals=(0.0, 1.0, 2.0, 3.0)),
        target="wind_speed",
    )

    BaseConfiguredEstimator.offset = 5.0

    with pytest.raises(ai_training.TrainingContractError, match="contract.*changed"):
        trainer.train_prepared(preparation)


def test_train_prepared_rejects_builtin_class_strategy_change():
    class TransformEstimator:
        transform = abs

        def fit(self, features, target):
            self.value = float(type(self).transform(float(np.mean(target))))
            return self

        def predict(self, features):
            return np.full(len(features), self.value)

    trainer = ResidualTrainer(estimator_factory=TransformEstimator)
    preparation = trainer.prepare_target(
        make_training_frame(residuals=(0.0, 1.0, 2.0, 3.0)),
        target="wind_speed",
    )

    TransformEstimator.transform = round

    with pytest.raises(ai_training.TrainingContractError, match="contract.*changed"):
        trainer.train_prepared(preparation)


def test_train_prepared_rejects_callable_instance_class_configuration_change():
    class ClassConfiguredFactory:
        offset = 0.0

        def __call__(self):
            return FixedResidualEstimator(type(self).offset)

    factory = ClassConfiguredFactory()
    trainer = ResidualTrainer(estimator_factory=factory)
    preparation = trainer.prepare_target(
        make_training_frame(residuals=(0.0, 1.0, 2.0, 3.0)),
        target="wind_speed",
    )

    ClassConfiguredFactory.offset = 5.0

    with pytest.raises(ai_training.TrainingContractError, match="contract.*changed"):
        trainer.train_prepared(preparation)


def test_training_contract_does_not_drop_call_count_configuration():
    class CallCountConfiguredFactory:
        def __init__(self, call_count):
            self.call_count = float(call_count)

        def __call__(self):
            return FixedResidualEstimator(self.call_count)

    frame = make_training_frame(residuals=(0.0, 1.0, 2.0, 3.0))

    zero = ResidualTrainer(
        estimator_factory=CallCountConfiguredFactory(0.0)
    ).prepare_target(frame, target="wind_speed")
    five = ResidualTrainer(
        estimator_factory=CallCountConfiguredFactory(5.0)
    ).prepare_target(frame, target="wind_speed")

    assert zero.training_contract_hash != five.training_contract_hash


def test_training_contract_does_not_treat_custom_append_as_observation():
    class AppendFactoryDelegate:
        def __init__(self, offset):
            self.offset = float(offset)

        def append(self):
            return FixedResidualEstimator(self.offset)

    def make_factory(offset):
        delegate = AppendFactoryDelegate(offset)

        def estimator_factory():
            return delegate.append()

        return estimator_factory

    frame = make_training_frame(residuals=(0.0, 1.0, 2.0, 3.0))
    zero = ResidualTrainer(estimator_factory=make_factory(0.0)).prepare_target(
        frame, target="wind_speed"
    )
    five = ResidualTrainer(estimator_factory=make_factory(5.0)).prepare_target(
        frame, target="wind_speed"
    )

    assert zero.training_contract_hash != five.training_contract_hash


def test_training_contract_preserves_mutable_alias_topology():
    def make_factory(left, right):
        def estimator_factory():
            offset = 0.0 if left is right else 5.0
            return FixedResidualEstimator(offset)

        estimator_factory.training_contract_descriptor = lambda: {
            "shared_configuration": left is right
        }
        return estimator_factory

    shared = []
    frame = make_training_frame(residuals=(0.0, 1.0, 2.0, 3.0))
    aliased = ResidualTrainer(
        estimator_factory=make_factory(shared, shared)
    ).prepare_target(frame, target="wind_speed")
    distinct = ResidualTrainer(
        estimator_factory=make_factory([], [])
    ).prepare_target(frame, target="wind_speed")

    assert aliased.training_contract_hash != distinct.training_contract_hash


def test_frozen_callable_contract_tracks_exception_table_semantics():
    def estimator_factory():
        try:
            return MeanResidualEstimator()
        except RuntimeError:
            return NonFiniteEstimator()

    original_code = estimator_factory.__code__
    assert original_code.co_exceptiontable
    frozen = ai_training.FrozenCallableContract.capture(estimator_factory)
    estimator_factory.__code__ = original_code.replace(co_exceptiontable=b"")
    try:
        with pytest.raises(
            ai_training.TrainingContractError,
            match="contract.*changed",
        ):
            frozen.verify(estimator_factory)
    finally:
        estimator_factory.__code__ = original_code


def test_training_contract_preserves_container_type_tags():
    frame = make_training_frame(residuals=(0.0, 1.0, 2.0, 3.0))

    list_contract = ResidualTrainer(
        estimator_factory=ConfiguredEstimatorFactory(values=[1, 2])
    ).prepare_target(frame, target="wind_speed")
    tuple_contract = ResidualTrainer(
        estimator_factory=ConfiguredEstimatorFactory(values=(1, 2))
    ).prepare_target(frame, target="wind_speed")

    assert list_contract.training_contract_hash != tuple_contract.training_contract_hash


def test_train_prepared_rejects_partial_configuration_changed_after_init():
    frame = make_training_frame(residuals=(0.0, 1.0, 2.0, 3.0))
    factory = partial(OrderSensitiveEstimator, offset=0.0)
    trainer = ResidualTrainer(estimator_factory=factory)
    preparation = trainer.prepare_target(frame, target="wind_speed")

    factory.keywords["offset"] = 7.0

    with pytest.raises(RuntimeError, match="factory.*contract|contract.*changed"):
        trainer.train_prepared(preparation)


def test_train_prepared_rejects_function_defaults_changed_after_init():
    frame = make_training_frame(residuals=(0.0, 1.0, 2.0, 3.0))

    def estimator_factory(offset=0.0):
        return OrderSensitiveEstimator(offset=offset)

    trainer = ResidualTrainer(estimator_factory=estimator_factory)
    preparation = trainer.prepare_target(frame, target="wind_speed")

    estimator_factory.__defaults__ = (7.0,)

    with pytest.raises(RuntimeError, match="factory.*contract|contract.*changed"):
        trainer.train_prepared(preparation)


def test_runtime_contract_includes_trainer_config_but_ignores_call_history():
    frame = make_training_frame(residuals=(0.0, 1.0, 2.0, 3.0))
    normal = ConfigurableTrainer(poor=False)
    poor = ConfigurableTrainer(poor=True)
    preparation = normal.prepare_target(frame, target="wind_speed")

    normal_hash = ai_training.training_runtime_contract_hash(normal, preparation)
    poor_hash = ai_training.training_runtime_contract_hash(poor, preparation)
    normal.calls.append(("observed", "only"))
    after_observation = ai_training.training_runtime_contract_hash(
        normal, preparation
    )

    assert normal_hash != poor_hash
    assert after_observation == normal_hash


def test_runtime_contract_accepts_builtin_residual_trainer():
    trainer, _, preparation = _temporal_preparation()

    contract_hash = ai_training.training_runtime_contract_hash(
        trainer, preparation
    )

    assert len(contract_hash) == 64


def test_runtime_contract_tracks_builtin_training_dependencies(monkeypatch):
    trainer, _, preparation = _temporal_preparation()
    original_hash = ai_training.training_runtime_contract_hash(
        trainer, preparation
    )

    def changed_metric_values(physical, truth, corrected):
        return {
            "baseline_mae": 0.0,
            "baseline_rmse": 0.0,
            "corrected_mae": 0.0,
            "corrected_rmse": 0.0,
        }

    monkeypatch.setattr(ai_training, "_metric_values", changed_metric_values)

    assert ai_training.training_runtime_contract_hash(
        trainer, preparation
    ) != original_hash


def test_training_contract_changes_with_dependency_versions(monkeypatch):
    frame = make_training_frame(residuals=(0.0, 1.0, 2.0, 3.0))
    monkeypatch.setattr(
        ai_training,
        "_dependency_versions",
        lambda: {"python": "3.11", "xgboost": "2.0"},
    )
    first = ResidualTrainer(
        estimator_factory=constant_factory
    ).prepare_target(frame, target="wind_speed")
    monkeypatch.setattr(
        ai_training,
        "_dependency_versions",
        lambda: {"python": "3.11", "xgboost": "3.0"},
    )
    changed = ResidualTrainer(
        estimator_factory=constant_factory
    ).prepare_target(frame, target="wind_speed")

    assert first.training_contract_hash != changed.training_contract_hash


def _temporal_preparation():
    trainer = ResidualTrainer(estimator_factory=constant_factory)
    frame = make_training_frame(residuals=(0.0, 1.0, 2.0, 3.0))
    return trainer, frame, trainer.prepare_target(frame, target="wind_speed")


@pytest.mark.parametrize(
    ("frame_attribute", "column"),
    [
        ("working", "wind_speed_truth"),
        ("feature_frame", "timestamp"),
    ],
)
def test_train_prepared_rejects_mutated_preparation_frames(
    frame_attribute,
    column,
):
    trainer, _, preparation = _temporal_preparation()
    frame = getattr(preparation, frame_attribute)
    if column == "timestamp":
        frame.loc[frame.index[0], column] += pd.Timedelta(days=30)
    else:
        frame.loc[frame.index[0], column] += 100.0

    with pytest.raises(ValueError, match="preparation|snapshot|integrity"):
        trainer.train_prepared(preparation)


def test_train_prepared_rejects_mutated_residual():
    trainer, _, preparation = _temporal_preparation()
    preparation.residual[0] += 100.0

    with pytest.raises(ValueError, match="preparation|snapshot|integrity"):
        trainer.train_prepared(preparation)


@pytest.mark.parametrize("field", ["physical", "truth"])
def test_train_prepared_rejects_mutated_source_arrays(field):
    trainer, _, preparation = _temporal_preparation()
    getattr(preparation, field)[0] += 100.0

    with pytest.raises(ValueError, match="preparation|snapshot|integrity"):
        trainer.train_prepared(preparation)


def test_train_prepared_rejects_mutated_model_features():
    trainer, _, preparation = _temporal_preparation()
    preparation.model_features.iloc[0, 0] += 100.0

    with pytest.raises(ValueError, match="preparation|snapshot|integrity"):
        trainer.train_prepared(preparation)


@pytest.mark.parametrize(
    "split",
    [
        (np.array([-1, 0]), np.array([1, 2, 3])),
        (np.array([0, 1]), np.array([1, 2, 3])),
        (np.array([0]), np.array([2, 3])),
    ],
    ids=("out_of_bounds", "overlap", "incomplete"),
)
def test_train_prepared_rejects_invalid_split_indices(split):
    trainer, _, preparation = _temporal_preparation()
    tampered = replace(preparation, split=split)

    with pytest.raises(ValueError, match="split|preparation|integrity"):
        trainer.train_prepared(tampered)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("input_data_hash", "0" * 64),
        ("evaluation_set_hash", "1" * 64),
        ("training_contract_hash", "2" * 64),
    ],
)
def test_train_prepared_rejects_forged_preparation_hashes(field, value):
    trainer, _, preparation = _temporal_preparation()
    tampered = replace(preparation, **{field: value})

    with pytest.raises(ValueError, match="hash|preparation|integrity"):
        trainer.train_prepared(tampered)


def test_temporal_split_may_exclude_rows_to_prevent_time_leakage():
    frame = make_training_frame(
        residuals=(0.0, 1.0, 2.0, 3.0, 4.0, 5.0),
        timestamps=pd.to_datetime(
            [
                "2025-01-01 00:00",
                "2025-01-01 00:30",
                "2025-01-01 01:00",
                "2025-01-01 00:30",
                "2025-01-01 01:00",
                "2025-01-01 01:30",
            ],
            utc=True,
        ),
    )
    frame["source_file_hash"] = ["a", "a", "a", "b", "b", "b"]
    trainer = ResidualTrainer(estimator_factory=constant_factory)

    preparation = trainer.prepare_target(frame, target="wind_speed")
    train_positions, holdout_positions = preparation.split

    assert len(set(train_positions) | set(holdout_positions)) < len(frame)
    assert trainer.train_prepared(preparation).metadata[
        "evaluation_mode"
    ] == "temporal_holdout"


def test_training_preparation_isolated_from_later_source_frame_changes():
    trainer, source, preparation = _temporal_preparation()
    expected_working = preparation.working.copy(deep=True)
    expected_hash = preparation.input_data_hash
    source.loc[:, "wind_speed_truth"] += 1000.0
    source.loc[:, "timestamp"] += pd.Timedelta(days=60)

    result = trainer.train_prepared(preparation)

    pd.testing.assert_frame_equal(preparation.working, expected_working)
    assert result.metadata["input_data_hash"] == expected_hash


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
        @classmethod
        def training_contract_descriptor(cls):
            return {"type": "recording-mean-residual-estimator"}

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
    assert result.training_outcome == "operational_fallback"
    assert result.metadata["training_outcome"] == "operational_fallback"
    prediction = result.bundle.model.predict(np.zeros((2, 1)))
    assert prediction.tolist() == [1.5, 1.5]


def test_constant_residual_is_a_data_fallback_not_an_operational_failure():
    result = ResidualTrainer(estimator_factory=constant_factory).train_target(
        make_training_frame(residuals=(1.0, 1.0, 1.0, 1.0)),
        target="wind_speed",
    )

    assert result.training_outcome == "data_fallback"
    assert result.metadata["training_outcome"] == "data_fallback"


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
    assert not predicted.loc[predicted.index[0], "used_ai"]
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


def test_training_parameter_metadata_never_persists_secret_object_content():
    result = ResidualTrainer(
        estimator_factory=SecretParameterEstimator
    ).train_target(
        make_training_frame(residuals=(0.0, 1.0, 2.0, 3.0)),
        target="wind_speed",
    )

    serialized = json.dumps(result.metadata, sort_keys=True)

    assert "secret-123" not in serialized
    parameters = result.metadata["training_params"]["parameters"]
    assert parameters["diagnostic_frame"]["type"].endswith("DataFrame")
    assert parameters["diagnostic_frame"]["shape"] == [1, 1]
    assert len(parameters["diagnostic_frame"]["content_hash"]) == 64
    assert parameters["artifact_path"]["type"].endswith("Path")
    assert parameters["opaque"] == {
        "type": f"{SecretReprObject.__module__}.SecretReprObject"
    }


def test_training_parameter_serialization_is_bounded_and_cycle_safe():
    cyclic = []
    cyclic.append(cyclic)
    payload = {
        "long": "x" * 10_000,
        "many": list(range(1_000)),
        "cyclic": cyclic,
    }

    serialized = json.dumps(ai_training._json_safe_training_value(payload))

    assert len(serialized) < 10_000
    assert "x" * 1_000 not in serialized
    assert "truncated" in serialized or "content_hash" in serialized


def test_sensitive_non_finite_training_parameter_is_json_safe():
    value = ai_training._json_safe_training_value(
        {"service_token": float("nan")}
    )

    serialized = json.dumps(value, allow_nan=False, sort_keys=True)

    assert "NaN" not in serialized
    assert value["service_token"]["type"].endswith("float")
    assert len(value["service_token"]["content_hash"]) == 64


def test_long_sensitive_parameter_key_never_leaks_value():
    secret = "VALUE-MUST-NOT-PERSIST"
    key = "customer_password_" + "x" * 256

    value = ai_training._json_safe_training_value({key: secret})
    serialized = json.dumps(value, allow_nan=False, sort_keys=True)

    assert secret not in serialized
    assert "password" not in serialized


def test_sensitive_string_subclass_key_cannot_override_detection():
    secret = "VALUE-MUST-NOT-PERSIST"
    lower_calls = []

    class MisleadingSensitiveKey(str):
        def lower(self):
            lower_calls.append(True)
            return "ordinary"

    value = ai_training._json_safe_training_value(
        {MisleadingSensitiveKey("service_token"): secret}
    )
    serialized = json.dumps(value, allow_nan=False, sort_keys=True)

    assert lower_calls == []
    assert secret not in serialized


def test_series_with_untrusted_dtype_never_calls_dtype_stringifier():
    secret = "VALUE-MUST-NOT-PERSIST"

    class SecretDType:
        def __str__(self):
            return secret

    class ForgedDTypeSeries(pd.Series):
        @property
        def dtype(self):
            return SecretDType()

    value = ai_training._json_safe_training_value(ForgedDTypeSeries([1.0]))
    serialized = json.dumps(value, allow_nan=False, sort_keys=True)

    assert secret not in serialized
    assert value["dtype"] == {
        "type": f"{SecretDType.__module__}.{SecretDType.__qualname__}"
    }


def test_ascii_bytes_sensitive_parameter_key_never_leaks_value():
    secret = "VALUE-MUST-NOT-PERSIST"

    value = ai_training._json_safe_training_value(
        {b"customer_password": secret}
    )
    serialized = json.dumps(value, allow_nan=False, sort_keys=True)

    assert secret not in serialized


def test_unknown_training_parameter_does_not_persist_dtype_content():
    secret = "VALUE-MUST-NOT-PERSIST"

    class SecretDType:
        name = secret

    class OpaqueParameter:
        dtype = SecretDType()

    value = ai_training._json_safe_training_value(OpaqueParameter())
    serialized = json.dumps(value, allow_nan=False, sort_keys=True)

    assert secret not in serialized
    assert value == {
        "type": f"{OpaqueParameter.__module__}.{OpaqueParameter.__qualname__}"
    }


def test_training_parameter_serialization_has_global_node_budget():
    shared = {"payload": "x" * 256}
    payload = shared
    for _ in range(5):
        payload = [payload] * 5

    value = ai_training._json_safe_training_value(payload)
    serialized = json.dumps(value, allow_nan=False, sort_keys=True)

    assert len(serialized) < 100_000
    assert "max_nodes" in serialized


def test_training_parameter_mapping_iteration_is_bounded_even_if_len_lies():
    class LyingMapping(dict):
        def __len__(self):
            return 0

    payload = LyingMapping({str(index): index for index in range(200)})

    value = ai_training._json_safe_training_value(payload)
    serialized = json.dumps(value, allow_nan=False, sort_keys=True)

    assert len(value) <= ai_training._MAX_TRAINING_METADATA_ITEMS + 1
    assert "max_items" in serialized


def test_wide_dataframe_training_summary_is_bounded():
    frame = pd.DataFrame(columns=[f"column-{index}" for index in range(200)])

    value = ai_training._json_safe_training_value(frame)
    serialized = json.dumps(value, allow_nan=False, sort_keys=True)

    assert value["column_count"] == 200
    assert len(value["column_hashes"]) <= ai_training._MAX_TRAINING_METADATA_ITEMS
    assert value["truncated"] == "max_columns"
    assert len(serialized) < 20_000


def test_huge_integer_training_parameters_are_json_safe_and_bounded():
    huge = 10**5_000

    value = ai_training._json_safe_training_value(
        {"ordinary": huge, "service_token": huge}
    )
    serialized = json.dumps(value, allow_nan=False, sort_keys=True)

    assert len(serialized) < 2_000
    assert value["ordinary"]["type"].endswith("int")
    assert len(value["ordinary"]["content_hash"]) == 64
    assert len(value["service_token"]["content_hash"]) == 64


def test_training_parameter_serialization_handles_self_referential_dataclass():
    @dataclass
    class RecursiveParameter:
        child: object = None

    parameter = RecursiveParameter()
    parameter.child = parameter

    value = ai_training._json_safe_training_value(parameter)
    serialized = json.dumps(value, allow_nan=False, sort_keys=True)

    assert len(serialized) < 1_000
    assert len(value["content_hash"]) == 64


def test_training_persists_feature_cadence_in_bundle_and_metadata():
    trainer = ResidualTrainer(
        estimator_factory=constant_factory,
        feature_builder=FeatureBuilder(cadence_minutes=60),
    )

    result = trainer.train_target(make_training_frame(), target="wind_speed")

    assert result.bundle.cadence_minutes == 60.0
    assert result.metadata["cadence_minutes"] == 60.0
