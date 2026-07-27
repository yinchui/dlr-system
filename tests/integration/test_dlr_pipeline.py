import hashlib
import json
from dataclasses import replace
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from affine import Affine
from pyproj import CRS
from rasterio.coords import BoundingBox
from sklearn.dummy import DummyRegressor
from xgboost import XGBRegressor

import modules.ai_training as ai_training
from modules import dlr_pipeline as dlr_pipeline_module
from modules.ai_prediction import FeatureBuilder, ModelBundle, ResidualPredictor
from modules.ai_training import ResidualTrainer
from modules.dlr_pipeline import (
    DlrPipeline,
    DlrPipelineResult,
    LongFrameThermalAdapter,
    ModelRunReport,
    derive_line_id,
)
from modules.model_registry import ModelKey, ModelLoadResult, ModelRegistry
from modules.terrain import DemGrid
from modules.weather_correction import CorrectionOptions


def _weather(role: str, *, truth_offset: bool = False) -> pd.DataFrame:
    timestamps = pd.to_datetime(
        [
            "2026-07-23 00:00",
            "2026-07-23 00:30",
            "2026-07-23 00:00",
            "2026-07-23 00:30",
        ]
    ).tz_localize("Asia/Shanghai")
    wind = np.array([2.0, 2.5, 3.0, 3.5])
    temp = np.array([30.0, 31.0, 28.0, 29.0])
    if truth_offset:
        wind = wind + 1.0
        temp = temp - 2.0
    return pd.DataFrame(
        {
            "tower_id": ["001", "001", "002", "002"],
            "timestamp": timestamps,
            "ambient_temp": temp,
            "wind_speed": wind,
            "wind_direction": [90.0, 100.0, 110.0, 120.0],
            "solar_radiation": [0.0, 10.0, 0.0, 20.0],
            "humidity": [30.0, 31.0, 32.0, 33.0],
            "elevation": [1000.0, 1000.0, 1200.0, 1200.0],
            "dataset_role": role,
            "source_file_hash": [f"{role}-source"] * 4,
        }
    )


def _conductor() -> dict:
    return {
        "D0": 0.0281,
        "R_low_25": 7.283e-5,
        "R_high_75": 8.688e-5,
        "R_high_200": 1.220e-4,
        "emissivity": 0.8,
        "absorptivity": 0.8,
        "max_allow_temp": 80.0,
        "latitude": 39.9,
        "longitude": 116.4,
        "line_azimuth": 90.0,
    }


class _LinearResidualRegressor:
    def __init__(self, **kwargs):
        self.parameters = kwargs

    def fit(self, features, target):
        physical = np.asarray(features.iloc[:, 0], dtype=float)
        design = np.column_stack([physical, np.ones(len(physical))])
        self.coefficients = np.linalg.lstsq(
            design, np.asarray(target, dtype=float), rcond=None
        )[0]
        return self

    def predict(self, features):
        physical = np.asarray(features.iloc[:, 0], dtype=float)
        return self.coefficients[0] * physical + self.coefficients[1]


def test_derived_line_id_is_stable_across_weather_and_coordinate_order():
    weather = _weather("physical").assign(
        longitude=[120.1, 120.1, 120.2, 120.2],
        latitude=[40.1, 40.1, 40.2, 40.2],
    )
    coordinates = {
        "002": {"lon": 120.2, "lat": 40.2},
        "001": {"lon": 120.1, "lat": 40.1},
    }

    expected = derive_line_id(weather, tower_coords=coordinates)
    reordered = derive_line_id(
        weather.iloc[::-1].reset_index(drop=True),
        tower_coords=dict(reversed(list(coordinates.items()))),
    )

    assert expected == reordered
    assert expected.startswith("line-")
    shifted_weather_coordinates = weather.assign(
        longitude=weather["longitude"] + 0.001,
        latitude=weather["latitude"] + 0.001,
    )
    assert expected == derive_line_id(
        shifted_weather_coordinates,
        tower_coords=coordinates,
    )
    assert derive_line_id(weather, tower_coords={}) != derive_line_id(
        shifted_weather_coordinates,
        tower_coords={},
    )
    assert expected != derive_line_id(
        weather,
        tower_coords=coordinates
        | {"002": {"lon": 120.2001, "lat": 40.2}},
    )
    assert expected != derive_line_id(
        pd.concat(
            [
                weather,
                weather.iloc[[0]].assign(tower_id="003", longitude=120.3),
            ],
            ignore_index=True,
        ),
        tower_coords=coordinates
        | {"003": {"lon": 120.3, "lat": 40.3}},
    )


def test_derived_line_id_without_coordinates_uses_stable_source_lineage():
    first = _weather("physical")
    second = first.iloc[::-1].copy()
    second["source_file_hash"] = "another-weather-upload"

    assert derive_line_id(first, tower_coords={}) == derive_line_id(
        first.iloc[::-1].reset_index(drop=True),
        tower_coords=None,
    )
    assert derive_line_id(first, tower_coords={}) != derive_line_id(
        second,
        tower_coords=None,
    )
    assert derive_line_id(first, tower_coords={}) != derive_line_id(
        first.loc[first["tower_id"] == "001"],
        tower_coords={},
    )


def test_derived_line_id_without_coordinates_or_lineage_uses_weather_content():
    first = _weather("physical").drop(columns="source_file_hash")
    changed = first.copy(deep=True)
    changed["ambient_temp"] += 10.0

    assert derive_line_id(first, tower_coords={}) == derive_line_id(
        first.iloc[::-1].reset_index(drop=True),
        tower_coords=None,
    )
    assert derive_line_id(first, tower_coords={}) != derive_line_id(
        changed,
        tower_coords=None,
    )


def test_partial_coordinates_make_derived_line_identity_nonpersistent():
    physical = _weather("physical")

    identity = dlr_pipeline_module.derive_line_identity(
        physical,
        tower_coords={"001": {"lon": 120.1, "lat": 40.1}},
    )

    assert identity.persistence_allowed is False
    assert identity.reason == "missing_coordinates"


def test_ambiguous_or_invalid_coordinates_are_not_persistent():
    physical = _weather("physical").assign(
        longitude=[120.1, 120.2, 121.0, 121.0],
        latitude=[40.1, 40.2, 41.0, 41.0],
    )

    ambiguous = dlr_pipeline_module.derive_line_identity(
        physical,
        tower_coords={},
    )
    invalid = dlr_pipeline_module.derive_line_identity(
        _weather("physical"),
        tower_coords={
            "001": {"lon": 999.0, "lat": 40.1},
            "002": {"lon": 121.0, "lat": 41.0},
        },
    )
    authoritative = dlr_pipeline_module.derive_line_identity(
        physical,
        tower_coords={
            "001": {"lon": 120.1, "lat": 40.1},
            "002": {"lon": 121.0, "lat": 41.0},
        },
    )

    assert ambiguous.persistence_allowed is False
    assert ambiguous.reason == "ambiguous_coordinates"
    assert invalid.persistence_allowed is False
    assert authoritative.persistence_allowed is True


def test_complete_coordinate_identity_is_stable_across_new_weather_batches():
    coordinates = {
        "001": {"lon": 120.1, "lat": 40.1},
        "002": {"lon": 121.0, "lat": 41.0},
    }
    first = _weather("physical")
    later = first.iloc[::-1].reset_index(drop=True)
    later["timestamp"] += pd.Timedelta(days=7)
    later["ambient_temp"] += 8.0
    later["source_file_hash"] = "new-export"

    first_identity = dlr_pipeline_module.derive_line_identity(
        first, tower_coords=coordinates
    )
    later_identity = dlr_pipeline_module.derive_line_identity(
        later, tower_coords=coordinates
    )
    other_line = dlr_pipeline_module.derive_line_identity(
        later,
        tower_coords=coordinates
        | {"002": {"lon": 121.001, "lat": 41.0}},
    )

    assert first_identity.persistence_allowed is True
    assert later_identity.persistence_allowed is True
    assert first_identity.line_id == later_identity.line_id
    assert other_line.line_id != first_identity.line_id


def test_nonpersistent_identity_uses_models_only_within_the_current_run(
    tmp_path,
):
    pipeline = DlrPipeline(model_root=tmp_path)
    identity = dlr_pipeline_module.derive_line_identity(
        _weather("physical"),
        tower_coords={"001": {"lon": 120.1, "lat": 40.1}},
    )
    run_kwargs = {
        "physical": _weather("physical"),
        "project_id": "project-a",
        "line_id": identity.line_id,
        "model_persistence_allowed": identity.persistence_allowed,
        "terrain_lookup": {},
        "ai_enabled": True,
        "conductor": _conductor(),
    }

    trained = pipeline.run(
        **run_kwargs,
        truth=_weather("truth", truth_offset=True),
    )
    later = pipeline.run(**run_kwargs, truth=None)

    assert len(trained.model_report.trained_targets) == 4
    assert len(trained.model_report.used_targets) == 4
    assert trained.model_report.loaded_targets == ()
    assert trained.comparison_weather[
        ["wind_speed_used_ai", "ambient_temp_used_ai"]
    ].to_numpy().all()
    assert later.model_report.loaded_targets == ()
    assert later.model_report.trained_targets == ()
    assert later.model_report.used_targets == ()
    assert pipeline.registry is None
    assert list(tmp_path.iterdir()) == []


def test_nonpersistent_identity_rejects_custom_estimator_backend(
    tmp_path,
):
    class ZeroResidualEstimator:
        def fit(self, features, target):
            return self

        def predict(self, features):
            return np.zeros(len(features), dtype=float)

    physical = _weather("physical")
    truth = _weather("truth", truth_offset=True)
    truth["wind_speed"] = physical["wind_speed"] + [-1.0, 1.0, -1.0, 1.0]
    truth["ambient_temp"] = physical["ambient_temp"] + [-1.0, 1.0, -1.0, 1.0]
    identity = dlr_pipeline_module.derive_line_identity(
        physical,
        tower_coords={"001": {"lon": 120.1, "lat": 40.1}},
    )
    result = DlrPipeline(
        model_root=tmp_path,
        trainer=ResidualTrainer(estimator_factory=ZeroResidualEstimator),
    ).run(
        physical=physical,
        truth=truth,
        project_id="project-a",
        line_id=identity.line_id,
        model_persistence_allowed=identity.persistence_allowed,
        terrain_lookup={},
        correction_options=CorrectionOptions(
            enable_vertical=False,
            enable_terrain=False,
            enable_desert=False,
            enable_wind_direction=False,
        ),
        ai_enabled=True,
        conductor=_conductor(),
    )

    assert result.model_report.trained_targets == ()
    assert result.model_report.used_targets == ()
    assert sum(
        fallback.reason == "unsupported_training_backend"
        for fallback in result.model_report.fallbacks
    ) == 4
    assert result.model_report.promotion_decisions == ()
    assert list(tmp_path.iterdir()) == []


def test_nonpersistent_training_rejects_descriptorless_backend(tmp_path):
    class DescriptorlessTrainer:
        def __init__(self):
            self.delegate = ResidualTrainer()
            self.feature_builder = self.delegate.feature_builder
            self.training_calls = []

        def prepare_target(self, frame, target, **kwargs):
            return self.delegate.prepare_target(frame, target, **kwargs)

        def train_prepared(self, preparation):
            self.training_calls.append(
                (preparation.tower_id, preparation.target)
            )
            return self.delegate.train_prepared(preparation)

        def train_target(self, frame, target, **kwargs):
            self.training_calls.append(
                (str(frame["tower_id"].iloc[0]), target)
            )
            return self.delegate.train_target(frame, target, **kwargs)

    trainer = DescriptorlessTrainer()
    pipeline = DlrPipeline(model_root=tmp_path, trainer=trainer)

    result = pipeline.run(
        physical=_weather("physical"),
        truth=_weather("truth", truth_offset=True),
        project_id="project-a",
        line_id="transient-line",
        model_persistence_allowed=False,
        terrain_lookup={},
        ai_enabled=True,
        conductor=_conductor(),
    )

    comparison = result.comparison_weather
    assert trainer.training_calls == []
    assert result.model_report.trained_targets == ()
    assert result.model_report.used_targets == ()
    assert sum(
        fallback.reason == "unsupported_training_backend"
        for fallback in result.model_report.fallbacks
    ) == 4
    np.testing.assert_array_equal(
        comparison["wind_speed_ai"], comparison["wind_speed_physical"]
    )
    np.testing.assert_array_equal(
        comparison["ambient_temp_ai"], comparison["ambient_temp_physical"]
    )
    assert result.max_currents.shape == (2, 2)
    assert pipeline.registry is None
    assert list(tmp_path.iterdir()) == []


def test_complete_coordinate_identity_reuses_models_for_new_weather_batch(
    tmp_path,
):
    coordinates = {
        "001": {"lon": 120.1, "lat": 40.1},
        "002": {"lon": 121.0, "lat": 41.0},
    }
    physical = _weather("physical")
    first_identity = dlr_pipeline_module.derive_line_identity(
        physical, tower_coords=coordinates
    )
    first = DlrPipeline(model_root=tmp_path).run(
        physical=physical,
        truth=_weather("truth", truth_offset=True),
        project_id="project-a",
        line_id=first_identity.line_id,
        model_persistence_allowed=first_identity.persistence_allowed,
        terrain_lookup={},
        coordinate_context=coordinates,
        ai_enabled=True,
        conductor=_conductor(),
    )
    later_physical = physical.copy(deep=True)
    later_physical["timestamp"] += pd.Timedelta(days=7)
    later_physical["source_file_hash"] = "new-export"
    later_identity = dlr_pipeline_module.derive_line_identity(
        later_physical, tower_coords=coordinates
    )

    later = DlrPipeline(model_root=tmp_path).run(
        physical=later_physical,
        project_id="project-a",
        line_id=later_identity.line_id,
        model_persistence_allowed=later_identity.persistence_allowed,
        terrain_lookup={},
        coordinate_context=coordinates,
        ai_enabled=True,
        conductor=_conductor(),
    )

    assert later_identity.line_id == first_identity.line_id
    assert later.model_report.loaded_targets == first.model_report.trained_targets
    assert later.model_report.used_targets == first.model_report.trained_targets


def _dem_context(values, *, crs="EPSG:4326") -> DemGrid:
    elevation = np.asarray(values)
    return DemGrid(
        elevation=elevation,
        mask=np.zeros(elevation.shape, dtype=bool),
        crs=CRS.from_user_input(crs),
        transform=Affine(0.01, 0.0, 120.0, 0.0, -0.01, 40.0),
        bounds=BoundingBox(120.0, 39.98, 120.02, 40.0),
        nodata=None,
    )


def _compatibility_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "tower_id": ["001", "002"],
            "elevation": [1000.0, 1200.0],
            "slope": [0.0, 0.0],
            "aspect": [0.0, 0.0],
            "source": ["dem", "dem"],
            "reason": [None, None],
        }
    )


def test_runtime_compatibility_hashes_actual_dem_crs_and_coordinates():
    frame = _compatibility_frame()
    coordinates = {
        "002": {"lon": 120.2, "lat": 40.2},
        "001": {"lon": 120.1, "lat": 40.1},
    }

    def compatibility(dem, coordinate_context=coordinates):
        return DlrPipeline._compatibility(
            frame,
            conductor=_conductor(),
            correction_options=CorrectionOptions(),
            interval_minutes=30,
            dem_context=dem,
            coordinate_context=coordinate_context,
        )

    baseline = compatibility(
        _dem_context(np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype="float32"))
    )
    reordered = compatibility(
        _dem_context(
            np.asfortranarray(
                np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype="float32")
            )
        ),
        {
            "001": {"lat": 40.1, "lon": 120.1},
            "002": {"lat": 40.2, "lon": 120.2},
        },
    )
    changed_content = compatibility(
        _dem_context(np.asarray([[1.0, 2.0], [3.0, 4.1]], dtype="float32"))
    )
    changed_dtype = compatibility(
        _dem_context(np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype="float64"))
    )
    changed_shape = compatibility(
        _dem_context(np.asarray([[1.0, 2.0, 3.0]], dtype="float32"))
    )
    changed_crs = compatibility(
        _dem_context(
            np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype="float32"),
            crs="EPSG:3857",
        )
    )
    changed_coordinates = compatibility(
        _dem_context(np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype="float32")),
        coordinates | {"002": {"lon": 120.2001, "lat": 40.2}},
    )

    assert baseline == reordered
    assert baseline.dem_hash != changed_content.dem_hash
    assert baseline.dem_hash != changed_dtype.dem_hash
    assert baseline.dem_hash != changed_shape.dem_hash
    assert baseline.dem_hash == changed_crs.dem_hash
    assert baseline.crs_hash != changed_crs.crs_hash
    assert baseline.coordinate_hash != changed_coordinates.coordinate_hash


def test_runtime_context_changes_invalidate_only_the_matching_model_field(
    tmp_path,
):
    pipeline = DlrPipeline(model_root=tmp_path)
    base_dem = _dem_context(
        np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype="float32")
    )
    coordinates = {
        "001": {"lon": 120.1, "lat": 40.1},
        "002": {"lon": 120.2, "lat": 40.2},
    }
    pipeline.run(
        physical=_weather("physical"),
        truth=_weather("truth", truth_offset=True),
        project_id="project-a",
        line_id="line-a",
        terrain_lookup={},
        dem_context=base_dem,
        coordinate_context=coordinates,
        ai_enabled=True,
        conductor=_conductor(),
    )

    cases = (
        (
            _dem_context(
                np.asarray([[1.0, 2.0], [3.0, 4.1]], dtype="float32")
            ),
            coordinates,
            "incompatible_dem_hash",
        ),
        (
            _dem_context(
                np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype="float32"),
                crs="EPSG:3857",
            ),
            coordinates,
            "incompatible_crs_hash",
        ),
        (
            base_dem,
            coordinates | {"002": {"lon": 120.2001, "lat": 40.2}},
            "incompatible_coordinate_hash",
        ),
    )
    for dem_context, coordinate_context, reason in cases:
        result = pipeline.run(
            physical=_weather("physical"),
            project_id="project-a",
            line_id="line-a",
            terrain_lookup={},
            dem_context=dem_context,
            coordinate_context=coordinate_context,
            ai_enabled=True,
            conductor=_conductor(),
        )
        assert not result.model_report.loaded_targets
        assert sum(
            fallback.reason == reason
            for fallback in result.model_report.fallbacks
        ) == 4


def test_pipeline_trains_missing_models_then_reuses_them(tmp_path):
    pipeline = DlrPipeline(model_root=tmp_path)
    terrain = {
        "001": {"elevation": 1000.0, "slope": 0.0, "aspect": 0.0},
        "002": {"elevation": 1200.0, "slope": 0.0, "aspect": 0.0},
    }

    first = pipeline.run(
        physical=_weather("physical"),
        truth=_weather("truth", truth_offset=True),
        project_id="project-a",
        line_id="line-a",
        interval_minutes=30,
        terrain_lookup=terrain,
        ai_enabled=True,
        conductor=_conductor(),
        truth_tolerance="5min",
    )
    second = pipeline.run(
        physical=_weather("physical"),
        truth=None,
        project_id="project-a",
        line_id="line-a",
        interval_minutes=30,
        terrain_lookup=terrain,
        ai_enabled=True,
        conductor=_conductor(),
    )

    assert first.model_report.trained_targets
    assert second.model_report.loaded_targets == first.model_report.trained_targets
    assert first.model_report.active_model_count == 4
    np.testing.assert_allclose(first.max_currents, second.max_currents)


def test_missing_sentinel_terrain_falls_back_to_finite_physical_dlr(tmp_path):
    terrain = {
        tower_id: {"elevation": 1000.0, "slope": -1.0e30, "aspect": 0.0}
        for tower_id in ("001", "002")
    }

    result = DlrPipeline(model_root=tmp_path).run(
        physical=_weather("physical"),
        truth=_weather("truth", truth_offset=True),
        project_id="project-a",
        line_id="line-a",
        interval_minutes=30,
        terrain_lookup=terrain,
        ai_enabled=True,
        conductor=_conductor(),
        truth_tolerance="5min",
    )

    assert result.model_report.trained_targets == ()
    assert result.model_report.used_targets == ()
    assert result.model_report.active_model_count == 0
    assert sum(
        fallback.reason == "training_failed:ValueError"
        for fallback in result.model_report.fallbacks
    ) == 4
    assert not result.comparison_weather[
        ["wind_speed_used_ai", "ambient_temp_used_ai"]
    ].to_numpy(dtype=bool).any()
    np.testing.assert_array_equal(
        result.comparison_weather["wind_speed_ai"],
        result.comparison_weather["wind_speed_physical"],
    )
    np.testing.assert_array_equal(
        result.comparison_weather["ambient_temp_ai"],
        result.comparison_weather["ambient_temp_physical"],
    )
    assert result.max_currents.size > 0
    assert np.isfinite(result.max_currents).all()


class _CountingTrainer:
    def __init__(self):
        self.delegate = ResidualTrainer()
        self.feature_builder = self.delegate.feature_builder
        self.calls = []

    def train_target(self, frame, target, **kwargs):
        self.calls.append((str(frame["tower_id"].iloc[0]), target))
        return self.delegate.train_target(frame, target, **kwargs)

    def prepare_target(self, frame, target, **kwargs):
        return self.delegate.prepare_target(frame, target, **kwargs)

    def train_prepared(self, preparation):
        self.calls.append((preparation.tower_id, preparation.target))
        return self.delegate.train_prepared(preparation)

class _ClaimingResidualTrainer(ResidualTrainer):
    def __init__(self):
        super().__init__()
        self.calls = []

    def prepare_target(self, frame, target, **kwargs):
        self.calls.append((str(frame["tower_id"].iloc[0]), target))
        return super().prepare_target(frame, target, **kwargs)


def _expected_pipeline_model_keys():
    return tuple(
        ModelKey("project-a", "line-a", tower_id, target)
        for tower_id in ("001", "002")
        for target in ("wind_speed", "ambient_temp")
    )


def _assert_unsupported_training_backend(result, expected_keys):
    assert result.model_report.loaded_targets == ()
    assert result.model_report.trained_targets == ()
    assert result.model_report.used_targets == ()
    assert result.model_report.promotion_decisions == ()
    assert tuple(
        (fallback.key, fallback.reason) for fallback in result.model_report.fallbacks
    ) == tuple((key, "unsupported_training_backend") for key in expected_keys)
    assert result.max_currents.size > 0
    assert np.isfinite(result.max_currents).all()
    assert (
        not result.comparison_weather[
            ["wind_speed_used_ai", "ambient_temp_used_ai"]
        ]
        .to_numpy(dtype=bool)
        .any()
    )


@pytest.mark.parametrize("model_persistence_allowed", [True, False])
def test_custom_trainer_cannot_train_or_persist(
    tmp_path,
    model_persistence_allowed,
):
    model_root = tmp_path / "models"
    trainer = _CountingTrainer()
    pipeline = DlrPipeline(model_root=model_root, trainer=trainer)

    result = pipeline.run(
        physical=_weather("physical"),
        truth=_weather("truth", truth_offset=True),
        project_id="project-a",
        line_id="line-a",
        model_persistence_allowed=model_persistence_allowed,
        terrain_lookup={},
        ai_enabled=True,
        conductor=_conductor(),
    )

    expected_keys = _expected_pipeline_model_keys()
    assert trainer.calls == []
    _assert_unsupported_training_backend(result, expected_keys)
    assert not list(model_root.rglob("model.joblib"))
    assert not list(model_root.rglob("*.attempts.json"))
    if model_persistence_allowed:
        assert pipeline.registry is not None
    else:
        assert pipeline.registry is None
        assert not model_root.exists()


def test_custom_estimator_factory_cannot_enter_pipeline_persistence(tmp_path):
    model_root = tmp_path / "models"
    trainer = ResidualTrainer(estimator_factory=_LinearResidualRegressor)
    prepare_calls = []
    original_prepare = trainer.prepare_target

    def counting_prepare(*args, **kwargs):
        prepare_calls.append((args, kwargs))
        return original_prepare(*args, **kwargs)

    trainer.prepare_target = counting_prepare

    result = DlrPipeline(model_root=model_root, trainer=trainer).run(
        physical=_weather("physical"),
        truth=_weather("truth", truth_offset=True),
        project_id="project-a",
        line_id="line-a",
        terrain_lookup={},
        ai_enabled=True,
        conductor=_conductor(),
    )

    expected_keys = _expected_pipeline_model_keys()
    assert prepare_calls == []
    _assert_unsupported_training_backend(result, expected_keys)
    assert not list(model_root.rglob("model.joblib"))
    assert not list(model_root.rglob("*.attempts.json"))


def test_mutated_exact_trainer_cannot_replace_persisted_model(
    tmp_path,
    monkeypatch,
):
    model_root = tmp_path / "models"
    registry = ModelRegistry(model_root)
    trainer = ResidualTrainer()
    training_calls = []
    original_train_prepared = trainer.train_prepared

    def forged_train_prepared(preparation):
        training_calls.append((preparation.tower_id, preparation.target))
        result = original_train_prepared(preparation)
        model = DummyRegressor(strategy="constant", constant=0.0)
        model.fit(
            np.zeros((1, len(result.bundle.feature_columns)), dtype=float),
            np.zeros(1, dtype=float),
        )
        return replace(
            result,
            bundle=replace(result.bundle, model=model),
        )

    monkeypatch.setattr(trainer, "train_prepared", forged_train_prepared)

    result = DlrPipeline(registry=registry, trainer=trainer).run(
        physical=_weather("physical"),
        truth=_weather("truth", truth_offset=True),
        project_id="project-a",
        line_id="line-a",
        terrain_lookup={},
        ai_enabled=True,
        conductor=_conductor(),
    )

    expected_keys = _expected_pipeline_model_keys()
    assert training_calls == []
    _assert_unsupported_training_backend(result, expected_keys)
    for key in expected_keys:
        assert not registry.path_for(key).exists()
        assert not registry.metadata_path_for(key).exists()
        assert not registry.manifest_path_for(key).exists()
        assert not registry.attempt_path_for(key).exists()


def test_mutated_exact_trainer_prepare_is_rejected_without_execution(
    tmp_path,
    monkeypatch,
):
    model_root = tmp_path / "models"
    registry = ModelRegistry(model_root)
    trainer = ResidualTrainer()
    preparation_calls = []
    original_prepare_target = trainer.prepare_target

    def forged_prepare_target(frame, target, **kwargs):
        preparation_calls.append((str(frame["tower_id"].iloc[0]), target))
        return original_prepare_target(frame, target, **kwargs)

    monkeypatch.setattr(trainer, "prepare_target", forged_prepare_target)

    result = DlrPipeline(registry=registry, trainer=trainer).run(
        physical=_weather("physical"),
        truth=_weather("truth", truth_offset=True),
        project_id="project-a",
        line_id="line-a",
        terrain_lookup={},
        ai_enabled=True,
        conductor=_conductor(),
    )

    expected_keys = _expected_pipeline_model_keys()
    assert preparation_calls == []
    _assert_unsupported_training_backend(result, expected_keys)
    for key in expected_keys:
        assert not registry.path_for(key).exists()
        assert not registry.metadata_path_for(key).exists()
        assert not registry.manifest_path_for(key).exists()
        assert not registry.attempt_path_for(key).exists()


def test_mutated_exact_trainer_cannot_poison_sealed_rejection_cache(
    tmp_path,
    monkeypatch,
):
    model_root = tmp_path / "models"
    registry = ModelRegistry(model_root)
    trainer = ResidualTrainer()
    training_calls = []
    original_train_prepared = trainer.train_prepared

    def forged_train_prepared(preparation):
        training_calls.append((preparation.tower_id, preparation.target))
        result = original_train_prepared(preparation)
        full_fit_metrics = dict(result.metadata["full_fit_metrics"])
        full_fit_metrics["corrected_mae"] = full_fit_metrics["baseline_mae"]
        full_fit_metrics["corrected_rmse"] = full_fit_metrics["baseline_rmse"]
        return replace(
            result,
            metadata={
                **result.metadata,
                "full_fit_metrics": full_fit_metrics,
            },
        )

    monkeypatch.setattr(trainer, "train_prepared", forged_train_prepared)
    run_kwargs = {
        "physical": _weather("physical"),
        "truth": _weather("truth", truth_offset=True),
        "project_id": "project-a",
        "line_id": "line-a",
        "terrain_lookup": {},
        "ai_enabled": True,
        "conductor": _conductor(),
    }

    rejected = DlrPipeline(registry=registry, trainer=trainer).run(**run_kwargs)

    expected_keys = _expected_pipeline_model_keys()
    assert training_calls == []
    _assert_unsupported_training_backend(rejected, expected_keys)
    assert all(not registry.attempt_path_for(key).exists() for key in expected_keys)

    recovered = DlrPipeline(model_root=model_root).run(**run_kwargs)

    assert recovered.model_report.trained_targets == expected_keys
    assert recovered.model_report.used_targets == expected_keys
    assert all(
        decision.promoted
        for decision in recovered.model_report.promotion_decisions
    )


@pytest.mark.parametrize(
    "trainer_factory",
    [_ClaimingResidualTrainer, _CountingTrainer],
    ids=["subclass", "duck"],
)
def test_unsupported_training_backend_rejects_claiming_trainers(
    tmp_path,
    trainer_factory,
):
    trainer = trainer_factory()
    trainer.production_eligible = True

    result = DlrPipeline(
        model_root=tmp_path / "models",
        trainer=trainer,
    ).run(
        physical=_weather("physical"),
        truth=_weather("truth", truth_offset=True),
        project_id="project-a",
        line_id="line-a",
        terrain_lookup={},
        ai_enabled=True,
        conductor=_conductor(),
    )

    assert trainer.calls == []
    _assert_unsupported_training_backend(
        result,
        _expected_pipeline_model_keys(),
    )


def test_unsupported_training_backend_is_not_cached_and_sealed_retry_succeeds(
    tmp_path,
):
    model_root = tmp_path / "models"
    run_kwargs = {
        "physical": _weather("physical"),
        "truth": _weather("truth", truth_offset=True),
        "project_id": "project-a",
        "line_id": "line-a",
        "terrain_lookup": {},
        "ai_enabled": True,
        "conductor": _conductor(),
    }

    rejected = DlrPipeline(
        model_root=model_root,
        trainer=_CountingTrainer(),
    ).run(**run_kwargs)
    assert rejected.model_report.promotion_decisions == ()
    assert not list(model_root.rglob("model.joblib"))
    assert not list(model_root.rglob("*.attempts.json"))

    recovered = DlrPipeline(
        model_root=model_root,
        trainer=ResidualTrainer(),
    ).run(**run_kwargs)

    expected_keys = _expected_pipeline_model_keys()
    assert recovered.model_report.trained_targets == expected_keys
    assert recovered.model_report.used_targets == expected_keys
    assert all(
        decision.promoted
        for decision in recovered.model_report.promotion_decisions
    )


def test_custom_trainer_without_training_need_keeps_physical_fallback(tmp_path):
    trainer = _CountingTrainer()

    result = DlrPipeline(
        model_root=tmp_path / "models",
        trainer=trainer,
    ).run(
        physical=_weather("physical"),
        truth=None,
        project_id="project-a",
        line_id="line-a",
        terrain_lookup={},
        ai_enabled=True,
        conductor=_conductor(),
    )

    assert trainer.calls == []
    assert result.model_report.trained_targets == ()
    assert result.model_report.used_targets == ()
    assert all(
        fallback.reason != "unsupported_training_backend"
        for fallback in result.model_report.fallbacks
    )
    assert result.max_currents.size > 0
    assert np.isfinite(result.max_currents).all()


class _PoorTemporalTrainer(_CountingTrainer):
    def train_prepared(self, preparation):
        result = super().train_prepared(preparation)
        metrics = dict(result.metrics)
        if not metrics:
            return result
        metrics["corrected_mae"] = metrics["baseline_mae"]
        metrics["corrected_rmse"] = metrics["baseline_rmse"]
        return replace(result, metrics=metrics)


class _AlwaysPoorTrainer(_CountingTrainer):
    @staticmethod
    def _rejectable(result):
        metrics = dict(result.metrics)
        if metrics:
            metrics["corrected_mae"] = metrics["baseline_mae"]
            metrics["corrected_rmse"] = metrics["baseline_rmse"]
            return replace(result, metrics=metrics)
        metadata = dict(result.metadata)
        full_fit_metrics = dict(metadata["full_fit_metrics"])
        full_fit_metrics["corrected_mae"] = full_fit_metrics["baseline_mae"]
        full_fit_metrics["corrected_rmse"] = full_fit_metrics["baseline_rmse"]
        metadata["full_fit_metrics"] = full_fit_metrics
        return replace(result, metadata=metadata)

    def train_target(self, frame, target, **kwargs):
        return self._rejectable(super().train_target(frame, target, **kwargs))

    def train_prepared(self, preparation):
        return self._rejectable(super().train_prepared(preparation))


class _PropertyTrackingTrainer:
    def __init__(self):
        self.accesses = []
        self.calls = []

    @property
    def feature_builder(self):
        self.accesses.append("feature_builder")
        accesses = self.accesses

        class TrackedFeatureBuilder:
            @property
            def cadence_minutes(self):
                accesses.append("cadence_minutes")
                return 30.0

        return TrackedFeatureBuilder()

    def prepare_target(self, *args, **kwargs):
        self.calls.append("prepare_target")
        raise AssertionError("custom trainer must be rejected before preparation")

    def train_prepared(self, *args, **kwargs):
        self.calls.append("train_prepared")
        raise AssertionError("custom trainer must be rejected before training")


def test_pipeline_does_not_retrain_loaded_models_when_truth_is_reuploaded(
    tmp_path,
):
    pipeline = DlrPipeline(model_root=tmp_path)
    run_kwargs = {
        "physical": _weather("physical"),
        "truth": _weather("truth", truth_offset=True),
        "project_id": "project-a",
        "line_id": "line-a",
        "terrain_lookup": {},
        "ai_enabled": True,
        "conductor": _conductor(),
    }

    first = pipeline.run(**run_kwargs)
    second = pipeline.run(**run_kwargs)

    assert len(first.model_report.trained_targets) == 4
    assert second.model_report.loaded_targets == first.model_report.trained_targets
    assert second.model_report.used_targets == first.model_report.trained_targets
    assert second.model_report.trained_targets == ()


def test_pipeline_rejects_custom_trainer_before_reading_injected_properties(
    tmp_path,
):
    trainer = _PropertyTrackingTrainer()
    registry = ModelRegistry(tmp_path)

    result = DlrPipeline(registry=registry, trainer=trainer).run(
        physical=_weather("physical"),
        truth=_weather("truth", truth_offset=True),
        project_id="project-a",
        line_id="line-a",
        terrain_lookup={},
        ai_enabled=True,
        conductor=_conductor(),
    )

    expected_keys = _expected_pipeline_model_keys()
    assert trainer.calls == []
    assert trainer.accesses == []
    _assert_unsupported_training_backend(result, expected_keys)
    assert all(not registry.path_for(key).exists() for key in expected_keys)
    assert all(not registry.attempt_path_for(key).exists() for key in expected_keys)


def _weather_with_additional_segment(
    role: str,
    *,
    truth_offset: bool = False,
    segment_count: int = 2,
) -> pd.DataFrame:
    segments = []
    for index in range(segment_count):
        segment = _weather(role, truth_offset=truth_offset)
        segment["timestamp"] += pd.Timedelta(days=2 * index)
        segment["source_file_hash"] = f"{role}-source-{index}"
        segments.append(segment)
    return pd.concat(segments, ignore_index=True).sort_values(
        ["tower_id", "timestamp"],
        kind="mergesort",
        ignore_index=True,
    )


def _sealed_poor_generalization_run_kwargs():
    physical = _weather_with_additional_segment(
        "physical",
        segment_count=3,
    )
    truth = _weather_with_additional_segment(
        "truth",
        segment_count=3,
    )
    residual_patterns = {
        "wind_speed": np.array([1.0, 2.0, 1.5, 2.5]),
        "ambient_temp": np.array([2.0, 3.0, 1.0, 2.5]),
    }
    for segment_index in range(3):
        segment_rows = truth["source_file_hash"].eq(
            f"truth-source-{segment_index}"
        )
        direction = 1.0 if segment_index < 2 else -1.0
        for target, pattern in residual_patterns.items():
            truth.loc[segment_rows, target] = (
                truth.loc[segment_rows, target].to_numpy(dtype=float)
                + direction * pattern
            )
    return {
        "physical": physical,
        "truth": truth,
        "project_id": "project-a",
        "line_id": "line-a",
        "terrain_lookup": {},
        "correction_options": CorrectionOptions(
            enable_vertical=False,
            enable_terrain=False,
            enable_desert=False,
            enable_wind_direction=False,
        ),
        "ai_enabled": True,
        "conductor": _conductor(),
        "truth_tolerance": "5min",
    }


def _assert_one_physical_fallback_per_key(result, expected_keys):
    assert result.model_report.trained_targets == ()
    assert result.model_report.used_targets == ()
    assert tuple(
        fallback.key for fallback in result.model_report.fallbacks
    ) == expected_keys
    assert not result.comparison_weather[
        ["wind_speed_used_ai", "ambient_temp_used_ai"]
    ].to_numpy(dtype=bool).any()
    assert result.max_currents.size > 0
    assert np.isfinite(result.max_currents).all()


def test_nonpersistent_sealed_xgboost_rejects_poor_generalization(tmp_path):
    model_root = tmp_path / "models"
    pipeline = DlrPipeline(model_root=model_root)

    result = pipeline.run(
        **_sealed_poor_generalization_run_kwargs(),
        model_persistence_allowed=False,
    )

    expected_keys = _expected_pipeline_model_keys()
    _assert_one_physical_fallback_per_key(result, expected_keys)
    assert tuple(
        fallback.reason for fallback in result.model_report.fallbacks
    ) == ("candidate_not_better_than_physical",) * 4
    assert result.model_report.promotion_decisions == ()
    assert pipeline.registry is None
    assert not model_root.exists()


def test_first_sealed_xgboost_rejection_is_cached_without_duplicate_fallbacks(
    tmp_path,
    monkeypatch,
):
    registry = ModelRegistry(tmp_path / "models")
    promoted_candidates = []
    original_promote = registry.promote

    def recording_promote(candidate, *, attempt=None):
        promoted_candidates.append(candidate)
        return original_promote(candidate, attempt=attempt)

    monkeypatch.setattr(registry, "promote", recording_promote)
    pipeline = DlrPipeline(registry=registry)
    run_kwargs = _sealed_poor_generalization_run_kwargs()
    expected_keys = _expected_pipeline_model_keys()

    first = pipeline.run(**run_kwargs)

    assert [
        (decision.key, decision.promoted, decision.reason)
        for decision in first.model_report.promotion_decisions
    ] == [
        (key, False, "candidate_not_better_than_physical")
        for key in expected_keys
    ]
    _assert_one_physical_fallback_per_key(first, expected_keys)
    assert len(promoted_candidates) == 4
    assert all(
        type(candidate.bundle.model) is XGBRegressor
        for candidate in promoted_candidates
    )
    assert all(
        candidate.metadata.training_outcome == "trained"
        for candidate in promoted_candidates
    )
    assert all(
        candidate.metadata.metrics["corrected_mae"]
        >= candidate.metadata.metrics["baseline_mae"]
        for candidate in promoted_candidates
    )
    sidecar_bytes = {}
    for key in expected_keys:
        sidecar = registry.attempt_path_for(key)
        entries = json.loads(sidecar.read_text(encoding="utf-8"))["entries"]
        assert len(entries) == 1
        assert entries[0]["reason"] == "candidate_not_better_than_physical"
        assert not registry.path_for(key).exists()
        sidecar_bytes[key] = sidecar.read_bytes()

    repeated = pipeline.run(**run_kwargs)

    _assert_one_physical_fallback_per_key(repeated, expected_keys)
    assert repeated.model_report.promotion_decisions == ()
    assert len(promoted_candidates) == 4
    assert all(
        registry.attempt_path_for(key).read_bytes() == sidecar_bytes[key]
        for key in expected_keys
    )


def test_sealed_rejection_cache_invalidates_when_estimator_contract_changes(
    tmp_path,
    monkeypatch,
):
    registry = ModelRegistry(tmp_path / "models")
    run_kwargs = _sealed_poor_generalization_run_kwargs()
    expected_keys = _expected_pipeline_model_keys()

    first = DlrPipeline(registry=registry).run(**run_kwargs)
    first_sidecars = {
        key: registry.attempt_path_for(key).read_bytes()
        for key in expected_keys
    }
    first_entries = {
        key: json.loads(first_sidecars[key].decode("utf-8"))["entries"][0]
        for key in expected_keys
    }
    changed_parameters = tuple(
        (
            name,
            value + 1 if name == "n_estimators" else value,
        )
        for name, value in ai_training._DEFAULT_ESTIMATOR_PARAMETERS
    )
    monkeypatch.setattr(
        ai_training,
        "_DEFAULT_ESTIMATOR_PARAMETERS",
        changed_parameters,
    )

    retried = DlrPipeline(registry=registry).run(**run_kwargs)

    assert [
        (decision.key, decision.promoted, decision.reason)
        for decision in retried.model_report.promotion_decisions
    ] == [
        (key, False, "candidate_not_better_than_physical")
        for key in expected_keys
    ]
    _assert_one_physical_fallback_per_key(first, expected_keys)
    _assert_one_physical_fallback_per_key(retried, expected_keys)
    for key in expected_keys:
        sidecar = registry.attempt_path_for(key)
        entries = json.loads(sidecar.read_text(encoding="utf-8"))["entries"]
        assert len(entries) == 2
        assert sidecar.read_bytes() != first_sidecars[key]
        assert entries[0] == first_entries[key]
        assert entries[1]["reason"] == "candidate_not_better_than_physical"
        assert entries[1]["fingerprint"] != entries[0]["fingerprint"]
        assert (
            entries[1]["training_contract_hash"]
            != entries[0]["training_contract_hash"]
        )
        assert not registry.path_for(key).exists()


def test_custom_trainer_does_not_taint_compatible_sealed_models(tmp_path):
    model_root = tmp_path / "models"
    run_kwargs = {
        "project_id": "project-a",
        "line_id": "line-a",
        "terrain_lookup": {},
        "ai_enabled": True,
        "conductor": _conductor(),
        "truth_tolerance": "5min",
    }
    sealed_pipeline = DlrPipeline(model_root=model_root)
    sealed_pipeline.run(
        **run_kwargs,
        physical=_weather("physical"),
        truth=_weather("truth", truth_offset=True),
    )
    expanded_physical = _weather_with_additional_segment("physical")
    expanded_truth = _weather_with_additional_segment(
        "truth", truth_offset=True
    )
    promoted = sealed_pipeline.run(
        **run_kwargs,
        physical=expanded_physical,
        truth=expanded_truth,
    )
    expected_keys = _expected_pipeline_model_keys()
    artifact_bytes = {
        path: path.read_bytes()
        for key in expected_keys
        for path in (
            sealed_pipeline.registry.path_for(key),
            sealed_pipeline.registry.metadata_path_for(key),
            sealed_pipeline.registry.manifest_path_for(key),
        )
    }
    trainer = _CountingTrainer()

    reused = DlrPipeline(
        model_root=model_root,
        trainer=trainer,
    ).run(
        **run_kwargs,
        physical=expanded_physical,
        truth=expanded_truth,
    )

    assert promoted.model_report.trained_targets == expected_keys
    assert trainer.calls == []
    assert reused.model_report.loaded_targets == expected_keys
    assert reused.model_report.trained_targets == ()
    assert reused.model_report.used_targets == expected_keys
    assert reused.model_report.fallbacks == ()
    assert all(
        path.read_bytes() == payload
        for path, payload in artifact_bytes.items()
    )


class _ContractCaptureRegistry:
    def __init__(self):
        self.calls = []

    def load_many(
        self,
        keys,
        *,
        expected_compatibility,
        expected_training_contract_hash,
        expected_backend_id,
    ):
        keys = tuple(keys)
        self.calls.append(
            (
                keys,
                expected_compatibility,
                expected_training_contract_hash,
                expected_backend_id,
            )
        )
        return {
            key: ModelLoadResult(None, None, "model_not_found") for key in keys
        }


def test_pipeline_loads_with_target_scoped_sealed_contracts(tmp_path):
    registry = _ContractCaptureRegistry()

    DlrPipeline(model_root=tmp_path, registry=registry).run(
        physical=_weather("physical"),
        project_id="project-a",
        line_id="line-a",
        interval_minutes=60,
        terrain_lookup={},
        ai_enabled=True,
        conductor=_conductor(),
    )

    assert len(registry.calls) == 1
    keys, _, expected_hashes, expected_backends = registry.calls[0]
    trainer = ResidualTrainer(
        feature_builder=FeatureBuilder(cadence_minutes=60)
    )
    for key in keys:
        physical_col = (
            "wind_speed_local"
            if key.target == "wind_speed"
            else "ambient_temp_local"
        )
        assert expected_hashes[key] == (
            ai_training.training_runtime_contract_hash_for_scope(
                trainer,
                target=key.target,
                physical_col=physical_col,
                truth_col=f"{key.target}_truth",
                feature_columns=trainer.feature_builder.feature_columns(
                    physical_col
                ),
                cadence_minutes=60,
            )
        )
        assert expected_backends[key] == trainer.sealed_estimator_spec.backend_id


def _active_sealed_pipeline(model_root):
    pipeline = DlrPipeline(model_root=model_root)
    run_kwargs = {
        "project_id": "project-a",
        "line_id": "line-a",
        "terrain_lookup": {},
        "ai_enabled": True,
        "conductor": _conductor(),
        "truth_tolerance": "5min",
    }
    pipeline.run(
        **run_kwargs,
        physical=_weather("physical"),
        truth=_weather("truth", truth_offset=True),
    )
    physical = _weather_with_additional_segment("physical")
    truth = _weather_with_additional_segment("truth", truth_offset=True)
    promoted = pipeline.run(
        **run_kwargs,
        physical=physical,
        truth=truth,
    )
    assert promoted.model_report.trained_targets == _expected_pipeline_model_keys()
    return pipeline, run_kwargs, physical, truth


def _changed_dependency_versions(package):
    versions = ai_training._dependency_versions()
    versions[package] = f"{versions[package]}-changed"
    return versions


def test_dependency_version_change_without_truth_uses_physical_weather(
    tmp_path,
    monkeypatch,
):
    pipeline, run_kwargs, physical, _ = _active_sealed_pipeline(
        tmp_path / "models"
    )
    changed_versions = _changed_dependency_versions("joblib")
    monkeypatch.setattr(
        ai_training,
        "_dependency_versions",
        lambda: changed_versions,
    )

    degraded = pipeline.run(
        **run_kwargs,
        physical=physical,
        truth=None,
    )

    assert degraded.model_report.loaded_targets == ()
    assert degraded.model_report.used_targets == ()
    assert {
        fallback.reason for fallback in degraded.model_report.fallbacks
    } == {"incompatible_training_contract_hash"}
    assert not degraded.comparison_weather[
        ["wind_speed_used_ai", "ambient_temp_used_ai"]
    ].to_numpy(dtype=bool).any()


def test_dependency_change_retrains_replaces_and_reuses_same_input_model(
    tmp_path,
    monkeypatch,
):
    pipeline, run_kwargs, physical, truth = _active_sealed_pipeline(
        tmp_path / "models"
    )
    changed_versions = _changed_dependency_versions("joblib")
    monkeypatch.setattr(
        ai_training,
        "_dependency_versions",
        lambda: changed_versions,
    )
    expected_keys = _expected_pipeline_model_keys()

    retrained = pipeline.run(
        **run_kwargs,
        physical=physical,
        truth=truth,
    )
    reused = pipeline.run(
        **run_kwargs,
        physical=physical,
        truth=None,
    )

    assert retrained.model_report.trained_targets == expected_keys
    assert all(
        decision.promoted
        for decision in retrained.model_report.promotion_decisions
    )
    assert reused.model_report.loaded_targets == expected_keys
    assert reused.model_report.used_targets == expected_keys
    assert reused.model_report.trained_targets == ()


def test_one_stale_training_contract_is_isolated_in_pipeline(tmp_path):
    pipeline, run_kwargs, physical, _ = _active_sealed_pipeline(
        tmp_path / "models"
    )
    expected_keys = _expected_pipeline_model_keys()
    stale_key = expected_keys[0]
    metadata_path = pipeline.registry.metadata_path_for(stale_key)
    manifest_path = pipeline.registry.manifest_path_for(stale_key)
    metadata = json.loads(metadata_path.read_text("utf-8"))
    metadata["training_contract_hash"] = "0" * 64
    metadata_bytes = (
        json.dumps(
            metadata,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")
    metadata_path.write_bytes(metadata_bytes)
    manifest = json.loads(manifest_path.read_text("utf-8"))
    manifest["metadata_checksum"] = hashlib.sha256(metadata_bytes).hexdigest()
    manifest_path.write_text(
        json.dumps(
            manifest,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n",
        encoding="utf-8",
    )

    isolated = pipeline.run(
        **run_kwargs,
        physical=physical,
        truth=None,
    )

    compatible_keys = tuple(key for key in expected_keys if key != stale_key)
    assert isolated.model_report.loaded_targets == compatible_keys
    assert isolated.model_report.used_targets == compatible_keys
    assert any(
        fallback.key == stale_key
        and fallback.reason == "incompatible_training_contract_hash"
        for fallback in isolated.model_report.fallbacks
    )


def test_sealed_spec_failure_never_accesses_existing_models_and_can_recover(
    tmp_path,
    monkeypatch,
):
    pipeline, run_kwargs, physical, _ = _active_sealed_pipeline(
        tmp_path / "models"
    )
    registry = pipeline.registry
    expected_keys = _expected_pipeline_model_keys()
    artifacts = {
        path: path.read_bytes()
        for key in expected_keys
        for path in (
            registry.path_for(key),
            registry.metadata_path_for(key),
            registry.manifest_path_for(key),
        )
    }
    sidecars = {
        registry.attempt_path_for(key): (
            registry.attempt_path_for(key).read_bytes()
            if registry.attempt_path_for(key).exists()
            else None
        )
        for key in expected_keys
    }
    original_loader = ai_training._load_xgb_regressor
    original_load_many = registry.load_many
    load_calls = []

    def unavailable():
        raise ImportError("temporary xgboost outage")

    def forbidden_load(*args, **kwargs):
        load_calls.append((args, kwargs))
        raise AssertionError("registry must not be read without a sealed spec")

    monkeypatch.setattr(ai_training, "_load_xgb_regressor", unavailable)
    monkeypatch.setattr(registry, "load_many", forbidden_load)

    degraded = pipeline.run(
        **run_kwargs,
        physical=physical,
        truth=None,
    )

    assert load_calls == []
    assert degraded.model_report.loaded_targets == ()
    assert degraded.model_report.used_targets == ()
    assert not degraded.comparison_weather[
        ["wind_speed_used_ai", "ambient_temp_used_ai"]
    ].to_numpy(dtype=bool).any()
    assert all(path.read_bytes() == payload for path, payload in artifacts.items())
    assert all(
        (path.read_bytes() if path.exists() else None) == payload
        for path, payload in sidecars.items()
    )

    monkeypatch.setattr(ai_training, "_load_xgb_regressor", original_loader)
    monkeypatch.setattr(registry, "load_many", original_load_many)
    recovered = pipeline.run(
        **run_kwargs,
        physical=physical,
        truth=None,
    )

    assert recovered.model_report.loaded_targets == expected_keys
    assert recovered.model_report.used_targets == expected_keys


def test_provisional_models_train_on_new_holdout_then_active_models_stop(
    tmp_path,
):
    pipeline = DlrPipeline(model_root=tmp_path)
    run_kwargs = {
        "project_id": "project-a",
        "line_id": "line-a",
        "terrain_lookup": {},
        "ai_enabled": True,
        "conductor": _conductor(),
        "truth_tolerance": "5min",
    }

    first = pipeline.run(
        **run_kwargs,
        physical=_weather("physical"),
        truth=_weather("truth", truth_offset=True),
    )
    keys = first.model_report.trained_targets
    first_metadata = {
        key: json.loads(
            pipeline.registry.metadata_path_for(key).read_text(encoding="utf-8")
        )
        for key in keys
    }

    expanded_physical = _weather_with_additional_segment("physical")
    expanded_truth = _weather_with_additional_segment(
        "truth", truth_offset=True
    )
    promoted = pipeline.run(
        **run_kwargs,
        physical=expanded_physical,
        truth=expanded_truth,
    )
    promoted_metadata = {
        key: json.loads(
            pipeline.registry.metadata_path_for(key).read_text(encoding="utf-8")
        )
        for key in keys
    }

    assert promoted.model_report.trained_targets == keys
    assert all(value["status"] == "active_provisional" for value in first_metadata.values())
    assert all(value["evaluation_mode"] == "full_fit" for value in first_metadata.values())
    assert all(value["status"] == "active" for value in promoted_metadata.values())
    assert all(
        value["evaluation_mode"] == "temporal_holdout"
        for value in promoted_metadata.values()
    )
    assert all(
        promoted_metadata[key]["input_data_hash"]
        != first_metadata[key]["input_data_hash"]
        for key in keys
    )

    repeated = pipeline.run(
        **run_kwargs,
        physical=expanded_physical,
        truth=expanded_truth,
    )
    newer = pipeline.run(
        **run_kwargs,
        physical=_weather_with_additional_segment(
            "physical", segment_count=3
        ),
        truth=_weather_with_additional_segment(
            "truth", truth_offset=True, segment_count=3
        ),
    )

    assert repeated.model_report.trained_targets == ()
    assert newer.model_report.trained_targets == ()
    assert repeated.model_report.loaded_targets == keys
    assert newer.model_report.loaded_targets == keys


def test_rejected_temporal_candidate_is_reported_and_not_retrained(tmp_path):
    registry = ModelRegistry(tmp_path, min_mae_improvement=999.0)
    pipeline = DlrPipeline(registry=registry)
    run_kwargs = {
        "project_id": "project-a",
        "line_id": "line-a",
        "terrain_lookup": {},
        "ai_enabled": True,
        "conductor": _conductor(),
        "truth_tolerance": "5min",
    }

    first = pipeline.run(
        **run_kwargs,
        physical=_weather("physical"),
        truth=_weather("truth", truth_offset=True),
    )
    champion_metadata = {
        key: registry.metadata_path_for(key).read_bytes()
        for key in first.model_report.trained_targets
    }
    expanded_physical = _weather_with_additional_segment("physical")
    expanded_truth = _weather_with_additional_segment(
        "truth", truth_offset=True
    )
    rejected = pipeline.run(
        **run_kwargs,
        physical=expanded_physical,
        truth=expanded_truth,
    )
    repeated = pipeline.run(
        **run_kwargs,
        physical=expanded_physical,
        truth=expanded_truth,
    )

    assert len(first.model_report.trained_targets) == 4
    assert rejected.model_report.trained_targets == ()
    assert [
        (decision.key, decision.promoted, decision.reason)
        for decision in rejected.model_report.promotion_decisions
    ] == [
        (key, False, "insufficient_mae_improvement")
        for key in first.model_report.trained_targets
    ]
    assert rejected.model_report.fallbacks == ()
    assert repeated.model_report.trained_targets == ()
    assert repeated.model_report.promotion_decisions == ()
    for key, original in champion_metadata.items():
        assert registry.metadata_path_for(key).read_bytes() == original
        assert registry.attempt_path_for(key).exists()


def test_poor_custom_trainer_is_rejected_before_candidate_evaluation(tmp_path):
    trainer = _PoorTemporalTrainer()
    registry = ModelRegistry(tmp_path)
    pipeline = DlrPipeline(registry=registry, trainer=trainer)
    run_kwargs = {
        "physical": _weather("physical"),
        "truth": _weather("truth", truth_offset=True),
        "project_id": "project-a",
        "line_id": "line-a",
        "terrain_lookup": {},
        "ai_enabled": True,
        "conductor": _conductor(),
    }

    first = pipeline.run(**run_kwargs)
    repeated = pipeline.run(**run_kwargs)

    expected_keys = _expected_pipeline_model_keys()
    assert trainer.calls == []
    _assert_unsupported_training_backend(first, expected_keys)
    _assert_unsupported_training_backend(repeated, expected_keys)
    assert all(not registry.attempt_path_for(key).exists() for key in expected_keys)
    assert all(not registry.path_for(key).exists() for key in expected_keys)


def test_custom_backend_rejection_has_no_sidecar_or_duplicate_fallbacks(tmp_path):
    trainer = _AlwaysPoorTrainer()
    registry = ModelRegistry(tmp_path)
    pipeline = DlrPipeline(registry=registry, trainer=trainer)
    run_kwargs = {
        "physical": _weather("physical"),
        "truth": _weather("truth", truth_offset=True),
        "project_id": "project-a",
        "line_id": "line-a",
        "terrain_lookup": {},
        "ai_enabled": True,
        "conductor": _conductor(),
    }

    first = pipeline.run(**run_kwargs)
    repeated = pipeline.run(**run_kwargs)

    expected_keys = _expected_pipeline_model_keys()
    assert trainer.calls == []
    _assert_unsupported_training_backend(first, expected_keys)
    _assert_unsupported_training_backend(repeated, expected_keys)
    assert all(not registry.attempt_path_for(key).exists() for key in expected_keys)
    assert all(not registry.path_for(key).exists() for key in expected_keys)


def test_temporary_import_failure_is_retried_without_rejection_sidecar(
    tmp_path,
    monkeypatch,
):
    original_loader = ai_training._load_xgb_regressor

    def unavailable():
        raise ImportError("temporary xgboost outage")

    physical = _weather_with_additional_segment("physical")
    truth = _weather_with_additional_segment("truth")
    alternating_residual = np.tile([-1.0, 1.0], len(truth) // 2)
    truth["wind_speed"] += alternating_residual
    truth["ambient_temp"] += alternating_residual
    registry = ModelRegistry(tmp_path)
    pipeline = DlrPipeline(registry=registry)
    run_kwargs = {
        "physical": physical,
        "truth": truth,
        "project_id": "project-a",
        "line_id": "line-a",
        "terrain_lookup": {},
        "ai_enabled": True,
        "conductor": _conductor(),
        "truth_tolerance": "5min",
    }
    expected_keys = tuple(
        ModelKey("project-a", "line-a", tower_id, target)
        for tower_id in ("001", "002")
        for target in ("wind_speed", "ambient_temp")
    )
    monkeypatch.setattr(ai_training, "_load_xgb_regressor", unavailable)

    degraded = pipeline.run(**run_kwargs)

    assert all(not registry.attempt_path_for(key).exists() for key in expected_keys)
    assert all(not registry.path_for(key).exists() for key in expected_keys)
    assert degraded.model_report.promotion_decisions == ()
    assert sum(
        fallback.reason == "training_failed:TrainingContractError"
        for fallback in degraded.model_report.fallbacks
    ) == 4

    monkeypatch.setattr(
        ai_training,
        "_load_xgb_regressor",
        original_loader,
    )

    recovered = pipeline.run(**run_kwargs)

    assert recovered.model_report.trained_targets == expected_keys
    assert all(
        decision.promoted for decision in recovered.model_report.promotion_decisions
    )


def test_unsupported_backend_is_retried_after_sealed_trainer_restored(tmp_path):
    poor_trainer = _AlwaysPoorTrainer()
    registry = ModelRegistry(tmp_path)
    run_kwargs = {
        "physical": _weather("physical"),
        "truth": _weather("truth", truth_offset=True),
        "project_id": "project-a",
        "line_id": "line-a",
        "terrain_lookup": {},
        "ai_enabled": True,
        "conductor": _conductor(),
    }
    rejected = DlrPipeline(registry=registry, trainer=poor_trainer).run(
        **run_kwargs
    )
    expected_keys = _expected_pipeline_model_keys()
    assert poor_trainer.calls == []
    _assert_unsupported_training_backend(rejected, expected_keys)
    assert all(not registry.attempt_path_for(key).exists() for key in expected_keys)

    retried = DlrPipeline(
        registry=ModelRegistry(tmp_path),
    ).run(**run_kwargs)

    assert retried.model_report.trained_targets == expected_keys
    assert retried.model_report.used_targets == expected_keys
    assert all(
        decision.promoted for decision in retried.model_report.promotion_decisions
    )


def test_attempt_cache_is_invalidated_when_mae_threshold_changes(tmp_path):
    high_registry = ModelRegistry(tmp_path, min_mae_improvement=999.0)
    high_pipeline = DlrPipeline(registry=high_registry)
    run_kwargs = {
        "project_id": "project-a",
        "line_id": "line-a",
        "terrain_lookup": {},
        "ai_enabled": True,
        "conductor": _conductor(),
        "truth_tolerance": "5min",
    }
    first = high_pipeline.run(
        **run_kwargs,
        physical=_weather("physical"),
        truth=_weather("truth", truth_offset=True),
    )
    expanded_physical = _weather_with_additional_segment("physical")
    expanded_truth = _weather_with_additional_segment(
        "truth", truth_offset=True
    )
    rejected = high_pipeline.run(
        **run_kwargs,
        physical=expanded_physical,
        truth=expanded_truth,
    )
    repeated = high_pipeline.run(
        **run_kwargs,
        physical=expanded_physical,
        truth=expanded_truth,
    )
    low_pipeline = DlrPipeline(
        registry=ModelRegistry(tmp_path, min_mae_improvement=0.0),
    )
    retrained = low_pipeline.run(
        **run_kwargs,
        physical=expanded_physical,
        truth=expanded_truth,
    )

    assert len(first.model_report.trained_targets) == 4
    assert rejected.model_report.trained_targets == ()
    assert repeated.model_report.promotion_decisions == ()
    assert retrained.model_report.trained_targets == first.model_report.trained_targets
    assert all(
        decision.promoted for decision in retrained.model_report.promotion_decisions
    )


def test_pipeline_retrains_only_the_corrupt_model_when_truth_is_available(
    tmp_path,
):
    pipeline = DlrPipeline(model_root=tmp_path)
    run_kwargs = {
        "physical": _weather("physical"),
        "truth": _weather("truth", truth_offset=True),
        "project_id": "project-a",
        "line_id": "line-a",
        "terrain_lookup": {},
        "ai_enabled": True,
        "conductor": _conductor(),
    }
    first = pipeline.run(**run_kwargs)
    damaged = ModelKey("project-a", "line-a", "001", "wind_speed")
    pipeline.registry.path_for(damaged).write_bytes(b"not-a-model")

    repaired = pipeline.run(**run_kwargs)

    assert repaired.model_report.loaded_targets == tuple(
        key for key in first.model_report.trained_targets if key != damaged
    )
    assert repaired.model_report.trained_targets == (damaged,)
    assert set(repaired.model_report.used_targets) == set(
        first.model_report.trained_targets
    )


def test_pipeline_interval_controls_training_bundle_and_model_compatibility(
    tmp_path,
):
    pipeline = DlrPipeline(model_root=tmp_path)
    no_correction = CorrectionOptions(
        enable_vertical=False,
        enable_terrain=False,
        enable_desert=False,
        enable_wind_direction=False,
    )
    first = pipeline.run(
        physical=_weather("physical"),
        truth=_weather("truth", truth_offset=True),
        project_id="project-a",
        line_id="line-a",
        interval_minutes=60,
        terrain_lookup={},
        correction_options=no_correction,
        ai_enabled=True,
        conductor=_conductor(),
    )
    key = ModelKey("project-a", "line-a", "001", "wind_speed")
    bundle = joblib.load(pipeline.registry.path_for(key))
    metadata = json.loads(
        pipeline.registry.metadata_path_for(key).read_text(encoding="utf-8")
    )

    assert first.model_report.trained_targets
    assert bundle.cadence_minutes == 60.0
    assert metadata["cadence_minutes"] == 60.0

    changed_interval = pipeline.run(
        physical=_weather("physical"),
        project_id="project-a",
        line_id="line-a",
        interval_minutes=30,
        terrain_lookup={},
        correction_options=no_correction,
        ai_enabled=True,
        conductor=_conductor(),
    )
    assert not changed_interval.model_report.loaded_targets
    assert any(
        fallback.reason == "incompatible_feature_version"
        for fallback in changed_interval.model_report.fallbacks
    )


def test_pipeline_does_not_validate_trainer_cadence_without_training(tmp_path):
    trainer = ResidualTrainer(feature_builder=FeatureBuilder(cadence_minutes=30))
    no_matching_truth = _weather("truth", truth_offset=True).assign(
        tower_id="999"
    )
    cases = (
        (False, None),
        (False, _weather("truth", truth_offset=True)),
        (True, None),
        (True, _weather("truth")),
        (True, no_matching_truth),
    )

    for index, (ai_enabled, truth) in enumerate(cases):
        result = DlrPipeline(
            model_root=tmp_path / str(index),
            trainer=trainer,
        ).run(
            physical=_weather("physical"),
            truth=truth,
            project_id="project-a",
            line_id="line-a",
            interval_minutes=60,
            terrain_lookup={},
            ai_enabled=ai_enabled,
            conductor=_conductor(),
        )
        assert result.max_currents.shape == (2, 2)
        assert not any(
            fallback.reason.startswith("training_failed:")
            for fallback in result.model_report.fallbacks
        )


def test_mismatched_trainer_cadence_fails_only_matched_training_keys(tmp_path):
    trainer = ResidualTrainer(feature_builder=FeatureBuilder(cadence_minutes=30))
    result = DlrPipeline(model_root=tmp_path, trainer=trainer).run(
        physical=_weather("physical"),
        truth=_weather("truth", truth_offset=True),
        project_id="project-a",
        line_id="line-a",
        interval_minutes=60,
        terrain_lookup={},
        ai_enabled=True,
        conductor=_conductor(),
    )

    assert result.max_currents.shape == (2, 2)
    assert not result.model_report.trained_targets
    assert sum(
        fallback.reason == "training_failed:ValueError"
        for fallback in result.model_report.fallbacks
    ) == 4


class _UnusedCompatibility:
    def __bool__(self):
        raise AssertionError("AI-disabled run evaluated model compatibility")


class _UnusedRegistry:
    def load_many(self, *args, **kwargs):
        raise AssertionError("AI-disabled run accessed model registry")

    def promote(self, *args, **kwargs):
        raise AssertionError("AI-disabled run accessed model registry")

    def load(self, *args, **kwargs):
        raise AssertionError("AI-disabled run accessed model registry")


def test_ai_disabled_runs_without_constructing_model_registry():
    pipeline = DlrPipeline(model_root="/dev/null/models")
    result = pipeline.run(
        physical=_weather("physical"),
        project_id="../not-a-model-project",
        line_id="invalid/model/line",
        terrain_lookup={},
        ai_enabled=False,
        conductor=_conductor(),
        model_compatibility=_UnusedCompatibility(),
    )

    assert result.max_currents.shape == (2, 2)
    assert result.model_report == ModelRunReport()
    assert pipeline.registry is None


def test_ai_disabled_does_not_access_injected_model_dependencies(tmp_path):
    registry = _UnusedRegistry()
    result = DlrPipeline(model_root=tmp_path, registry=registry).run(
        physical=_weather("physical"),
        project_id="project-a",
        line_id="line-a",
        terrain_lookup={},
        ai_enabled=False,
        conductor=_conductor(),
        model_compatibility=_UnusedCompatibility(),
    )

    assert result.max_currents.shape == (2, 2)
    assert result.model_report == ModelRunReport()


def test_invalid_optional_truth_still_calculates_physical_dlr(tmp_path):
    from modules import weather_upload

    class InvalidTruthUpload:
        name = "truth.csv"

        @staticmethod
        def getvalue():
            return b"not,a,weather\n1,2,3\n"

    truth = weather_upload.normalize_optional_truth_weather(
        [InvalidTruthUpload()],
        ai_enabled=True,
    )
    result = DlrPipeline(model_root=tmp_path).run(
        physical=_weather("physical"),
        truth=truth.snapshot,
        project_id="project-a",
        line_id="line-a",
        terrain_lookup={},
        ai_enabled=True,
        conductor=_conductor(),
    )

    assert truth.snapshot is None
    assert truth.warning
    assert result.max_currents.shape == (2, 2)
    assert not result.model_report.trained_targets


class _SpyThermalAdapter:
    def __init__(self):
        self.last_conductor = None
        self.last_weather_columns = None

    def calculate_from_long_frame(self, weather, *, base_params):
        self.last_conductor = dict(base_params)
        self.last_weather_columns = tuple(weather.columns)
        tower_count = weather["tower_id"].nunique()
        time_count = weather["timestamp"].nunique()
        return {
            "max_currents": np.full((tower_count, time_count), 1000.0),
            "corrected_winds": np.ones((tower_count, time_count)),
            "local_temps": np.full((tower_count, time_count), 25.0),
        }


def test_pipeline_publishes_factored_steady_ratings_and_metadata(tmp_path):
    result = DlrPipeline(
        model_root=tmp_path,
        thermal_adapter=_SpyThermalAdapter(),
    ).run(
        physical=_weather("physical"),
        project_id="project-a",
        line_id="line-a",
        terrain_lookup={},
        ai_enabled=False,
        conductor=_conductor(),
    )
    expected = np.full((2, 2), 800.0)

    np.testing.assert_array_equal(result.max_currents, expected)
    np.testing.assert_array_equal(
        result.thermal_result["max_currents"], expected
    )
    assert result.thermal_result["safety_factor"] == 0.8
    legacy = result.to_legacy_line_data()
    np.testing.assert_array_equal(legacy["max_currents"], expected)
    assert legacy["safety_factor"] == 0.8


def _assert_model_preparation_fallback(result, exception_name):
    assert result.max_currents.shape == (2, 2)
    assert result.model_report.loaded_targets == ()
    assert result.model_report.trained_targets == ()
    assert result.model_report.used_targets == ()
    assert any(
        fallback.key is None
        and fallback.reason
        == f"model_preparation_failed:{exception_name}"
        for fallback in result.model_report.fallbacks
    )


def test_ai_model_root_failure_falls_back_to_physical_dlr():
    spy = _SpyThermalAdapter()

    result = DlrPipeline(
        model_root="/dev/null/models",
        thermal_adapter=spy,
    ).run(
        physical=_weather("physical"),
        project_id="project-a",
        line_id="line-a",
        terrain_lookup={},
        ai_enabled=True,
        conductor=_conductor(),
    )

    _assert_model_preparation_fallback(result, "ValueError")
    assert spy.last_conductor == _conductor()


def test_ai_compatibility_failure_falls_back_to_physical_dlr(tmp_path):
    class CompatibilityFailurePipeline(DlrPipeline):
        @staticmethod
        def _compatibility(*args, **kwargs):
            raise RuntimeError("compatibility fingerprint failed")

    spy = _SpyThermalAdapter()
    result = CompatibilityFailurePipeline(
        model_root=tmp_path,
        thermal_adapter=spy,
    ).run(
        physical=_weather("physical"),
        project_id="project-a",
        line_id="line-a",
        terrain_lookup={},
        ai_enabled=True,
        conductor=_conductor(),
    )

    _assert_model_preparation_fallback(result, "RuntimeError")
    assert spy.last_conductor == _conductor()


def test_ai_initial_registry_load_failure_falls_back_to_physical_dlr(tmp_path):
    class LoadFailureRegistry:
        def load_many(self, *args, **kwargs):
            raise OSError("registry unavailable")

    spy = _SpyThermalAdapter()
    result = DlrPipeline(
        model_root=tmp_path,
        registry=LoadFailureRegistry(),
        thermal_adapter=spy,
    ).run(
        physical=_weather("physical"),
        project_id="project-a",
        line_id="line-a",
        terrain_lookup={},
        ai_enabled=True,
        conductor=_conductor(),
    )

    _assert_model_preparation_fallback(result, "OSError")
    assert spy.last_conductor == _conductor()


def test_invalid_model_key_falls_back_to_physical_dlr(tmp_path):
    spy = _SpyThermalAdapter()
    result = DlrPipeline(model_root=tmp_path, thermal_adapter=spy).run(
        physical=_weather("physical"),
        project_id="../invalid-project",
        line_id="invalid/model/line",
        terrain_lookup={},
        ai_enabled=True,
        conductor=_conductor(),
    )

    _assert_model_preparation_fallback(result, "ValueError")
    assert spy.last_conductor == _conductor()


def test_model_preparation_boundary_does_not_swallow_thermal_failure():
    class FailingThermalAdapter:
        def calculate_from_long_frame(self, weather, *, base_params):
            raise RuntimeError("thermal calculation failed")

    with pytest.raises(RuntimeError, match="thermal calculation failed"):
        DlrPipeline(
            model_root="/dev/null/models",
            thermal_adapter=FailingThermalAdapter(),
        ).run(
            physical=_weather("physical"),
            project_id="project-a",
            line_id="line-a",
            terrain_lookup={},
            ai_enabled=True,
            conductor=_conductor(),
        )


def test_pipeline_passes_selected_conductor_and_never_truth_to_dlr(tmp_path):
    spy = _SpyThermalAdapter()
    conductor = _conductor() | {"D0": 0.0338}
    result = DlrPipeline(model_root=tmp_path, thermal_adapter=spy).run(
        physical=_weather("physical"),
        truth=_weather("truth", truth_offset=True),
        project_id="project-a",
        line_id="line-a",
        interval_minutes=30,
        terrain_lookup={},
        ai_enabled=False,
        conductor=conductor,
        truth_tolerance="5min",
    )

    assert spy.last_conductor["D0"] == conductor["D0"]
    assert all("truth" not in column.lower() for column in spy.last_weather_columns)
    assert result.max_currents.shape[0] == 2


def test_pipeline_uses_only_the_common_tower_timestamps(tmp_path):
    physical = _weather("physical").iloc[[0, 1, 2, 3]].copy()
    extra_early = physical.iloc[[0]].copy()
    extra_early["timestamp"] = pd.Timestamp(
        "2026-07-22 23:30", tz="Asia/Shanghai"
    )
    extra_late = physical.iloc[[2]].copy()
    extra_late["timestamp"] = pd.Timestamp(
        "2026-07-23 01:00", tz="Asia/Shanghai"
    )
    physical = pd.concat(
        [extra_early, physical, extra_late], ignore_index=True
    ).sort_values(["tower_id", "timestamp"], ignore_index=True)
    physical["timestamp"] = pd.to_datetime(
        physical["timestamp"], utc=True
    ).dt.tz_convert("Asia/Shanghai")

    result = DlrPipeline(model_root=tmp_path).run(
        physical=physical,
        project_id="project-a",
        line_id="line-a",
        interval_minutes=30,
        terrain_lookup={},
        ai_enabled=False,
        conductor=_conductor(),
    )

    expected = pd.to_datetime(
        ["2026-07-23 00:00", "2026-07-23 00:30"]
    ).tz_localize("Asia/Shanghai")
    assert pd.DatetimeIndex(result.final_weather["timestamp"].unique()).equals(
        expected
    )
    assert result.max_currents.shape == (2, 2)


def test_pipeline_rejects_fully_overlapping_truth_content_but_calculates_dlr(
    tmp_path,
):
    physical = _weather("physical")
    truth = _weather("truth").iloc[::-1].reset_index(drop=True)
    truth["source_file_hash"] = "different-upload"

    result = DlrPipeline(model_root=tmp_path).run(
        physical=physical,
        truth=truth,
        project_id="project-a",
        line_id="line-a",
        terrain_lookup={},
        ai_enabled=True,
        conductor=_conductor(),
    )

    assert result.max_currents.shape == (2, 2)
    assert not result.model_report.trained_targets
    assert any(
        fallback.reason == "truth_rejected_overlapping_content"
        for fallback in result.model_report.fallbacks
    )


def test_pipeline_rejects_same_source_hash_when_weather_content_differs(tmp_path):
    physical = _weather("physical")
    truth = _weather("truth", truth_offset=True)
    truth["source_file_hash"] = physical["source_file_hash"]

    result = DlrPipeline(model_root=tmp_path).run(
        physical=physical,
        truth=truth,
        project_id="project-a",
        line_id="line-a",
        terrain_lookup={},
        ai_enabled=True,
        conductor=_conductor(),
    )

    assert result.max_currents.shape == (2, 2)
    assert not result.model_report.trained_targets
    assert any(
        fallback.reason == "truth_rejected_overlapping_source_hash"
        for fallback in result.model_report.fallbacks
    )


def test_pipeline_rejects_shared_hash_from_full_upload_lineage(tmp_path):
    physical = _weather("physical")
    truth = _weather("truth", truth_offset=True)
    physical["source_file_hash"] = "physical-retained-row"
    truth["source_file_hash"] = "truth-retained-row"
    physical.attrs["source_file_hashes"] = ("physical-only", "shared-file")
    truth.attrs["source_file_hashes"] = ("truth-only", "shared-file")

    result = DlrPipeline(model_root=tmp_path).run(
        physical=physical,
        truth=truth,
        project_id="project-a",
        line_id="line-a",
        terrain_lookup={},
        ai_enabled=True,
        conductor=_conductor(),
    )

    assert result.max_currents.shape == (2, 2)
    assert not result.model_report.trained_targets
    assert any(
        fallback.reason == "truth_rejected_overlapping_source_hash"
        for fallback in result.model_report.fallbacks
    )


def test_pipeline_has_no_union_fill_when_towers_do_not_overlap(tmp_path):
    physical = _weather("physical")
    physical.loc[physical["tower_id"] == "002", "timestamp"] += pd.Timedelta(
        days=1
    )

    with pytest.raises(ValueError, match="没有共同时间戳"):
        DlrPipeline(model_root=tmp_path).run(
            physical=physical,
            project_id="project-a",
            line_id="line-a",
            terrain_lookup={},
            ai_enabled=False,
            conductor=_conductor(),
        )


def test_legacy_projection_keeps_tower_by_common_time_matrices(tmp_path):
    result = DlrPipeline(model_root=tmp_path).run(
        physical=_weather("physical"),
        project_id="project-a",
        line_id="line-a",
        terrain_lookup={},
        ai_enabled=False,
        conductor=_conductor(),
    )

    legacy = result.to_legacy_line_data()

    assert {
        "positions",
        "times",
        "datetimes",
        "elevations",
        "solar",
        "temps",
        "winds",
        "angles",
        "terrain_data",
        "correction_details",
        "max_currents",
        "corrected_winds",
        "local_temps",
    } <= legacy.keys()
    for key in (
        "solar",
        "temps",
        "winds",
        "angles",
        "max_currents",
        "corrected_winds",
        "local_temps",
    ):
        assert np.asarray(legacy[key]).shape == (2, 2)


def test_pipeline_result_is_a_defensive_snapshot_with_read_only_arrays(
    tmp_path,
):
    base = DlrPipeline(model_root=tmp_path).run(
        physical=_weather("physical"),
        project_id="project-a",
        line_id="line-a",
        terrain_lookup={},
        ai_enabled=False,
        conductor=_conductor(),
    )
    physical = base.physical_weather
    physical["object_payload"] = [
        {"values": [index]} for index in range(len(physical))
    ]
    physical.attrs["nested"] = {"values": [1]}
    terrain = base.terrain_corrected_weather
    final = base.final_weather
    comparison = base.comparison_weather
    max_currents = np.asarray(base.max_currents).copy()
    nested_array = np.asarray([1.0, 2.0])
    thermal_result = {
        key: value for key, value in base.thermal_result.items()
    }
    thermal_result["nested"] = {
        "array": nested_array,
        "items": [{"value": 1}],
    }

    result = DlrPipelineResult(
        physical_weather=physical,
        terrain_corrected_weather=terrain,
        final_weather=final,
        comparison_weather=comparison,
        thermal_result=thermal_result,
        max_currents=max_currents,
        model_report=base.model_report,
        weather_metrics=base.weather_metrics,
        input_hash=base.input_hash,
    )
    physical.loc[0, "object_payload"]["values"].append(99)
    physical.attrs["nested"]["values"].append(99)
    max_currents[0, 0] = -1.0
    nested_array[0] = -1.0
    thermal_result["nested"]["items"][0]["value"] = 99

    first_view = result.physical_weather
    assert first_view.loc[0, "object_payload"] == {"values": [0]}
    assert first_view.attrs["nested"] == {"values": [1]}
    first_view.loc[0, "object_payload"]["values"].append(2)
    first_view.attrs["nested"]["values"].append(2)
    assert result.physical_weather.loc[0, "object_payload"] == {"values": [0]}
    assert result.physical_weather.attrs["nested"] == {"values": [1]}

    assert result.max_currents[0, 0] != -1.0
    assert not result.max_currents.flags.writeable
    with pytest.raises(ValueError):
        result.max_currents[0, 0] = 0.0
    assert result.thermal_result["nested"]["array"][0] == 1.0
    assert result.thermal_result["nested"]["items"][0]["value"] == 1
    with pytest.raises(TypeError):
        result.thermal_result["new"] = "value"
    with pytest.raises(TypeError):
        result.thermal_result["nested"]["new"] = "value"
    with pytest.raises(ValueError):
        result.thermal_result["nested"]["array"][0] = 0.0

    expected_currents = result.max_currents.copy()
    exposed_currents = result.max_currents
    exposed_currents.setflags(write=True)
    exposed_currents[0, 0] = -999.0
    assert result.max_currents is not exposed_currents
    np.testing.assert_array_equal(result.max_currents, expected_currents)

    expected_winds = result.thermal_result["corrected_winds"].copy()
    exposed_thermal = result.thermal_result
    exposed_winds = exposed_thermal["corrected_winds"]
    exposed_nested = exposed_thermal["nested"]["array"]
    exposed_winds.setflags(write=True)
    exposed_nested.setflags(write=True)
    exposed_winds[0, 0] = -999.0
    exposed_nested[0] = -999.0
    fresh_thermal = result.thermal_result
    assert fresh_thermal is not exposed_thermal
    np.testing.assert_array_equal(
        fresh_thermal["corrected_winds"], expected_winds
    )
    np.testing.assert_array_equal(
        fresh_thermal["nested"]["array"], [1.0, 2.0]
    )
    legacy = result.to_legacy_line_data()
    np.testing.assert_array_equal(legacy["max_currents"], expected_currents)
    np.testing.assert_array_equal(legacy["corrected_winds"], expected_winds)


def test_pipeline_result_rejects_invalid_input_hash(tmp_path):
    base = DlrPipeline(model_root=tmp_path).run(
        physical=_weather("physical"),
        project_id="project-a",
        line_id="line-a",
        terrain_lookup={},
        ai_enabled=False,
        conductor=_conductor(),
    )
    result_kwargs = {
        "physical_weather": base.physical_weather,
        "terrain_corrected_weather": base.terrain_corrected_weather,
        "final_weather": base.final_weather,
        "comparison_weather": base.comparison_weather,
        "thermal_result": base.thermal_result,
        "max_currents": base.max_currents,
        "model_report": base.model_report,
        "weather_metrics": base.weather_metrics,
    }

    for invalid_hash in ("", "not-a-hash", "A" * 64):
        with pytest.raises(ValueError, match="64 位小写 SHA-256"):
            DlrPipelineResult(**result_kwargs, input_hash=invalid_hash)


def test_legacy_projection_remains_independent_and_mutable(tmp_path):
    result = DlrPipeline(model_root=tmp_path).run(
        physical=_weather("physical"),
        project_id="project-a",
        line_id="line-a",
        terrain_lookup={},
        ai_enabled=False,
        conductor=_conductor(),
    )
    baseline = result.to_legacy_line_data()
    legacy = result.to_legacy_line_data()

    assert legacy["winds"].flags.writeable
    assert legacy["max_currents"].flags.writeable
    legacy["winds"][0, 0] = -1.0
    legacy["max_currents"][0, 0] = -1.0
    legacy["correction_details"]["winds_orig"][0, 0] = -1.0
    legacy["terrain_data"][0]["source"] = "changed"
    legacy["comparison_weather"].loc[0, "wind_speed_ai"] = -1.0
    legacy["new"] = "value"

    fresh = result.to_legacy_line_data()
    np.testing.assert_array_equal(fresh["winds"], baseline["winds"])
    np.testing.assert_array_equal(
        fresh["max_currents"], baseline["max_currents"]
    )
    np.testing.assert_array_equal(
        fresh["correction_details"]["winds_orig"],
        baseline["correction_details"]["winds_orig"],
    )
    assert fresh["terrain_data"][0]["source"] != "changed"
    assert fresh["comparison_weather"].loc[0, "wind_speed_ai"] != -1.0
    assert "new" not in fresh


class _FailingTransientAdapter(_SpyThermalAdapter):
    def calculate_transient_from_long_frame(
        self, weather, *, base_params, request, steady_result
    ):
        raise RuntimeError("transient failed")


class _SuccessfulTransientAdapter(_SpyThermalAdapter):
    def calculate_transient_from_long_frame(
        self, weather, *, base_params, request, steady_result
    ):
        return {
            "max_currents": np.full_like(
                np.asarray(steady_result["max_currents"], dtype=float),
                1200.0,
            ),
            "window_start_hour": 0.0,
            "window_end_hour": 1.0,
        }


class _InvalidTransientMetadataAdapter(_SpyThermalAdapter):
    def calculate_transient_from_long_frame(
        self, weather, *, base_params, request, steady_result
    ):
        return {
            "max_currents": np.asarray(steady_result["max_currents"]),
            "window_start_hour": "invalid",
            "window_end_hour": 1.0,
        }


class _NonFiniteTransientResultAdapter(_SpyThermalAdapter):
    def calculate_transient_from_long_frame(
        self, weather, *, base_params, request, steady_result
    ):
        return {
            "max_currents": np.full_like(
                np.asarray(steady_result["max_currents"], dtype=float),
                np.nan,
            ),
            "window_start_hour": 0.0,
            "window_end_hour": 1.0,
        }


def test_transient_failure_falls_back_to_same_steady_result(tmp_path):
    adapter = _FailingTransientAdapter()
    result = DlrPipeline(model_root=tmp_path, thermal_adapter=adapter).run(
        physical=_weather("physical"),
        project_id="project-a",
        line_id="line-a",
        terrain_lookup={},
        ai_enabled=False,
        conductor=_conductor(),
        transient_request={"window_minutes": 15},
    )

    np.testing.assert_array_equal(
        result.thermal_result["transient_result"]["max_currents"],
        result.max_currents,
    )
    np.testing.assert_array_equal(
        result.max_currents,
        np.full((2, 2), 800.0),
    )
    assert not np.array_equal(
        result.max_currents,
        np.full((2, 2), 640.0),
    )
    assert result.transient_fallbacks == ("transient_failed:RuntimeError",)


def test_successful_transient_ratings_are_factored_once(tmp_path):
    conductor = _conductor() | {
        "materials": [
            {"type": "aluminum", "density": 1.116},
            {"type": "steel", "density": 0.5126},
        ]
    }
    result = DlrPipeline(
        model_root=tmp_path,
        thermal_adapter=_SuccessfulTransientAdapter(),
    ).run(
        physical=_weather("physical"),
        project_id="project-a",
        line_id="line-a",
        terrain_lookup={},
        ai_enabled=False,
        conductor=conductor,
        transient_request={"start_hour": 0.0, "end_hour": 1.0},
    )

    np.testing.assert_array_equal(
        result.max_currents,
        np.full((2, 2), 800.0),
    )
    np.testing.assert_array_equal(
        result.thermal_result["transient_result"]["max_currents"],
        np.full((2, 2), 960.0),
    )
    assert result.thermal_result["safety_factor"] == 0.8
    assert not result.transient_fallbacks


def test_invalid_transient_metadata_falls_back_and_uses_steady_input_hash(tmp_path):
    steady = DlrPipeline(
        model_root=tmp_path / "steady",
        thermal_adapter=_InvalidTransientMetadataAdapter(),
    ).run(
        physical=_weather("physical"),
        project_id="project-a",
        line_id="line-a",
        terrain_lookup={},
        ai_enabled=False,
        conductor=_conductor(),
    )
    fallback = DlrPipeline(
        model_root=tmp_path / "fallback",
        thermal_adapter=_InvalidTransientMetadataAdapter(),
    ).run(
        physical=_weather("physical"),
        project_id="project-a",
        line_id="line-a",
        terrain_lookup={},
        ai_enabled=False,
        conductor=_conductor(),
        transient_request={"window_minutes": 15},
    )

    np.testing.assert_array_equal(
        fallback.thermal_result["transient_result"]["max_currents"],
        fallback.max_currents,
    )
    assert fallback.transient_fallbacks == ("transient_failed:ValueError",)
    assert fallback.input_hash == steady.input_hash


def test_nonfinite_transient_currents_fall_back_to_steady_result(tmp_path):
    conductor = _conductor() | {
        "materials": [
            {"type": "aluminum", "density": 1.116},
            {"type": "steel", "density": 0.5126},
        ]
    }
    fallback = DlrPipeline(
        model_root=tmp_path,
        thermal_adapter=_NonFiniteTransientResultAdapter(),
    ).run(
        physical=_weather("physical"),
        project_id="project-a",
        line_id="line-a",
        terrain_lookup={},
        ai_enabled=False,
        conductor=conductor,
        transient_request={"window_minutes": 15},
    )

    np.testing.assert_array_equal(
        fallback.thermal_result["transient_result"]["max_currents"],
        fallback.max_currents,
    )
    assert fallback.transient_fallbacks == ("transient_failed:ValueError",)


def test_all_weather_stages_keep_the_full_project_line_tower_key(tmp_path):
    result = DlrPipeline(model_root=tmp_path).run(
        physical=_weather("physical"),
        project_id="project-a",
        line_id="line-a",
        terrain_lookup={},
        ai_enabled=False,
        conductor=_conductor(),
    )

    for frame in (
        result.physical_weather,
        result.terrain_corrected_weather,
        result.final_weather,
        result.comparison_weather,
    ):
        assert {"project_id", "line_id", "tower_id", "timestamp"} <= set(
            frame.columns
        )
        assert set(frame["project_id"]) == {"project-a"}
        assert set(frame["line_id"]) == {"line-a"}


class _ZeroResidualModel:
    def predict(self, features):
        return np.zeros(len(features), dtype=float)


class _PredictionContractRegistry:
    def __init__(self, *, missing_feature=True):
        self.missing_feature = missing_feature

    def load_many(
        self,
        keys,
        *,
        expected_compatibility,
        expected_training_contract_hash,
        expected_backend_id,
    ):
        assert set(expected_training_contract_hash) == set(keys)
        assert set(expected_backend_id) == set(keys)
        loaded = {}
        for key in keys:
            physical_column = (
                "wind_speed_local"
                if key.target == "wind_speed"
                else "ambient_temp_local"
            )
            feature_columns = [physical_column]
            if (
                self.missing_feature
                and key.tower_id == "001"
                and key.target == "wind_speed"
            ):
                feature_columns = ["missing_contract_feature"]
            loaded[key] = ModelLoadResult(
                bundle=ModelBundle(
                    target_name=key.target,
                    feature_columns=feature_columns,
                    model=_ZeroResidualModel(),
                    residual_bounds=(-10.0, 10.0),
                    line_id=key.line_id,
                    tower_id=key.tower_id,
                    cadence_minutes=30.0,
                ),
                metadata=None,
            )
        return loaded


def test_prediction_contract_failure_isolated_to_one_tower_target(tmp_path):
    failed_key = ModelKey("project-a", "line-a", "001", "wind_speed")
    result = DlrPipeline(
        model_root=tmp_path,
        registry=_PredictionContractRegistry(),
    ).run(
        physical=_weather("physical"),
        project_id="project-a",
        line_id="line-a",
        terrain_lookup={},
        ai_enabled=True,
        conductor=_conductor(),
    )

    tower_one = result.comparison_weather.loc[
        result.comparison_weather["tower_id"] == "001"
    ]
    assert result.max_currents.shape == (2, 2)
    assert not tower_one["wind_speed_used_ai"].any()
    np.testing.assert_array_equal(
        tower_one["wind_speed_ai"], tower_one["wind_speed_physical"]
    )
    assert tower_one["ambient_temp_used_ai"].all()
    assert result.model_report.active_model_count == 3
    assert any(
        fallback.key == failed_key
        and fallback.reason == "prediction_failed:ValueError"
        for fallback in result.model_report.fallbacks
    )


def test_prediction_input_drops_outputs_and_incomplete_key_rolls_back(
    tmp_path,
    monkeypatch,
):
    original_predict = ResidualPredictor.predict
    leaked_output_columns = set()

    def incomplete_predict(self, frame, target_name, physical_col):
        if str(frame["tower_id"].iloc[0]) == "001" and target_name == "wind_speed":
            output_columns = {"final", "residual", "used_ai", "fallback_reason"}
            for target in ("wind_speed", "ambient_temp"):
                output_columns.update(
                    {
                        f"{target}_final",
                        f"{target}_residual",
                        f"{target}_used_ai",
                        f"{target}_fallback_reason",
                    }
                )
            leaked_output_columns.update(output_columns & set(frame.columns))
            predicted = frame.copy(deep=True)
            predicted["wind_speed_final"] = predicted[physical_col] + 1.0
            predicted["used_ai"] = True
            predicted["fallback_reason"] = ""
            return predicted
        return original_predict(
            self,
            frame,
            target_name=target_name,
            physical_col=physical_col,
        )

    monkeypatch.setattr(ResidualPredictor, "predict", incomplete_predict)
    result = DlrPipeline(
        model_root=tmp_path,
        registry=_PredictionContractRegistry(missing_feature=False),
    ).run(
        physical=_weather("physical"),
        project_id="project-a",
        line_id="line-a",
        terrain_lookup={},
        ai_enabled=True,
        conductor=_conductor(),
    )

    comparison = result.comparison_weather
    failed_rows = comparison["tower_id"] == "001"
    assert not leaked_output_columns
    np.testing.assert_array_equal(
        comparison.loc[failed_rows, "wind_speed_ai"],
        comparison.loc[failed_rows, "wind_speed_physical"],
    )
    assert not comparison.loc[failed_rows, "wind_speed_used_ai"].any()
    final = result.final_weather
    local = result.terrain_corrected_weather
    np.testing.assert_array_equal(
        final.loc[final["tower_id"] == "001", "wind_speed"],
        local.loc[local["tower_id"] == "001", "wind_speed_local"],
    )
    assert any(
        fallback.key == ModelKey("project-a", "line-a", "001", "wind_speed")
        and fallback.reason == "prediction_failed:KeyError"
        for fallback in result.model_report.fallbacks
    )


def test_inconsistent_non_ai_prediction_rolls_back_the_entire_key(
    tmp_path,
    monkeypatch,
):
    original_predict = ResidualPredictor.predict

    def inconsistent_predict(self, frame, target_name, physical_col):
        if str(frame["tower_id"].iloc[0]) == "001" and target_name == "wind_speed":
            predicted = frame.copy(deep=True)
            predicted["wind_speed_residual"] = 0.5
            predicted["wind_speed_final"] = predicted[physical_col] + 0.5
            predicted["used_ai"] = False
            predicted["fallback_reason"] = ""
            return predicted
        return original_predict(
            self,
            frame,
            target_name=target_name,
            physical_col=physical_col,
        )

    monkeypatch.setattr(ResidualPredictor, "predict", inconsistent_predict)
    result = DlrPipeline(
        model_root=tmp_path,
        registry=_PredictionContractRegistry(missing_feature=False),
    ).run(
        physical=_weather("physical"),
        project_id="project-a",
        line_id="line-a",
        terrain_lookup={},
        ai_enabled=True,
        conductor=_conductor(),
    )

    comparison = result.comparison_weather
    failed_rows = comparison["tower_id"] == "001"
    np.testing.assert_array_equal(
        comparison.loc[failed_rows, "wind_speed_ai"],
        comparison.loc[failed_rows, "wind_speed_physical"],
    )
    assert not comparison.loc[failed_rows, "wind_speed_used_ai"].any()
    assert any(
        fallback.key == ModelKey("project-a", "line-a", "001", "wind_speed")
        and fallback.reason == "prediction_failed:ValueError"
        for fallback in result.model_report.fallbacks
    )


def test_one_tower_target_training_failure_does_not_disable_other_models(
    tmp_path,
    monkeypatch,
):
    original_train_prepared = ResidualTrainer.train_prepared

    def selectively_fail(self, preparation):
        if (
            preparation.tower_id == "001"
            and preparation.target == "wind_speed"
        ):
            raise RuntimeError("tower target failed")
        return original_train_prepared(self, preparation)

    monkeypatch.setattr(ResidualTrainer, "train_prepared", selectively_fail)
    result = DlrPipeline(model_root=tmp_path).run(
        physical=_weather("physical"),
        truth=_weather("truth", truth_offset=True),
        project_id="project-a",
        line_id="line-a",
        terrain_lookup={},
        ai_enabled=True,
        conductor=_conductor(),
    )

    failed_key = ModelKey("project-a", "line-a", "001", "wind_speed")
    successful_keys = tuple(
        key for key in _expected_pipeline_model_keys() if key != failed_key
    )
    assert result.model_report.trained_targets == successful_keys
    assert result.model_report.used_targets == successful_keys
    assert result.model_report.active_model_count == 3
    tower_one = result.comparison_weather.loc[
        result.comparison_weather["tower_id"] == "001"
    ]
    assert not tower_one["wind_speed_used_ai"].any()
    assert tower_one["ambient_temp_used_ai"].all()
    assert any(
        fallback.key == failed_key
        and fallback.reason == "training_failed:RuntimeError"
        for fallback in result.model_report.fallbacks
    )


def test_one_tower_target_without_aligned_truth_does_not_disable_other_models(
    tmp_path,
):
    truth = _weather("truth", truth_offset=True)
    truth.loc[truth["tower_id"] == "001", "wind_speed"] = np.nan

    result = DlrPipeline(model_root=tmp_path).run(
        physical=_weather("physical"),
        truth=truth,
        project_id="project-a",
        line_id="line-a",
        terrain_lookup={},
        ai_enabled=True,
        conductor=_conductor(),
    )

    failed_key = ModelKey("project-a", "line-a", "001", "wind_speed")
    successful_keys = tuple(
        key for key in _expected_pipeline_model_keys() if key != failed_key
    )
    assert result.model_report.trained_targets == successful_keys
    assert result.model_report.used_targets == successful_keys
    assert result.model_report.active_model_count == 3
    tower_one = result.comparison_weather.loc[
        result.comparison_weather["tower_id"] == "001"
    ]
    assert not tower_one["wind_speed_used_ai"].any()
    assert tower_one["ambient_temp_used_ai"].all()
    assert any(
        fallback.key == failed_key
        and fallback.reason == "no_aligned_truth"
        for fallback in result.model_report.fallbacks
    )


def test_conductor_change_invalidates_saved_weather_models(tmp_path):
    pipeline = DlrPipeline(model_root=tmp_path)
    pipeline.run(
        physical=_weather("physical"),
        truth=_weather("truth", truth_offset=True),
        project_id="project-a",
        line_id="line-a",
        terrain_lookup={},
        ai_enabled=True,
        conductor=_conductor(),
    )
    changed_conductor = _conductor() | {"D0": 0.0338}

    changed = pipeline.run(
        physical=_weather("physical"),
        project_id="project-a",
        line_id="line-a",
        terrain_lookup={},
        ai_enabled=True,
        conductor=changed_conductor,
    )

    assert not changed.model_report.loaded_targets
    assert changed.model_report.active_model_count == 0
    assert sum(
        fallback.reason == "incompatible_conductor_hash"
        for fallback in changed.model_report.fallbacks
    ) == 4


def test_pipeline_joins_terrain_metadata_before_single_correction(tmp_path):
    terrain = {
        "001": {
            "elevation": 1500.0,
            "slope": 20.0,
            "aspect": 180.0,
            "source": "dem",
            "reason": None,
        },
        "002": {
            "elevation": 1800.0,
            "slope": 5.0,
            "aspect": 90.0,
            "source": "default",
            "reason": "nodata",
        },
    }
    result = DlrPipeline(model_root=tmp_path).run(
        physical=_weather("physical"),
        project_id="project-a",
        line_id="line-a",
        terrain_lookup=terrain,
        correction_options=CorrectionOptions(
            enable_vertical=False,
            enable_terrain=True,
            enable_desert=False,
            enable_wind_direction=True,
        ),
        ai_enabled=False,
        conductor=_conductor(),
    )

    tower_one = result.terrain_corrected_weather.loc[
        result.terrain_corrected_weather["tower_id"] == "001"
    ]
    tower_two = result.terrain_corrected_weather.loc[
        result.terrain_corrected_weather["tower_id"] == "002"
    ]
    assert tower_one["elevation"].eq(1500.0).all()
    assert tower_one["source"].eq("dem").all()
    assert tower_two["reason"].eq("nodata").all()
    assert result.terrain_corrected_weather["correction_stage"].eq(
        "terrain_corrected"
    ).all()
    assert result.final_weather["correction_stage"].eq("final").all()


def test_final_weather_is_a_truth_free_thermal_whitelist(tmp_path):
    truth = _weather("truth", truth_offset=True)
    truth["private_truth_note"] = "must not leak"
    result = DlrPipeline(model_root=tmp_path).run(
        physical=_weather("physical"),
        truth=truth,
        project_id="project-a",
        line_id="line-a",
        terrain_lookup={},
        ai_enabled=False,
        conductor=_conductor(),
    )

    assert tuple(result.final_weather.columns) == (
        "project_id",
        "line_id",
        "tower_id",
        "timestamp",
        "ambient_temp",
        "wind_speed",
        "wind_direction",
        "wind_angle_deg",
        "solar_radiation",
        "humidity",
        "elevation",
        "slope",
        "aspect",
        "source",
        "reason",
        "correction_stage",
    )
    assert all("truth" not in column.lower() for column in result.final_weather)
    assert "wind_speed_truth" in result.comparison_weather
    assert "ambient_temp_truth" in result.comparison_weather


class _RecordingLineAnalyzer:
    def __init__(self):
        self.calls = []
        self.transient_calls = []

    def calculate_max_current_for_points(self, **kwargs):
        self.calls.append(kwargs)
        return {
            "max_currents": np.asarray(kwargs["solar"], dtype=float)[None, :],
            "corrected_winds": np.asarray(kwargs["winds"], dtype=float),
            "local_temps": np.asarray(kwargs["temps"], dtype=float),
        }

    def find_max_current_for_window(
        self, env_params, base_static, params, dt_hours, start_hour=0, end_hour=2
    ):
        self.transient_calls.append(
            {
                "env_params": env_params,
                "base_static": base_static,
                "params": params,
                "dt_hours": dt_hours,
                "start_hour": start_hour,
                "end_hour": end_hour,
            }
        )
        return float(base_static) + len(self.transient_calls)


def test_long_frame_adapter_preserves_each_tower_solar_and_conductor():
    analyzer = _RecordingLineAnalyzer()
    frame = pd.DataFrame(
        {
            "tower_id": ["001", "001", "002", "002"],
            "timestamp": pd.to_datetime(
                [
                    "2026-07-23 00:00",
                    "2026-07-23 00:30",
                    "2026-07-23 00:00",
                    "2026-07-23 00:30",
                ]
            ).tz_localize("Asia/Shanghai"),
            "ambient_temp": [30.0, 31.0, 28.0, 29.0],
            "wind_speed": [2.0, 2.5, 3.0, 3.5],
            "wind_angle_deg": [90.0, 80.0, 70.0, 60.0],
            "solar_radiation": [100.0, 200.0, 700.0, 800.0],
            "elevation": [1000.0, 1000.0, 1200.0, 1200.0],
        }
    )
    conductor = _conductor() | {"D0": 0.0338}

    result = LongFrameThermalAdapter(analyzer).calculate_from_long_frame(
        frame, base_params=conductor
    )

    assert len(analyzer.calls) == 2
    np.testing.assert_array_equal(analyzer.calls[0]["solar"], [100.0, 200.0])
    np.testing.assert_array_equal(analyzer.calls[1]["solar"], [700.0, 800.0])
    assert all(call["base_params"]["D0"] == 0.0338 for call in analyzer.calls)
    np.testing.assert_array_equal(
        result["max_currents"], [[100.0, 200.0], [700.0, 800.0]]
    )


def test_long_frame_adapter_calculates_transient_with_each_tower_weather():
    analyzer = _RecordingLineAnalyzer()
    adapter = LongFrameThermalAdapter(analyzer)
    frame = pd.DataFrame(
        {
            "tower_id": ["001", "001", "002", "002"],
            "timestamp": pd.to_datetime(
                [
                    "2026-07-23 00:00",
                    "2026-07-23 00:30",
                    "2026-07-23 00:00",
                    "2026-07-23 00:30",
                ]
            ).tz_localize("Asia/Shanghai"),
            "ambient_temp": [30.0, 31.0, 28.0, 29.0],
            "wind_speed": [2.0, 2.5, 3.0, 3.5],
            "wind_angle_deg": [90.0, 80.0, 70.0, 60.0],
            "solar_radiation": [100.0, 200.0, 700.0, 800.0],
            "elevation": [1000.0, 1000.0, 1200.0, 1200.0],
        }
    )
    steady = adapter.calculate_from_long_frame(frame, base_params=_conductor())

    transient = adapter.calculate_transient_from_long_frame(
        frame,
        base_params=_conductor(),
        request={"window_minutes": 15},
        steady_result=steady,
    )

    assert len(analyzer.transient_calls) == 2
    np.testing.assert_array_equal(
        analyzer.transient_calls[0]["env_params"]["solar"], [100.0, 200.0]
    )
    np.testing.assert_array_equal(
        analyzer.transient_calls[1]["env_params"]["solar"], [700.0, 800.0]
    )
    assert analyzer.transient_calls[0]["params"]["D0"] == _conductor()["D0"]
    assert transient["max_currents"].shape == steady["max_currents"].shape


def test_corrupt_model_falls_back_only_for_its_tower_target(tmp_path):
    pipeline = DlrPipeline(model_root=tmp_path)
    first = pipeline.run(
        physical=_weather("physical"),
        truth=_weather("truth", truth_offset=True),
        project_id="project-a",
        line_id="line-a",
        terrain_lookup={},
        ai_enabled=True,
        conductor=_conductor(),
    )
    damaged = ModelKey("project-a", "line-a", "001", "wind_speed")
    assert damaged in first.model_report.trained_targets
    pipeline.registry.path_for(damaged).write_bytes(b"not-a-model")

    second = pipeline.run(
        physical=_weather("physical"),
        project_id="project-a",
        line_id="line-a",
        terrain_lookup={},
        ai_enabled=True,
        conductor=_conductor(),
    )

    assert damaged not in second.model_report.loaded_targets
    assert second.model_report.active_model_count == 3
    assert any(
        fallback.key == damaged and fallback.reason == "corrupt_model"
        for fallback in second.model_report.fallbacks
    )
    tower_one = second.comparison_weather.loc[
        second.comparison_weather["tower_id"] == "001"
    ]
    assert not tower_one["wind_speed_used_ai"].any()
    assert tower_one["ambient_temp_used_ai"].all()


def _page_source() -> str:
    return (Path(__file__).parents[2] / "dispatch_app_st.py").read_text(
        encoding="utf-8"
    )


def test_page_adds_truth_upload_and_normalizes_each_role_once():
    source = _page_source()

    assert '"上传真实气象数据"' in source
    truth_upload_start = source.index('"上传真实气象数据"')
    truth_upload_end = source.index(")", truth_upload_start)
    assert "accept_multiple_files=True" in source[
        truth_upload_start:truth_upload_end
    ]
    assert source.count("normalize_uploaded_weather_files(") == 1
    assert source.count("normalize_optional_truth_weather(") == 1
    assert 'role="physical"' in source
    assert "truth_normalization.warning" in source
    assert "st.warning(truth_normalization.warning)" in source
    assert "physical_weather_snapshot" in source
    assert "truth_weather_snapshot" in source


def test_page_main_button_is_a_thin_pipeline_adapter():
    source = _page_source()
    button_start = source.index("if btn_generate and weather_files:")
    button_end = source.index("# 结果展示", button_start)
    button_block = source[button_start:button_end]

    assert "DlrPipeline(" in button_block
    assert ".run(" in button_block
    assert "conductor=st.session_state.conductor_params" in button_block
    assert "result.to_legacy_line_data()" in button_block
    assert "load_weather_data_from_files(" not in button_block
    assert "process_weather_data(" not in button_block
    assert "convert_to_analysis_format(" not in button_block
    assert "apply_weather_corrections(" not in button_block
    assert "calculate_max_current_for_points(" not in button_block


def test_page_derives_line_namespace_and_passes_runtime_contexts():
    source = _page_source()
    button_start = source.index("if btn_generate and weather_files:")
    button_end = source.index("# 结果展示", button_start)
    button_block = source[button_start:button_end]

    assert "derive_line_identity(" in button_block
    assert "line_id=line_identity.line_id" in button_block
    assert (
        "model_persistence_allowed=line_identity.persistence_allowed"
        in button_block
    )
    assert 'line_id="main-line"' not in button_block
    assert "tower_coords=st.session_state.tower_coords" in button_block
    assert "dem_context=st.session_state.dem_data" in button_block
    assert "coordinate_context=st.session_state.tower_coords" in button_block


def test_page_reports_weather_error_and_has_no_random_dlr_ai_demo():
    source = _page_source()
    ai_start = source.index("# ---- AI预测部分 ----")
    ai_block = source[ai_start:]

    assert "IEEE 738-2023" in source
    assert "IEEE 738-2013" not in source
    assert "np.random.seed" not in ai_block
    assert '"风速 MAE"' in ai_block
    assert '"温度 MAE"' in ai_block
    assert 'metric("已启用模型数"' in ai_block
    assert "wind_speed_truth" in ai_block
    assert "ambient_temp_truth" in ai_block
