from __future__ import annotations

import dataclasses
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional

import numpy as np
import pandas as pd

from config.config import DEFAULT_INTERVAL_MINUTES, MODEL_DIR
from modules.ai_prediction import ResidualPredictor
from modules.ai_training import ResidualTrainer
from modules.model_registry import (
    ModelCompatibility,
    ModelKey,
    ModelRegistry,
    candidate_from_training_result,
)
from modules.thermal_engine import LineAnalyzer, ThermalCalculator
from modules.weather_correction import CorrectionOptions, WeatherCorrectionService
from modules.weather_pipeline import (
    AlignmentReport,
    align_physical_and_truth,
    resample_weather_by_tower,
)
from modules.weather_upload import (
    WeatherUploadResult,
    ensure_distinct_dataset_hashes,
)


_TARGETS = ("wind_speed", "ambient_temp")
_LOCAL_COLUMNS = {
    "wind_speed": "wind_speed_local",
    "ambient_temp": "ambient_temp_local",
}


def _weather_axes(frame: pd.DataFrame) -> tuple[tuple[str, ...], pd.DatetimeIndex]:
    tower_ids = tuple(sorted(frame["tower_id"].astype(str).unique()))
    timestamps = pd.DatetimeIndex(
        frame["timestamp"].drop_duplicates().sort_values()
    )
    return tower_ids, timestamps


def _fractional_hours(timestamps: pd.DatetimeIndex) -> np.ndarray:
    return (
        timestamps.hour
        + timestamps.minute / 60.0
        + timestamps.second / 3600.0
    ).to_numpy(dtype=float)


def _long_frame_matrix(
    frame: pd.DataFrame,
    column: str,
    tower_ids: tuple[str, ...],
    timestamps: pd.DatetimeIndex,
) -> np.ndarray:
    selected = frame.loc[
        frame["tower_id"].astype(str).isin(tower_ids)
        & frame["timestamp"].isin(timestamps),
        ["tower_id", "timestamp", column],
    ].copy()
    selected["tower_id"] = selected["tower_id"].astype(str)
    pivoted = selected.pivot(
        index="tower_id", columns="timestamp", values=column
    ).reindex(index=tower_ids, columns=timestamps)
    values = pivoted.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError(f"兼容投影 {column} 不能包含缺失或非有限值")
    return values


@dataclass(frozen=True)
class WeatherMetrics:
    wind_speed_mae: Optional[float] = None
    wind_speed_rmse: Optional[float] = None
    ambient_temp_mae: Optional[float] = None
    ambient_temp_rmse: Optional[float] = None


@dataclass(frozen=True)
class ModelFallback:
    key: Optional[ModelKey]
    reason: str


@dataclass(frozen=True)
class ModelRunReport:
    trained_targets: tuple[ModelKey, ...] = ()
    loaded_targets: tuple[ModelKey, ...] = ()
    used_targets: tuple[ModelKey, ...] = ()
    fallbacks: tuple[ModelFallback, ...] = ()
    alignment: Optional[AlignmentReport] = None

    @property
    def active_model_count(self) -> int:
        return len(set(self.used_targets))


@dataclass(frozen=True)
class DlrPipelineResult:
    physical_weather: pd.DataFrame
    terrain_corrected_weather: pd.DataFrame
    final_weather: pd.DataFrame
    comparison_weather: pd.DataFrame
    thermal_result: Mapping[str, Any]
    max_currents: np.ndarray
    model_report: ModelRunReport
    weather_metrics: WeatherMetrics
    transient_fallbacks: tuple[str, ...] = ()

    def to_legacy_line_data(self) -> dict[str, Any]:
        tower_ids = tuple(self.thermal_result.get("tower_ids", ()))
        timestamps = self.thermal_result.get("timestamps")
        default_towers, default_timestamps = _weather_axes(self.final_weather)
        tower_ids = tower_ids or default_towers
        timestamps = pd.DatetimeIndex(
            default_timestamps if timestamps is None else timestamps
        )

        def matrix(frame: pd.DataFrame, column: str) -> np.ndarray:
            return _long_frame_matrix(frame, column, tower_ids, timestamps)

        physical_winds = matrix(self.physical_weather, "wind_speed")
        physical_temps = matrix(self.physical_weather, "ambient_temp")
        physical_solar = matrix(self.physical_weather, "solar_radiation")
        local_solar = matrix(
            self.terrain_corrected_weather, "solar_radiation_local"
        )
        vertical_factors = matrix(
            self.terrain_corrected_weather, "vertical_wind_factor"
        )
        terrain_factors = matrix(
            self.terrain_corrected_weather, "terrain_wind_factor"
        )

        elevations = []
        terrain_data = {}
        for index, tower_id in enumerate(tower_ids):
            tower = self.final_weather.loc[
                self.final_weather["tower_id"].astype(str) == tower_id
            ]
            row = tower.iloc[0]
            elevations.append(float(row["elevation"]))
            terrain_data[index] = {
                "tower_id": tower_id,
                "elevation": float(row["elevation"]),
                "slope": float(row["slope"]),
                "aspect": float(row["aspect"]),
                "source": row["source"],
                "reason": row["reason"],
            }

        fractional_hours = _fractional_hours(timestamps)
        solar = matrix(self.final_weather, "solar_radiation")
        temps = matrix(self.final_weather, "ambient_temp")
        winds = matrix(self.final_weather, "wind_speed")
        angles = matrix(self.final_weather, "wind_angle_deg")
        max_currents = np.asarray(self.max_currents, dtype=float).copy()
        corrected_winds = np.asarray(
            self.thermal_result["corrected_winds"], dtype=float
        ).copy()
        local_temps = np.asarray(
            self.thermal_result["local_temps"], dtype=float
        ).copy()
        expected_shape = (len(tower_ids), len(timestamps))
        for name, values in (
            ("solar", solar),
            ("temps", temps),
            ("winds", winds),
            ("angles", angles),
            ("max_currents", max_currents),
            ("corrected_winds", corrected_winds),
            ("local_temps", local_temps),
        ):
            if values.shape != expected_shape:
                raise ValueError(f"兼容投影 {name} 维度不一致")

        daylight = solar.mean(axis=0) > 10.0
        sunrise = float(fractional_hours[daylight][0]) if daylight.any() else 6.0
        sunset = float(fractional_hours[daylight][-1]) if daylight.any() else 18.0
        return {
            "points_km": np.arange(len(tower_ids), dtype=float),
            "positions": list(tower_ids),
            "times": fractional_hours,
            "datetimes": timestamps,
            "elevations": np.asarray(elevations, dtype=float),
            "solar": solar,
            "temps": temps,
            "winds": winds,
            "angles": angles,
            "terrain_data": terrain_data,
            "correction_details": {
                "winds_orig": physical_winds,
                "temps_orig": physical_temps,
                "solar_orig": physical_solar,
                "vertical_factors": vertical_factors,
                "terrain_factors": terrain_factors,
                "desert_solar_delta": local_solar - physical_solar,
                "wind_dir_factors": np.ones(expected_shape, dtype=float),
            },
            "max_currents": max_currents,
            "corrected_winds": corrected_winds,
            "local_temps": local_temps,
            "sunrise": sunrise,
            "sunset": sunset,
            "comparison_weather": self.comparison_weather.copy(deep=True),
            "model_report": self.model_report,
            "weather_metrics": self.weather_metrics,
            "correction_stage": "final",
        }


class LongFrameThermalAdapter:
    """Adapt final per-tower weather to the existing IEEE 738 line analyzer."""

    def __init__(self, analyzer: Optional[LineAnalyzer] = None):
        self.analyzer = analyzer or LineAnalyzer(ThermalCalculator())

    def calculate_from_long_frame(
        self,
        weather: pd.DataFrame,
        *,
        base_params: Mapping[str, Any],
    ) -> dict[str, Any]:
        tower_ids, timestamps = _weather_axes(weather)
        times = _fractional_hours(timestamps)
        currents = []
        corrected_winds = []
        local_temps = []

        for tower_id in tower_ids:
            tower = weather.loc[
                weather["tower_id"].astype(str) == tower_id
            ].sort_values("timestamp", kind="mergesort")
            if not pd.DatetimeIndex(tower["timestamp"]).equals(timestamps):
                raise ValueError("热核输入必须使用所有杆塔的共同时间戳")
            elevation = float(
                pd.to_numeric(tower["elevation"], errors="raise").iloc[0]
            )
            result = self.analyzer.calculate_max_current_for_points(
                observation_points=np.asarray([tower_id], dtype=object),
                elevations=np.asarray([elevation], dtype=float),
                temps=tower["ambient_temp"].to_numpy(dtype=float)[None, :],
                winds=tower["wind_speed"].to_numpy(dtype=float)[None, :],
                angles=tower["wind_angle_deg"].to_numpy(dtype=float)[None, :],
                solar=tower["solar_radiation"].to_numpy(dtype=float),
                times=times,
                max_temp=float(base_params.get("max_allow_temp", 80.0)),
                base_params=base_params,
                terrain_data=None,
            )
            currents.append(np.asarray(result["max_currents"], dtype=float)[0])
            corrected_winds.append(
                np.asarray(result["corrected_winds"], dtype=float)[0]
            )
            local_temps.append(np.asarray(result["local_temps"], dtype=float)[0])

        max_currents = np.vstack(currents)
        bottleneck_indices = np.argmin(max_currents, axis=0)
        return {
            "max_currents": max_currents,
            "corrected_winds": np.vstack(corrected_winds),
            "local_temps": np.vstack(local_temps),
            "bottleneck_tower_ids": np.asarray(
                [tower_ids[index] for index in bottleneck_indices], dtype=object
            ),
            "tower_ids": tuple(tower_ids),
            "timestamps": timestamps,
        }

    def calculate_transient_from_long_frame(
        self,
        weather: pd.DataFrame,
        *,
        base_params: Mapping[str, Any],
        request: Mapping[str, Any],
        steady_result: Mapping[str, Any],
    ) -> dict[str, Any]:
        if not isinstance(request, Mapping):
            raise TypeError("transient request 必须是映射")
        if not isinstance(steady_result, Mapping):
            raise TypeError("steady_result 必须是映射")

        tower_ids, timestamps = _weather_axes(weather)
        if len(timestamps) < 2:
            raise ValueError("暂态计算至少需要两个共同时间戳")
        elapsed_hours = (
            (timestamps - timestamps[0]).total_seconds() / 3600.0
        ).to_numpy(dtype=float)
        intervals = np.diff(elapsed_hours)
        if (
            not np.isfinite(intervals).all()
            or np.any(intervals <= 0.0)
            or not np.allclose(intervals, intervals[0])
        ):
            raise ValueError("暂态计算要求共同时间戳等间隔且严格递增")
        dt_hours = float(intervals[0])

        start_hour = float(request.get("start_hour", elapsed_hours[0]))
        if "end_hour" in request:
            end_hour = float(request["end_hour"])
        elif "window_minutes" in request:
            end_hour = start_hour + float(request["window_minutes"]) / 60.0
        else:
            end_hour = float(elapsed_hours[-1])
        if not np.isfinite([start_hour, end_hour]).all() or end_hour < start_hour:
            raise ValueError("暂态窗口必须是有限且非递减的时间范围")

        steady_currents = np.asarray(
            steady_result["max_currents"], dtype=float
        )
        expected_shape = (len(tower_ids), len(timestamps))
        if steady_currents.shape != expected_shape:
            raise ValueError("稳态结果维度与暂态天气不一致")

        transient_rows = []
        for tower_index, tower_id in enumerate(tower_ids):
            tower = weather.loc[
                weather["tower_id"].astype(str) == tower_id
            ].sort_values("timestamp", kind="mergesort")
            if not pd.DatetimeIndex(tower["timestamp"]).equals(timestamps):
                raise ValueError("暂态热核输入必须使用所有杆塔的共同时间戳")
            elevation = float(
                pd.to_numeric(tower["elevation"], errors="raise").iloc[0]
            )
            environment = {
                "times": elapsed_hours.copy(),
                "temp": tower["ambient_temp"].to_numpy(dtype=float),
                "wind": tower["wind_speed"].to_numpy(dtype=float),
                "angle": tower["wind_angle_deg"].to_numpy(dtype=float),
                "solar": tower["solar_radiation"].to_numpy(dtype=float),
                "elevation": np.full(len(timestamps), elevation, dtype=float),
            }
            base_static = float(np.min(steady_currents[tower_index]))
            rating = self.analyzer.find_max_current_for_window(
                env_params=environment,
                base_static=base_static,
                params=dict(base_params),
                dt_hours=dt_hours,
                start_hour=start_hour,
                end_hour=end_hour,
            )
            transient_rows.append(
                np.full(len(timestamps), float(rating), dtype=float)
            )
        return {
            "max_currents": np.vstack(transient_rows),
            "tower_ids": tuple(tower_ids),
            "timestamps": timestamps,
            "window_start_hour": start_hour,
            "window_end_hour": end_hour,
        }


def _stable_value(value: Any):
    if dataclasses.is_dataclass(value):
        return _stable_value(dataclasses.asdict(value))
    if isinstance(value, Mapping):
        return {
            str(key): _stable_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_stable_value(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float):
        if not np.isfinite(value):
            return str(value)
        return value
    if value is None or isinstance(value, (str, int, bool)):
        return value
    return repr(value)


def _stable_hash(value: Any) -> str:
    payload = json.dumps(
        _stable_value(value),
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


class DlrPipeline:
    def __init__(
        self,
        model_root: Path | str = MODEL_DIR,
        *,
        correction_service: Optional[WeatherCorrectionService] = None,
        trainer: Optional[ResidualTrainer] = None,
        registry: Optional[ModelRegistry] = None,
        thermal_adapter: Optional[Any] = None,
    ):
        self.correction_service = correction_service or WeatherCorrectionService()
        self.trainer = trainer or ResidualTrainer()
        self.registry = registry or ModelRegistry(model_root)
        self.thermal_adapter = thermal_adapter or LongFrameThermalAdapter()

    @staticmethod
    def _weather_frame(value, *, role: str) -> pd.DataFrame:
        if isinstance(value, WeatherUploadResult):
            frame = value.frame
        elif isinstance(value, pd.DataFrame):
            frame = value
        else:
            raise TypeError(f"{role} 必须是规范化 DataFrame 或 WeatherUploadResult")
        result = frame.copy(deep=True)
        if "dataset_role" not in result.columns:
            raise ValueError(f"{role} 缺少 dataset_role")
        if not result.empty and not result["dataset_role"].eq(role).all():
            raise ValueError(f"{role} dataset_role 必须全部为 {role}")
        return result

    @staticmethod
    def _terrain_for_tower(
        terrain_lookup: Optional[Mapping[Any, Any]], tower_id: str
    ) -> Mapping[str, Any]:
        if not isinstance(terrain_lookup, Mapping):
            return {}
        for key in (tower_id, str(tower_id)):
            value = terrain_lookup.get(key)
            if isinstance(value, Mapping):
                return value
        return {}

    def _join_terrain(
        self,
        frame: pd.DataFrame,
        terrain_lookup: Optional[Mapping[Any, Any]],
    ) -> pd.DataFrame:
        result = frame.copy(deep=True)
        elevation = pd.to_numeric(result.get("elevation", 1000.0), errors="coerce")
        elevations = []
        slopes = []
        aspects = []
        sources = []
        reasons = []
        for row_position, tower_id in enumerate(result["tower_id"].astype(str)):
            terrain = self._terrain_for_tower(terrain_lookup, tower_id)
            terrain_elevation = terrain.get("elevation", elevation.iloc[row_position])
            elevations.append(float(terrain_elevation))
            slopes.append(float(terrain.get("slope", 0.0)))
            aspects.append(float(terrain.get("aspect", 0.0)))
            sources.append(str(terrain.get("source", "weather")))
            reasons.append(terrain.get("reason"))
        result["elevation"] = elevations
        result["slope"] = slopes
        result["aspect"] = aspects
        result["source"] = sources
        result["reason"] = reasons
        return result

    @staticmethod
    def _joined_terrain_lookup(frame: pd.DataFrame) -> dict[str, dict[str, Any]]:
        lookup = {}
        for tower_id, tower in frame.groupby("tower_id", sort=False):
            row = tower.iloc[0]
            lookup[str(tower_id)] = {
                "elevation": row["elevation"],
                "slope": row["slope"],
                "aspect": row["aspect"],
                "source": row["source"],
                "reason": row["reason"],
            }
        return lookup

    @staticmethod
    def _compatibility(
        frame: pd.DataFrame,
        *,
        conductor: Mapping[str, Any],
        correction_options: CorrectionOptions,
    ) -> ModelCompatibility:
        coordinate_columns = [
            column
            for column in ("tower_id", "longitude", "latitude")
            if column in frame.columns
        ]
        coordinates = (
            frame.loc[:, coordinate_columns]
            .drop_duplicates()
            .sort_values(coordinate_columns, kind="mergesort")
            .to_dict(orient="records")
        )
        terrain_projection = frame.loc[
            :, ["tower_id", "elevation", "slope", "aspect", "source", "reason"]
        ].drop_duplicates().sort_values("tower_id", kind="mergesort")
        return ModelCompatibility(
            dem_hash=_stable_hash(terrain_projection.to_dict(orient="records")),
            crs_hash=_stable_hash("crs-unavailable-v1"),
            coordinate_hash=_stable_hash(coordinates),
            conductor_hash=_stable_hash(conductor),
            feature_version="weather-features-v1",
            correction_config_hash=_stable_hash(correction_options),
        )

    @staticmethod
    def _alignment_physical(frame: pd.DataFrame) -> pd.DataFrame:
        conflicting_aliases = {
            "wind_speed_physical",
            "ambient_temp_physical",
            "solar_radiation_physical",
        }
        return frame.drop(
            columns=[column for column in conflicting_aliases if column in frame],
        )

    @staticmethod
    def _alignment_truth(frame: pd.DataFrame) -> pd.DataFrame:
        allowed = [
            column
            for column in (
                "tower_id",
                "timestamp",
                "ambient_temp",
                "wind_speed",
                "wind_direction",
                "solar_radiation",
                "humidity",
                "elevation",
                "measurement_height",
                "dataset_role",
                "source_file_hash",
            )
            if column in frame.columns
        ]
        return frame.loc[:, allowed].copy(deep=True)

    @staticmethod
    def _common_time_weather(frame: pd.DataFrame) -> pd.DataFrame:
        timestamp_sets = [
            set(tower["timestamp"].tolist())
            for _, tower in frame.groupby("tower_id", sort=False)
        ]
        if not timestamp_sets:
            raise ValueError("物理气象数据不能为空")
        common = set.intersection(*timestamp_sets)
        if not common:
            raise ValueError("所有杆塔没有共同时间戳，无法进入 DLR 热核")
        return frame.loc[frame["timestamp"].isin(common)].sort_values(
            ["tower_id", "timestamp"], kind="mergesort", ignore_index=True
        )

    @staticmethod
    def _metrics(comparison: pd.DataFrame) -> WeatherMetrics:
        values = {}
        for target in _TARGETS:
            truth_column = f"{target}_truth"
            ai_column = f"{target}_ai"
            if truth_column not in comparison:
                values[f"{target}_mae"] = None
                values[f"{target}_rmse"] = None
                continue
            truth = pd.to_numeric(comparison[truth_column], errors="coerce")
            predicted = pd.to_numeric(comparison[ai_column], errors="coerce")
            valid = np.isfinite(truth) & np.isfinite(predicted)
            if not valid.any():
                values[f"{target}_mae"] = None
                values[f"{target}_rmse"] = None
                continue
            error = predicted.loc[valid].to_numpy() - truth.loc[valid].to_numpy()
            values[f"{target}_mae"] = float(np.mean(np.abs(error)))
            values[f"{target}_rmse"] = float(np.sqrt(np.mean(np.square(error))))
        return WeatherMetrics(**values)

    def run(
        self,
        physical,
        truth=None,
        *,
        project_id: str,
        line_id: str,
        interval_minutes: int = DEFAULT_INTERVAL_MINUTES,
        terrain_lookup: Optional[Mapping[Any, Any]] = None,
        correction_options: Optional[CorrectionOptions] = None,
        ai_enabled: bool = False,
        conductor: Mapping[str, Any],
        truth_tolerance: Any = "30min",
        model_compatibility: Optional[ModelCompatibility] = None,
        transient_request: Optional[Mapping[str, Any]] = None,
    ) -> DlrPipelineResult:
        options = correction_options or CorrectionOptions()
        physical_input = self._weather_frame(physical, role="physical")
        physical_weather = resample_weather_by_tower(
            physical_input, interval_minutes=interval_minutes
        )
        physical_weather["project_id"] = str(project_id)
        physical_weather["line_id"] = str(line_id)
        terrain_joined = self._join_terrain(physical_weather, terrain_lookup)
        terrain_corrected = self.correction_service.apply(
            terrain_joined,
            self._joined_terrain_lookup(terrain_joined),
            options,
        )

        compatibility = model_compatibility or self._compatibility(
            terrain_corrected,
            conductor=conductor,
            correction_options=options,
        )
        keys = [
            ModelKey(str(project_id), str(line_id), str(tower_id), target)
            for tower_id in sorted(terrain_corrected["tower_id"].astype(str).unique())
            for target in _TARGETS
        ]
        loaded = {}
        loaded_targets = []
        trained_targets = []
        used_targets = []
        fallbacks: list[ModelFallback] = []
        aligned = None
        alignment_report = None

        if ai_enabled:
            loaded = self.registry.load_many(
                keys,
                expected_compatibility={key: compatibility for key in keys},
            )
            loaded_targets = [
                key for key in keys if loaded[key].bundle is not None
            ]

        if ai_enabled and truth is not None:
            try:
                truth_frame = self._weather_frame(truth, role="truth")
                ensure_distinct_dataset_hashes(physical_input, truth_frame)
                aligned, alignment_report = align_physical_and_truth(
                    self._alignment_physical(terrain_corrected),
                    self._alignment_truth(truth_frame),
                    tolerance=truth_tolerance,
                    roughness_alpha=options.roughness_alpha,
                    temp_lapse_rate=options.temp_lapse_rate,
                )
            except ValueError as exc:
                reason = (
                    "truth_rejected_same_source_hash"
                    if "同一文件哈希" in str(exc)
                    else f"truth_alignment_failed:{type(exc).__name__}"
                )
                fallbacks.append(
                    ModelFallback(None, reason)
                )
                aligned = None
            except Exception as exc:
                fallbacks.append(
                    ModelFallback(None, f"truth_alignment_failed:{type(exc).__name__}")
                )
                aligned = None

            if aligned is not None:
                matched = aligned.loc[aligned["truth_timestamp"].notna()].copy()
                for key in keys:
                    tower_training = matched.loc[
                        matched["tower_id"].astype(str) == key.tower_id
                    ].copy()
                    truth_column = f"{key.target}_truth"
                    physical_column = _LOCAL_COLUMNS[key.target]
                    finite = (
                        np.isfinite(pd.to_numeric(tower_training.get(truth_column), errors="coerce"))
                        & np.isfinite(pd.to_numeric(tower_training.get(physical_column), errors="coerce"))
                    )
                    tower_training = tower_training.loc[finite]
                    if tower_training.empty:
                        fallbacks.append(ModelFallback(key, "no_aligned_truth"))
                        continue
                    try:
                        training = self.trainer.train_target(
                            tower_training,
                            key.target,
                            physical_col=physical_column,
                            truth_col=truth_column,
                        )
                        candidate = candidate_from_training_result(
                            training,
                            project_id=str(project_id),
                            model_version=f"train-{training.metadata['input_data_hash'][:24]}",
                            compatibility=compatibility,
                        )
                        decision = self.registry.promote(candidate)
                        if decision.promoted:
                            trained_targets.append(key)
                            loaded[key] = self.registry.load(
                                key, expected_compatibility=compatibility
                            )
                        elif loaded.get(key) is None or loaded[key].bundle is None:
                            fallbacks.append(ModelFallback(key, decision.reason))
                    except Exception as exc:
                        fallbacks.append(
                            ModelFallback(key, f"training_failed:{type(exc).__name__}")
                        )

        prediction = terrain_corrected.copy(deep=True)
        for target in _TARGETS:
            prediction[f"{target}_final"] = prediction[_LOCAL_COLUMNS[target]]
            prediction[f"{target}_residual"] = 0.0
            prediction[f"{target}_used_ai"] = False
            prediction[f"{target}_fallback_reason"] = (
                "model_unavailable" if ai_enabled else "ai_disabled"
            )

        if ai_enabled:
            for key in keys:
                load_result = loaded.get(key)
                if load_result is None or load_result.bundle is None:
                    reason = (
                        load_result.fallback_reason
                        if load_result is not None
                        else "model_unavailable"
                    )
                    fallbacks.append(ModelFallback(key, reason))
                    continue
                row_mask = prediction["tower_id"].astype(str) == key.tower_id
                tower = prediction.loc[row_mask].copy()
                predicted = ResidualPredictor(
                    {key.target: load_result.bundle}
                ).predict(
                    tower,
                    target_name=key.target,
                    physical_col=_LOCAL_COLUMNS[key.target],
                )
                prediction.loc[row_mask, f"{key.target}_final"] = predicted[
                    f"{key.target}_final"
                ].to_numpy()
                prediction.loc[row_mask, f"{key.target}_residual"] = predicted[
                    f"{key.target}_residual"
                ].to_numpy()
                prediction.loc[row_mask, f"{key.target}_used_ai"] = predicted[
                    "used_ai"
                ].to_numpy(dtype=bool)
                prediction.loc[row_mask, f"{key.target}_fallback_reason"] = predicted[
                    "fallback_reason"
                ].to_numpy(dtype=object)
                if predicted["used_ai"].any():
                    used_targets.append(key)

        comparison = prediction.loc[
            :,
            ["project_id", "line_id", "tower_id", "timestamp"],
        ].copy()
        for target in _TARGETS:
            comparison[f"{target}_physical"] = prediction[_LOCAL_COLUMNS[target]].to_numpy()
            comparison[f"{target}_ai"] = prediction[f"{target}_final"].to_numpy()
            comparison[f"{target}_used_ai"] = prediction[
                f"{target}_used_ai"
            ].to_numpy(dtype=bool)
            comparison[f"{target}_fallback_reason"] = prediction[
                f"{target}_fallback_reason"
            ].to_numpy(dtype=object)
            comparison[f"{target}_truth"] = np.nan
        if aligned is not None:
            for target in _TARGETS:
                comparison[f"{target}_truth"] = pd.to_numeric(
                    aligned[f"{target}_truth"], errors="coerce"
                ).to_numpy()

        final_weather = pd.DataFrame(
            {
                "project_id": str(project_id),
                "line_id": str(line_id),
                "tower_id": prediction["tower_id"].astype(str),
                "timestamp": prediction["timestamp"],
                "ambient_temp": prediction["ambient_temp_final"],
                "wind_speed": prediction["wind_speed_final"],
                "wind_direction": prediction["wind_direction"],
                "wind_angle_deg": prediction["wind_angle_deg"],
                "solar_radiation": prediction["solar_radiation_local"],
                "humidity": prediction["humidity"],
                "elevation": prediction["elevation"],
                "slope": prediction["slope"],
                "aspect": prediction["aspect"],
                "source": prediction["source"],
                "reason": prediction["reason"],
                "correction_stage": "final",
            }
        )
        final_weather = self._common_time_weather(final_weather)
        steady_result = self.thermal_adapter.calculate_from_long_frame(
            final_weather,
            base_params=conductor,
        )
        thermal_result = dict(steady_result)
        max_currents = np.asarray(steady_result["max_currents"], dtype=float)
        transient_fallbacks = ()
        if transient_request is not None:
            try:
                transient_result = (
                    self.thermal_adapter.calculate_transient_from_long_frame(
                        final_weather,
                        base_params=conductor,
                        request=transient_request,
                        steady_result=steady_result,
                    )
                )
                if not isinstance(transient_result, Mapping):
                    raise TypeError("暂态结果必须是映射")
                thermal_result["transient_result"] = dict(transient_result)
            except Exception as exc:
                reason = f"transient_failed:{type(exc).__name__}"
                transient_fallbacks = (reason,)
                thermal_result["transient_result"] = {
                    "max_currents": max_currents.copy(),
                    "fallback_reason": reason,
                    "used_steady_fallback": True,
                }
        report = ModelRunReport(
            trained_targets=tuple(trained_targets),
            loaded_targets=tuple(loaded_targets),
            used_targets=tuple(used_targets),
            fallbacks=tuple(fallbacks),
            alignment=alignment_report,
        )
        return DlrPipelineResult(
            physical_weather=physical_weather.copy(deep=True),
            terrain_corrected_weather=terrain_corrected.copy(deep=True),
            final_weather=final_weather.copy(deep=True),
            comparison_weather=comparison.copy(deep=True),
            thermal_result=thermal_result,
            max_currents=max_currents.copy(),
            model_report=report,
            weather_metrics=self._metrics(comparison),
            transient_fallbacks=transient_fallbacks,
        )
