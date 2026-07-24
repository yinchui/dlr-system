from __future__ import annotations

import copy
import dataclasses
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Optional

import numpy as np
import pandas as pd
from pyproj import CRS

from config.config import DEFAULT_INTERVAL_MINUTES, MODEL_DIR
from modules.ai_prediction import FeatureBuilder, ResidualPredictor
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
)


_TARGETS = ("wind_speed", "ambient_temp")
_LOCAL_COLUMNS = {
    "wind_speed": "wind_speed_local",
    "ambient_temp": "ambient_temp_local",
}
_GENERIC_PREDICTION_OUTPUT_COLUMNS = {
    "final",
    "residual",
    "used_ai",
    "fallback_reason",
}
_CONTENT_COLUMNS = (
    "tower_id",
    "timestamp",
    "ambient_temp",
    "wind_speed",
    "wind_direction",
    "solar_radiation",
    "humidity",
    "elevation",
    "measurement_height",
    "longitude",
    "latitude",
)


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


def _canonical_weather_content(frame: pd.DataFrame) -> pd.DataFrame:
    projection = pd.DataFrame(index=frame.index)
    projection["tower_id"] = frame["tower_id"].astype(str)
    timestamps = pd.to_datetime(frame["timestamp"], utc=True)
    projection["timestamp"] = timestamps.astype("int64")
    for column in _CONTENT_COLUMNS[2:]:
        if column not in frame.columns:
            projection[column] = np.nan
            continue
        projection[column] = pd.to_numeric(
            frame[column], errors="coerce"
        ).astype(float)
    return projection.sort_values(
        list(_CONTENT_COLUMNS), kind="mergesort", ignore_index=True
    )


def _weather_content_overlaps(
    physical: pd.DataFrame,
    truth: pd.DataFrame,
) -> bool:
    if len(physical) != len(truth):
        return False
    return _canonical_weather_content(physical).equals(
        _canonical_weather_content(truth)
    )


def _prediction_input(frame: pd.DataFrame) -> pd.DataFrame:
    output_columns = set(_GENERIC_PREDICTION_OUTPUT_COLUMNS)
    for target in _TARGETS:
        output_columns.update(
            {
                f"{target}_final",
                f"{target}_residual",
                f"{target}_used_ai",
                f"{target}_fallback_reason",
            }
        )
    return frame.drop(
        columns=[column for column in output_columns if column in frame.columns]
    )


def _validated_prediction_output(
    predicted: pd.DataFrame,
    expected_index: pd.Index,
    target: str,
    physical: Any,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if not isinstance(predicted, pd.DataFrame):
        raise TypeError("prediction output must be a pandas DataFrame")
    if len(predicted) != len(expected_index):
        raise ValueError("prediction output length does not match input")
    if not predicted.index.equals(expected_index):
        raise ValueError("prediction output index does not match input")
    required = {
        f"{target}_final",
        f"{target}_residual",
        "used_ai",
        "fallback_reason",
    }
    missing = sorted(required - set(predicted.columns))
    if missing:
        raise KeyError(f"prediction output missing columns: {', '.join(missing)}")
    physical_values = np.asarray(physical, dtype=float).reshape(-1)
    if len(physical_values) != len(predicted) or not np.isfinite(
        physical_values
    ).all():
        raise ValueError("prediction physical values must be finite and aligned")
    final = pd.to_numeric(
        predicted[f"{target}_final"], errors="raise"
    ).to_numpy(dtype=float)
    residual = pd.to_numeric(
        predicted[f"{target}_residual"], errors="raise"
    ).to_numpy(dtype=float)
    if not np.isfinite(final).all() or not np.isfinite(residual).all():
        raise ValueError("prediction output must be finite")
    with np.errstate(over="ignore", invalid="ignore"):
        expected_final = physical_values + residual
    if not np.allclose(final, expected_final, rtol=1e-12, atol=1e-12):
        raise ValueError("prediction final must equal physical plus residual")
    used_series = predicted["used_ai"]
    if not used_series.map(lambda value: isinstance(value, (bool, np.bool_))).all():
        raise TypeError("prediction used_ai must contain booleans")
    used = used_series.to_numpy(dtype=bool)
    reason_series = predicted["fallback_reason"]
    if not reason_series.map(lambda value: isinstance(value, str)).all():
        raise TypeError("prediction fallback_reason must contain strings")
    reasons = reason_series.to_numpy(dtype=object)
    fallback_rows = ~used
    if fallback_rows.any():
        if not np.allclose(
            residual[fallback_rows], 0.0, rtol=0.0, atol=1e-12
        ) or not np.allclose(
            final[fallback_rows],
            physical_values[fallback_rows],
            rtol=1e-12,
            atol=1e-12,
        ):
            raise ValueError("non-AI predictions must use physical values")
        if any(not str(reason).strip() for reason in reasons[fallback_rows]):
            raise ValueError("non-AI predictions require a fallback reason")
    return final, residual, used, reasons


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


def _dataframe_snapshot(frame: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("weather stages must be pandas DataFrames")
    result = frame.copy(deep=True)
    result.attrs = copy.deepcopy(frame.attrs)
    for column in result.select_dtypes(include=["object"]).columns:
        result[column] = result[column].map(copy.deepcopy)
    return result


def _readonly_array(value: Any, *, dtype=None) -> np.ndarray:
    source = np.asarray(value, dtype=dtype)
    result = np.array(source, copy=True, subok=False)
    if result.dtype.hasobject:
        for index in np.ndindex(result.shape):
            result[index] = copy.deepcopy(source[index])
    result.setflags(write=False)
    return result


def _freeze_snapshot(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return _readonly_array(value)
    if isinstance(value, Mapping):
        return MappingProxyType(
            {
                copy.deepcopy(key): _freeze_snapshot(item)
                for key, item in value.items()
            }
        )
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_snapshot(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return frozenset(_freeze_snapshot(item) for item in value)
    if isinstance(value, pd.DataFrame):
        return _dataframe_snapshot(value)
    return copy.deepcopy(value)


@dataclass(frozen=True, init=False, eq=False)
class DlrPipelineResult:
    _physical_weather: pd.DataFrame
    _terrain_corrected_weather: pd.DataFrame
    _final_weather: pd.DataFrame
    _comparison_weather: pd.DataFrame
    _thermal_result: Mapping[str, Any]
    _max_currents: np.ndarray
    model_report: ModelRunReport
    weather_metrics: WeatherMetrics
    transient_fallbacks: tuple[str, ...] = ()

    def __init__(
        self,
        physical_weather: pd.DataFrame,
        terrain_corrected_weather: pd.DataFrame,
        final_weather: pd.DataFrame,
        comparison_weather: pd.DataFrame,
        thermal_result: Mapping[str, Any],
        max_currents: np.ndarray,
        model_report: ModelRunReport,
        weather_metrics: WeatherMetrics,
        transient_fallbacks: tuple[str, ...] = (),
    ):
        if not isinstance(thermal_result, Mapping):
            raise TypeError("thermal_result must be a mapping")
        weather_stages = {
            "_physical_weather": physical_weather,
            "_terrain_corrected_weather": terrain_corrected_weather,
            "_final_weather": final_weather,
            "_comparison_weather": comparison_weather,
        }
        for name, frame in weather_stages.items():
            object.__setattr__(self, name, _dataframe_snapshot(frame))
        object.__setattr__(self, "_thermal_result", _freeze_snapshot(thermal_result))
        object.__setattr__(
            self,
            "_max_currents",
            _readonly_array(max_currents, dtype=float),
        )
        object.__setattr__(self, "model_report", model_report)
        object.__setattr__(self, "weather_metrics", weather_metrics)
        object.__setattr__(
            self, "transient_fallbacks", tuple(transient_fallbacks)
        )

    @property
    def physical_weather(self) -> pd.DataFrame:
        return _dataframe_snapshot(self._physical_weather)

    @property
    def terrain_corrected_weather(self) -> pd.DataFrame:
        return _dataframe_snapshot(self._terrain_corrected_weather)

    @property
    def final_weather(self) -> pd.DataFrame:
        return _dataframe_snapshot(self._final_weather)

    @property
    def comparison_weather(self) -> pd.DataFrame:
        return _dataframe_snapshot(self._comparison_weather)

    @property
    def thermal_result(self) -> Mapping[str, Any]:
        return _freeze_snapshot(self._thermal_result)

    @property
    def max_currents(self) -> np.ndarray:
        return _readonly_array(self._max_currents, dtype=float)

    def to_legacy_line_data(self) -> dict[str, Any]:
        tower_ids = tuple(self._thermal_result.get("tower_ids", ()))
        timestamps = self._thermal_result.get("timestamps")
        default_towers, default_timestamps = _weather_axes(self._final_weather)
        tower_ids = tower_ids or default_towers
        timestamps = pd.DatetimeIndex(
            default_timestamps if timestamps is None else timestamps
        )

        def matrix(frame: pd.DataFrame, column: str) -> np.ndarray:
            return _long_frame_matrix(frame, column, tower_ids, timestamps)

        physical_winds = matrix(self._physical_weather, "wind_speed")
        physical_temps = matrix(self._physical_weather, "ambient_temp")
        physical_solar = matrix(self._physical_weather, "solar_radiation")
        local_solar = matrix(
            self._terrain_corrected_weather, "solar_radiation_local"
        )
        vertical_factors = matrix(
            self._terrain_corrected_weather, "vertical_wind_factor"
        )
        terrain_factors = matrix(
            self._terrain_corrected_weather, "terrain_wind_factor"
        )

        elevations = []
        terrain_data = {}
        for index, tower_id in enumerate(tower_ids):
            tower = self._final_weather.loc[
                self._final_weather["tower_id"].astype(str) == tower_id
            ]
            row = tower.iloc[0]
            elevations.append(float(row["elevation"]))
            terrain_data[index] = {
                "tower_id": tower_id,
                "elevation": float(row["elevation"]),
                "slope": float(row["slope"]),
                "aspect": float(row["aspect"]),
                "source": copy.deepcopy(row["source"]),
                "reason": copy.deepcopy(row["reason"]),
            }

        fractional_hours = _fractional_hours(timestamps)
        solar = matrix(self._final_weather, "solar_radiation")
        temps = matrix(self._final_weather, "ambient_temp")
        winds = matrix(self._final_weather, "wind_speed")
        angles = matrix(self._final_weather, "wind_angle_deg")
        max_currents = np.asarray(self._max_currents, dtype=float).copy()
        corrected_winds = np.asarray(
            self._thermal_result["corrected_winds"], dtype=float
        ).copy()
        local_temps = np.asarray(
            self._thermal_result["local_temps"], dtype=float
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
            "comparison_weather": _dataframe_snapshot(
                self._comparison_weather
            ),
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


def _array_fingerprint(value: Any) -> dict[str, Any]:
    array = np.asarray(value)
    if array.dtype.hasobject:
        raise TypeError("runtime context arrays cannot use object dtype")
    contiguous = np.ascontiguousarray(array)
    return {
        "dtype": array.dtype.str,
        "shape": list(array.shape),
        "content_sha256": hashlib.sha256(
            contiguous.tobytes(order="C")
        ).hexdigest(),
    }


def _context_value(context: Any, name: str, default: Any = None) -> Any:
    if isinstance(context, Mapping):
        return context.get(name, default)
    return getattr(context, name, default)


def _canonical_crs(value: Any) -> str:
    if value is None:
        return "crs-unavailable-v1"
    crs = CRS.from_user_input(value)
    authority = crs.to_authority()
    if authority is not None:
        return f"{authority[0].upper()}:{authority[1]}"
    return crs.to_wkt(version="WKT2_2019", pretty=False)


def _dem_context_hashes(context: Any) -> tuple[str, str]:
    elevation = _context_value(context, "elevation")
    if elevation is None:
        raise ValueError("dem_context must contain elevation")
    mask = _context_value(
        context,
        "mask",
        np.zeros(np.asarray(elevation).shape, dtype=bool),
    )
    transform = _context_value(context, "transform")
    transform_values = (
        None
        if transform is None
        else [float(value) for value in tuple(transform)[:6]]
    )
    bounds = _context_value(context, "bounds")
    bounds_values = (
        None if bounds is None else [float(value) for value in tuple(bounds)]
    )
    payload = {
        "version": "dem-context-v1",
        "elevation": _array_fingerprint(elevation),
        "mask": _array_fingerprint(mask),
        "transform": transform_values,
        "bounds": bounds_values,
        "nodata": _context_value(context, "nodata"),
    }
    crs = _canonical_crs(_context_value(context, "crs"))
    return _stable_hash(payload), _stable_hash(crs)


def _coordinate_context_hash(context: Mapping[Any, Any]) -> str:
    coordinates = []
    for tower_id, value in context.items():
        if not isinstance(value, Mapping):
            continue
        longitude = _finite_coordinate(value.get("lon", value.get("longitude")))
        latitude = _finite_coordinate(value.get("lat", value.get("latitude")))
        if longitude is None or latitude is None:
            continue
        coordinates.append((str(tower_id), longitude, latitude))
    return _stable_hash(sorted(coordinates))


def _finite_coordinate(value: Any) -> Optional[float]:
    try:
        coordinate = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(coordinate):
        return None
    return 0.0 if coordinate == 0.0 else coordinate


def derive_line_id(
    weather: pd.DataFrame,
    *,
    tower_coords: Optional[Mapping[Any, Any]] = None,
) -> str:
    """Derive a stable model namespace from tower topology and coordinates."""
    if not isinstance(weather, pd.DataFrame):
        raise TypeError("weather must be a pandas DataFrame")
    if "tower_id" not in weather.columns:
        raise ValueError("weather must contain tower_id")
    tower_ids = tuple(sorted(weather["tower_id"].astype(str).unique()))
    if not tower_ids:
        raise ValueError("weather must contain at least one tower")

    coordinates_by_id = {
        str(key): value
        for key, value in (tower_coords or {}).items()
        if isinstance(value, Mapping)
    }
    weather_coordinates: dict[str, set[tuple[float, float]]] = {}
    if {"longitude", "latitude"} <= set(weather.columns):
        projection = weather.loc[
            :, ["tower_id", "longitude", "latitude"]
        ].copy()
        projection["tower_id"] = projection["tower_id"].astype(str)
        for row in projection.itertuples(index=False):
            longitude = _finite_coordinate(row.longitude)
            latitude = _finite_coordinate(row.latitude)
            if longitude is not None and latitude is not None:
                weather_coordinates.setdefault(str(row.tower_id), set()).add(
                    (longitude, latitude)
                )

    selected_coordinates = []
    for tower_id in tower_ids:
        coordinates = coordinates_by_id.get(tower_id, {})
        longitude = _finite_coordinate(
            coordinates.get("lon", coordinates.get("longitude"))
        )
        latitude = _finite_coordinate(
            coordinates.get("lat", coordinates.get("latitude"))
        )
        if longitude is not None and latitude is not None:
            selected_coordinates.append((tower_id, longitude, latitude))
            continue
        selected_coordinates.extend(
            (tower_id, weather_lon, weather_lat)
            for weather_lon, weather_lat in sorted(
                weather_coordinates.get(tower_id, set())
            )
        )

    payload = {
        "version": "line-identity-v2",
        "tower_ids": tower_ids,
        "coordinates": selected_coordinates,
    }
    mode = "coordinates" if selected_coordinates else "topology"
    return f"line-{mode}-{_stable_hash(payload)[:24]}"


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
        self.trainer = trainer
        self.registry = registry or ModelRegistry(model_root)
        self.thermal_adapter = thermal_adapter or LongFrameThermalAdapter()

    def _trainer_for_interval(self, interval_minutes: float) -> ResidualTrainer:
        cadence = float(interval_minutes)
        if self.trainer is None:
            return ResidualTrainer(
                feature_builder=FeatureBuilder(cadence_minutes=cadence)
            )
        feature_builder = getattr(self.trainer, "feature_builder", None)
        configured_cadence = getattr(feature_builder, "cadence_minutes", None)
        if configured_cadence is None:
            if not np.isclose(cadence, float(DEFAULT_INTERVAL_MINUTES)):
                raise ValueError(
                    "injected trainer must declare cadence_minutes for "
                    "a non-default pipeline cadence"
                )
            return self.trainer
        if not np.isclose(float(configured_cadence), cadence):
            raise ValueError(
                "injected trainer cadence does not match interval_minutes"
            )
        return self.trainer

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
        interval_minutes: float,
        dem_context: Any = None,
        coordinate_context: Optional[Mapping[Any, Any]] = None,
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
        if dem_context is None:
            dem_hash = _stable_hash(terrain_projection.to_dict(orient="records"))
            crs_hash = _stable_hash("crs-unavailable-v1")
        else:
            dem_hash, crs_hash = _dem_context_hashes(dem_context)
        coordinate_hash = (
            _stable_hash(coordinates)
            if coordinate_context is None
            else _coordinate_context_hash(coordinate_context)
        )
        return ModelCompatibility(
            dem_hash=dem_hash,
            crs_hash=crs_hash,
            coordinate_hash=coordinate_hash,
            conductor_hash=_stable_hash(conductor),
            feature_version=(
                "weather-features-v1-cadence-"
                f"{float(interval_minutes):.12g}m"
            ),
            correction_config_hash=_stable_hash(correction_options),
        )

    @staticmethod
    def _alignment_physical(frame: pd.DataFrame) -> pd.DataFrame:
        conflicting_aliases = {
            "wind_speed_physical",
            "ambient_temp_physical",
            "solar_radiation_physical",
            "source_file_hash",
        }
        result = frame.drop(
            columns=[column for column in conflicting_aliases if column in frame],
        )
        result.attrs.pop("source_file_hashes", None)
        return result

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
            )
            if column in frame.columns
        ]
        result = frame.loc[:, allowed].copy(deep=True)
        result.attrs.pop("source_file_hashes", None)
        return result

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
        dem_context: Any = None,
        coordinate_context: Optional[Mapping[Any, Any]] = None,
        correction_options: Optional[CorrectionOptions] = None,
        ai_enabled: bool = False,
        conductor: Mapping[str, Any],
        truth_tolerance: Any = "30min",
        model_compatibility: Optional[ModelCompatibility] = None,
        transient_request: Optional[Mapping[str, Any]] = None,
    ) -> DlrPipelineResult:
        options = correction_options or CorrectionOptions()
        trainer = None
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
            interval_minutes=interval_minutes,
            dem_context=dem_context,
            coordinate_context=coordinate_context,
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
                if _weather_content_overlaps(physical_input, truth_frame):
                    fallbacks.append(
                        ModelFallback(
                            None,
                            "truth_rejected_overlapping_content",
                        )
                    )
                    truth_frame = None
                if truth_frame is None:
                    aligned = None
                else:
                    aligned, alignment_report = align_physical_and_truth(
                        self._alignment_physical(terrain_corrected),
                        self._alignment_truth(truth_frame),
                        tolerance=truth_tolerance,
                        roughness_alpha=options.roughness_alpha,
                        temp_lapse_rate=options.temp_lapse_rate,
                    )
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
                        if trainer is None:
                            trainer = self._trainer_for_interval(interval_minutes)
                        training = trainer.train_target(
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
                tower = _prediction_input(prediction.loc[row_mask])
                try:
                    predicted = ResidualPredictor(
                        {key.target: load_result.bundle}
                    ).predict(
                        tower,
                        target_name=key.target,
                        physical_col=_LOCAL_COLUMNS[key.target],
                    )
                    final, residual, used_ai, prediction_reasons = (
                        _validated_prediction_output(
                            predicted,
                            tower.index,
                            key.target,
                            tower[_LOCAL_COLUMNS[key.target]],
                        )
                    )
                    candidate = prediction.copy(deep=True)
                    candidate.loc[row_mask, f"{key.target}_final"] = final
                    candidate.loc[row_mask, f"{key.target}_residual"] = residual
                    candidate.loc[row_mask, f"{key.target}_used_ai"] = used_ai
                    candidate.loc[
                        row_mask, f"{key.target}_fallback_reason"
                    ] = prediction_reasons
                    prediction = candidate
                except Exception as exc:
                    reason = f"prediction_failed:{type(exc).__name__}"
                    prediction.loc[row_mask, f"{key.target}_final"] = prediction.loc[
                        row_mask, _LOCAL_COLUMNS[key.target]
                    ].to_numpy()
                    prediction.loc[row_mask, f"{key.target}_residual"] = 0.0
                    prediction.loc[row_mask, f"{key.target}_used_ai"] = False
                    prediction.loc[
                        row_mask, f"{key.target}_fallback_reason"
                    ] = reason
                    fallbacks.append(ModelFallback(key, reason))
                    continue
                if used_ai.any():
                    used_targets.append(key)
                else:
                    failed_reasons = {
                        str(reason)
                        for reason in prediction_reasons
                        if not pd.isna(reason)
                        if str(reason).startswith("prediction_failed:")
                    }
                    fallbacks.extend(
                        ModelFallback(key, reason)
                        for reason in sorted(failed_reasons)
                    )

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
