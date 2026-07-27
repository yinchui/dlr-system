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

from config.config import DEFAULT_INTERVAL_MINUTES, DLR_SAFETY_FACTOR, MODEL_DIR
from modules.ai_prediction import FeatureBuilder, ResidualPredictor
from modules.ai_training import (
    ResidualTrainer,
    TrainingContractError,
    bind_training_result_contract,
    training_runtime_contract_hash,
    training_runtime_contract_hash_for_scope,
)
from modules.model_registry import (
    ModelCompatibility,
    ModelKey,
    ModelLoadResult,
    ModelRegistry,
    candidate_admission_reason,
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
class ModelPromotionDecision:
    key: ModelKey
    promoted: bool
    reason: str


@dataclass(frozen=True)
class ModelRunReport:
    trained_targets: tuple[ModelKey, ...] = ()
    loaded_targets: tuple[ModelKey, ...] = ()
    used_targets: tuple[ModelKey, ...] = ()
    fallbacks: tuple[ModelFallback, ...] = ()
    promotion_decisions: tuple[ModelPromotionDecision, ...] = ()
    alignment: Optional[AlignmentReport] = None

    @property
    def active_model_count(self) -> int:
        return len(set(self.used_targets))


def _authoritative_fallbacks(
    fallbacks: list[ModelFallback],
) -> tuple[ModelFallback, ...]:
    result = []
    keyed = set()
    global_reasons = set()
    for fallback in fallbacks:
        if fallback.key is None:
            if fallback.reason in global_reasons:
                continue
            global_reasons.add(fallback.reason)
        else:
            if fallback.key in keyed:
                continue
            keyed.add(fallback.key)
        result.append(fallback)
    return tuple(result)


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


def publish_dlr_currents(values: Any) -> np.ndarray:
    currents = np.asarray(values, dtype=float)
    if not np.isfinite(currents).all() or np.any(currents < 0.0):
        raise ValueError("DLR 额定值必须为有限非负值")
    return currents * DLR_SAFETY_FACTOR


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
    input_hash: str
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
        input_hash: str,
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
        if not isinstance(input_hash, str):
            raise TypeError("input_hash must be a string")
        if len(input_hash) != 64 or any(
            character not in "0123456789abcdef" for character in input_hash
        ):
            raise ValueError("input_hash 必须是 64 位小写 SHA-256")
        object.__setattr__(self, "input_hash", input_hash)

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
            "safety_factor": float(self._thermal_result["safety_factor"]),
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


def _dlr_input_hash(
    final_weather: pd.DataFrame,
    *,
    conductor: Mapping[str, Any],
    transient_window: Optional[Mapping[str, Any]],
) -> str:
    weather_columns = (
        "tower_id",
        "timestamp",
        "ambient_temp",
        "wind_speed",
        "wind_angle_deg",
        "solar_radiation",
        "elevation",
    )
    missing = [
        column for column in weather_columns if column not in final_weather.columns
    ]
    if missing:
        raise ValueError(f"DLR 哈希缺少热核输入列: {', '.join(missing)}")
    weather = final_weather.loc[:, weather_columns].copy()
    weather["tower_id"] = weather["tower_id"].astype(str)
    weather["timestamp"] = pd.to_datetime(
        weather["timestamp"], utc=True
    ).astype("int64")
    for column in weather_columns[2:]:
        values = pd.to_numeric(weather[column], errors="raise").astype(float)
        if not np.isfinite(values.to_numpy()).all():
            raise ValueError(f"DLR 哈希热核输入列 {column} 必须为有限值")
        weather[column] = values.mask(values == 0.0, 0.0)
    weather = weather.sort_values(
        ["tower_id", "timestamp"], kind="mergesort", ignore_index=True
    )
    weather["elevation"] = weather.groupby(
        "tower_id", sort=False
    )["elevation"].transform("first")
    conductor_values = dict(conductor)
    steady_conductor = {
        key: ThermalCalculator._finite_number(conductor_values, key)
        for key in (
            "D0",
            "R_low_25",
            "R_high_75",
            "R_high_200",
            "emissivity",
            "absorptivity",
        )
    }
    steady_conductor["max_allow_temp"] = ThermalCalculator._finite_number(
        conductor_values,
        "max_allow_temp",
        80.0,
    )
    steady_conductor = {
        key: 0.0 if value == 0.0 else value
        for key, value in steady_conductor.items()
    }
    transient_inputs = None
    if transient_window is not None:
        window_values = dict(transient_window)
        window_start = ThermalCalculator._finite_number(
            window_values, "window_start_hour"
        )
        window_end = ThermalCalculator._finite_number(
            window_values, "window_end_hour"
        )
        if window_end < window_start:
            raise ValueError("window_end_hour 必须不小于 window_start_hour")
        timestamp_ns = weather["timestamp"].to_numpy(dtype=np.int64)
        weather_end = float(timestamp_ns.max() - timestamp_ns.min()) / 3.6e12
        effective_start = max(window_start, 0.0)
        effective_end = min(window_end, weather_end)
        if effective_end < effective_start:
            effective_window = {"empty": True}
        else:
            effective_window = {
                "window_start_hour": (
                    0.0 if effective_start == 0.0 else effective_start
                ),
                "window_end_hour": (
                    0.0 if effective_end == 0.0 else effective_end
                ),
            }
        transient_inputs = {
            "window": effective_window,
            "heat_capacity_j_per_m_c": (
                ThermalCalculator().calculate_heat_capacity(conductor_values)
            ),
        }
        explicit_initial = None
        if "T_s" in conductor_values:
            explicit_initial = ThermalCalculator._finite_number(
                conductor_values, "T_s"
            )
        initial_temperatures = []
        for tower_id, tower in weather.groupby("tower_id", sort=False):
            initial_temp = (
                explicit_initial
                if explicit_initial is not None
                else float(tower["ambient_temp"].iloc[0])
            )
            initial_temperatures.append(
                {
                    "tower_id": str(tower_id),
                    "initial_temp_c": (
                        0.0 if initial_temp == 0.0 else initial_temp
                    ),
                }
            )
        transient_inputs["initial_temperatures"] = initial_temperatures
    return _stable_hash(
        {
            "version": "dlr-thermal-input-v2",
            "weather": weather.to_dict(orient="records"),
            "conductor": steady_conductor,
            "transient": transient_inputs,
        }
    )


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


def _valid_coordinate_pair(
    longitude: Any,
    latitude: Any,
) -> Optional[tuple[float, float]]:
    lon = _finite_coordinate(longitude)
    lat = _finite_coordinate(latitude)
    if lon is None or lat is None:
        return None
    if not -180.0 <= lon <= 180.0 or not -90.0 <= lat <= 90.0:
        return None
    return lon, lat


def _nonempty_text_values(values: Any) -> tuple[str, ...]:
    if values is None:
        return ()
    if isinstance(values, str):
        candidates = (values,)
    else:
        try:
            candidates = tuple(values)
        except TypeError:
            candidates = (values,)
    result = set()
    for value in candidates:
        if value is None:
            continue
        missing = pd.isna(value)
        if isinstance(missing, (bool, np.bool_)) and missing:
            continue
        text = str(value).strip()
        if text:
            result.add(text)
    return tuple(sorted(result))


def _line_source_lineage(
    weather: Any,
    frame: pd.DataFrame,
) -> tuple[str, ...]:
    if isinstance(weather, WeatherUploadResult):
        file_hashes = _nonempty_text_values(
            item.sha256 for item in weather.files
        )
        if file_hashes:
            return file_hashes
    for attribute in ("source_file_hashes", "source_file_hash"):
        hashes = _nonempty_text_values(frame.attrs.get(attribute))
        if hashes:
            return hashes
    if "source_file_hash" in frame.columns:
        return _nonempty_text_values(frame["source_file_hash"])
    return ()


@dataclass(frozen=True)
class DerivedLineIdentity:
    line_id: str
    persistence_allowed: bool
    reason: str


def derive_line_identity(
    weather: Any,
    *,
    tower_coords: Optional[Mapping[Any, Any]] = None,
) -> DerivedLineIdentity:
    """Derive a persistent identity only from a complete coordinate topology."""
    frame = weather.frame if isinstance(weather, WeatherUploadResult) else weather
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("weather must be a DataFrame or WeatherUploadResult")
    if "tower_id" not in frame.columns:
        raise ValueError("weather must contain tower_id")
    tower_ids = tuple(sorted(frame["tower_id"].astype(str).unique()))
    if not tower_ids:
        raise ValueError("weather must contain at least one tower")

    explicit_coordinates: dict[str, set[tuple[float, float]]] = {}
    for key, value in (tower_coords or {}).items():
        if not isinstance(value, Mapping):
            continue
        pair = _valid_coordinate_pair(
            value.get("lon", value.get("longitude")),
            value.get("lat", value.get("latitude")),
        )
        if pair is not None:
            explicit_coordinates.setdefault(str(key), set()).add(pair)

    weather_coordinates: dict[str, set[tuple[float, float]]] = {}
    if {"longitude", "latitude"} <= set(frame.columns):
        projection = frame.loc[
            :, ["tower_id", "longitude", "latitude"]
        ].copy()
        projection["tower_id"] = projection["tower_id"].astype(str)
        for row in projection.itertuples(index=False):
            pair = _valid_coordinate_pair(row.longitude, row.latitude)
            if pair is not None:
                weather_coordinates.setdefault(str(row.tower_id), set()).add(pair)

    coordinate_candidates = {}
    for tower_id in tower_ids:
        candidates = explicit_coordinates.get(tower_id)
        if not candidates:
            candidates = weather_coordinates.get(tower_id, set())
        coordinate_candidates[tower_id] = tuple(sorted(candidates))

    ambiguous = any(
        len(candidates) > 1 for candidates in coordinate_candidates.values()
    )
    missing = any(
        len(candidates) == 0 for candidates in coordinate_candidates.values()
    )
    if not ambiguous and not missing:
        selected_coordinates = tuple(
            (tower_id, *coordinate_candidates[tower_id][0])
            for tower_id in tower_ids
        )
        payload = {
            "version": "line-identity-v4",
            "tower_ids": tower_ids,
            "coordinates": selected_coordinates,
        }
        return DerivedLineIdentity(
            line_id=f"line-coordinates-{_stable_hash(payload)[:24]}",
            persistence_allowed=True,
            reason="complete_coordinates",
        )

    payload = {
        "version": "line-runtime-v1",
        "tower_ids": tower_ids,
        "coordinate_candidates": coordinate_candidates,
        "source_file_hashes": _line_source_lineage(weather, frame),
        "weather_content_hash": _stable_hash(
            _canonical_weather_content(frame).to_dict(orient="records")
        ),
    }
    return DerivedLineIdentity(
        line_id=f"line-runtime-{_stable_hash(payload)[:24]}",
        persistence_allowed=False,
        reason="ambiguous_coordinates" if ambiguous else "missing_coordinates",
    )


def derive_line_id(
    weather: Any,
    *,
    tower_coords: Optional[Mapping[Any, Any]] = None,
) -> str:
    """Compatibility wrapper returning only the derived namespace string."""
    return derive_line_identity(weather, tower_coords=tower_coords).line_id


def _begin_registry_pipeline_run(registry: Any) -> None:
    begin = getattr(registry, "begin_pipeline_run", None)
    if callable(begin):
        begin()


def _end_registry_pipeline_run(registry: Any) -> None:
    end = getattr(registry, "end_pipeline_run", None)
    if callable(end):
        end()


def _registry_model_operations_available(registry: Any) -> bool:
    available = getattr(registry, "model_operations_available", None)
    return True if not callable(available) else bool(available())


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
        self.model_root = model_root
        self.registry = registry
        self.thermal_adapter = thermal_adapter or LongFrameThermalAdapter()

    def _registry_for_ai(self) -> ModelRegistry:
        if self.registry is None:
            self.registry = ModelRegistry(self.model_root)
        return self.registry

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

    def _sealed_trainer_for_interval(self, interval_minutes: float) -> ResidualTrainer:
        if self.trainer is not None:
            if type(self.trainer) is not ResidualTrainer:
                raise TrainingContractError("unsupported_training_backend")
            if not object.__getattribute__(
                self.trainer, "production_eligible"
            ):
                raise TrainingContractError("unsupported_training_backend")
        trainer = self._trainer_for_interval(interval_minutes)
        if type(trainer) is not ResidualTrainer or not trainer.production_eligible:
            raise TrainingContractError("unsupported_training_backend")
        trainer_state = object.__getattribute__(trainer, "__dict__")
        if any(
            method_name in trainer_state
            for method_name in ("prepare_target", "train_prepared")
        ):
            raise TrainingContractError("unsupported_training_backend")
        if self.trainer is None:
            return trainer
        return ResidualTrainer(
            feature_builder=FeatureBuilder(
                cadence_minutes=float(interval_minutes)
            )
        )

    @staticmethod
    def _sealed_load_contracts(
        keys: list[ModelKey],
        interval_minutes: float,
    ) -> tuple[ResidualTrainer, dict[ModelKey, str], dict[ModelKey, str]]:
        trainer = ResidualTrainer(
            feature_builder=FeatureBuilder(
                cadence_minutes=float(interval_minutes)
            )
        )
        spec = trainer.sealed_estimator_spec
        if not trainer.production_eligible or spec is None:
            raise TrainingContractError("sealed training backend is unavailable")

        contract_hashes = {}
        backend_ids = {}
        for key in keys:
            physical_column = _LOCAL_COLUMNS[key.target]
            contract_hashes[key] = training_runtime_contract_hash_for_scope(
                trainer,
                target=key.target,
                physical_col=physical_column,
                truth_col=f"{key.target}_truth",
                feature_columns=trainer.feature_builder.feature_columns(
                    physical_column
                ),
                cadence_minutes=float(interval_minutes),
            )
            backend_ids[key] = spec.backend_id
        return trainer, contract_hashes, backend_ids

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
        model_persistence_allowed: bool = True,
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

        compatibility = None
        keys = []
        registry = None
        registry_run = None
        loaded = {}
        loaded_targets = []
        trained_targets = []
        used_targets = []
        fallbacks: list[ModelFallback] = []
        promotion_decisions: list[ModelPromotionDecision] = []
        aligned = None
        alignment_report = None
        model_ready = False
        sealed_runtime_trainer = None
        sealed_preparation_failed = False
        expected_training_contract_hashes = {}
        expected_backend_ids = {}

        if ai_enabled:
            try:
                keys = [
                    ModelKey(str(project_id), str(line_id), str(tower_id), target)
                    for tower_id in sorted(
                        terrain_corrected["tower_id"].astype(str).unique()
                    )
                    for target in _TARGETS
                ]
                if model_persistence_allowed:
                    try:
                        (
                            sealed_runtime_trainer,
                            expected_training_contract_hashes,
                            expected_backend_ids,
                        ) = self._sealed_load_contracts(
                            keys,
                            interval_minutes,
                        )
                    except TrainingContractError:
                        sealed_preparation_failed = True
                        sealed_runtime_trainer = None
                        loaded = {
                            key: ModelLoadResult(
                                None,
                                None,
                                "training_failed:TrainingContractError",
                            )
                            for key in keys
                        }
                    if not sealed_preparation_failed:
                        compatibility = (
                            model_compatibility
                            if model_compatibility is not None
                            else self._compatibility(
                                terrain_corrected,
                                conductor=conductor,
                                correction_options=options,
                                interval_minutes=interval_minutes,
                                dem_context=dem_context,
                                coordinate_context=coordinate_context,
                            )
                        )
                        registry = self._registry_for_ai()
                        registry_run = registry
                        _begin_registry_pipeline_run(registry)
                        loaded = registry.load_many(
                            keys,
                            expected_compatibility={
                                key: compatibility for key in keys
                            },
                            expected_training_contract_hash=(
                                expected_training_contract_hashes
                            ),
                            expected_backend_id=expected_backend_ids,
                        )
                        loaded_targets = [
                            key for key in keys if loaded[key].bundle is not None
                        ]
                model_ready = True
            except Exception as exc:
                keys = []
                loaded = {}
                registry = None
                compatibility = None
                fallbacks.append(
                    ModelFallback(
                        None,
                        f"model_preparation_failed:{type(exc).__name__}",
                    )
                )

        if model_ready and truth is not None and not sealed_preparation_failed:
            try:
                truth_frame = self._weather_frame(truth, role="truth")
                physical_lineage = set(
                    _line_source_lineage(physical, physical_input)
                )
                truth_lineage = set(_line_source_lineage(truth, truth_frame))
                if physical_lineage.intersection(truth_lineage):
                    fallbacks.append(
                        ModelFallback(
                            None,
                            "truth_rejected_overlapping_source_hash",
                        )
                    )
                    truth_frame = None
                elif _weather_content_overlaps(physical_input, truth_frame):
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
                    if (
                        model_persistence_allowed
                        and registry is not None
                        and not _registry_model_operations_available(registry)
                    ):
                        break
                    load_result = loaded.get(key)
                    loaded_provisional = (
                        load_result is not None
                        and load_result.bundle is not None
                        and load_result.metadata is not None
                        and load_result.metadata.status == "active_provisional"
                        and load_result.metadata.evaluation_mode == "full_fit"
                    )
                    if (
                        load_result is not None
                        and load_result.bundle is not None
                        and not loaded_provisional
                    ):
                        continue
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
                        if load_result is None or load_result.bundle is None:
                            fallbacks.append(ModelFallback(key, "no_aligned_truth"))
                        continue
                    try:
                        if trainer is None:
                            if self.trainer is None:
                                trainer = (
                                    sealed_runtime_trainer
                                    or self._sealed_trainer_for_interval(
                                        interval_minutes
                                    )
                                )
                            else:
                                eligible_trainer = (
                                    self._sealed_trainer_for_interval(
                                        interval_minutes
                                    )
                                )
                                trainer = (
                                    sealed_runtime_trainer
                                    or eligible_trainer
                                )
                        attempt = None
                        preparation = trainer.prepare_target(
                            tower_training,
                            key.target,
                            physical_col=physical_column,
                            truth_col=truth_column,
                        )
                        runtime_contract_hash = training_runtime_contract_hash(
                            trainer,
                            preparation,
                        )
                        if model_persistence_allowed:
                            if (
                                runtime_contract_hash
                                != expected_training_contract_hashes[key]
                            ):
                                raise TrainingContractError(
                                    "training runtime contract differs from "
                                    "the preload contract"
                                )
                            attempt = registry.build_attempt(
                                key,
                                input_data_hash=preparation.input_data_hash,
                                evaluation_set_hash=(
                                    preparation.evaluation_set_hash
                                ),
                                training_contract_hash=runtime_contract_hash,
                                backend_id=expected_backend_ids[key],
                                feature_version=compatibility.feature_version,
                                champion=(
                                    load_result.metadata
                                    if load_result is not None
                                    and load_result.bundle is not None
                                    else None
                                ),
                            )
                            was_rejected = registry.was_rejected(attempt)
                            if not _registry_model_operations_available(registry):
                                break
                            if was_rejected:
                                continue
                            if loaded_provisional and (
                                preparation.input_data_hash
                                == load_result.metadata.input_data_hash
                                or preparation.evaluation_mode
                                != "temporal_holdout"
                            ):
                                continue
                        training = trainer.train_prepared(preparation)
                        if (
                            training_runtime_contract_hash(
                                trainer,
                                preparation,
                            )
                            != runtime_contract_hash
                        ):
                            raise TrainingContractError(
                                "trainer runtime contract changed during training"
                            )
                        training = bind_training_result_contract(
                            training,
                            runtime_contract_hash,
                        )
                        if not model_persistence_allowed:
                            admission_reason = candidate_admission_reason(
                                evaluation_mode=training.metadata[
                                    "evaluation_mode"
                                ],
                                evaluation_set_hash=training.metadata.get(
                                    "evaluation_set_hash"
                                ),
                                metrics=training.metrics,
                                full_fit_metrics=training.metadata.get(
                                    "full_fit_metrics"
                                ),
                                training_outcome=training.training_outcome,
                            )
                            if admission_reason:
                                fallbacks.append(
                                    ModelFallback(key, admission_reason)
                                )
                                continue
                            trained_targets.append(key)
                            loaded[key] = ModelLoadResult(
                                bundle=training.bundle,
                                metadata=None,
                            )
                        else:
                            candidate = candidate_from_training_result(
                                training,
                                project_id=str(project_id),
                                model_version=(
                                    "train-"
                                    f"{training.metadata['input_data_hash'][:24]}"
                                ),
                                compatibility=compatibility,
                            )
                            decision = registry.promote(
                                candidate,
                                attempt=attempt,
                            )
                            promotion_decisions.append(
                                ModelPromotionDecision(
                                    key=key,
                                    promoted=decision.promoted,
                                    reason=decision.reason,
                                )
                            )
                            if decision.promoted:
                                trained_targets.append(key)
                                loaded[key] = registry.load(
                                    key,
                                    expected_compatibility=compatibility,
                                    expected_training_contract_hash=(
                                        expected_training_contract_hashes[key]
                                    ),
                                    expected_backend_id=(
                                        expected_backend_ids[key]
                                    ),
                                )
                    except TrainingContractError as exc:
                        if load_result is None or load_result.bundle is None:
                            reason = (
                                "unsupported_training_backend"
                                if str(exc) == "unsupported_training_backend"
                                else "training_failed:TrainingContractError"
                            )
                            fallbacks.append(ModelFallback(key, reason))
                    except Exception as exc:
                        if load_result is None or load_result.bundle is None:
                            fallbacks.append(
                                ModelFallback(
                                    key,
                                    f"training_failed:{type(exc).__name__}",
                                )
                            )

        if registry_run is not None:
            _end_registry_pipeline_run(registry_run)

        prediction = terrain_corrected.copy(deep=True)
        for target in _TARGETS:
            prediction[f"{target}_final"] = prediction[_LOCAL_COLUMNS[target]]
            prediction[f"{target}_residual"] = 0.0
            prediction[f"{target}_used_ai"] = False
            prediction[f"{target}_fallback_reason"] = (
                "model_unavailable" if ai_enabled else "ai_disabled"
            )

        if model_ready:
            for key in keys:
                load_result = loaded.get(key)
                if load_result is None or load_result.bundle is None:
                    reason = (
                        load_result.fallback_reason
                        if load_result is not None
                        else "model_unavailable"
                    ) or "model_unavailable"
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
        raw_steady_currents = np.asarray(
            steady_result["max_currents"], dtype=float
        )
        max_currents = publish_dlr_currents(raw_steady_currents)
        thermal_result["max_currents"] = max_currents.copy()
        thermal_result["safety_factor"] = DLR_SAFETY_FACTOR
        transient_fallbacks = ()
        input_hash = None
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
                transient_currents = np.asarray(
                    transient_result.get("max_currents"), dtype=float
                )
                if transient_currents.shape != raw_steady_currents.shape:
                    raise ValueError("暂态载流量维度必须与稳态结果一致")
                published_transient_currents = publish_dlr_currents(
                    transient_currents
                )
                transient_window = {
                    "window_start_hour": transient_result[
                        "window_start_hour"
                    ],
                    "window_end_hour": transient_result["window_end_hour"],
                }
                input_hash = _dlr_input_hash(
                    final_weather,
                    conductor=conductor,
                    transient_window=transient_window,
                )
                validated_transient = dict(transient_result)
                validated_transient["max_currents"] = (
                    published_transient_currents
                )
                thermal_result["transient_result"] = validated_transient
            except Exception as exc:
                reason = f"transient_failed:{type(exc).__name__}"
                transient_fallbacks = (reason,)
                thermal_result["transient_result"] = {
                    "max_currents": max_currents.copy(),
                    "fallback_reason": reason,
                    "used_steady_fallback": True,
                }
        if input_hash is None:
            input_hash = _dlr_input_hash(
                final_weather,
                conductor=conductor,
                transient_window=None,
            )
        report = ModelRunReport(
            trained_targets=tuple(trained_targets),
            loaded_targets=tuple(loaded_targets),
            used_targets=tuple(used_targets),
            fallbacks=_authoritative_fallbacks(fallbacks),
            promotion_decisions=tuple(promotion_decisions),
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
            input_hash=input_hash,
        )
