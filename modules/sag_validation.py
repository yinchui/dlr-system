from __future__ import annotations

import math
import uuid
from dataclasses import dataclass, field, fields
from enum import Enum
from types import MappingProxyType
from typing import Any, Callable, Mapping, Optional

import numpy as np
import pandas as pd

from config.config import SAG_VALIDATION_DEFAULTS
from modules.data_processor import normalize_tower_id


_ANGLE_COLUMNS = ("angle_deg", "inclination", "inclination_angle")
_TOWER_COLUMNS = ("tower_id", "tower")
_TIMESTAMP_COLUMNS = ("timestamp", "datetime")
_EARTH_RADIUS_M = 6_371_008.8
_ABSOLUTE_ZERO_C = -273.15
_MAX_SAG_TEMPERATURE_C = 1000.0


class ParameterSource(str, Enum):
    MEASURED = "measured"
    DERIVED = "derived"
    DEFAULT = "default"


@dataclass(frozen=True)
class SourcedValue:
    value: float
    source: ParameterSource
    detail: str = ""

    def __post_init__(self) -> None:
        value = _finite_float(self.value)
        if value is None:
            raise ValueError("sourced value must be finite")
        if not isinstance(self.source, ParameterSource):
            raise TypeError("source must be a ParameterSource")
        if not isinstance(self.detail, str):
            raise TypeError("detail must be a string")
        object.__setattr__(self, "value", value)


@dataclass(frozen=True)
class InclinationRecord:
    tower_id: str
    timestamp: Any
    angle_deg: float
    sample_index: int


@dataclass(frozen=True)
class SagValidationSnapshot:
    source_run_id: str
    tower_ids: tuple[str, ...]
    timestamps: tuple[Any, ...]
    coordinates: Mapping[str, Mapping[str, float]]
    conductor_params: Mapping[str, Any]
    original_currents: tuple[tuple[float, ...], ...]
    ambient_temperatures: tuple[tuple[float, ...], ...]
    wind_speeds: tuple[tuple[float, ...], ...]
    wind_angles: tuple[tuple[float, ...], ...]
    solar_radiation: tuple[tuple[float, ...], ...]
    elevations: tuple[tuple[float, ...], ...]
    operating_currents: Optional[tuple[tuple[float, ...], ...]] = None

    def __post_init__(self) -> None:
        if not isinstance(self.source_run_id, str) or not self.source_run_id:
            raise ValueError("source_run_id must be a non-empty string")
        if not self.tower_ids or not self.timestamps:
            raise ValueError("snapshot axes cannot be empty")
        tower_ids = tuple(str(value) for value in self.tower_ids)
        timestamps = tuple(_deep_freeze(value) for value in self.timestamps)
        if len(set(tower_ids)) != len(tower_ids):
            raise ValueError("snapshot tower IDs must be unique")
        object.__setattr__(self, "tower_ids", tower_ids)
        object.__setattr__(self, "timestamps", timestamps)
        object.__setattr__(self, "coordinates", _deep_freeze(self.coordinates))
        object.__setattr__(
            self, "conductor_params", _deep_freeze(self.conductor_params)
        )
        expected_shape = (len(tower_ids), len(timestamps))
        for name in (
            "original_currents",
            "ambient_temperatures",
            "wind_speeds",
            "wind_angles",
            "solar_radiation",
            "elevations",
        ):
            array = np.asarray(getattr(self, name), dtype=float)
            if array.shape != expected_shape or not np.isfinite(array).all():
                raise ValueError(f"{name} must be a finite snapshot matrix")
            if name == "original_currents" and (array < 0.0).any():
                raise ValueError("original_currents must be non-negative")
            object.__setattr__(self, name, _frozen_matrix(array.copy()))
        if self.operating_currents is not None:
            operating = np.asarray(self.operating_currents, dtype=float)
            if (
                operating.shape != expected_shape
                or not np.isfinite(operating).all()
                or (operating < 0.0).any()
            ):
                raise ValueError(
                    "operating_currents must be a finite non-negative matrix"
                )
            object.__setattr__(
                self, "operating_currents", _frozen_matrix(operating.copy())
            )


@dataclass(frozen=True)
class ResolvedSagParameters:
    span_m: SourcedValue
    unit_weight_n_m: SourcedValue
    reference_tension_n: SourcedValue
    reference_temp_c: SourcedValue
    elastic_modulus_pa: SourcedValue
    area_m2: SourcedValue
    thermal_expansion_per_c: SourcedValue
    ambient_temp_c: SourcedValue
    theoretical_temp_c: SourcedValue
    original_current_a: SourcedValue
    recalculated_current_a: SourcedValue
    operating_current_a: Optional[SourcedValue] = None

    def values(self) -> tuple[SourcedValue, ...]:
        return tuple(
            value
            for field in fields(self)
            if isinstance((value := getattr(self, field.name)), SourcedValue)
        )


def _finite_float(value: Any) -> Optional[float]:
    if isinstance(value, (bool, np.bool_)) or value is None:
        return None
    try:
        converted = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return converted if math.isfinite(converted) else None


def _positive_float(value: Any) -> Optional[float]:
    converted = _finite_float(value)
    return converted if converted is not None and converted > 0.0 else None


def _nonnegative_float(value: Any) -> Optional[float]:
    converted = _finite_float(value)
    return converted if converted is not None and converted >= 0.0 else None


def _deep_freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {str(key): _deep_freeze(item) for key, item in value.items()}
        )
    if isinstance(value, np.ndarray):
        return tuple(_deep_freeze(item) for item in value.tolist())
    if isinstance(value, (list, tuple)):
        return tuple(_deep_freeze(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return frozenset(_deep_freeze(item) for item in value)
    if isinstance(value, np.generic):
        return value.item()
    return value


def _column_matching(frame: pd.DataFrame, candidates: tuple[str, ...], chinese: str):
    normalized = {str(column).strip().lower(): column for column in frame.columns}
    for candidate in candidates:
        if candidate in normalized:
            return normalized[candidate]
    for column in frame.columns:
        if chinese and chinese in str(column):
            return column
    return None


def _snapshot_timezone(snapshot: SagValidationSnapshot):
    for value in snapshot.timestamps:
        if isinstance(value, pd.Timestamp) and value.tzinfo is not None:
            return value.tz
    return None


def _parse_timestamp(value: Any, timezone) -> Any:
    if isinstance(value, (int, np.integer)):
        return int(value)
    try:
        timestamp = pd.Timestamp(value)
    except (TypeError, ValueError):
        return pd.NaT
    if pd.isna(timestamp):
        return pd.NaT
    if timestamp.tzinfo is None and timezone is not None:
        try:
            timestamp = timestamp.tz_localize(timezone)
        except (TypeError, ValueError):
            return pd.NaT
    return timestamp


def normalize_inclination_dataframe(
    frame: pd.DataFrame,
    *,
    selected_tower_id: str,
    snapshot: SagValidationSnapshot,
) -> pd.DataFrame:
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("inclination input must be a DataFrame")
    if not isinstance(snapshot, SagValidationSnapshot):
        raise TypeError("snapshot must be a SagValidationSnapshot")
    angle_column = _column_matching(frame, _ANGLE_COLUMNS, "倾角")
    if angle_column is None:
        raise ValueError("倾角列是必需字段")

    output = frame.copy(deep=True).reset_index(drop=True)
    output["sample_index"] = np.arange(len(output), dtype=int)
    output["angle_deg"] = pd.to_numeric(output[angle_column], errors="coerce")

    tower_column = _column_matching(frame, _TOWER_COLUMNS, "杆塔")
    if tower_column is None:
        selected = normalize_tower_id(selected_tower_id)
        if selected not in snapshot.tower_ids:
            raise ValueError("选定杆塔不在 DLR 快照中")
        output["tower_id"] = selected
    else:
        output["tower_id"] = output[tower_column].map(normalize_tower_id)

    timestamp_column = _column_matching(frame, _TIMESTAMP_COLUMNS, "时间")
    if timestamp_column is None:
        if len(output) == len(snapshot.timestamps):
            output["timestamp"] = list(snapshot.timestamps)
            output["timestamp_source"] = "snapshot"
        else:
            output["timestamp"] = output["sample_index"]
            output["timestamp_source"] = "sequence"
    else:
        timezone = _snapshot_timezone(snapshot)
        output["timestamp"] = output[timestamp_column].map(
            lambda value: _parse_timestamp(value, timezone)
        )
        output["timestamp_source"] = "measured"
    return output


def _matrix(
    value: Any,
    *,
    tower_count: int,
    time_count: int,
    name: str,
    default: float,
) -> np.ndarray:
    if value is None:
        return np.full((tower_count, time_count), default, dtype=float)
    array = np.asarray(value, dtype=float)
    if array.shape == (tower_count, time_count):
        result = array.copy()
    elif array.shape == (time_count,):
        result = np.broadcast_to(array, (tower_count, time_count)).copy()
    elif array.shape == (tower_count,):
        result = np.broadcast_to(array[:, None], (tower_count, time_count)).copy()
    else:
        raise ValueError(f"{name} shape does not match snapshot axes")
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must contain only finite values")
    return result


def _frozen_matrix(value: np.ndarray) -> tuple[tuple[float, ...], ...]:
    return tuple(tuple(float(item) for item in row) for row in value)


def build_sag_snapshot(
    line_data: Mapping[str, Any],
    conductor_params: Mapping[str, Any],
    *,
    tower_coords: Optional[Mapping[str, Mapping[str, Any]]] = None,
    source_run_id: Optional[str] = None,
) -> SagValidationSnapshot:
    if not isinstance(line_data, Mapping):
        raise TypeError("line_data must be a mapping")
    if not isinstance(conductor_params, Mapping):
        raise TypeError("conductor_params must be a mapping")
    raw_tower_ids = line_data.get("positions") or line_data.get("tower_ids")
    if raw_tower_ids is None:
        raise ValueError("line_data must contain positions")
    tower_ids = tuple(normalize_tower_id(value) for value in raw_tower_ids)
    if not tower_ids or len(set(tower_ids)) != len(tower_ids):
        raise ValueError("snapshot tower IDs must be unique and non-empty")

    raw_timestamps = line_data.get("datetimes")
    if raw_timestamps is None:
        raw_timestamps = line_data.get("timestamps")
    if raw_timestamps is None:
        raw_timestamps = line_data.get("times")
    if raw_timestamps is None:
        raise ValueError("line_data must contain datetimes or times")
    timestamps = tuple(raw_timestamps)
    if not timestamps:
        raise ValueError("snapshot timestamps cannot be empty")

    shape = {"tower_count": len(tower_ids), "time_count": len(timestamps)}
    if "max_currents" not in line_data:
        raise ValueError("line_data must contain max_currents")
    original = _matrix(
        line_data.get("max_currents"),
        **shape,
        name="max_currents",
        default=0.0,
    )
    if (original < 0.0).any():
        raise ValueError("max_currents must be non-negative")
    ambient = _matrix(
        line_data.get("local_temps", line_data.get("temps")),
        **shape,
        name="local_temps",
        default=float(SAG_VALIDATION_DEFAULTS["reference_temp_c"]),
    )
    winds = _matrix(
        line_data.get("winds", line_data.get("corrected_winds")),
        **shape,
        name="winds",
        default=0.0,
    )
    angles = _matrix(
        line_data.get("angles"),
        **shape,
        name="angles",
        default=90.0,
    )
    solar = _matrix(
        line_data.get("solar"),
        **shape,
        name="solar",
        default=0.0,
    )
    elevations = _matrix(
        line_data.get("elevations"),
        **shape,
        name="elevations",
        default=0.0,
    )

    operating = None
    for key in ("operating_currents", "actual_currents", "measured_currents"):
        if key in line_data:
            operating_array = _matrix(
                line_data[key],
                **shape,
                name=key,
                default=0.0,
            )
            if (operating_array < 0.0).any():
                raise ValueError(f"{key} must be non-negative")
            operating = _frozen_matrix(operating_array)
            break

    coordinate_source = tower_coords
    if coordinate_source is None:
        coordinate_source = line_data.get("tower_coords", {})
    coordinates = {}
    for tower_id in tower_ids:
        raw = coordinate_source.get(tower_id, {}) if coordinate_source else {}
        lon = _finite_float(raw.get("lon", raw.get("longitude"))) if raw else None
        lat = _finite_float(raw.get("lat", raw.get("latitude"))) if raw else None
        if lon is not None and lat is not None:
            coordinates[tower_id] = {"lon": lon, "lat": lat}

    run_id = source_run_id or line_data.get("run_id") or uuid.uuid4().hex
    return SagValidationSnapshot(
        source_run_id=str(run_id),
        tower_ids=tower_ids,
        timestamps=timestamps,
        coordinates=_deep_freeze(coordinates),
        conductor_params=_deep_freeze(dict(conductor_params)),
        original_currents=_frozen_matrix(original),
        ambient_temperatures=_frozen_matrix(ambient),
        wind_speeds=_frozen_matrix(winds),
        wind_angles=_frozen_matrix(angles),
        solar_radiation=_frozen_matrix(solar),
        elevations=_frozen_matrix(elevations),
        operating_currents=operating,
    )


def _row_value(row: Mapping[str, Any], *names: str) -> Any:
    for name in names:
        if name in row:
            return row[name]
    return None


def _sourced_positive(
    row: Mapping[str, Any],
    row_names: tuple[str, ...],
    conductor: Mapping[str, Any],
    conductor_name: str,
    default_name: str,
) -> SourcedValue:
    measured = _positive_float(_row_value(row, *row_names))
    if measured is not None:
        return SourcedValue(measured, ParameterSource.MEASURED, "inclination_input")
    catalog = _positive_float(conductor.get(conductor_name))
    if catalog is not None:
        return SourcedValue(catalog, ParameterSource.MEASURED, "conductor_catalog")
    return SourcedValue(
        float(SAG_VALIDATION_DEFAULTS[default_name]),
        ParameterSource.DEFAULT,
        f"SAG_VALIDATION_DEFAULTS.{default_name}",
    )


def _sourced_finite(
    row: Mapping[str, Any],
    row_names: tuple[str, ...],
    conductor: Mapping[str, Any],
    conductor_name: str,
    default_name: str,
) -> SourcedValue:
    measured = _finite_float(_row_value(row, *row_names))
    if measured is not None:
        return SourcedValue(measured, ParameterSource.MEASURED, "inclination_input")
    catalog = _finite_float(conductor.get(conductor_name))
    if catalog is not None:
        return SourcedValue(catalog, ParameterSource.MEASURED, "conductor_catalog")
    return SourcedValue(
        float(SAG_VALIDATION_DEFAULTS[default_name]),
        ParameterSource.DEFAULT,
        f"SAG_VALIDATION_DEFAULTS.{default_name}",
    )


def _material_mass_per_length(conductor: Mapping[str, Any]) -> Optional[float]:
    materials = conductor.get("materials")
    if not isinstance(materials, (list, tuple)) or not materials:
        return None
    components = []
    for material in materials:
        if not isinstance(material, Mapping):
            return None
        component = _positive_float(material.get("mass", material.get("density")))
        if component is None:
            return None
        components.append(component)
    total = sum(components)
    return total if math.isfinite(total) and total > 0.0 else None


def _haversine_distance(first: Mapping[str, float], second: Mapping[str, float]) -> float:
    lon1, lat1 = math.radians(first["lon"]), math.radians(first["lat"])
    lon2, lat2 = math.radians(second["lon"]), math.radians(second["lat"])
    delta_lon = lon2 - lon1
    delta_lat = lat2 - lat1
    haversine = (
        math.sin(delta_lat / 2.0) ** 2
        + math.cos(lat1) * math.cos(lat2) * math.sin(delta_lon / 2.0) ** 2
    )
    return 2.0 * _EARTH_RADIUS_M * math.asin(min(1.0, math.sqrt(haversine)))


def _derived_span(snapshot: SagValidationSnapshot, tower_id: str) -> Optional[float]:
    try:
        index = snapshot.tower_ids.index(tower_id)
    except ValueError:
        return None
    adjacent_index = index + 1 if index + 1 < len(snapshot.tower_ids) else index - 1
    if adjacent_index < 0:
        return None
    adjacent_id = snapshot.tower_ids[adjacent_index]
    if tower_id not in snapshot.coordinates or adjacent_id not in snapshot.coordinates:
        return None
    distance = _haversine_distance(
        snapshot.coordinates[tower_id], snapshot.coordinates[adjacent_id]
    )
    return distance if math.isfinite(distance) and distance > 0.0 else None


def _sample_indices(
    row: Mapping[str, Any], snapshot: SagValidationSnapshot
) -> tuple[Optional[int], Optional[int]]:
    try:
        tower_id = normalize_tower_id(row.get("tower_id"))
        tower_index = snapshot.tower_ids.index(tower_id)
    except (ValueError, TypeError):
        return None, None

    timestamp = row.get("timestamp")
    for index, candidate in enumerate(snapshot.timestamps):
        if timestamp == candidate:
            return tower_index, index
    if row.get("timestamp_source") == "measured":
        return tower_index, None
    sample_index = row.get("sample_index")
    if isinstance(sample_index, (int, np.integer)) and 0 <= int(sample_index) < len(
        snapshot.timestamps
    ):
        return tower_index, int(sample_index)
    if isinstance(timestamp, (int, np.integer)) and 0 <= int(timestamp) < len(
        snapshot.timestamps
    ):
        return tower_index, int(timestamp)
    return tower_index, None


def _weather_at(
    snapshot: SagValidationSnapshot, tower_index: int, time_index: int
) -> dict[str, float]:
    return {
        "T_a": snapshot.ambient_temperatures[tower_index][time_index],
        "wind_speed": snapshot.wind_speeds[tower_index][time_index],
        "wind_angle": snapshot.wind_angles[tower_index][time_index],
        "solar_radiation": snapshot.solar_radiation[tower_index][time_index],
        "elevation": snapshot.elevations[tower_index][time_index],
    }


def resolve_sag_parameters(
    *,
    inclination_row: Mapping[str, Any],
    snapshot: SagValidationSnapshot,
    conductor: Optional[Mapping[str, Any]] = None,
    temperature_solver: Optional[Callable[[Mapping[str, Any], float], float]] = None,
    current_recalculator: Optional[Callable[[Mapping[str, Any]], float]] = None,
) -> ResolvedSagParameters:
    if isinstance(inclination_row, pd.Series):
        inclination_row = inclination_row.to_dict()
    elif not isinstance(inclination_row, Mapping):
        raise TypeError("inclination_row must be a mapping")
    if not isinstance(snapshot, SagValidationSnapshot):
        raise TypeError("snapshot must be a SagValidationSnapshot")
    conductor_values = dict(snapshot.conductor_params)
    if conductor is not None:
        if not isinstance(conductor, Mapping):
            raise TypeError("conductor must be a mapping")
        conductor_values.update(conductor)

    tower_id = normalize_tower_id(inclination_row.get("tower_id"))
    measured_span = _positive_float(
        _row_value(inclination_row, "span_m", "档距", "档距(m)")
    )
    if measured_span is not None:
        span = SourcedValue(
            measured_span, ParameterSource.MEASURED, "inclination_input"
        )
    elif (derived_span := _derived_span(snapshot, tower_id)) is not None:
        span = SourcedValue(
            derived_span, ParameterSource.DERIVED, "adjacent_tower_coordinates"
        )
    else:
        span = SourcedValue(
            float(SAG_VALIDATION_DEFAULTS["span_m"]),
            ParameterSource.DEFAULT,
            "SAG_VALIDATION_DEFAULTS.span_m",
        )

    measured_weight = _positive_float(
        _row_value(
            inclination_row,
            "unit_weight_n_m",
            "单位重量",
            "单位重量(N/m)",
        )
    )
    if measured_weight is not None:
        unit_weight = SourcedValue(
            measured_weight, ParameterSource.MEASURED, "inclination_input"
        )
    else:
        mass = _positive_float(
            _row_value(
                inclination_row,
                "mass_per_length_kg_m",
                "单位质量",
                "单位质量(kg/m)",
            )
        )
        mass_detail = "inclination_input"
        if mass is None:
            mass = _positive_float(conductor_values.get("mass_per_length_kg_m"))
            mass_detail = "conductor_catalog"
        if mass is None:
            mass = _material_mass_per_length(conductor_values)
            mass_detail = "conductor_catalog_material_components"
        if mass is not None:
            unit_weight = SourcedValue(
                mass * float(SAG_VALIDATION_DEFAULTS["gravity_m_s2"]),
                ParameterSource.DERIVED,
                f"{mass_detail}:mass_times_gravity",
            )
        else:
            default_mass = _positive_float(
                SAG_VALIDATION_DEFAULTS.get("mass_per_length_kg_m")
            )
            if default_mass is None:
                default_mass = 1.0
            unit_weight = SourcedValue(
                default_mass * float(SAG_VALIDATION_DEFAULTS["gravity_m_s2"]),
                ParameterSource.DEFAULT,
                "default_mass_times_gravity",
            )

    reference_tension = _sourced_positive(
        inclination_row,
        ("reference_tension_n", "initial_tension_n", "初始张力"),
        conductor_values,
        "reference_tension_n",
        "reference_tension_n",
    )
    reference_temp = _sourced_finite(
        inclination_row,
        ("reference_temp_c", "initial_temp_c", "初始温度"),
        conductor_values,
        "reference_temp_c",
        "reference_temp_c",
    )
    elastic_modulus = _sourced_positive(
        inclination_row,
        ("elastic_modulus_pa", "弹性模量"),
        conductor_values,
        "elastic_modulus_pa",
        "elastic_modulus_pa",
    )
    area = _sourced_positive(
        inclination_row,
        ("area_m2", "截面积", "截面积(m2)"),
        conductor_values,
        "area_m2",
        "area_m2",
    )
    expansion = _sourced_positive(
        inclination_row,
        ("thermal_expansion_per_c", "线膨胀系数"),
        conductor_values,
        "thermal_expansion_per_c",
        "thermal_expansion_per_c",
    )

    tower_index, time_index = _sample_indices(inclination_row, snapshot)
    if tower_index is None or time_index is None:
        raise ValueError("inclination sample does not match the DLR snapshot")
    weather = _weather_at(snapshot, tower_index, time_index)
    measured_ambient = _finite_float(
        _row_value(inclination_row, "ambient_temp_c", "环境温度")
    )
    if measured_ambient is not None:
        ambient = SourcedValue(
            measured_ambient, ParameterSource.MEASURED, "inclination_input"
        )
        weather["T_a"] = measured_ambient
    else:
        ambient = SourcedValue(
            weather["T_a"], ParameterSource.DERIVED, "DLR_weather_snapshot"
        )

    measured_original = _nonnegative_float(
        _row_value(inclination_row, "original_current_a", "原始额定电流")
    )
    if measured_original is not None:
        original_current = SourcedValue(
            measured_original, ParameterSource.MEASURED, "inclination_input"
        )
    else:
        original_current = SourcedValue(
            snapshot.original_currents[tower_index][time_index],
            ParameterSource.DERIVED,
            "DLR_rating_snapshot",
        )

    operating_value = _nonnegative_float(
        _row_value(inclination_row, "operating_current_a", "实际运行电流")
    )
    operating_source = ParameterSource.MEASURED
    operating_detail = "inclination_input"
    if operating_value is None and snapshot.operating_currents is not None:
        operating_value = _nonnegative_float(
            snapshot.operating_currents[tower_index][time_index]
        )
        operating_source = ParameterSource.DERIVED
        operating_detail = "DLR_operating_current_snapshot"
    operating_current = (
        SourcedValue(operating_value, operating_source, operating_detail)
        if operating_value is not None
        else None
    )

    thermal_weather = {**conductor_values, **weather}
    if operating_current is not None and temperature_solver is not None:
        theoretical_value = _finite_float(
            temperature_solver(thermal_weather, operating_current.value)
        )
    else:
        theoretical_value = None
    if theoretical_value is None:
        theoretical = SourcedValue(
            reference_temp.value,
            ParameterSource.DEFAULT,
            "reference_temperature_without_operating_current_solution",
        )
    else:
        theoretical = SourcedValue(
            theoretical_value,
            ParameterSource.DERIVED,
            "injected_IEEE738_temperature_solver",
        )

    recalculated_value = None
    if current_recalculator is not None:
        recalculated_value = _nonnegative_float(
            current_recalculator(thermal_weather)
        )
    if recalculated_value is None:
        recalculated_current = SourcedValue(
            original_current.value,
            ParameterSource.DEFAULT,
            "original_rating_without_recalculation",
        )
    else:
        recalculated_current = SourcedValue(
            recalculated_value,
            ParameterSource.DERIVED,
            "injected_IEEE738_current_recalculator",
        )

    return ResolvedSagParameters(
        span_m=span,
        unit_weight_n_m=unit_weight,
        reference_tension_n=reference_tension,
        reference_temp_c=reference_temp,
        elastic_modulus_pa=elastic_modulus,
        area_m2=area,
        thermal_expansion_per_c=expansion,
        ambient_temp_c=ambient,
        theoretical_temp_c=theoretical,
        original_current_a=original_current,
        recalculated_current_a=recalculated_current,
        operating_current_a=operating_current,
    )


class SagState(str, Enum):
    NORMAL = "normal"
    RISK = "risk"
    RECOVERY = "recovery"
    INVALID = "invalid"


@dataclass(frozen=True)
class SagValidationConfig:
    formula_version: str = str(SAG_VALIDATION_DEFAULTS["formula_version"])
    min_angle_deg: float = float(SAG_VALIDATION_DEFAULTS["min_angle_deg"])
    max_angle_deg: float = float(SAG_VALIDATION_DEFAULTS["max_angle_deg"])
    base_threshold_c: float = float(SAG_VALIDATION_DEFAULTS["base_threshold_c"])
    recovery_ratio: float = float(SAG_VALIDATION_DEFAULTS["recovery_ratio"])
    recovery_samples: int = int(SAG_VALIDATION_DEFAULTS["recovery_samples"])
    recovery_alpha: float = float(SAG_VALIDATION_DEFAULTS["recovery_alpha"])
    convergence_tolerance_a: float = 1.0
    threshold_reference_span_m: float = 300.0
    angle_mad_weight: float = 2.0
    wind_mad_weight: float = 0.25
    span_weight: float = 1.0
    history_error_weight: float = 0.25
    history_window: int = 12

    def __post_init__(self) -> None:
        if not isinstance(self.formula_version, str) or not self.formula_version:
            raise ValueError("formula_version must be a non-empty string")
        numeric_positive = (
            "min_angle_deg",
            "max_angle_deg",
            "base_threshold_c",
            "convergence_tolerance_a",
            "threshold_reference_span_m",
        )
        for name in numeric_positive:
            if _positive_float(getattr(self, name)) is None:
                raise ValueError(f"{name} must be positive and finite")
        if self.min_angle_deg >= self.max_angle_deg or self.max_angle_deg >= 90.0:
            raise ValueError("angle bounds must satisfy 0 < min < max < 90")
        for name in (
            "angle_mad_weight",
            "wind_mad_weight",
            "span_weight",
            "history_error_weight",
        ):
            if _nonnegative_float(getattr(self, name)) is None:
                raise ValueError(f"{name} must be finite and non-negative")
        for name in ("recovery_ratio", "recovery_alpha"):
            value = _finite_float(getattr(self, name))
            if value is None or not 0.0 < value < 1.0:
                raise ValueError(f"{name} must be between zero and one")
        for name in ("recovery_samples", "history_window"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")


@dataclass(frozen=True)
class DeratingResult:
    factor: float
    derated_current_a: float
    checked_current_a: float


@dataclass(frozen=True)
class SagValidationResult:
    result_id: str
    tower_id: str
    timestamp: Any
    sample_index: int
    state: SagState
    angle_deg: float
    horizontal_tension_n: float
    measured_temp_c: float
    theoretical_temp_c: float
    ambient_temp_c: float
    temperature_error_c: float
    threshold_c: float
    derating_factor: float
    original_current_a: float
    recalculated_current_a: float
    checked_current_a: float
    final_current_a: float
    formula_version: str
    valid: bool = True
    error_code: str = ""
    fallback_reason: str = ""
    parameter_sources: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.state, SagState):
            raise TypeError("state must be a SagState")
        if not isinstance(self.sample_index, int) or self.sample_index < 0:
            raise ValueError("sample_index must be a non-negative integer")
        for name in (
            "angle_deg",
            "horizontal_tension_n",
            "measured_temp_c",
            "theoretical_temp_c",
            "ambient_temp_c",
            "temperature_error_c",
            "threshold_c",
            "derating_factor",
            "original_current_a",
            "recalculated_current_a",
            "checked_current_a",
            "final_current_a",
        ):
            value = _finite_float(getattr(self, name))
            if value is None:
                raise ValueError(f"{name} must be finite")
            object.__setattr__(self, name, value)
        if not isinstance(self.valid, bool):
            raise TypeError("valid must be a bool")
        object.__setattr__(
            self,
            "parameter_sources",
            _deep_freeze(dict(self.parameter_sources)),
        )


@dataclass(frozen=True)
class SagValidationBatchResult:
    rows: tuple[SagValidationResult, ...]
    formula_version: str
    source_run_id: str = ""

    def __iter__(self):
        return iter(self.rows)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index):
        return self.rows[index]


@dataclass
class _TowerState:
    state: SagState = SagState.NORMAL
    last_output_a: Optional[float] = None
    recovery_count: int = 0
    angle_history: tuple[float, ...] = ()
    wind_history: tuple[float, ...] = ()
    error_history: tuple[float, ...] = ()


def _required_finite(value: Any, name: str) -> float:
    result = _finite_float(value)
    if result is None:
        raise ValueError(f"{name} must be finite")
    return result


def _required_positive(value: Any, name: str) -> float:
    result = _positive_float(value)
    if result is None:
        raise ValueError(f"{name} must be positive and finite")
    return result


def _required_nonnegative(value: Any, name: str) -> float:
    result = _nonnegative_float(value)
    if result is None:
        raise ValueError(f"{name} must be finite and non-negative")
    return result


def horizontal_tension(
    *, weight_n_m: float, span_m: float, angle_deg: float
) -> float:
    weight = _required_positive(weight_n_m, "weight_n_m")
    span = _required_positive(span_m, "span_m")
    angle = _required_finite(angle_deg, "angle_deg")
    if not 0.0 < angle < 90.0:
        raise ValueError("angle_deg must be between zero and 90 degrees")
    tangent = math.tan(math.radians(angle))
    if not math.isfinite(tangent) or tangent <= 0.0:
        raise ValueError("angle_deg produces an invalid tangent")
    result = weight * span / (2.0 * tangent)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError("horizontal tension is invalid")
    return result


def infer_mean_temperature(
    *,
    current_tension_n: float,
    reference_tension_n: float,
    elastic_modulus_pa: float,
    area_m2: float,
    thermal_expansion_per_c: float,
    reference_temp_c: float,
) -> float:
    current_tension = _required_positive(current_tension_n, "current_tension_n")
    reference_tension = _required_positive(
        reference_tension_n, "reference_tension_n"
    )
    elastic_modulus = _required_positive(
        elastic_modulus_pa, "elastic_modulus_pa"
    )
    area = _required_positive(area_m2, "area_m2")
    expansion = _required_positive(
        thermal_expansion_per_c, "thermal_expansion_per_c"
    )
    reference_temp = _required_finite(reference_temp_c, "reference_temp_c")
    denominator = elastic_modulus * area * expansion
    if not math.isfinite(denominator) or denominator <= 0.0:
        raise ValueError("temperature denominator is invalid")
    result = reference_temp + (reference_tension - current_tension) / denominator
    if not math.isfinite(result):
        raise ValueError("inferred mean temperature is invalid")
    return result


def compute_derating(
    *,
    ambient_temp_c: float,
    theoretical_temp_c: float,
    measured_temp_c: float,
    original_current_a: float,
    recalculated_current_a: float,
) -> DeratingResult:
    ambient = _required_finite(ambient_temp_c, "ambient_temp_c")
    theoretical = _required_finite(theoretical_temp_c, "theoretical_temp_c")
    measured = _required_finite(measured_temp_c, "measured_temp_c")
    original = _required_nonnegative(original_current_a, "original_current_a")
    recalculated = _required_nonnegative(
        recalculated_current_a, "recalculated_current_a"
    )
    denominator = measured - ambient
    numerator = theoretical - ambient
    if denominator <= 0.0 or numerator < 0.0:
        raise ValueError("temperature rise margin is invalid")
    ratio = min(1.0, max(0.0, numerator / denominator))
    factor = math.sqrt(ratio)
    derated = original * factor
    checked = min(original, recalculated, derated)
    if not np.isfinite([factor, derated, checked]).all():
        raise ValueError("derating result is not finite")
    return DeratingResult(factor, derated, checked)


def _median_absolute_deviation(values) -> float:
    finite = np.asarray(
        [value for value in (_finite_float(item) for item in values) if value is not None],
        dtype=float,
    )
    if finite.size < 2:
        return 0.0
    median = float(np.median(finite))
    return float(np.median(np.abs(finite - median)))


def adaptive_temperature_threshold(
    *,
    base_threshold_c: float,
    angle_samples=(),
    wind_samples=(),
    span_m: float,
    historical_errors=(),
    reference_span_m: float = 300.0,
    angle_mad_weight: float = 2.0,
    wind_mad_weight: float = 0.25,
    span_weight: float = 1.0,
    history_error_weight: float = 0.25,
) -> float:
    base = _required_nonnegative(base_threshold_c, "base_threshold_c")
    span = _required_positive(span_m, "span_m")
    reference_span = _required_positive(reference_span_m, "reference_span_m")
    weights = tuple(
        _required_nonnegative(value, name)
        for name, value in (
            ("angle_mad_weight", angle_mad_weight),
            ("wind_mad_weight", wind_mad_weight),
            ("span_weight", span_weight),
            ("history_error_weight", history_error_weight),
        )
    )
    angle_uncertainty = _median_absolute_deviation(angle_samples)
    wind_uncertainty = _median_absolute_deviation(wind_samples)
    span_uncertainty = max(0.0, span / reference_span - 1.0)
    finite_errors = [
        abs(value)
        for value in (_finite_float(item) for item in historical_errors)
        if value is not None
    ]
    history_uncertainty = float(np.median(finite_errors)) if finite_errors else 0.0
    threshold = (
        base
        + weights[0] * angle_uncertainty
        + weights[1] * wind_uncertainty
        + weights[2] * span_uncertainty
        + weights[3] * history_uncertainty
    )
    if not math.isfinite(threshold):
        raise ValueError("adaptive threshold is invalid")
    return threshold


def _direct_parameters(row: Mapping[str, Any]) -> ResolvedSagParameters:
    def sourced_positive(name: str) -> SourcedValue:
        return SourcedValue(
            _required_positive(row.get(name), name),
            ParameterSource.MEASURED,
            "validation_input",
        )

    def sourced_finite(name: str) -> SourcedValue:
        return SourcedValue(
            _required_finite(row.get(name), name),
            ParameterSource.MEASURED,
            "validation_input",
        )

    def sourced_nonnegative(name: str) -> SourcedValue:
        return SourcedValue(
            _required_nonnegative(row.get(name), name),
            ParameterSource.MEASURED,
            "validation_input",
        )

    original = sourced_nonnegative("original_current_a")
    recalculated_value = row.get("recalculated_current_a", original.value)
    return ResolvedSagParameters(
        span_m=sourced_positive("span_m"),
        unit_weight_n_m=sourced_positive("unit_weight_n_m"),
        reference_tension_n=sourced_positive("reference_tension_n"),
        reference_temp_c=sourced_finite("reference_temp_c"),
        elastic_modulus_pa=sourced_positive("elastic_modulus_pa"),
        area_m2=sourced_positive("area_m2"),
        thermal_expansion_per_c=sourced_positive("thermal_expansion_per_c"),
        ambient_temp_c=sourced_finite("ambient_temp_c"),
        theoretical_temp_c=sourced_finite("theoretical_temp_c"),
        original_current_a=original,
        recalculated_current_a=SourcedValue(
            _required_nonnegative(
                recalculated_value, "recalculated_current_a"
            ),
            ParameterSource.MEASURED,
            "validation_input",
        ),
    )


class SagValidationService:
    def __init__(
        self,
        *,
        config: Optional[SagValidationConfig] = None,
        temperature_solver: Optional[
            Callable[[Mapping[str, Any], float], float]
        ] = None,
        current_recalculator: Optional[Callable[[Mapping[str, Any]], float]] = None,
    ) -> None:
        self.config = config or SagValidationConfig()
        if not isinstance(self.config, SagValidationConfig):
            raise TypeError("config must be a SagValidationConfig")
        self.temperature_solver = temperature_solver
        self.current_recalculator = current_recalculator
        self._tower_states: dict[str, _TowerState] = {}

    def _parameter_sources(
        self, parameters: ResolvedSagParameters
    ) -> Mapping[str, str]:
        return {
            field.name: value.source.value
            for field in fields(parameters)
            if isinstance((value := getattr(parameters, field.name)), SourcedValue)
        }

    def _threshold(
        self,
        memory: _TowerState,
        *,
        angle_deg: float,
        wind_speed: float,
        span_m: float,
    ) -> float:
        config = self.config
        return adaptive_temperature_threshold(
            base_threshold_c=config.base_threshold_c,
            angle_samples=(*memory.angle_history, angle_deg),
            wind_samples=(*memory.wind_history, wind_speed),
            span_m=span_m,
            historical_errors=memory.error_history,
            reference_span_m=config.threshold_reference_span_m,
            angle_mad_weight=config.angle_mad_weight,
            wind_mad_weight=config.wind_mad_weight,
            span_weight=config.span_weight,
            history_error_weight=config.history_error_weight,
        )

    def _remember_valid(
        self,
        memory: _TowerState,
        *,
        angle_deg: float,
        wind_speed: float,
        error_c: float,
    ) -> None:
        window = self.config.history_window
        memory.angle_history = (*memory.angle_history, angle_deg)[-window:]
        memory.wind_history = (*memory.wind_history, wind_speed)[-window:]
        memory.error_history = (*memory.error_history, error_c)[-window:]

    def _invalid_result(
        self,
        *,
        row: Mapping[str, Any],
        tower_id: str,
        sample_index: int,
        memory: _TowerState,
        error_code: str,
        parameters: Optional[ResolvedSagParameters] = None,
    ) -> SagValidationResult:
        original = _nonnegative_float(row.get("original_current_a"))
        if parameters is not None:
            original = parameters.original_current_a.value
        if original is None:
            original = memory.last_output_a if memory.last_output_a is not None else 0.0
        final = memory.last_output_a if memory.last_output_a is not None else original
        recalculated = (
            parameters.recalculated_current_a.value
            if parameters is not None
            else original
        )
        ambient = (
            parameters.ambient_temp_c.value if parameters is not None else 0.0
        )
        theoretical = (
            parameters.theoretical_temp_c.value
            if parameters is not None
            else ambient
        )
        sources = self._parameter_sources(parameters) if parameters else {}
        return SagValidationResult(
            result_id=uuid.uuid4().hex,
            tower_id=tower_id,
            timestamp=row.get("timestamp", sample_index),
            sample_index=sample_index,
            state=SagState.INVALID,
            angle_deg=_finite_float(row.get("angle_deg")) or 0.0,
            horizontal_tension_n=0.0,
            measured_temp_c=theoretical,
            theoretical_temp_c=theoretical,
            ambient_temp_c=ambient,
            temperature_error_c=0.0,
            threshold_c=self.config.base_threshold_c,
            derating_factor=1.0,
            original_current_a=original,
            recalculated_current_a=recalculated,
            checked_current_a=final,
            final_current_a=final,
            formula_version=self.config.formula_version,
            valid=False,
            error_code=error_code,
            fallback_reason=error_code,
            parameter_sources=sources,
        )

    def _validate_row(
        self,
        row: Mapping[str, Any],
        *,
        sample_index: int,
        snapshot: Optional[SagValidationSnapshot],
        conductor: Optional[Mapping[str, Any]],
    ) -> SagValidationResult:
        try:
            tower_id = normalize_tower_id(row.get("tower_id"))
        except (TypeError, ValueError):
            tower_id = str(row.get("tower_id", f"invalid-{sample_index}"))
        memory = self._tower_states.setdefault(tower_id, _TowerState())
        angle = _finite_float(row.get("angle_deg"))
        if (
            angle is None
            or angle < self.config.min_angle_deg
            or angle > self.config.max_angle_deg
        ):
            return self._invalid_result(
                row=row,
                tower_id=tower_id,
                sample_index=sample_index,
                memory=memory,
                error_code="invalid_angle",
            )

        parameters = None
        try:
            if snapshot is None:
                parameters = _direct_parameters(row)
            else:
                parameters = resolve_sag_parameters(
                    inclination_row=row,
                    snapshot=snapshot,
                    conductor=conductor,
                    temperature_solver=self.temperature_solver,
                    current_recalculator=self.current_recalculator,
                )
            tension = horizontal_tension(
                weight_n_m=parameters.unit_weight_n_m.value,
                span_m=parameters.span_m.value,
                angle_deg=angle,
            )
            measured_temp = infer_mean_temperature(
                current_tension_n=tension,
                reference_tension_n=parameters.reference_tension_n.value,
                elastic_modulus_pa=parameters.elastic_modulus_pa.value,
                area_m2=parameters.area_m2.value,
                thermal_expansion_per_c=parameters.thermal_expansion_per_c.value,
                reference_temp_c=parameters.reference_temp_c.value,
            )
            if any(
                temperature <= _ABSOLUTE_ZERO_C
                or temperature > _MAX_SAG_TEMPERATURE_C
                for temperature in (
                    parameters.ambient_temp_c.value,
                    parameters.theoretical_temp_c.value,
                    measured_temp,
                )
            ):
                return self._invalid_result(
                    row=row,
                    tower_id=tower_id,
                    sample_index=sample_index,
                    memory=memory,
                    error_code="nonphysical_temperature",
                    parameters=parameters,
                )
            error_c = measured_temp - parameters.theoretical_temp_c.value
            wind_speed = _nonnegative_float(row.get("wind_speed"))
            if wind_speed is None and snapshot is not None:
                tower_index, time_index = _sample_indices(row, snapshot)
                if tower_index is not None and time_index is not None:
                    wind_speed = snapshot.wind_speeds[tower_index][time_index]
            if wind_speed is None:
                wind_speed = 0.0
            threshold = self._threshold(
                memory,
                angle_deg=angle,
                wind_speed=wind_speed,
                span_m=parameters.span_m.value,
            )
            if not np.isfinite([tension, measured_temp, error_c, threshold]).all():
                raise ValueError("non_finite_formula_result")
        except Exception as exc:
            return self._invalid_result(
                row=row,
                tower_id=tower_id,
                sample_index=sample_index,
                memory=memory,
                error_code=f"calculation_failed:{type(exc).__name__}",
                parameters=parameters,
            )

        original = parameters.original_current_a.value
        recalculated = parameters.recalculated_current_a.value
        risk = (
            measured_temp > parameters.theoretical_temp_c.value
            and error_c > threshold
        )
        factor = 1.0
        checked = original
        if risk:
            try:
                derating = compute_derating(
                    ambient_temp_c=parameters.ambient_temp_c.value,
                    theoretical_temp_c=parameters.theoretical_temp_c.value,
                    measured_temp_c=measured_temp,
                    original_current_a=original,
                    recalculated_current_a=recalculated,
                )
            except Exception as exc:
                return self._invalid_result(
                    row=row,
                    tower_id=tower_id,
                    sample_index=sample_index,
                    memory=memory,
                    error_code=f"derating_failed:{type(exc).__name__}",
                    parameters=parameters,
                )
            factor = derating.factor
            checked = derating.checked_current_a
            final = checked
            state = SagState.RISK
            memory.recovery_count = 0
        else:
            recovery_threshold = threshold * self.config.recovery_ratio
            below_recovery = error_c < recovery_threshold
            if memory.state is SagState.RISK:
                if below_recovery:
                    memory.recovery_count += 1
                else:
                    memory.recovery_count = 0
                if memory.recovery_count >= self.config.recovery_samples:
                    previous = (
                        memory.last_output_a
                        if memory.last_output_a is not None
                        else original
                    )
                    final = (
                        self.config.recovery_alpha * original
                        + (1.0 - self.config.recovery_alpha) * previous
                    )
                    state = SagState.RECOVERY
                else:
                    final = (
                        memory.last_output_a
                        if memory.last_output_a is not None
                        else original
                    )
                    state = SagState.RISK
            elif memory.state is SagState.RECOVERY:
                previous = (
                    memory.last_output_a
                    if memory.last_output_a is not None
                    else original
                )
                if below_recovery:
                    final = (
                        self.config.recovery_alpha * original
                        + (1.0 - self.config.recovery_alpha) * previous
                    )
                    if abs(original - final) < self.config.convergence_tolerance_a:
                        final = original
                        state = SagState.NORMAL
                        memory.recovery_count = 0
                    else:
                        state = SagState.RECOVERY
                else:
                    final = previous
                    state = SagState.RECOVERY
            else:
                final = original
                state = SagState.NORMAL
                memory.recovery_count = 0

        if not np.isfinite([factor, checked, final]).all():
            return self._invalid_result(
                row=row,
                tower_id=tower_id,
                sample_index=sample_index,
                memory=memory,
                error_code="non_finite_output",
                parameters=parameters,
            )
        memory.state = state
        memory.last_output_a = float(final)
        self._remember_valid(
            memory,
            angle_deg=angle,
            wind_speed=wind_speed,
            error_c=error_c,
        )
        return SagValidationResult(
            result_id=uuid.uuid4().hex,
            tower_id=tower_id,
            timestamp=row.get("timestamp", sample_index),
            sample_index=sample_index,
            state=state,
            angle_deg=angle,
            horizontal_tension_n=tension,
            measured_temp_c=measured_temp,
            theoretical_temp_c=parameters.theoretical_temp_c.value,
            ambient_temp_c=parameters.ambient_temp_c.value,
            temperature_error_c=error_c,
            threshold_c=threshold,
            derating_factor=factor,
            original_current_a=original,
            recalculated_current_a=recalculated,
            checked_current_a=checked,
            final_current_a=final,
            formula_version=self.config.formula_version,
            parameter_sources=self._parameter_sources(parameters),
        )

    def validate_batch(
        self,
        records,
        *,
        snapshot: Optional[SagValidationSnapshot] = None,
        conductor: Optional[Mapping[str, Any]] = None,
    ) -> SagValidationBatchResult:
        if isinstance(records, pd.DataFrame):
            rows = records.to_dict(orient="records")
        else:
            try:
                rows = [dict(row) for row in records]
            except (TypeError, ValueError) as exc:
                raise TypeError("records must contain mapping rows") from exc
        results = tuple(
            self._validate_row(
                row,
                sample_index=index,
                snapshot=snapshot,
                conductor=conductor,
            )
            for index, row in enumerate(rows)
        )
        return SagValidationBatchResult(
            rows=results,
            formula_version=self.config.formula_version,
            source_run_id=snapshot.source_run_id if snapshot is not None else "",
        )
