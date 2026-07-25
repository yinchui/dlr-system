from __future__ import annotations

import math
import uuid
from dataclasses import dataclass, fields
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
        else:
            output["timestamp"] = output["sample_index"]
    else:
        timezone = _snapshot_timezone(snapshot)
        output["timestamp"] = output[timestamp_column].map(
            lambda value: _parse_timestamp(value, timezone)
        )
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
    sample_index = row.get("sample_index")
    if isinstance(sample_index, (int, np.integer)) and 0 <= int(sample_index) < len(
        snapshot.timestamps
    ):
        return tower_index, int(sample_index)
    if isinstance(timestamp, (int, np.integer)) and 0 <= int(timestamp) < len(
        snapshot.timestamps
    ):
        return tower_index, int(timestamp)
    return tower_index, 0


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
