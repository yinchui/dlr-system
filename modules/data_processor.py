# modules/data_processor.py
from dataclasses import dataclass
from datetime import datetime
import re
from typing import Optional
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import numpy as np
import pandas as pd
from pytz.exceptions import AmbiguousTimeError, NonExistentTimeError

from config.config import PHYSICAL_BOUNDS, PROJECT_TIMEZONE


@dataclass
class WeatherDataset:
    positions: list
    timestamps: np.ndarray
    times_float: np.ndarray
    elevations: dict
    temps: dict
    wind_speeds: dict
    wind_dirs: dict
    solar: np.ndarray
    humidity: dict


@dataclass(frozen=True)
class DataQualityReport:
    input_rows: int
    valid_rows: int
    dropped_rows: int
    duplicate_rows: int
    reasons: dict[str, int]


@dataclass(frozen=True)
class CanonicalWeatherResult:
    frame: pd.DataFrame
    report: DataQualityReport


COLUMN_ALIASES = {
    "位置": "position",
    "日期": "date",
    "时刻": "time_str",
    "环境温度": "ambient_temp",
    "风速": "wind_speed",
    "风向": "wind_direction",
    "太阳辐射强度": "solar_radiation",
    "太阳辐射": "solar_radiation",
    "相对湿度": "humidity",
    "海拔": "elevation",
}

_WEATHER_REQUIRED_COLUMNS = (
    "position",
    "date",
    "time_str",
    "ambient_temp",
    "wind_speed",
    "wind_direction",
)

CANONICAL_WEATHER_COLUMNS = (
    "tower_id",
    "timestamp",
    "ambient_temp",
    "wind_speed",
    "wind_direction",
    "solar_radiation",
    "humidity",
    "elevation",
    "dataset_role",
    "source_file_hash",
)


def _clean_column_name(value) -> str:
    """Normalize header punctuation so ASCII/full-width unit spellings match."""
    return (
        str(value)
        .strip()
        .replace("（", "(")
        .replace("）", ")")
        .replace("／", "/")
        .replace("℃", "C")
        .replace("°", "deg")
    )


def _find_column(columns, predicate):
    for column in columns:
        if predicate(_clean_column_name(column)):
            return column
    return None


def _tower_number(value):
    numeric = pd.to_numeric(value, errors="coerce")
    if pd.notna(numeric):
        return int(numeric)
    match = re.search(r"(\d+)", str(value))
    return int(match.group(1)) if match else np.nan


def normalize_tower_id(value) -> str:
    text = str(value).strip()
    label_match = re.search(r"(?<![\d.])(\d+)\s*号$", text)
    if label_match:
        return label_match.group(1)
    if re.fullmatch(r"\d+", text):
        return text

    numeric = pd.to_numeric(text, errors="coerce")
    if pd.notna(numeric) and np.isfinite(numeric) and numeric >= 0:
        integer = int(numeric)
        if numeric == integer:
            return str(integer)
    raise ValueError("无法解析杆塔编号")


def _parse_datetime_value(value):
    if pd.isna(value):
        return pd.NaT
    try:
        return pd.Timestamp(value)
    except (TypeError, ValueError):
        return pd.NaT


def _normalize_tower_time_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    columns = list(df.columns)
    time_col = _find_column(columns, lambda c: c == "时间")
    tower_col = _find_column(columns, lambda c: "杆塔" in c)
    wind_col = _find_column(columns, lambda c: "风速WS" in c or "WS(m/s)" in c)
    if not (time_col and tower_col and wind_col):
        return None

    normalized = pd.DataFrame(index=df.index)
    normalized["position"] = df[tower_col].map(_tower_number)
    parsed_time = df[time_col].map(_parse_datetime_value)
    normalized["date"] = parsed_time.map(
        lambda value: value.strftime("%Y-%m-%d") if pd.notna(value) else np.nan
    )
    normalized["time_str"] = parsed_time.map(
        lambda value: value.strftime("%H:%M") if pd.notna(value) else np.nan
    )
    normalized["wind_speed"] = pd.to_numeric(df[wind_col], errors="coerce")

    direction_col = _find_column(columns, lambda c: "风向WD" in c or "WD(" in c)
    temp_col = _find_column(columns, lambda c: "温度TEM" in c or "TEM(" in c)
    humidity_col = _find_column(columns, lambda c: "相对湿度RHU" in c or "RHU(" in c)
    normalized["wind_direction"] = (
        pd.to_numeric(df[direction_col], errors="coerce") if direction_col else np.nan
    )
    normalized["ambient_temp"] = (
        pd.to_numeric(df[temp_col], errors="coerce") if temp_col else np.nan
    )
    normalized["humidity"] = (
        pd.to_numeric(df[humidity_col], errors="coerce") if humidity_col else 50.0
    )

    longitude_col = _find_column(columns, lambda c: c == "经度" or "经度" in c)
    latitude_col = _find_column(columns, lambda c: c == "纬度" or "纬度" in c)
    if longitude_col:
        normalized["longitude"] = pd.to_numeric(df[longitude_col], errors="coerce")
    if latitude_col:
        normalized["latitude"] = pd.to_numeric(df[latitude_col], errors="coerce")

    normalized["solar_radiation"] = 0.0
    normalized["elevation"] = 1000.0
    normalized.attrs["input_format"] = "tower_time"
    return normalized


def normalize_weather_input_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Convert legacy and tower/time weather tables to the canonical schema."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError("气象数据必须是 pandas DataFrame")

    source = df.copy()
    source.columns = [str(column).strip() for column in source.columns]
    tower_time = _normalize_tower_time_dataframe(source)
    if tower_time is not None:
        normalized = tower_time
    else:
        rename_map = {}
        for column in source.columns:
            if column in _WEATHER_REQUIRED_COLUMNS:
                continue
            if "位置" in column:
                rename_map[column] = "position"
            elif "日期" in column:
                rename_map[column] = "date"
            elif "时刻" in column:
                rename_map[column] = "time_str"
            elif "环境温度" in column:
                rename_map[column] = "ambient_temp"
            elif "风速" in column and "相对湿度" not in column:
                rename_map[column] = "wind_speed"
            elif "风向" in column:
                rename_map[column] = "wind_direction"
            elif "太阳辐射" in column:
                rename_map[column] = "solar_radiation"
            elif "海拔" in column:
                rename_map[column] = "elevation"
            elif "相对湿度" in column:
                rename_map[column] = "humidity"

        normalized = source.rename(columns=rename_map).copy()
        normalized.attrs["input_format"] = "legacy"
        if "date" not in normalized:
            normalized["date"] = datetime.now().strftime("%Y-%m-%d")
        if "solar_radiation" not in normalized:
            normalized["solar_radiation"] = 0.0
        if "elevation" not in normalized:
            normalized["elevation"] = 1000.0
        if "humidity" not in normalized:
            normalized["humidity"] = 50.0

    if "position" in normalized:
        normalized["position"] = normalized["position"].map(_tower_number)
    for column in ("ambient_temp", "wind_speed", "wind_direction", "solar_radiation", "elevation", "humidity"):
        if column in normalized:
            normalized[column] = pd.to_numeric(normalized[column], errors="coerce")

    missing = [column for column in _WEATHER_REQUIRED_COLUMNS if column not in normalized.columns]
    if missing:
        raise ValueError(
            "缺少必需字段: " + ", ".join(missing) +
            "；检测到的原始列: " + ", ".join(map(str, source.columns))
        )
    return normalized


def _combine_date_and_time(date_value, time_value):
    if pd.isna(date_value) or pd.isna(time_value):
        return pd.NaT
    try:
        date_part = pd.Timestamp(date_value).normalize()
    except (TypeError, ValueError):
        return pd.NaT

    if hasattr(time_value, "hour"):
        hours = time_value.hour
        minutes = time_value.minute
        seconds = time_value.second
        microseconds = getattr(time_value, "microsecond", 0)
        return date_part + pd.Timedelta(
            hours=hours,
            minutes=minutes,
            seconds=seconds,
            microseconds=microseconds,
        )

    try:
        return date_part + pd.to_timedelta(str(time_value).strip())
    except (TypeError, ValueError):
        try:
            parsed_time = pd.Timestamp(time_value)
        except (TypeError, ValueError):
            return pd.NaT
        return date_part + pd.Timedelta(
            hours=parsed_time.hour,
            minutes=parsed_time.minute,
            seconds=parsed_time.second,
            microseconds=parsed_time.microsecond,
        )


def _canonical_source_frame(df: pd.DataFrame) -> pd.DataFrame:
    source = df.copy(deep=True)
    source.columns = [str(column).strip() for column in source.columns]

    if {"tower_id", "timestamp"}.issubset(source.columns):
        missing = [
            column
            for column in ("ambient_temp", "wind_speed", "wind_direction")
            if column not in source.columns
        ]
        if missing:
            raise ValueError("缺少必需字段: " + ", ".join(missing))
        prepared = pd.DataFrame(index=source.index)
        prepared["tower_id_raw"] = source["tower_id"]
        prepared["timestamp_raw"] = source["timestamp"]
        for column in (
            "ambient_temp",
            "wind_speed",
            "wind_direction",
            "solar_radiation",
            "humidity",
            "elevation",
            "longitude",
            "latitude",
        ):
            if column in source.columns:
                prepared[column] = source[column]
        for column, default in (
            ("solar_radiation", 0.0),
            ("humidity", 50.0),
            ("elevation", 1000.0),
        ):
            if column not in prepared:
                prepared[column] = default
        return prepared

    normalized = normalize_weather_input_dataframe(source)
    columns = list(source.columns)
    if normalized.attrs.get("input_format") == "tower_time":
        tower_column = _find_column(columns, lambda c: "杆塔" in c)
        time_column = _find_column(columns, lambda c: c == "时间")
        timestamp_raw = source[time_column]
    else:
        tower_column = _find_column(
            columns,
            lambda c: c in {"position", "tower_id"} or "位置" in c or "杆塔" in c,
        )
        timestamp_raw = pd.Series(
            (
                _combine_date_and_time(date_value, time_value)
                for date_value, time_value in zip(
                    normalized["date"], normalized["time_str"]
                )
            ),
            index=source.index,
        )

    prepared = pd.DataFrame(index=source.index)
    prepared["tower_id_raw"] = source[tower_column]
    prepared["timestamp_raw"] = timestamp_raw
    for column in (
        "ambient_temp",
        "wind_speed",
        "wind_direction",
        "solar_radiation",
        "humidity",
        "elevation",
    ):
        prepared[column] = normalized[column]

    for canonical_name, chinese_name in (
        ("longitude", "经度"),
        ("latitude", "纬度"),
    ):
        if canonical_name in normalized.columns:
            prepared[canonical_name] = normalized[canonical_name]
            continue
        source_column = _find_column(
            columns,
            lambda c, name=chinese_name: c == name or name in c,
        )
        if source_column is not None:
            prepared[canonical_name] = source[source_column]
    return prepared


def _parse_timestamp(value, target_timezone: ZoneInfo):
    if pd.isna(value):
        return pd.NaT
    try:
        timestamp = pd.Timestamp(value)
        if timestamp.tzinfo is None:
            return timestamp.tz_localize(
                target_timezone,
                ambiguous="raise",
                nonexistent="raise",
            )
        return timestamp.tz_convert(target_timezone)
    except (TypeError, ValueError, AmbiguousTimeError, NonExistentTimeError):
        return pd.NaT


def canonicalize_weather_frame(
    df: pd.DataFrame,
    role: str,
    timezone: str = PROJECT_TIMEZONE,
    source_hash: str = "",
) -> CanonicalWeatherResult:
    if not isinstance(df, pd.DataFrame):
        raise TypeError("气象数据必须是 pandas DataFrame")
    if role not in {"physical", "truth"}:
        raise ValueError("role 仅允许 physical 或 truth")
    try:
        target_timezone = ZoneInfo(timezone)
    except (TypeError, ZoneInfoNotFoundError) as exc:
        raise ValueError(f"无效时区: {timezone}") from exc

    prepared = _canonical_source_frame(df)
    canonical = pd.DataFrame(index=prepared.index)

    def parse_tower(value):
        try:
            return normalize_tower_id(value)
        except ValueError:
            return None

    canonical["tower_id"] = prepared["tower_id_raw"].map(parse_tower)
    timestamps = [
        _parse_timestamp(value, target_timezone)
        for value in prepared["timestamp_raw"]
    ]
    canonical["timestamp"] = pd.array(
        timestamps,
        dtype=pd.DatetimeTZDtype(tz=target_timezone),
    )
    for column in (
        "ambient_temp",
        "wind_speed",
        "wind_direction",
        "solar_radiation",
        "humidity",
        "elevation",
        "longitude",
        "latitude",
    ):
        if column in prepared.columns:
            canonical[column] = pd.to_numeric(prepared[column], errors="coerce")

    canonical["solar_radiation"] = canonical["solar_radiation"].fillna(0.0)
    canonical["humidity"] = canonical["humidity"].fillna(50.0)
    canonical["elevation"] = canonical["elevation"].fillna(1000.0)
    canonical["dataset_role"] = role
    canonical["source_file_hash"] = "" if source_hash is None else str(source_hash)

    invalid = pd.Series(False, index=canonical.index)
    reasons: dict[str, int] = {}

    def mark_invalid(reason: str, mask):
        nonlocal invalid
        mask = pd.Series(mask, index=canonical.index).fillna(False).astype(bool)
        count = int(mask.sum())
        if count:
            reasons[reason] = count
            invalid = invalid | mask

    mark_invalid("invalid_tower_id", canonical["tower_id"].isna())
    mark_invalid("invalid_timestamp", canonical["timestamp"].isna())

    for column in ("ambient_temp", "wind_speed", "wind_direction"):
        mark_invalid(f"missing_{column}", canonical[column].isna())

    wind_min, wind_max = PHYSICAL_BOUNDS["wind_speed"]
    wind_present = canonical["wind_speed"].notna()
    mark_invalid(
        "wind_speed_out_of_range",
        wind_present
        & (
            ~np.isfinite(canonical["wind_speed"])
            | (canonical["wind_speed"] < wind_min)
            | (canonical["wind_speed"] > wind_max)
        ),
    )

    temp_min, temp_max = PHYSICAL_BOUNDS["ambient_temp"]
    temp_present = canonical["ambient_temp"].notna()
    mark_invalid(
        "ambient_temp_out_of_range",
        temp_present
        & (
            ~np.isfinite(canonical["ambient_temp"])
            | (canonical["ambient_temp"] < temp_min)
            | (canonical["ambient_temp"] > temp_max)
        ),
    )

    direction_present = canonical["wind_direction"].notna()
    mark_invalid(
        "wind_direction_out_of_range",
        direction_present
        & (
            ~np.isfinite(canonical["wind_direction"])
            | (canonical["wind_direction"] < 0.0)
            | (canonical["wind_direction"] > 360.0)
        ),
    )
    canonical.loc[canonical["wind_direction"] == 360.0, "wind_direction"] = 0.0

    mark_invalid(
        "humidity_out_of_range",
        ~np.isfinite(canonical["humidity"])
        | (canonical["humidity"] < 0.0)
        | (canonical["humidity"] > 100.0),
    )
    mark_invalid(
        "solar_radiation_out_of_range",
        ~np.isfinite(canonical["solar_radiation"])
        | (canonical["solar_radiation"] < 0.0),
    )
    mark_invalid("elevation_not_finite", ~np.isfinite(canonical["elevation"]))
    for column in ("longitude", "latitude"):
        if column in canonical.columns:
            mark_invalid(f"{column}_not_finite", ~np.isfinite(canonical[column]))

    valid = canonical.loc[~invalid].copy()
    duplicate_mask = valid.duplicated(subset=["tower_id", "timestamp"], keep="first")
    duplicate_rows = int(duplicate_mask.sum())
    if duplicate_rows:
        reasons["duplicate_tower_timestamp"] = duplicate_rows
        valid = valid.loc[~duplicate_mask].copy()

    optional_location_columns = [
        column for column in ("longitude", "latitude") if column in valid.columns
    ]
    output_columns = list(CANONICAL_WEATHER_COLUMNS[:-2])
    output_columns.extend(optional_location_columns)
    output_columns.extend(CANONICAL_WEATHER_COLUMNS[-2:])
    frame = valid.loc[:, output_columns].reset_index(drop=True)
    input_rows = len(df)
    report = DataQualityReport(
        input_rows=input_rows,
        valid_rows=len(frame),
        dropped_rows=input_rows - len(frame),
        duplicate_rows=duplicate_rows,
        reasons=reasons,
    )
    return CanonicalWeatherResult(frame=frame, report=report)


def normalize_weather_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    normalized = df.rename(columns={k: v for k, v in COLUMN_ALIASES.items() if k in df.columns}).copy()
    normalized["date_obj"] = pd.to_datetime(normalized["date"], errors="coerce").dt.date
    time_objs = pd.to_datetime(normalized["time_str"], format="%H:%M", errors="coerce")
    if time_objs.isna().all():
        time_objs = pd.to_datetime(normalized["time_str"], errors="coerce")
    normalized["time_obj"] = time_objs.dt.time
    normalized["timestamp"] = normalized.apply(
        lambda row: datetime.combine(row["date_obj"], row["time_obj"]),
        axis=1,
    )
    min_ts = normalized["timestamp"].min()
    normalized["time_hour_float"] = (normalized["timestamp"] - min_ts).dt.total_seconds() / 3600.0
    normalized["solar_radiation"] = normalized["solar_radiation"].fillna(0) if "solar_radiation" in normalized else 0
    normalized["humidity"] = normalized["humidity"].fillna(50) if "humidity" in normalized else 50
    normalized["elevation"] = normalized["elevation"].fillna(1000) if "elevation" in normalized else 1000
    return normalized


def build_weather_dataset(df: pd.DataFrame) -> WeatherDataset:
    normalized = normalize_weather_dataframe(df)
    positions = sorted(normalized["position"].unique())
    time_index = normalized[["timestamp", "time_hour_float"]].drop_duplicates().sort_values("timestamp")
    return WeatherDataset(
        positions=positions,
        timestamps=time_index["timestamp"].values,
        times_float=time_index["time_hour_float"].values,
        elevations={pos: normalized.loc[normalized["position"] == pos, "elevation"].values for pos in positions},
        temps={pos: normalized.loc[normalized["position"] == pos, "ambient_temp"].values for pos in positions},
        wind_speeds={pos: normalized.loc[normalized["position"] == pos, "wind_speed"].values for pos in positions},
        wind_dirs={pos: normalized.loc[normalized["position"] == pos, "wind_direction"].values for pos in positions},
        solar=normalized.groupby("timestamp")["solar_radiation"].mean().values,
        humidity={pos: normalized.loc[normalized["position"] == pos, "humidity"].values for pos in positions},
    )


def interpolate_analysis_dataset(dataset: WeatherDataset, interval_minutes: int, terrain_lookup: Optional[dict] = None):
    num_times = int(((dataset.times_float[-1] - dataset.times_float[0]) * 60) / interval_minutes) + 1
    times_new = np.linspace(dataset.times_float[0], dataset.times_float[-1], num_times)
    temps = np.zeros((len(dataset.positions), num_times))
    winds = np.zeros((len(dataset.positions), num_times))
    angles = np.zeros((len(dataset.positions), num_times))
    elevations = np.zeros(len(dataset.positions))

    for idx, pos in enumerate(dataset.positions):
        temps[idx, :] = np.interp(times_new, dataset.times_float, dataset.temps[pos])
        winds[idx, :] = np.interp(times_new, dataset.times_float, dataset.wind_speeds[pos])
        angles[idx, :] = np.interp(times_new, dataset.times_float, dataset.wind_dirs[pos]) % 360
        elevations[idx] = np.mean(dataset.elevations[pos])

    return {
        "positions": dataset.positions,
        "times": times_new,
        "temps": temps,
        "winds": winds,
        "angles": angles,
        "elevations": elevations,
        "solar": np.interp(times_new, dataset.times_float, dataset.solar),
        "terrain_data": terrain_lookup or {},
    }
