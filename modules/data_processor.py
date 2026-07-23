# modules/data_processor.py
from dataclasses import dataclass
from datetime import datetime
import re
from typing import Optional

import numpy as np
import pandas as pd


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


def _normalize_tower_time_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    columns = list(df.columns)
    time_col = _find_column(columns, lambda c: c == "时间")
    tower_col = _find_column(columns, lambda c: "杆塔" in c)
    wind_col = _find_column(columns, lambda c: "风速WS" in c or "WS(m/s)" in c)
    if not (time_col and tower_col and wind_col):
        return None

    normalized = pd.DataFrame(index=df.index)
    normalized["position"] = df[tower_col].map(_tower_number)
    parsed_time = pd.to_datetime(df[time_col], errors="coerce")
    normalized["date"] = parsed_time.dt.strftime("%Y-%m-%d")
    normalized["time_str"] = parsed_time.dt.strftime("%H:%M")
    normalized["wind_speed"] = pd.to_numeric(df[wind_col], errors="coerce")

    direction_col = _find_column(columns, lambda c: "风向WD" in c or "WD(" in c)
    temp_col = _find_column(columns, lambda c: "温度TEM" in c or "TEM(" in c)
    humidity_col = _find_column(columns, lambda c: "相对湿度RHU" in c or "RHU(" in c)
    normalized["wind_direction"] = (
        pd.to_numeric(df[direction_col], errors="coerce") if direction_col else 0.0
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
