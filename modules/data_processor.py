# modules/data_processor.py
from dataclasses import dataclass
from datetime import datetime
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
