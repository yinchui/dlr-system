# modules/weather_correction.py
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from config.config import CORRECTION_DEFAULTS
from modules.data_processor import normalize_weather_dataframe


@dataclass
class CorrectionOptions:
    enable_vertical: bool = True
    enable_terrain: bool = True
    enable_desert: bool = True
    enable_wind_direction: bool = True
    ref_height_m: float = CORRECTION_DEFAULTS["ref_height_m"]
    line_height_m: float = CORRECTION_DEFAULTS["line_height_m"]
    roughness_alpha: float = CORRECTION_DEFAULTS["roughness_alpha"]
    temp_lapse_rate: float = CORRECTION_DEFAULTS["temp_lapse_rate"]
    humidity_factor: float = CORRECTION_DEFAULTS["humidity_factor"]
    ground_albedo: float = CORRECTION_DEFAULTS["ground_albedo"]
    line_azimuth_deg: float = CORRECTION_DEFAULTS["line_azimuth_deg"]


class WeatherCorrectionService:
    ANGLE_FACTORS = [
        (0, 30, 1.15),
        (30, 60, 0.95),
        (60, 90, 0.98),
        (90, 120, 0.90),
        (120, 150, 0.88),
        (150, 181, 0.90),
    ]

    def apply(self, df: pd.DataFrame, terrain_lookup: Optional[dict], options: CorrectionOptions) -> pd.DataFrame:
        # Try to normalize if it has date/time columns, otherwise use as-is
        if "date" in df.columns and "time_str" in df.columns:
            corrected = normalize_weather_dataframe(df).copy()
        else:
            corrected = df.copy()

        corrected["wind_speed_raw"] = corrected["wind_speed"]
        corrected["ambient_temp_raw"] = corrected["ambient_temp"]

        corrected["wind_speed_corrected"] = corrected["wind_speed"]
        corrected["ambient_temp_corrected"] = corrected["ambient_temp"]
        corrected["wind_angle_factor"] = 1.0

        for idx, row in corrected.iterrows():
            terrain = (terrain_lookup or {}).get(row["position"], {"slope": 0.0, "aspect": 0.0, "elevation": 1000.0})
            wind = float(row["wind_speed"])
            temp = float(row["ambient_temp"])

            if options.enable_vertical:
                wind = wind * (options.line_height_m / options.ref_height_m) ** options.roughness_alpha
                lapse = options.temp_lapse_rate * (options.line_height_m - options.ref_height_m)
                temp = temp - lapse

            if options.enable_terrain:
                slope = terrain["slope"]
                aspect = terrain["aspect"]
                impact = np.cos(np.radians(float(row["wind_direction"]) - aspect))
                if slope >= 2 and impact < -0.5:
                    wind = wind * (1.0 + min((slope / 45.0) * 0.4, 0.3))
                elif slope >= 2 and impact > 0.5:
                    wind = wind * (1.0 - min((slope / 45.0) * 0.4, 0.3))

            if options.enable_desert:
                humidity_term = 1 - max(0, 50 - float(row.get("humidity", 50))) / 1000
                radiation_term = 1 + float(row.get("solar_radiation", 0)) / 4000
                temp = temp * humidity_term * radiation_term

            if options.enable_wind_direction:
                angle = abs((float(row["wind_direction"]) - options.line_azimuth_deg) % 180)
                factor = next(coeff for lower, upper, coeff in self.ANGLE_FACTORS if lower <= angle < upper)
                corrected.loc[idx, "wind_angle_factor"] = factor
                wind = wind * factor

            corrected.loc[idx, "wind_speed_corrected"] = wind
            corrected.loc[idx, "ambient_temp_corrected"] = temp

        return corrected
