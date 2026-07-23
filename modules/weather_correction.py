import copy
from dataclasses import dataclass
from typing import Any, Mapping, Optional

import numpy as np
import pandas as pd

from config.config import CORRECTION_DEFAULTS


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
    ground_temp_offset: float = 15.0
    line_azimuth_deg: float = CORRECTION_DEFAULTS["line_azimuth_deg"]


class WeatherCorrectionService:
    """Apply local weather corrections once and retain the physical measurements."""

    @staticmethod
    def _finite(value: Any, default: float = 0.0, minimum: Optional[float] = None) -> float:
        try:
            number = float(value)
        except (TypeError, ValueError):
            number = default
        if not np.isfinite(number):
            number = default
        if minimum is not None:
            number = max(minimum, number)
        return float(number)

    @classmethod
    def _terrain_for_row(
        cls, terrain_lookup: Optional[Mapping[Any, Any]], row: pd.Series, index: Any
    ) -> Mapping[str, Any]:
        if not isinstance(terrain_lookup, Mapping):
            return {}

        keys = []
        for column in ("tower_id", "position"):
            if column in row and not pd.isna(row[column]):
                keys.extend((row[column], str(row[column])))
        keys.extend((index, str(index)))

        seen = set()
        for key in keys:
            try:
                marker = (type(key), key)
                if marker in seen:
                    continue
                seen.add(marker)
                terrain = terrain_lookup.get(key)
            except TypeError:
                continue
            if isinstance(terrain, Mapping):
                return terrain
        return {}

    @classmethod
    def _desert_solar_radiation(cls, solar: float, ambient_temp: float, options: CorrectionOptions) -> float:
        albedo = min(cls._finite(options.ground_albedo, 0.0, minimum=0.0), 1.0)
        offset = min(cls._finite(options.ground_temp_offset, 0.0, minimum=0.0), 100.0)
        air_temp_k = np.clip(ambient_temp, -100.0, 100.0) + 273.15
        ground_temp_k = air_temp_k + offset
        longwave_extra = 5.67e-8 * (ground_temp_k**4 - air_temp_k**4) * 0.15
        return cls._finite(solar * (1.0 + albedo * 0.3) + longwave_extra, solar, minimum=0.0)

    @classmethod
    def _wind_angle(cls, wind_direction: float, options: CorrectionOptions) -> float:
        if not options.enable_wind_direction:
            return 90.0
        azimuth = cls._finite(options.line_azimuth_deg) % 360.0
        difference = abs((wind_direction - azimuth) % 360.0)
        return min(difference, 360.0 - difference, abs(180.0 - difference))

    def apply(self, df: pd.DataFrame, terrain_lookup: Optional[dict], options: CorrectionOptions) -> pd.DataFrame:
        if not isinstance(df, pd.DataFrame):
            raise TypeError("df 必须是 pandas DataFrame")
        if "correction_stage" in df.columns:
            existing_stages = df["correction_stage"].dropna().astype(str)
            if (existing_stages != "original").any():
                raise ValueError("气象数据已经修正，不能重复应用局地修正")

        corrected = df.copy(deep=True)
        corrected.attrs = copy.deepcopy(df.attrs)

        wind_physical = corrected.get("wind_speed", pd.Series(0.0, index=corrected.index)).map(
            lambda value: self._finite(value, minimum=0.0)
        )
        temp_physical = corrected.get("ambient_temp", pd.Series(0.0, index=corrected.index)).map(
            lambda value: self._finite(value)
        )
        solar_physical = corrected.get("solar_radiation", pd.Series(0.0, index=corrected.index)).map(
            lambda value: self._finite(value, minimum=0.0)
        )

        corrected["wind_speed_physical"] = wind_physical
        corrected["ambient_temp_physical"] = temp_physical
        corrected["solar_radiation_physical"] = solar_physical

        local_winds = []
        local_temps = []
        local_solar = []
        wind_angles = []
        vertical_factors = []
        terrain_factors = []

        valid_heights = (
            self._finite(options.ref_height_m, minimum=0.0) > 0.0
            and self._finite(options.line_height_m, minimum=0.0) > 0.0
        )
        if valid_heights:
            height_ratio = min(
                self._finite(options.line_height_m, minimum=0.0)
                / self._finite(options.ref_height_m, minimum=1.0),
                1000.0,
            )
            roughness_alpha = min(self._finite(options.roughness_alpha, minimum=0.0), 1.0)
            vertical_factor = self._finite(height_ratio**roughness_alpha, 1.0, minimum=0.0)
            lapse = self._finite(options.temp_lapse_rate) * (
                self._finite(options.line_height_m) - self._finite(options.ref_height_m)
            )
            lapse = float(np.clip(lapse, -100.0, 100.0))
        else:
            vertical_factor = 1.0
            lapse = 0.0

        for row_position, (index, row) in enumerate(corrected.iterrows()):
            wind = self._finite(wind_physical.iloc[row_position], minimum=0.0)
            temp = self._finite(temp_physical.iloc[row_position])
            solar = self._finite(solar_physical.iloc[row_position], minimum=0.0)
            wind_direction = self._finite(row.get("wind_direction", 0.0)) % 360.0

            current_vertical_factor = vertical_factor if options.enable_vertical else 1.0
            if options.enable_vertical:
                wind *= current_vertical_factor
                temp -= lapse

            terrain = self._terrain_for_row(terrain_lookup, row, index)
            slope = min(self._finite(terrain.get("slope", 0.0), minimum=0.0), 90.0)
            aspect = self._finite(terrain.get("aspect", 0.0)) % 360.0
            current_terrain_factor = 1.0
            if options.enable_terrain and slope >= 2.0:
                impact = np.cos(np.radians(wind_direction - aspect))
                magnitude = min((slope / 45.0) * 0.4, 0.3)
                if impact < -0.5:
                    current_terrain_factor = 1.0 + magnitude
                elif impact > 0.5:
                    current_terrain_factor = 1.0 - magnitude
                wind *= current_terrain_factor

            if options.enable_desert:
                solar = self._desert_solar_radiation(solar, temp, options)

            local_winds.append(self._finite(wind, minimum=0.0))
            local_temps.append(self._finite(temp))
            local_solar.append(self._finite(solar, minimum=0.0))
            wind_angles.append(self._wind_angle(wind_direction, options))
            vertical_factors.append(current_vertical_factor)
            terrain_factors.append(current_terrain_factor)

        corrected["wind_speed_local"] = local_winds
        corrected["ambient_temp_local"] = local_temps
        corrected["solar_radiation_local"] = local_solar
        corrected["wind_angle_deg"] = wind_angles
        corrected["vertical_wind_factor"] = vertical_factors
        corrected["terrain_wind_factor"] = terrain_factors
        corrected["correction_stage"] = "terrain_corrected"

        # Backward-compatible columns are aliases only; all math above is single-pass.
        corrected["wind_speed_raw"] = corrected["wind_speed_physical"]
        corrected["ambient_temp_raw"] = corrected["ambient_temp_physical"]
        corrected["wind_speed_corrected"] = corrected["wind_speed_local"]
        corrected["ambient_temp_corrected"] = corrected["ambient_temp_local"]
        corrected["wind_angle_factor"] = 1.0
        return corrected
