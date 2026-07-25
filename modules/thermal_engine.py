import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Integral, Number
from typing import Dict, List, Tuple

import numpy as np


# ==============================================================================
# 核心物理计算类
# ==============================================================================


@dataclass(frozen=True)
class HeatBalanceResult:
    q_convection_natural: float
    q_convection_low_re: float
    q_convection_high_re: float
    q_convection: float
    q_radiation: float
    q_solar: float
    resistance: float
    current_a: float


class ThermalCalculator:
    """基于 IEEE Std 738-2023 的裸导线电流-温度关系计算器。"""

    _MAX_CONDUCTOR_TEMPERATURE_C = 1004.0
    _STEADY_STATE_RESIDUAL_TOLERANCE_W_PER_M = 1e-6
    _MAX_TRANSIENT_STEP_SECONDS = 10.0
    _MAX_TRANSIENT_SUBSTEPS = 1_000_000

    def __init__(self):
        # --- 基础材料参数 ---
        self.material_properties = {
            'aluminum': {'cp': 955},
            'copper': {'cp': 423},
            'steel': {'cp': 476},
            'aluminum_clad_steel': {'cp': 534}
        }

        self.solar_coeff_clear = {
            'SI': {'A': -42.2391, 'B': 63.8044, 'C': -1.9220, 'D': 3.46921e-2,
                   'E': -3.61118e-4, 'F': 1.94318e-6, 'G': -4.07608e-9},
        }

        self.elevation_coeff = {
            'SI': {'A': 1, 'B': 1.148e-4, 'C': -1.108e-8},
            'US': {'A': 1, 'B': 3.500e-5, 'C': -1.000e-9}
        }

    def calculate_heat_balance(self, params: Dict) -> HeatBalanceResult:
        """一次计算 IEEE 738 稳态热平衡的全部分项。"""
        local_params = dict(params)
        q_cn, q_c1, q_c2, q_c = self._calculate_convection_components(local_params)
        q_r = self._calculate_radiation(local_params)
        q_s = self._calculate_solar_gain(local_params)
        resistance = self._calculate_resistance(local_params)
        current = math.sqrt(max(q_c + q_r - q_s, 0.0) / resistance)
        return HeatBalanceResult(
            q_convection_natural=q_cn,
            q_convection_low_re=q_c1,
            q_convection_high_re=q_c2,
            q_convection=q_c,
            q_radiation=q_r,
            q_solar=q_s,
            resistance=resistance,
            current_a=current,
        )

    def calculate_steady_state_current(self, params: Dict) -> float:
        """计算稳态电流。"""
        return self.calculate_heat_balance(params).current_a

    @staticmethod
    def wind_angle_factor(angle: float) -> float:
        """返回归一到导线轴线 0..90 度后的 IEEE 738 风向因子。"""
        if isinstance(angle, bool):
            raise ValueError("wind_angle must be a finite number")
        try:
            wind_angle = float(angle)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("wind_angle must be a finite number") from exc
        if not math.isfinite(wind_angle):
            raise ValueError("wind_angle must be a finite number")

        effective_angle = wind_angle % 180.0
        if effective_angle > 90.0:
            effective_angle = 180.0 - effective_angle
        phi = math.radians(effective_angle)
        return (
            1.194
            - math.cos(phi)
            + 0.194 * math.cos(2.0 * phi)
            + 0.368 * math.sin(2.0 * phi)
        )

    def calculate_steady_state_temperature(self, params: Dict, current: float,
                                           max_iter: int = 100, tol: float = 1e-3) -> float:
        """已知电流推导温度"""
        current = self._finite_number({'current': current}, 'current')
        if current < 0.0:
            raise ValueError("current must be nonnegative")
        if isinstance(max_iter, bool) or not isinstance(max_iter, Integral) or max_iter <= 0:
            raise ValueError("max_iter must be a positive integer")
        tol = self._finite_number({'tol': tol}, 'tol')
        if tol <= 0.0:
            raise ValueError("tol must be greater than zero")
        params = dict(params)
        ambient_temp = self._finite_number(params, 'T_a')
        if ambient_temp >= self._MAX_CONDUCTOR_TEMPERATURE_C:
            raise ValueError("T_a is outside the physical temperature range")

        def heat_residual(conductor_temp: float) -> float:
            trial = {**params, 'T_s': conductor_temp, 'T_avg': conductor_temp}
            balance = self.calculate_heat_balance(trial)
            return (
                current ** 2 * balance.resistance
                + balance.q_solar
                - balance.q_convection
                - balance.q_radiation
            )

        def residual_is_verified(residual: float) -> bool:
            return (
                abs(residual)
                <= self._STEADY_STATE_RESIDUAL_TOLERANCE_W_PER_M
            )

        low = ambient_temp
        low_residual = heat_residual(low)
        if residual_is_verified(low_residual):
            return low

        high = min(
            max(200.0, low + 1.0), self._MAX_CONDUCTOR_TEMPERATURE_C
        )
        high_residual = heat_residual(high)
        if residual_is_verified(high_residual):
            return high
        while high_residual > 0.0 and high < self._MAX_CONDUCTOR_TEMPERATURE_C:
            high = min(
                low + 2.0 * (high - low),
                self._MAX_CONDUCTOR_TEMPERATURE_C,
            )
            high_residual = heat_residual(high)
            if residual_is_verified(high_residual):
                return high
        if high_residual > 0.0:
            raise ValueError("steady-state root is outside the physical temperature range")

        for _ in range(max_iter):
            mid = (low + high) / 2.0
            if mid == low or mid == high:
                candidate_temp, candidate_residual = min(
                    ((low, low_residual), (high, high_residual)),
                    key=lambda candidate: abs(candidate[1]),
                )
                if residual_is_verified(candidate_residual):
                    return candidate_temp
                raise ValueError("steady-state root did not converge within max_iter")
            mid_residual = heat_residual(mid)
            if residual_is_verified(mid_residual):
                return mid
            if mid_residual > 0.0:
                low = mid
                low_residual = mid_residual
            else:
                high = mid
                high_residual = mid_residual
            if high - low < tol:
                candidate_temp, candidate_residual = min(
                    ((low, low_residual), (high, high_residual)),
                    key=lambda candidate: abs(candidate[1]),
                )
                if residual_is_verified(candidate_residual):
                    return candidate_temp
        raise ValueError("steady-state root did not converge within max_iter")

    def calculate_transient_temperature(self, params: Dict, time_steps: List[float],
                                        initial_temp: float, current_profile: List[float]) -> List[float]:
        """计算暂态温度变化"""
        if not isinstance(params, Mapping):
            raise ValueError("params must be a mapping")
        initial_temp = self._finite_number(
            {'initial_temp': initial_temp}, 'initial_temp'
        )
        if initial_temp <= -273.15:
            raise ValueError("initial_temp must be above absolute zero")
        validated_steps = self._finite_numeric_vector(time_steps, 'time_steps')
        validated_currents = self._finite_numeric_vector(
            current_profile, 'current_profile'
        )
        if len(validated_steps) != len(validated_currents):
            raise ValueError("time_steps and current_profile must have the same length")
        if np.any(validated_steps <= 0.0):
            raise ValueError("time_steps values must be greater than zero")
        if np.any(validated_currents < 0.0):
            raise ValueError("current_profile values must be nonnegative")

        substep_counts = self._validated_transient_substep_counts(
            validated_steps
        )

        params = dict(params)

        temps = [initial_temp]
        current_temp = initial_temp
        heat_capacity = self.calculate_heat_capacity(params)
        if len(validated_steps) == 0:
            self._calculate_transient_heat_terms(
                {**params, 'T_avg': current_temp, 'T_s': current_temp}
            )

        for dt, current, substep_count in zip(
            validated_steps, validated_currents, substep_counts
        ):
            remaining = dt
            for _ in range(substep_count):
                substep = min(self._MAX_TRANSIENT_STEP_SECONDS, remaining)
                trial = {**params, 'T_avg': current_temp, 'T_s': current_temp}
                q_convection, q_radiation, q_solar, resistance = (
                    self._calculate_transient_heat_terms(trial)
                )
                heat_flow = (
                    resistance * current ** 2
                    + q_solar
                    - q_convection
                    - q_radiation
                )
                current_temp += heat_flow * substep / heat_capacity
                if not math.isfinite(current_temp) or current_temp <= -273.15:
                    raise ValueError("transient temperature is outside the physical range")
                remaining -= substep
            temps.append(current_temp)

        return temps

    def _calculate_transient_heat_terms(
        self, params: Dict
    ) -> Tuple[float, float, float, float]:
        """返回允许热流反向的暂态 qc、qr、qs 和 R。"""
        q_convection = self._calculate_convection_components(
            params, allow_heat_gain=True
        )[3]
        q_radiation = self._calculate_radiation(params, allow_heat_gain=True)
        q_solar = self._calculate_solar_gain(params)
        resistance = self._calculate_resistance(params)
        return q_convection, q_radiation, q_solar, resistance

    # --------------------------------------------------------------------------
    # IEEE 738 分项计算方法
    # --------------------------------------------------------------------------

    def calculate_convection(self, params: Dict) -> float:
        """兼容入口：计算取自然与强制对流最大值后的散热。"""
        return self._calculate_convection_components(dict(params))[3]

    def calculate_radiation(self, params: Dict) -> float:
        """兼容入口：计算辐射散热。"""
        return self._calculate_radiation(dict(params))

    def calculate_solar_gain(self, params: Dict) -> float:
        """兼容入口：计算太阳热增益。"""
        return self._calculate_solar_gain(dict(params))

    def calculate_resistance(self, params: Dict) -> float:
        """兼容入口：计算导线交流电阻。"""
        return self._calculate_resistance(dict(params))

    @staticmethod
    def _finite_number(params: Dict, key: str, default=None) -> float:
        if key in params:
            value = params[key]
        elif default is not None:
            value = default
        else:
            raise ValueError(f"{key} is required")
        if isinstance(value, (bool, np.bool_)):
            raise ValueError(f"{key} must be a finite number")
        try:
            number = float(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"{key} must be a finite number") from exc
        if not math.isfinite(number):
            raise ValueError(f"{key} must be a finite number")
        return number

    @staticmethod
    def _finite_numeric_vector(values, key: str) -> np.ndarray:
        if isinstance(values, np.ndarray) and np.issubdtype(values.dtype, np.bool_):
            raise ValueError(f"{key} must be a one-dimensional finite numeric sequence")
        try:
            objects = np.asarray(values, dtype=object)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(
                f"{key} must be a one-dimensional finite numeric sequence"
            ) from exc
        if objects.ndim != 1:
            raise ValueError(f"{key} must be a one-dimensional finite numeric sequence")
        if any(
            isinstance(value, (bool, np.bool_, str, bytes))
            for value in objects.flat
        ):
            raise ValueError(f"{key} must be a one-dimensional finite numeric sequence")
        try:
            vector = np.asarray(values, dtype=float)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(
                f"{key} must be a one-dimensional finite numeric sequence"
            ) from exc
        if vector.ndim != 1 or not np.all(np.isfinite(vector)):
            raise ValueError(f"{key} must be a one-dimensional finite numeric sequence")
        return vector.copy()

    def _validated_transient_substep_counts(self, time_steps) -> List[int]:
        substep_counts = []
        total_substeps = 0
        for time_step in time_steps:
            if isinstance(time_step, (bool, np.bool_, str, bytes)):
                raise ValueError("time_steps values must be finite numbers")
            try:
                seconds = float(time_step)
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError("time_steps values must be finite numbers") from exc
            if not math.isfinite(seconds) or seconds <= 0.0:
                raise ValueError(
                    "time_steps values must be finite and greater than zero"
                )
            substep_count = max(
                1,
                math.ceil(seconds / self._MAX_TRANSIENT_STEP_SECONDS),
            )
            total_substeps += substep_count
            if total_substeps > self._MAX_TRANSIENT_SUBSTEPS:
                raise ValueError("transient substep limit exceeded")
            substep_counts.append(substep_count)
        return substep_counts

    @classmethod
    def _positive_number(cls, params: Dict, key: str, default=None) -> float:
        value = cls._finite_number(params, key, default)
        if value <= 0.0:
            raise ValueError(f"{key} must be greater than zero")
        return value

    @classmethod
    def _fraction(cls, params: Dict, key: str) -> float:
        value = cls._finite_number(params, key)
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{key} must be between zero and one")
        return value

    @classmethod
    def _physical_temperatures(cls, params: Dict) -> Tuple[float, float]:
        conductor = cls._finite_number(params, 'T_s')
        ambient = cls._finite_number(params, 'T_a')
        if conductor <= -273.15:
            raise ValueError("T_s must be above absolute zero")
        if ambient <= -273.15:
            raise ValueError("T_a must be above absolute zero")
        return conductor, ambient

    @classmethod
    def _temperatures(cls, params: Dict) -> Tuple[float, float]:
        conductor, ambient = cls._physical_temperatures(params)
        if conductor < ambient:
            raise ValueError("T_s must be greater than or equal to T_a")
        return conductor, ambient

    def _calculate_convection_components(
        self, params: Dict, *, allow_heat_gain: bool = False
    ) -> Tuple[float, float, float, float]:
        diameter = self._positive_number(params, 'D0')
        if allow_heat_gain:
            conductor_temp, ambient_temp = self._physical_temperatures(params)
        else:
            conductor_temp, ambient_temp = self._temperatures(params)
        wind_speed = self._finite_number(params, 'wind_speed')
        if wind_speed < 0.0:
            raise ValueError("wind_speed must be nonnegative")
        wind_angle = self._finite_number(params, 'wind_angle', 90.0)
        elevation = self._finite_number(params, 'elevation', 0.0)

        film_temp = (conductor_temp + ambient_temp) / 2.0
        temperature_difference = conductor_temp - ambient_temp
        difference_magnitude = abs(temperature_difference)
        heat_flow_sign = -1.0 if temperature_difference < 0.0 else 1.0
        density = (
            1.293 - 1.525e-4 * elevation + 6.379e-9 * elevation ** 2
        ) / (1.0 + 0.00367 * film_temp)
        viscosity = (
            1.458e-6 * (film_temp + 273.15) ** 1.5 / (film_temp + 383.4)
        )
        conductivity = (
            2.424e-2 + 7.477e-5 * film_temp - 4.407e-9 * film_temp ** 2
        )
        if not all(
            math.isfinite(value) and value > 0.0
            for value in (density, viscosity, conductivity)
        ):
            raise ValueError("T_s and T_a produce invalid air properties")

        q_natural = (
            3.645
            * density ** 0.5
            * diameter ** 0.75
            * difference_magnitude ** 1.25
        )
        if wind_speed == 0.0:
            q_natural *= heat_flow_sign
            return q_natural, 0.0, 0.0, q_natural

        reynolds = diameter * density * wind_speed / viscosity
        angle_factor = self.wind_angle_factor(wind_angle)
        q_low_re = (
            angle_factor
            * (1.01 + 1.35 * reynolds ** 0.52)
            * conductivity
            * difference_magnitude
        )
        q_high_re = (
            angle_factor
            * 0.754
            * reynolds ** 0.60
            * conductivity
            * difference_magnitude
        )
        q_convection = max(q_natural, q_low_re, q_high_re)
        return tuple(
            heat_flow_sign * value
            for value in (q_natural, q_low_re, q_high_re, q_convection)
        )

    def _calculate_radiation(
        self, params: Dict, *, allow_heat_gain: bool = False
    ) -> float:
        diameter = self._positive_number(params, 'D0')
        emissivity = self._fraction(params, 'emissivity')
        if allow_heat_gain:
            conductor_temp, ambient_temp = self._physical_temperatures(params)
        else:
            conductor_temp, ambient_temp = self._temperatures(params)
        return 17.8 * diameter * emissivity * (
            ((conductor_temp + 273.0) / 100.0) ** 4
            - ((ambient_temp + 273.0) / 100.0) ** 4
        )

    def _calculate_solar_gain(self, params: Dict) -> float:
        absorptivity = self._fraction(params, 'absorptivity')
        diameter = self._positive_number(params, 'D0')
        if 'solar_radiation' in params:
            radiation = self._finite_number(params, 'solar_radiation')
            if radiation < 0.0:
                raise ValueError("solar_radiation must be nonnegative")
            return absorptivity * radiation * diameter

        latitude = self._finite_number(params, 'latitude')
        if not -90.0 <= latitude <= 90.0:
            raise ValueError("latitude must be between -90 and 90")
        day = self._finite_number(params, 'day_of_year')
        if not 1.0 <= day <= 366.0:
            raise ValueError("day_of_year must be between 1 and 366")
        solar_time = self._finite_number(params, 'time')
        if not 0.0 <= solar_time <= 24.0:
            raise ValueError("time must be between 0 and 24")
        line_azimuth = self._finite_number(params, 'line_azimuth', 0.0)
        self._finite_number(params, 'elevation', 0.0)

        solar_altitude = self.calculate_solar_altitude(params)
        if solar_altitude <= 0.0:
            return 0.0
        solar_azimuth = self.calculate_solar_azimuth(params)
        incidence_cosine = (
            math.cos(math.radians(solar_altitude))
            * math.cos(math.radians(solar_azimuth - line_azimuth))
        )
        incidence_angle = math.acos(max(-1.0, min(1.0, incidence_cosine)))
        clear_sky_radiation = self.calculate_solar_radiation(params, solar_altitude)
        corrected_radiation = self.calculate_elevation_corrected_radiation(
            params, clear_sky_radiation
        )
        if not math.isfinite(corrected_radiation) or corrected_radiation < 0.0:
            raise ValueError("elevation produces invalid solar radiation")
        return absorptivity * corrected_radiation * math.sin(incidence_angle) * diameter

    def _calculate_resistance(self, params: Dict) -> float:
        if 'T_avg' in params:
            average_temp = self._finite_number(params, 'T_avg')
        else:
            average_temp = self._finite_number(params, 'T_s')
        if average_temp <= -273.15:
            raise ValueError("T_avg must be above absolute zero")

        resistance_25 = self._positive_number(params, 'R_low_25', 7.283e-5)
        resistance_75 = self._positive_number(params, 'R_high_75', 8.688e-5)
        resistance_200 = self._positive_number(params, 'R_high_200', 1.220e-4)

        if average_temp <= 100.0:
            resistance = resistance_25 + (
                (resistance_75 - resistance_25) / (75.0 - 25.0)
            ) * (average_temp - 25.0)
        else:
            resistance = resistance_25 + (
                (resistance_200 - resistance_25) / (200.0 - 25.0)
            ) * (average_temp - 25.0)
        if not math.isfinite(resistance) or resistance <= 0.0:
            raise ValueError("T_avg produces nonpositive resistance")
        return resistance

    def calculate_heat_capacity(self, params: Dict) -> float:
        """按各材料单位长度质量计算导线总热容量，单位 J/(m*°C)。"""
        materials = params.get('materials')
        if (
            not isinstance(materials, Sequence)
            or isinstance(materials, (str, bytes))
            or not materials
        ):
            raise ValueError("materials must be a non-empty sequence")

        total = 0.0
        for index, material in enumerate(materials):
            if not isinstance(material, Mapping):
                raise ValueError(f"materials[{index}] must be a mapping")
            mat_type = material.get('type')
            if mat_type not in self.material_properties:
                raise ValueError(f"materials[{index}].type is unsupported")
            mass_key = 'mass' if 'mass' in material else 'density'
            if mass_key not in material:
                raise ValueError(f"materials[{index}].mass is required")
            mass = self._finite_number(material, mass_key)
            if mass <= 0.0:
                raise ValueError(f"materials[{index}].mass must be greater than zero")
            cp = self.material_properties[mat_type]['cp']
            total += mass * cp
        if not math.isfinite(total) or total <= 0.0:
            raise ValueError("materials produce invalid heat capacity")
        return total

    # --------------------------------------------------------------------------
    # 太阳位置与辅助计算
    # --------------------------------------------------------------------------

    def calculate_solar_altitude(self, params: Dict) -> float:
        lat = params['latitude']
        delta = self.calculate_solar_declination(params)
        omega = self.calculate_hour_angle(params)
        sin_Hc = math.cos(math.radians(lat)) * math.cos(math.radians(delta)) * math.cos(math.radians(omega)) + \
                 math.sin(math.radians(lat)) * math.sin(math.radians(delta))
        sin_Hc = max(min(sin_Hc, 1.0), -1.0)
        return max(0, math.degrees(math.asin(sin_Hc)))

    def calculate_solar_declination(self, params: Dict) -> float:
        day = params['day_of_year']
        return 23.45 * math.sin(math.radians(((284 + day) / 365) * 360))

    def calculate_hour_angle(self, params: Dict) -> float:
        time = params['time']
        return (time - 12) * 15

    def calculate_solar_azimuth(self, params: Dict) -> float:
        lat = params['latitude']
        delta = self.calculate_solar_declination(params)
        omega = self.calculate_hour_angle(params)
        numerator = math.sin(math.radians(omega))
        denominator = math.sin(math.radians(lat)) * math.cos(math.radians(omega)) - \
                      math.cos(math.radians(lat)) * math.tan(math.radians(delta))
        return (math.degrees(math.atan2(numerator, denominator)) + 180.0) % 360.0

    def calculate_solar_radiation(self, params: Dict, Hc: float) -> float:
        coeff = self.solar_coeff_clear['SI']
        Hc_rad = Hc
        Qs = coeff['A'] + coeff['B'] * Hc_rad + coeff['C'] * Hc_rad ** 2 + \
             coeff['D'] * Hc_rad ** 3 + coeff['E'] * Hc_rad ** 4 + \
             coeff['F'] * Hc_rad ** 5 + coeff['G'] * Hc_rad ** 6
        return max(0, Qs)

    def calculate_elevation_corrected_radiation(self, params: Dict, Qs: float) -> float:
        H_e = params.get('elevation', 0)
        coeff = self.elevation_coeff['SI']
        K_solar = coeff['A'] + coeff['B'] * H_e + coeff['C'] * H_e ** 2
        return Qs * K_solar


# ==============================================================================
# 环境生成器
# ==============================================================================

class EnvironmentGenerator:
    def __init__(self):
        pass

    def calculate_sunrise_sunset(self, lat: float, day: int) -> Tuple[float, float]:
        """计算日出日落"""
        delta = 23.45 * math.sin(math.radians(((284 + day) / 365) * 360))
        cos_omega = -math.tan(math.radians(lat)) * math.tan(math.radians(delta))
        cos_omega = max(min(cos_omega, 1.0), -1.0)
        omega = math.degrees(math.acos(cos_omega)) / 15
        return round(12 - omega, 1), round(12 + omega, 1)


# ==============================================================================
# 线路分析器
# ==============================================================================

class LineAnalyzer:
    _MIN_TIME_TO_MAX_STEP_SECONDS = 1e-3
    _MAX_TIME_TO_MAX_ITERATIONS = 128
    _TIME_TO_MAX_TOLERANCE_SECONDS = 1e-6

    def __init__(self, calculator: ThermalCalculator):
        self.calculator = calculator

    def calculate_max_current_for_points(self, observation_points: np.ndarray, elevations: np.ndarray,
                                         temps: np.ndarray, winds: np.ndarray, angles: np.ndarray,
                                         solar: np.ndarray, times: np.ndarray, max_temp: float = 80,
                                         base_params: Dict = None, terrain_data: Dict = None) -> Dict:
        if base_params is None:
            raise ValueError("base_params must explicitly select a conductor")
        if not isinstance(base_params, Mapping):
            raise ValueError("base_params must be a mapping")
        conductor_params = self._validated_conductor_params(base_params)
        if terrain_data is not None:
            if not isinstance(terrain_data, Mapping) or len(terrain_data) > 0:
                raise ValueError("地形和气象修正必须在上游一次完成")

        points = np.asarray(observation_points, dtype=object)
        if points.ndim != 1 or len(points) == 0:
            raise ValueError("observation_points must be a non-empty one-dimensional array")
        time_shape = np.asarray(times).shape
        if len(time_shape) != 1:
            raise ValueError("times must be one-dimensional")
        time_values = self._validated_array(times, "times", time_shape)
        if len(time_values) == 0:
            raise ValueError("times must not be empty")

        num_points = len(points)
        num_times = len(time_values)
        elevation_values = self._validated_array(
            elevations, "elevations", (num_points,)
        )
        temp_values = self._validated_array(
            temps, "temps", (num_points, num_times)
        )
        wind_values = self._validated_array(
            winds, "winds", (num_points, num_times)
        )
        angle_values = self._validated_array(
            angles, "angles", (num_points, num_times)
        )
        solar_values = self._validated_scalar_or_array(
            solar, "solar", num_times
        )
        if np.any(temp_values <= -273.15):
            raise ValueError("temps must be above absolute zero")
        if np.any(wind_values < 0.0):
            raise ValueError("winds must be nonnegative")
        if np.any(solar_values < 0.0):
            raise ValueError("solar must be nonnegative")

        maximum_temp = self.calculator._finite_number(
            {"max_temp": max_temp}, "max_temp"
        )
        if maximum_temp <= -273.15:
            raise ValueError("max_temp must be above absolute zero")
        if np.any(temp_values > maximum_temp):
            raise ValueError("max_temp must be greater than or equal to temps")

        max_currents = np.zeros((num_points, num_times), dtype=float)

        for i in range(num_points):
            for j in range(num_times):
                params = dict(conductor_params)
                params.update({
                    'T_s': maximum_temp,
                    'T_avg': maximum_temp,
                    'T_a': temp_values[i, j],
                    'wind_speed': wind_values[i, j],
                    'wind_angle': angle_values[i, j],
                    'elevation': elevation_values[i],
                    'time': time_values[j],
                    'solar_radiation': solar_values[j],
                })
                max_currents[i, j] = self.calculator.calculate_steady_state_current(
                    params
                )

        point_identifiers = [
            value
            if not isinstance(value, Number) or isinstance(value, bool)
            else f"position_{index}"
            for index, value in enumerate(points.tolist())
        ]
        bottleneck_indices = np.argmin(max_currents, axis=0)
        return {
            'max_currents': max_currents,
            'corrected_winds': wind_values.copy(),
            'local_temps': temp_values.copy(),
            'bottleneck_tower_ids': np.asarray(
                [point_identifiers[index] for index in bottleneck_indices],
                dtype=object,
            ),
        }

    def _validated_conductor_params(self, base_params: Mapping) -> Dict:
        conductor_params = dict(base_params)
        for key in ('D0', 'R_low_25', 'R_high_75', 'R_high_200'):
            self.calculator._positive_number(conductor_params, key)
        for key in ('emissivity', 'absorptivity'):
            self.calculator._fraction(conductor_params, key)
        return conductor_params

    def _validated_transient_conductor_params(self, params: Mapping) -> Dict:
        if not isinstance(params, Mapping):
            raise ValueError("params must be a mapping")
        conductor_params = self._validated_conductor_params(params)
        self.calculator.calculate_heat_capacity(conductor_params)
        return conductor_params

    @staticmethod
    def _validated_array(values, name: str, expected_shape: Tuple[int, ...]) -> np.ndarray:
        if LineAnalyzer._contains_boolean(values):
            raise ValueError(f"{name} must contain finite numbers")
        try:
            array = np.asarray(values, dtype=float)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"{name} must contain finite numbers") from exc
        if array.shape != expected_shape:
            raise ValueError(f"{name} must have shape {expected_shape}")
        if not np.all(np.isfinite(array)):
            raise ValueError(f"{name} must contain finite numbers")
        return array.copy()

    @staticmethod
    def _validated_scalar_or_array(
        values, name: str, expected_length: int
    ) -> np.ndarray:
        if LineAnalyzer._contains_boolean(values):
            raise ValueError(f"{name} must contain finite numbers")
        try:
            objects = np.asarray(values, dtype=object)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"{name} must contain finite numbers") from exc
        if objects.ndim == 0:
            scalar = objects.item()
            if isinstance(scalar, (str, bytes)):
                raise ValueError(f"{name} must contain finite numbers")
            try:
                number = float(scalar)
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError(f"{name} must contain finite numbers") from exc
            if not math.isfinite(number):
                raise ValueError(f"{name} must contain finite numbers")
            return np.full(expected_length, number, dtype=float)
        return LineAnalyzer._validated_array(
            values, name, (expected_length,)
        )

    def calculate_time_to_max_temp(self, params: Dict, current: float, max_temp: float,
                                   initial_temp: float, time_step: float = 10) -> float:
        """计算达到限值的时间"""
        conductor_params = self._validated_transient_conductor_params(params)
        validated_current = self.calculator._finite_number(
            {'current': current}, 'current'
        )
        if validated_current < 0.0:
            raise ValueError("current must be nonnegative")
        target_temp = self.calculator._finite_number(
            {'max_temp': max_temp}, 'max_temp'
        )
        current_temp = self.calculator._finite_number(
            {'initial_temp': initial_temp}, 'initial_temp'
        )
        if target_temp <= -273.15:
            raise ValueError("max_temp must be above absolute zero")
        if current_temp <= -273.15:
            raise ValueError("initial_temp must be above absolute zero")
        requested_step = self.calculator._finite_number(
            {'time_step': time_step}, 'time_step'
        )
        if requested_step < self._MIN_TIME_TO_MAX_STEP_SECONDS:
            raise ValueError(
                "time_step must be at least "
                f"{self._MIN_TIME_TO_MAX_STEP_SECONDS} seconds"
            )

        initial_heat_flow = self._transient_net_heat_flow(
            conductor_params, validated_current, current_temp
        )
        if target_temp <= current_temp:
            return 0.0

        target_heat_flow = self._transient_net_heat_flow(
            conductor_params, validated_current, target_temp
        )
        ambient_temp = self.calculator._finite_number(
            conductor_params, 'T_a'
        )
        target_is_reachable = target_heat_flow > 0.0
        if target_temp >= ambient_temp:
            target_balance = self.calculator.calculate_heat_balance(
                {
                    **conductor_params,
                    'T_s': target_temp,
                    'T_avg': target_temp,
                }
            )
            required_joule_heat = (
                target_balance.q_convection
                + target_balance.q_radiation
                - target_balance.q_solar
            )
            if required_joule_heat >= 0.0:
                target_is_reachable = (
                    validated_current > target_balance.current_a
                )
        if initial_heat_flow <= 0.0 or not target_is_reachable:
            return float('inf')
        if math.nextafter(current_temp, math.inf) >= target_temp:
            return 0.0

        evaluation_count = 0

        def temperature_at(elapsed: float) -> float:
            nonlocal evaluation_count
            if evaluation_count >= self._MAX_TIME_TO_MAX_ITERATIONS:
                raise ValueError("time-to-max iteration limit exceeded")
            if not math.isfinite(elapsed) or elapsed <= 0.0:
                raise ValueError("time-to-max search exhausted its time range")
            evaluation_count += 1
            return self.calculator.calculate_transient_temperature(
                params=conductor_params,
                time_steps=[elapsed],
                initial_temp=current_temp,
                current_profile=[validated_current],
            )[-1]

        safe_time = 0.0
        unsafe_time = requested_step
        while True:
            candidate_temp = temperature_at(unsafe_time)
            if candidate_temp >= target_temp:
                break
            safe_time = unsafe_time
            unsafe_time *= 2.0
            if not math.isfinite(unsafe_time):
                raise ValueError("time-to-max search exhausted its time range")

        while unsafe_time - safe_time > self._TIME_TO_MAX_TOLERANCE_SECONDS:
            midpoint = (safe_time + unsafe_time) / 2.0
            midpoint_temp = temperature_at(midpoint)
            if midpoint_temp <= target_temp:
                safe_time = midpoint
            else:
                unsafe_time = midpoint

        if safe_time > 0.0 and temperature_at(safe_time) > target_temp:
            raise ValueError("time-to-max safe bound verification failed")
        return safe_time

    def _transient_net_heat_flow(
        self, params: Mapping, current: float, conductor_temp: float
    ) -> float:
        trial = {
            **params,
            'T_s': conductor_temp,
            'T_avg': conductor_temp,
        }
        q_convection, q_radiation, q_solar, resistance = (
            self.calculator._calculate_transient_heat_terms(trial)
        )
        try:
            heat_flow = (
                resistance * current ** 2
                + q_solar
                - q_convection
                - q_radiation
            )
        except OverflowError as exc:
            raise ValueError("current produces invalid transient heat flow") from exc
        if not math.isfinite(heat_flow):
            raise ValueError("current produces invalid transient heat flow")
        return heat_flow

    def find_max_current_for_window(self, env_params, base_static, params, dt_hours, start_hour=0, end_hour=2):
        """日前调度：寻找时间窗口内的最大允许电流"""
        interval_hours = self.calculator._finite_number(
            {'dt_hours': dt_hours}, 'dt_hours'
        )
        if interval_hours <= 0.0:
            raise ValueError("dt_hours must be greater than zero")
        if not isinstance(params, Mapping):
            raise ValueError("params must be a mapping")
        conductor_params = self._validated_transient_conductor_params(params)
        target_temp = self.calculator._finite_number(
            conductor_params, 'max_allow_temp'
        )
        if target_temp <= -273.15:
            raise ValueError("max_allow_temp must be above absolute zero")
        low = self.calculator._finite_number(
            {'base_static': base_static}, 'base_static'
        )
        if low < 0.0:
            raise ValueError("base_static must be nonnegative")
        times = self._one_dimensional_finite_array(env_params, 'times')
        temperatures = self._one_dimensional_finite_array(env_params, 'temp')
        if len(times) != len(temperatures):
            raise ValueError("temp must have the same length as times")
        if len(times) == 0:
            raise ValueError("times must not be empty")
        time_differences = np.diff(times)
        if np.any(time_differences <= 0.0):
            raise ValueError("times must be strictly increasing")
        if not np.allclose(
            time_differences,
            interval_hours,
            rtol=1e-9,
            atol=1e-12,
        ):
            raise ValueError("dt_hours must match every interval in times")
        weather = self._validated_dynamic_weather(env_params, len(times))
        initial_temp = self._initial_conductor_temperature(
            conductor_params, weather
        )

        start = self.calculator._finite_number(
            {'start_hour': start_hour}, 'start_hour'
        )
        end = self.calculator._finite_number({'end_hour': end_hour}, 'end_hour')
        if end < start:
            raise ValueError("end_hour must be greater than or equal to start_hour")
        self._validate_dynamic_thermal_inputs(
            conductor_params, weather, initial_temp
        )
        window_start = max(start, float(times[0]))
        window_end = min(end, float(times[-1]))
        if window_end < window_start:
            return 0.0

        augmented_times = np.unique(
            np.concatenate(
                (
                    times[times <= window_end],
                    np.asarray([window_start, window_end], dtype=float),
                )
            )
        )
        weather_indices = (
            np.searchsorted(times, augmented_times, side="right") - 1
        )
        augmented_weather = {
            key: values[weather_indices].copy()
            for key, values in weather.items()
        }
        time_indices = np.flatnonzero(
            (augmented_times >= window_start)
            & (augmented_times <= window_end)
        )
        steps = (np.diff(augmented_times) * 3600.0).tolist()

        def maximum_temperature(current: float) -> float:
            profile = [current] * len(steps)
            temperatures_for_current = self._integrate_dynamic_profile(
                params=conductor_params,
                weather=augmented_weather,
                time_steps=steps,
                current_profile=profile,
                initial_temp=initial_temp,
            )
            return float(np.max(temperatures_for_current[time_indices]))

        def current_is_safe(current: float) -> bool:
            try:
                return maximum_temperature(current) <= target_temp
            except (OverflowError, ValueError):
                return False

        if maximum_temperature(0.0) > target_temp:
            raise ValueError("当前天气和温度限制下不存在可行电流")

        if current_is_safe(low):
            high = max(low * 3.0, 1.0)
            for _ in range(40):
                if not current_is_safe(high):
                    break
                low = high
                high *= 2.0
                if not math.isfinite(high):
                    raise ValueError("无法建立不可行电流上界")
            else:
                raise ValueError("无法建立不可行电流上界")
        else:
            high = low
            low = 0.0

        for _ in range(15):
            mid = (low + high) / 2
            if current_is_safe(mid):
                low = mid
            else:
                high = mid
        return low

    def generate_current_profile(self, max_curr, times, sunrise, sunset):
        """生成显示用的电流曲线"""
        base = np.ones_like(times) * max_curr
        noise = np.random.normal(0, max_curr * 0.02, size=len(times))
        return base + noise

    def calculate_dynamic_temperature(self, env_params, params, current_profile, dt_hours):
        """计算全时段温度"""
        interval_hours = self.calculator._finite_number(
            {'dt_hours': dt_hours}, 'dt_hours'
        )
        if interval_hours <= 0.0:
            raise ValueError("dt_hours must be greater than zero")
        conductor_params = self._validated_transient_conductor_params(params)
        currents = self._one_dimensional_finite_array(
            {'current_profile': current_profile}, 'current_profile'
        )
        if len(currents) == 0:
            raise ValueError("current_profile must not be empty")
        if np.any(currents < 0.0):
            raise ValueError("current_profile must be nonnegative")
        if isinstance(env_params, Mapping) and 'times' in env_params:
            times = self._one_dimensional_finite_array(env_params, 'times')
            if len(times) != len(currents):
                raise ValueError("times must have the same length as the samples")
            time_differences = np.diff(times)
            if np.any(time_differences <= 0.0):
                raise ValueError("times must be strictly increasing")
            if not np.allclose(
                time_differences,
                interval_hours,
                rtol=1e-9,
                atol=1e-12,
            ):
                raise ValueError("dt_hours must match every interval in times")
        weather = self._validated_dynamic_weather(env_params, len(currents))
        initial_temp = self._initial_conductor_temperature(
            conductor_params, weather
        )
        steps = [interval_hours * 3600.0] * (len(currents) - 1)
        result = self._integrate_dynamic_profile(
            params=conductor_params,
            weather=weather,
            time_steps=steps,
            current_profile=currents[:-1].tolist(),
            initial_temp=initial_temp,
        )
        return result, result.copy()

    def _validated_dynamic_weather(
        self, env_params, sample_count: int
    ) -> Dict[str, np.ndarray]:
        if not isinstance(env_params, Mapping):
            raise ValueError("env_params must be a mapping")

        fields = (
            (("temp",), "T_a", True),
            (("wind", "wind_speed", "winds"), "wind_speed", False),
            (("angle", "wind_angle", "angles"), "wind_angle", False),
            (("solar", "solar_radiation"), "solar_radiation", False),
            (("elevation", "elevations"), "elevation", False),
            (("time", "times"), "time", False),
        )
        weather = {}
        for aliases, thermal_key, required in fields:
            present = [alias for alias in aliases if alias in env_params]
            if not present:
                if required:
                    raise ValueError(f"{aliases[0]} is required")
                continue
            if len(present) > 1:
                raise ValueError(
                    f"{thermal_key} weather aliases are ambiguous: {present}"
                )
            source_key = present[0]
            values = self._one_dimensional_finite_array(env_params, source_key)
            if len(values) != sample_count:
                raise ValueError(
                    f"{source_key} must have the same length as the samples"
                )
            weather[thermal_key] = values

        if np.any(weather['T_a'] <= -273.15):
            raise ValueError("temp must be above absolute zero")
        if 'wind_speed' in weather and np.any(weather['wind_speed'] < 0.0):
            raise ValueError("wind must be nonnegative")
        if 'solar_radiation' in weather and np.any(
            weather['solar_radiation'] < 0.0
        ):
            raise ValueError("solar must be nonnegative")
        return weather

    def _initial_conductor_temperature(
        self, params: Dict, weather: Dict[str, np.ndarray]
    ) -> float:
        if not isinstance(params, Mapping):
            raise ValueError("params must be a mapping")
        initial = params.get('T_s', weather['T_a'][0])
        initial_temp = self.calculator._finite_number(
            {'initial_temp': initial}, 'initial_temp'
        )
        if initial_temp <= -273.15:
            raise ValueError("initial_temp must be above absolute zero")
        return initial_temp

    def _validate_dynamic_thermal_inputs(
        self,
        params: Mapping,
        weather: Dict[str, np.ndarray],
        initial_temp: float,
    ) -> None:
        original_params = dict(params)
        for key, values in weather.items():
            original_params.setdefault(key, values[0])
        self.calculator.calculate_transient_temperature(
            original_params, [], initial_temp, []
        )

        for index in range(len(weather['T_a'])):
            effective_params = dict(params)
            effective_params.update(
                {key: values[index] for key, values in weather.items()}
            )
            self.calculator.calculate_transient_temperature(
                effective_params, [], initial_temp, []
            )

    def _integrate_dynamic_profile(
        self,
        params: Dict,
        weather: Dict[str, np.ndarray],
        time_steps: List[float],
        current_profile: List[float],
        initial_temp: float,
    ) -> np.ndarray:
        if len(time_steps) != len(current_profile):
            raise ValueError("time_steps and current_profile must have the same length")
        self.calculator._validated_transient_substep_counts(time_steps)

        temperatures = [initial_temp]
        current_temp = initial_temp
        if not time_steps:
            validation_params = dict(params)
            validation_params.update(
                {key: values[0] for key, values in weather.items()}
            )
            self.calculator.calculate_transient_temperature(
                validation_params, [], current_temp, []
            )
            return np.asarray(temperatures, dtype=float)

        for index, (time_step, current) in enumerate(
            zip(time_steps, current_profile)
        ):
            interval_params = dict(params)
            interval_params.update(
                {key: values[index] for key, values in weather.items()}
            )
            current_temp = self.calculator.calculate_transient_temperature(
                params=interval_params,
                time_steps=[time_step],
                initial_temp=current_temp,
                current_profile=[current],
            )[-1]
            temperatures.append(current_temp)
        return np.asarray(temperatures, dtype=float)

    @staticmethod
    def _one_dimensional_finite_array(container, key: str) -> np.ndarray:
        if not isinstance(container, Mapping) or key not in container:
            raise ValueError(f"{key} is required")
        if LineAnalyzer._contains_boolean(container[key]):
            raise ValueError(f"{key} must contain finite numbers")
        try:
            values = np.asarray(container[key], dtype=float)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"{key} must contain finite numbers") from exc
        if values.ndim != 1:
            raise ValueError(f"{key} must be one-dimensional")
        if not np.all(np.isfinite(values)):
            raise ValueError(f"{key} must contain finite numbers")
        return values.copy()

    @staticmethod
    def _contains_boolean(values) -> bool:
        try:
            objects = np.asarray(values, dtype=object)
        except (TypeError, ValueError):
            return False
        return any(isinstance(value, (bool, np.bool_)) for value in objects.flat)
