import math
from dataclasses import dataclass
from numbers import Integral
from typing import Dict, List, Tuple, Optional

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
        except (TypeError, ValueError) as exc:
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

        low = ambient_temp
        low_residual = heat_residual(low)
        if low_residual == 0.0:
            return low

        high = min(
            max(200.0, low + 1.0), self._MAX_CONDUCTOR_TEMPERATURE_C
        )
        high_residual = heat_residual(high)
        while high_residual > 0.0 and high < self._MAX_CONDUCTOR_TEMPERATURE_C:
            high = min(
                low + 2.0 * (high - low),
                self._MAX_CONDUCTOR_TEMPERATURE_C,
            )
            high_residual = heat_residual(high)
        if high_residual > 0.0:
            raise ValueError("steady-state root is outside the physical temperature range")

        for _ in range(max_iter):
            mid = (low + high) / 2.0
            if mid == low or mid == high:
                raise ValueError("steady-state root did not converge within max_iter")
            mid_residual = heat_residual(mid)
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
                if (
                    abs(candidate_residual)
                    <= self._STEADY_STATE_RESIDUAL_TOLERANCE_W_PER_M
                ):
                    return candidate_temp
        raise ValueError("steady-state root did not converge within max_iter")

    def calculate_transient_temperature(self, params: Dict, time_steps: List[float],
                                        initial_temp: float, current_profile: List[float]) -> List[float]:
        """计算暂态温度变化"""
        initial_temp = self._finite_number(
            {'initial_temp': initial_temp}, 'initial_temp'
        )
        if initial_temp <= -273.15:
            raise ValueError("initial_temp must be above absolute zero")
        if len(time_steps) != len(current_profile):
            raise ValueError("time_steps and current_profile must have the same length")

        validated_steps = []
        validated_currents = []
        for time_step in time_steps:
            value = self._finite_number({'time_steps': time_step}, 'time_steps')
            if value <= 0.0:
                raise ValueError("time_steps values must be greater than zero")
            validated_steps.append(value)
        for profile_current in current_profile:
            value = self._finite_number(
                {'current_profile': profile_current}, 'current_profile'
            )
            if value < 0.0:
                raise ValueError("current_profile values must be nonnegative")
            validated_currents.append(value)

        params = dict(params)

        temps = [initial_temp]
        current_temp = initial_temp

        mc_p = self.calculate_heat_capacity(params)
        if mc_p <= 0:
            return [initial_temp] * (len(time_steps) + 1)

        for i, dt in enumerate(validated_steps):
            params['T_avg'] = current_temp
            params['T_s'] = current_temp

            q_c = self.calculate_convection(params)
            q_r = self.calculate_radiation(params)
            q_s = self.calculate_solar_gain(params)
            r = self.calculate_resistance(params)
            current = validated_currents[i]

            delta_T = (1 / mc_p) * (r * current ** 2 + q_s - q_c - q_r) * dt
            current_temp += delta_T
            temps.append(current_temp)

        return temps

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
        if isinstance(value, bool):
            raise ValueError(f"{key} must be a finite number")
        try:
            number = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{key} must be a finite number") from exc
        if not math.isfinite(number):
            raise ValueError(f"{key} must be a finite number")
        return number

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
    def _temperatures(cls, params: Dict) -> Tuple[float, float]:
        conductor = cls._finite_number(params, 'T_s')
        ambient = cls._finite_number(params, 'T_a')
        if conductor <= -273.15:
            raise ValueError("T_s must be above absolute zero")
        if ambient <= -273.15:
            raise ValueError("T_a must be above absolute zero")
        if conductor < ambient:
            raise ValueError("T_s must be greater than or equal to T_a")
        return conductor, ambient

    def _calculate_convection_components(self, params: Dict) -> Tuple[float, float, float, float]:
        diameter = self._positive_number(params, 'D0')
        conductor_temp, ambient_temp = self._temperatures(params)
        wind_speed = self._finite_number(params, 'wind_speed')
        if wind_speed < 0.0:
            raise ValueError("wind_speed must be nonnegative")
        wind_angle = self._finite_number(params, 'wind_angle', 90.0)
        elevation = self._finite_number(params, 'elevation', 0.0)

        film_temp = (conductor_temp + ambient_temp) / 2.0
        temperature_difference = conductor_temp - ambient_temp
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
            * temperature_difference ** 1.25
        )
        if wind_speed == 0.0:
            return q_natural, 0.0, 0.0, q_natural

        reynolds = diameter * density * wind_speed / viscosity
        angle_factor = self.wind_angle_factor(wind_angle)
        q_low_re = (
            angle_factor
            * (1.01 + 1.35 * reynolds ** 0.52)
            * conductivity
            * temperature_difference
        )
        q_high_re = (
            angle_factor
            * 0.754
            * reynolds ** 0.60
            * conductivity
            * temperature_difference
        )
        return q_natural, q_low_re, q_high_re, max(q_natural, q_low_re, q_high_re)

    def _calculate_radiation(self, params: Dict) -> float:
        diameter = self._positive_number(params, 'D0')
        emissivity = self._fraction(params, 'emissivity')
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
        """计算热容量"""
        materials = params['materials']
        total = 0.0
        for material in materials:
            mat_type = material['type']
            mass = material.get('mass', material.get('density', 0))
            cp = self.material_properties.get(mat_type, {'cp': 0})['cp']
            total += mass * cp
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
    def __init__(self, calculator: ThermalCalculator):
        self.calculator = calculator

    def calculate_max_current_for_points(self, observation_points: np.ndarray, elevations: np.ndarray,
                                         temps: np.ndarray, winds: np.ndarray, angles: np.ndarray,
                                         solar: np.ndarray, times: np.ndarray, max_temp: float = 80,
                                         base_params: Dict = None, terrain_data: Dict = None) -> Dict:
        """
        批量计算全线最大载流量，并返回修正后的微气象数据

        返回:
            Dict: {
                'max_currents': np.ndarray,
                'corrected_winds': np.ndarray,
                'local_temps': np.ndarray
            }
        """
        num_points = len(observation_points)
        num_times = len(times)

        # 初始化结果矩阵
        max_currents = np.zeros((num_points, num_times))
        corrected_winds = np.zeros((num_points, num_times))  # 新增：存储地形修正后的风速
        local_temps = np.zeros((num_points, num_times))  # 新增：存储环境温度

        if base_params is None:
            base_params = {
                'D0': 0.0369, 'emissivity': 0.8, 'absorptivity': 0.8,
                'R_low_25': 7.283e-5, 'R_high_75': 8.688e-5, 'latitude': 40,
                'day_of_year': 201, 'line_azimuth': 90,
                'materials': [{'type': 'aluminum', 'mass': 1.116}, {'type': 'steel', 'mass': 0.5126}]
            }

        for i in range(num_points):
            for j in range(num_times):
                params = base_params.copy()

                current_solar = solar[j] if isinstance(solar, np.ndarray) and len(solar) == num_times else (
                    solar if isinstance(solar, (float, int)) else 0)

                # 设置当前时空点的环境参数
                params.update({
                    'T_s': max_temp,
                    'T_avg': max_temp,
                    'T_a': temps[i, j],
                    'wind_speed': winds[i, j],
                    'wind_speed_original': winds[i, j],
                    'wind_angle': angles[i, j],
                    'wind_direction_original': angles[i, j],
                    'elevation': elevations[i],
                    'time': times[j],
                    'solar_radiation': current_solar
                })

                # 注入地形数据
                if terrain_data and i in terrain_data:
                    terrain = terrain_data[i]
                    params['slope'] = terrain.get('slope', 0)
                    params['aspect'] = terrain.get('aspect', 0)

                # 计算载流量
                # 注意：calculate_steady_state_current 会在内部修改 params['wind_speed'] 为修正后风速
                current_val = self.calculator.calculate_steady_state_current(params)

                # 保存结果
                max_currents[i, j] = current_val
                corrected_winds[i, j] = params['wind_speed']  # 获取修正后的风速
                local_temps[i, j] = params['T_a']  # 获取该点的环境温度

        # 返回字典结构
        return {
            'max_currents': max_currents,
            'corrected_winds': corrected_winds,
            'local_temps': local_temps
        }

    def calculate_time_to_max_temp(self, params: Dict, current: float, max_temp: float,
                                   initial_temp: float, time_step: float = 10) -> float:
        """计算达到限值的时间"""
        current_temp = initial_temp
        time = 0.0

        mc_p = self.calculator.calculate_heat_capacity(params)
        if mc_p <= 0:
            return float('inf')

        while current_temp < max_temp:
            params['T_avg'] = current_temp
            params['T_s'] = current_temp

            q_c = self.calculator.calculate_convection(params)
            q_r = self.calculator.calculate_radiation(params)
            q_s = self.calculator.calculate_solar_gain(params)
            r = self.calculator.calculate_resistance(params)

            delta_T = (1 / mc_p) * (r * current ** 2 + q_s - q_c - q_r) * time_step

            if delta_T <= 0:
                return float('inf')

            current_temp += delta_T
            time += time_step
            if time > 7200:
                return float('inf')

        return time

    def find_max_current_for_window(self, env_params, base_static, params, dt_hours, start_hour=0, end_hour=2):
        """日前调度：寻找时间窗口内的最大允许电流"""
        time_mask = (env_params['times'] >= start_hour) & (env_params['times'] <= end_hour)
        time_indices = np.where(time_mask)[0]
        if len(time_indices) == 0:
            return 0

        target_temp = params['max_allow_temp']
        low = base_static
        high = base_static * 3.0

        for _ in range(15):
            mid = (low + high) / 2
            steps = (np.diff(env_params['times']) * 3600).tolist()
            current_profile = [mid] * len(steps)

            temps = self.calculator.calculate_transient_temperature(
                params, steps, env_params['temp'][0], current_profile
            )

            relevant_temps = np.array(temps)[time_indices]
            max_t = np.max(relevant_temps)

            if max_t <= target_temp:
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
        steps = [dt_hours * 3600] * len(current_profile)
        temps = self.calculator.calculate_transient_temperature(
            params, steps, env_params['temp'][0], current_profile
        )
        if len(temps) > len(current_profile):
            temps = temps[:len(current_profile)]
        return np.array(temps), np.array(temps)
