import math
from typing import Dict, List, Tuple, Optional

import numpy as np
from scipy.interpolate import interp1d


# ==============================================================================
# 核心物理计算类 - 集成地形修正和沙漠环境优化
# ==============================================================================

class ThermalCalculator:
    """
    基于IEEE Std 738-2023的裸导线电流-温度关系计算器
    集成：
    1. 地形微气候修正 (slope, aspect based wind correction)
    2. 沙漠环境优化 (ShaGeHuangCalculator)
    """

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

        # --- 沙漠/戈壁环境修正参数 ---
        self.SAND_WIND_COEFF = 1  # 风沙对流散热修正
        self.HUMIDITY_CORRECTION = 0.95  # 湿度对流物性修正
        self.GROUND_ALBEDO = 0.35  # 沙地反照率

        # --- 地形微气候修正参数 ---
        self.ALPHA_ROUGHNESS = 0.15  # 地表粗糙度指数
        self.REF_HEIGHT_GRID = 10.0  # 参考高度 (气象站高度)
        self.LINE_AVG_HEIGHT = 20.0  # 导线平均高度

    def apply_micro_climate_corrections(self, grid_wind_speed: float, grid_wind_dir: float,
                                        slope: float, aspect: float) -> Tuple[float, float]:
        """
        核心地形修正函数 - 从 dynamic_temperature_calculation.py 集成
        """
        # 1. 垂直高度修正 - Hellman指数法
        v_vertical_corrected = grid_wind_speed * (self.LINE_AVG_HEIGHT / self.REF_HEIGHT_GRID) ** self.ALPHA_ROUGHNESS

        # 2. 地形风速修正 - 基于坡度和相对风向
        if slope < 2:
            # 平坦地形
            topo_factor = 1.0
        else:
            # 有起伏的地形
            angle_diff_rad = math.radians(grid_wind_dir - aspect)
            impact = math.cos(angle_diff_rad)  # 风向与坡向的相关性

            if impact < -0.5:
                # 迎风坡 - 风速加速
                topo_factor = 1.0 + min((slope / 45.0) * 0.4, 0.3)
            elif impact > 0.5:
                # 背风坡 - 风速减弱
                topo_factor = 1.0 - min((slope / 45.0) * 0.4, 0.3)
            else:
                # 斜侧风 - 风速基本不变
                topo_factor = 1.0

        v_final = v_vertical_corrected * topo_factor
        return v_final, topo_factor

    def calculate_steady_state_current(self, params: Dict) -> float:
        """计算稳态电流"""
        # 应用地形修正 (如果提供了地形参数)
        if 'slope' in params and 'aspect' in params:
            v_grid = params.get('wind_speed_original', params['wind_speed'])
            wind_dir = params.get('wind_direction_original', 90)
            v_corrected, _ = self.apply_micro_climate_corrections(
                v_grid, wind_dir, params['slope'], params['aspect']
            )
            # 注意：此处修改了传入的字典引用
            params['wind_speed'] = v_corrected

        q_c = self.calculate_convection(params)
        q_r = self.calculate_radiation(params)
        q_s = self.calculate_solar_gain(params)
        r = self.calculate_resistance(params)

        if (q_c + q_r - q_s) <= 0:
            return 0.0
        return math.sqrt((q_c + q_r - q_s) / r)

    def calculate_steady_state_temperature(self, params: Dict, current: float,
                                           max_iter: int = 100, tol: float = 1e-3) -> float:
        """已知电流推导温度"""
        if 'slope' in params and 'aspect' in params:
            v_grid = params.get('wind_speed_original', params['wind_speed'])
            wind_dir = params.get('wind_direction_original', 90)
            v_corrected, _ = self.apply_micro_climate_corrections(
                v_grid, wind_dir, params['slope'], params['aspect']
            )
            params['wind_speed'] = v_corrected

        Ta = params['T_a']
        low = Ta
        high = 200.0

        for _ in range(max_iter):
            mid = (low + high) / 2
            params['T_s'] = mid
            params['T_avg'] = mid

            r = self.calculate_resistance(params)
            left = current ** 2 * r

            q_c = self.calculate_convection(params)
            q_r = self.calculate_radiation(params)
            q_s = self.calculate_solar_gain(params)
            right = q_c + q_r - q_s

            if left > right:
                low = mid
            else:
                high = mid
            if high - low < tol:
                return mid
        return (low + high) / 2

    def calculate_transient_temperature(self, params: Dict, time_steps: List[float],
                                        initial_temp: float, current_profile: List[float]) -> List[float]:
        """计算暂态温度变化"""
        if 'slope' in params and 'aspect' in params:
            v_grid = params.get('wind_speed_original', params['wind_speed'])
            wind_dir = params.get('wind_direction_original', 90)
            v_corrected, _ = self.apply_micro_climate_corrections(
                v_grid, wind_dir, params['slope'], params['aspect']
            )
            params['wind_speed'] = v_corrected

        temps = [initial_temp]
        current_temp = initial_temp

        mc_p = self.calculate_heat_capacity(params)
        if mc_p <= 0:
            return [initial_temp] * (len(time_steps) + 1)

        for i, dt in enumerate(time_steps):
            params['T_avg'] = current_temp
            params['T_s'] = current_temp

            q_c = self.calculate_convection(params)
            q_r = self.calculate_radiation(params)
            q_s = self.calculate_solar_gain(params)
            r = self.calculate_resistance(params)
            current = current_profile[i]

            delta_T = (1 / mc_p) * (r * current ** 2 + q_s - q_c - q_r) * dt
            current_temp += delta_T
            temps.append(current_temp)

        return temps

    # --------------------------------------------------------------------------
    # 分项计算方法 (沙漠/戈壁优化版)
    # --------------------------------------------------------------------------

    def calculate_convection(self, params: Dict) -> float:
        """
        计算对流散热 (ShaGeHuang沙漠优化版)
        包含：湿度修正、风沙系数修正、IEEE738简化Nu计算
        """
        D0 = params['D0']
        T_avg = params['T_avg']
        T_a = params['T_a']
        wind_speed = params['wind_speed']  # 已是修正后的本地风速
        angle = params.get('wind_angle', 90)
        elevation = params.get('elevation', 0)

        # 1. 物理参数计算 (含湿度修正)
        p_air = (1.293 - 1.525e-4 * elevation + 6.379e-9 * elevation ** 2) / (1 + 0.00367 * T_avg)

        # 粘度和导热系数应用湿度修正
        mu_air = ((1.458e-6 * (T_avg + 273.15) ** 1.5) / (T_avg + 383.4)) * self.HUMIDITY_CORRECTION
        k_air = (2.424e-2 + 7.477e-5 * T_avg - 4.407e-9 * T_avg ** 2) * self.HUMIDITY_CORRECTION

        # 2. 有效风速修正 (含沙风修正)
        effective_wind = wind_speed * self.SAND_WIND_COEFF

        if effective_wind < 0.1:
            return 0.0

        # 3. 雷诺数与努塞尔数计算
        Re = (D0 * effective_wind * p_air) / mu_air

        # IEEE 738 简化逻辑
        Nu_low = 0.324 * Re ** 0.55
        Nu_high = 0.052 * Re ** 0.8
        Nu_forced = max(Nu_low, Nu_high)

        # 风向修正
        phi = math.radians(angle)
        K_angle = 1.194 - math.cos(phi) + 0.194 * math.cos(2 * phi) + 0.368 * math.sin(2 * phi)

        q_c = Nu_forced * K_angle * k_air * (T_avg - T_a)
        return max(q_c, 0.0)

    def calculate_radiation(self, params: Dict) -> float:
        """计算辐射散热"""
        D0 = params['D0']
        epsilon = params['emissivity']
        Ts = params['T_s']
        Ta = params['T_a']

        Ts_k = Ts + 273
        Ta_k = Ta + 273
        return 17.8 * D0 * epsilon * (((Ts_k / 100) ** 4) - ((Ta_k / 100) ** 4))

    def calculate_solar_gain(self, params: Dict) -> float:
        """
        计算太阳热增益 (数据驱动版 + 地面反射增强)
        """
        alpha = params['absorptivity']
        D0 = params['D0']

        # 如果提供了实测太阳辐射数据
        if 'solar_radiation' in params:
            global_radiation = params['solar_radiation']

            # 直接吸收部分
            direct_gain = alpha * global_radiation * D0

            # 地面反射增强 (沙漠特有)
            ground_gain = alpha * global_radiation * self.GROUND_ALBEDO * 0.5 * D0

            return direct_gain + ground_gain

        # 否则使用晴空模型计算理论值
        Hc = self.calculate_solar_altitude(params)
        if Hc > 0:
            Zc = self.calculate_solar_azimuth(params)
            Zl = params.get('line_azimuth', 0)
            theta = math.acos(math.cos(math.radians(Hc)) * math.cos(math.radians(Zc - Zl)))
            Qs_theory = self.calculate_solar_radiation(params, Hc)
            Qse_theory = self.calculate_elevation_corrected_radiation(params, Qs_theory)
            return alpha * Qse_theory * math.sin(theta) * D0
        else:
            return 0.0

    def calculate_resistance(self, params: Dict) -> float:
        """计算电阻 (线性插值)"""
        T_avg = params['T_avg']

        if T_avg > 100:
            T_low, T_high = 25, 200
            R_low = params.get('R_low_25', 7.283e-5)
            R_high = params.get('R_high_200', 1.220e-4)
        else:
            T_low, T_high = 25, 75
            R_low = params.get('R_low_25', 7.283e-5)
            R_high = params.get('R_high_75', 8.688e-5)

        return ((R_high - R_low) / (T_high - T_low)) * (T_avg - T_low) + R_low

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
        if abs(denominator) < 1e-10:
            chi = 0.0
        else:
            chi = numerator / denominator
        if omega < 0:
            C = 180 if chi < 0 else 0
        else:
            C = 360 if chi < 0 else 180
        Zc = C + math.degrees(math.atan(chi))
        return Zc % 360

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
            current_profile = [mid] * len(env_params['times'])

            steps = [(env_params['times'][1] - env_params['times'][0]) * 3600] * len(current_profile)
            if len(steps) > 1:
                steps = steps[:-1]

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
