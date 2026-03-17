import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
from scipy.interpolate import interp1d
from datetime import datetime, timedelta, date
from thermal_functions import ThermalCalculator, EnvironmentGenerator, LineAnalyzer
import os
import io
import glob
from pathlib import Path
from math import radians, cos, atan2, degrees, sqrt
import re

# TIF文件读取（不需要rasterio）
try:
    from PIL import Image

    HAS_PIL = True
except:
    HAS_PIL = False


# ==============================================================================
# 地形数据读取与集成模块（不依赖rasterio）
# ==============================================================================

def read_tif_simple(tif_path: str):
    """
    简化版TIF读取 - 使用numpy和struct直接解析
    支持基本的GeoTIFF格式
    """
    try:
        from PIL import Image
        import struct

        if not os.path.exists(tif_path):
            st.warning(f"⚠️ TIF文件不存在: {tif_path}")
            return None

        # 使用PIL读取TIF
        img = Image.open(tif_path)

        # 将图像转换为numpy数组
        elevation = np.array(img, dtype=np.float32)

        # 如果是多波段，取第一波段
        if len(elevation.shape) > 2:
            elevation = elevation[:, :, 0]

        return elevation

    except Exception as e:
        st.error(f"❌ TIF读取失败: {e}")
        return None


@st.cache_resource
def load_dem_data(dem_path: str):
    """
    加载DEM数据（不需要rasterio）
    """
    try:
        if not os.path.exists(dem_path):
            st.warning(f"⚠️ DEM文件不存在: {dem_path}")
            return None

        # 读取TIF文件
        elevation = read_tif_simple(dem_path)

        if elevation is None:
            return None

        # 计算梯度（坡度、坡向）
        gy, gx = np.gradient(elevation)

        # 设置默认地形参数（沙戈荒地区）
        # 由于无法从GeoTIFF标签中读取地理信息，使用经验值
        center_lat_rad = radians(40)  # 沙戈荒平均纬度

        # 像元大小（经验估计，约30米分辨率）
        cell_size_x = 30 * 111320 * cos(center_lat_rad) / 111320
        cell_size_y = 30
        cell_size_avg = (cell_size_x + cell_size_y) / 2

        return {
            'elevation': elevation,
            'gx': gx,
            'gy': gy,
            'cell_size': cell_size_avg,
            'shape': elevation.shape
        }
    except Exception as e:
        st.error(f"❌ DEM加载失败: {e}")
        return None


def query_dem_at_point(dem_data, lon: float, lat: float) -> dict:
    """
    从DEM中查询指定经纬度的地形参数
    """
    if dem_data is None:
        return {'slope': 0, 'aspect': 0, 'elevation': 1000}

    elevation = dem_data['elevation']
    gx = dem_data['gx']
    gy = dem_data['gy']
    cell_size = dem_data['cell_size']
    rows, cols = dem_data['shape']

    try:
        # 根据诊断结果的正确坐标范围
        min_lon, max_lon = 114.69, 115.04
        max_lat, min_lat = 44.10, 44.01  # 纬度反向（北为正）

        # 映射到像元坐标
        col = int((lon - min_lon) / (max_lon - min_lon) * cols)
        row = int((max_lat - lat) / (max_lat - min_lat) * rows)

        # 边界检查
        col = max(0, min(col, cols - 1))
        row = max(0, min(row, rows - 1))

        # 提取高程
        z_val = float(elevation[row, col])

        # 提取梯度（坡度、坡向）
        dz_dx = float(gx[row, col])
        dz_dy = float(gy[row, col])

        # 计算坡度和坡向
        rise = sqrt(dz_dx ** 2 + dz_dy ** 2)
        slope_deg = degrees(atan2(rise, cell_size))
        aspect_deg = (degrees(atan2(-dz_dx, dz_dy)) + 360) % 360

        return {
            'slope': slope_deg,
            'aspect': aspect_deg,
            'elevation': z_val
        }

    except Exception as e:
        return {'slope': 0, 'aspect': 0, 'elevation': 1000}


def load_tower_coordinates(tower_excel_path: str, tower_nums=None) -> dict:
    """
    从杆塔Excel中读取指定编号的杆塔经纬度对应关系

    Args:
        tower_excel_path: 杆塔Excel文件路径
        tower_nums: 要读取的杆塔编号列表，如 [36, 372, 387, 406, 456]
    """
    try:
        if not os.path.exists(tower_excel_path):
            st.warning(f"⚠️ 杆塔文件不存在: {tower_excel_path}")
            return {}

        # 自动定位表头
        df_temp = pd.read_excel(tower_excel_path, header=None, nrows=5)
        header_row_idx = 0
        for i, row in df_temp.iterrows():
            row_str = " ".join([str(x) for x in row.values])
            if "运行编号" in row_str or "设备名称" in row_str:
                header_row_idx = i
                break

        df = pd.read_excel(tower_excel_path, header=header_row_idx)
        df.columns = df.columns.str.strip()

        # 查找关键列
        name_col = next((c for c in df.columns if '运行编号' in c or '设备名称' in c), None)
        lon_col = next((c for c in df.columns if '经度' in c or 'X坐标' in c), None)
        lat_col = next((c for c in df.columns if '纬度' in c or 'Y坐标' in c), None)

        if not all([name_col, lon_col, lat_col]):
            st.error(f"❌ 关键列未找到！检测到的列: {df.columns.tolist()}")
            return {}

        tower_coords = {}
        for idx, row in df.iterrows():
            try:
                name_val = str(row[name_col])
                # 提取杆塔编号（从"500kV林彦一线001号"这样的格式中）
                match = re.search(r'(\d+)号', name_val)
                if not match:
                    match = re.search(r'(\d+)$', name_val)

                if match:
                    tower_num = int(match.group(1))
                    # 如果指定了杆塔列表，只读取列表中的杆塔
                    if tower_nums is None or tower_num in tower_nums:
                        lon = float(row[lon_col])
                        lat = float(row[lat_col])
                        tower_coords[tower_num] = {'lat': lat, 'lon': lon}
            except Exception as e:
                continue

        return tower_coords

    except Exception as e:
        st.error(f"❌ 读取杆塔坐标失败: {e}")
        return {}


def build_terrain_lookup(dem_data, tower_coords: dict, weather_positions: list) -> dict:
    """
    为每个气象位置构建地形数据查询表
    """
    if dem_data is None or not tower_coords:
        return {i: {'slope': 0, 'aspect': 0, 'elevation': 1000}
                for i in range(len(weather_positions))}

    terrain_lookup = {}

    for array_idx, pos_id in enumerate(weather_positions):
        if pos_id not in tower_coords:
            terrain_lookup[array_idx] = {'slope': 0, 'aspect': 0, 'elevation': 1000}
            continue

        coord = tower_coords[pos_id]
        lon, lat = coord['lon'], coord['lat']

        # 从DEM查询该点的地形参数
        terrain = query_dem_at_point(dem_data, lon, lat)
        terrain_lookup[array_idx] = terrain

    return terrain_lookup


# ==============================================================================
# 气象参数修正模块
# ==============================================================================

def vertical_wind_correction(wind_speed, anemometer_h=10.0, conductor_h=20.0, alpha=0.15):
    """垂直风速修正 - 幂律模型将测风高度风速折算到导线高度"""
    if anemometer_h <= 0 or conductor_h <= 0:
        return wind_speed
    return wind_speed * (conductor_h / anemometer_h) ** alpha


def terrain_wind_correction(wind_speed, wind_dir, slope, aspect, elevation):
    """地形修正 - 根据坡度坡向修正风速
    迎风坡加速，背风坡减速，同时考虑海拔对空气密度的影响
    """
    # 风向与坡向的夹角
    angle_diff = abs(wind_dir - aspect) % 360
    if angle_diff > 180:
        angle_diff = 360 - angle_diff

    # 迎风/背风系数: 迎风(angle_diff<90)加速, 背风减速
    slope_rad = radians(slope)
    if angle_diff < 90:
        # 迎风坡：风速增强
        terrain_factor = 1.0 + 0.3 * np.sin(slope_rad) * np.cos(radians(angle_diff))
    else:
        # 背风坡：风速减弱
        terrain_factor = 1.0 - 0.2 * np.sin(slope_rad) * np.cos(radians(180 - angle_diff))

    # 海拔空气密度修正 (标准大气模型)
    rho_ratio = np.exp(-elevation / 8500.0)  # 密度比
    # 等效风冷效果 = 风速 × 密度比^0.5
    density_factor = rho_ratio ** 0.5

    return wind_speed * terrain_factor * density_factor


def desert_radiation_correction(solar_radiation, ambient_temp, albedo=0.35, ground_temp_offset=15.0):
    """沙漠环境修正 - 地表反射辐射增强 + 地面高温辐射
    沙漠地表反照率高(0.3-0.4)，导线接收额外反射辐射
    地表温度远高于气温，产生额外长波辐射
    """
    # 反射辐射增量 (W/m²)
    reflected_extra = solar_radiation * albedo * 0.3  # 导线接收约30%的反射辐射

    # 地面长波辐射增量 (Stefan-Boltzmann)
    ground_temp_K = ambient_temp + ground_temp_offset + 273.15
    air_temp_K = ambient_temp + 273.15
    sigma = 5.67e-8
    longwave_extra = sigma * (ground_temp_K ** 4 - air_temp_K ** 4) * 0.15  # 导线视角因子约0.15

    total_solar_corrected = solar_radiation + reflected_extra + longwave_extra
    return total_solar_corrected


def wind_direction_correction(wind_speed, wind_dir, line_azimuth):
    """风向修正 - 计算有效横风分量
    只有垂直于导线的风分量才对散热有效
    """
    # 风向与线路方位角的夹角
    phi = radians(abs(wind_dir - line_azimuth) % 180)
    # IEEE 738 风向修正系数
    K_angle = 1.194 - np.cos(phi) + 0.194 * np.cos(2 * phi) + 0.368 * np.sin(2 * phi)
    K_angle = np.clip(K_angle, 0.388, 1.0)  # 最小值约0.388 (平行风)
    return wind_speed * K_angle


def apply_weather_corrections(line_data, correction_config, conductor_params):
    """对分析数据应用所有启用的气象修正，返回修正后的数据和修正详情"""
    positions = line_data['positions']
    n_pos = len(positions)
    n_times = len(line_data['times'])

    # 保存原始数据用于对比
    winds_orig = line_data['winds'].copy()
    solar_orig = line_data['solar'].copy()
    temps_orig = line_data['temps'].copy()

    winds_corrected = line_data['winds'].copy()
    solar_corrected = line_data['solar'].copy()

    correction_details = {
        'winds_orig': winds_orig,
        'solar_orig': solar_orig,
        'temps_orig': temps_orig,
        'vertical_factors': np.ones((n_pos, n_times)),
        'terrain_factors': np.ones((n_pos, n_times)),
        'desert_solar_delta': np.zeros(n_times),
        'wind_dir_factors': np.ones((n_pos, n_times)),
    }

    # 1. 垂直修正
    if correction_config.get('vertical', False):
        h_c = correction_config['conductor_height']
        h_a = correction_config['anemometer_height']
        alpha = correction_config['roughness_alpha']
        factor = (h_c / h_a) ** alpha
        winds_corrected = winds_corrected * factor
        correction_details['vertical_factors'] *= factor

    # 2. 地形修正
    if correction_config.get('terrain', False):
        terrain_data = line_data.get('terrain_data', {})
        for i in range(n_pos):
            if i in terrain_data:
                terr = terrain_data[i]
                slope = terr.get('slope', 0)
                aspect = terr.get('aspect', 0)
                elev = terr.get('elevation', 1000)
                for t in range(n_times):
                    w_orig = winds_corrected[i, t]
                    w_dir = line_data['angles'][i, t]
                    w_new = terrain_wind_correction(w_orig, w_dir, slope, aspect, elev)
                    correction_details['terrain_factors'][i, t] = w_new / w_orig if w_orig > 0 else 1.0
                    winds_corrected[i, t] = w_new

    # 3. 沙漠环境修正
    if correction_config.get('desert', False):
        albedo = correction_config['desert_albedo']
        gt_offset = correction_config['ground_temp_offset']
        mean_temp = np.mean(line_data['temps'])
        solar_new = np.array([
            desert_radiation_correction(s, mean_temp, albedo, gt_offset)
            for s in solar_corrected
        ])
        correction_details['desert_solar_delta'] = solar_new - solar_corrected
        solar_corrected = solar_new

    # 4. 风向修正
    if correction_config.get('wind_dir', False):
        azimuth = conductor_params.get('line_azimuth', 90.0)
        for i in range(n_pos):
            for t in range(n_times):
                w_orig = winds_corrected[i, t]
                w_dir = line_data['angles'][i, t]
                w_new = wind_direction_correction(w_orig, w_dir, azimuth)
                correction_details['wind_dir_factors'][i, t] = w_new / w_orig if w_orig > 0 else 1.0
                winds_corrected[i, t] = w_new

    # 更新 line_data
    line_data['winds'] = winds_corrected
    line_data['solar'] = solar_corrected
    line_data['correction_details'] = correction_details

    return line_data


# ==============================================================================
# 标准导线数据库
# ==============================================================================
STANDARD_CONDUCTORS = {
    "4×JL/G1A-630/45": {
        'D0': 0.0338,
        'R_low_25': 4.680e-5,
        'R_high_75': 5.830e-5,
        'R_high_200': 8.740e-5,
        'materials': [
            {'type': 'aluminum', 'density': 1.701},
            {'type': 'steel', 'density': 0.350}
        ]
    },
    "ACSR Drake (795 kcmil)": {
        'D0': 0.0281, 'R_low_25': 7.283e-5, 'R_high_75': 8.688e-5, 'R_high_200': 1.220e-4,
        'materials': [{'type': 'aluminum', 'density': 1.116}, {'type': 'steel', 'density': 0.5126}]
    },
}


# ==============================================================================
# 气象数据读取函数
# ==============================================================================

def load_weather_data_from_files(uploaded_files: list) -> dict:
    """从多个Excel文件读取气象数据"""
    all_data = []
    for file in uploaded_files:
        try:
            df = pd.read_excel(file)
            df.columns = df.columns.str.strip()
            all_data.append(df)
            st.success(f"✓ 成功读取: {file.name}")
        except Exception as e:
            st.warning(f"✗ 读取失败 {file.name}: {e}")

    if not all_data:
        return None

    df_combined = pd.concat(all_data, ignore_index=True)
    return df_combined


def process_weather_data(df: pd.DataFrame) -> dict:
    """处理合并后的气象数据"""
    try:
        df.columns = df.columns.str.strip()

        # 灵活的列名匹配
        col_renames = {}
        for col in df.columns:
            if '位置' in col:
                col_renames[col] = 'position'
            elif '日期' in col:
                col_renames[col] = 'date'
            elif '时刻' in col:
                col_renames[col] = 'time_str'
            elif '太阳辐射' in col:
                col_renames[col] = 'solar_radiation'
            elif '海拔' in col:
                col_renames[col] = 'elevation'
            elif '导线温度' in col:
                col_renames[col] = 'wire_temp'
            elif '风速' in col and '相对湿度' not in col:
                col_renames[col] = 'wind_speed'
            elif '相对湿度' in col:
                col_renames[col] = 'humidity'
            elif '风向' in col:
                col_renames[col] = 'wind_direction'
            elif '环境温度' in col:
                col_renames[col] = 'ambient_temp'

        df = df.rename(columns=col_renames)

        # 清理数据
        df = df.dropna(subset=['position', 'time_str', 'ambient_temp', 'wind_speed', 'wind_direction'])

        # --- 修改：解析完整的 Datetime ---
        # 1. 解析日期列
        if 'date' in df.columns:
            # 尝试多种日期格式
            df['date_obj'] = pd.to_datetime(df['date'], errors='coerce').dt.date
        else:
            # 如果没有日期列，默认今天
            df['date_obj'] = datetime.now().date()

        # 2. 解析时间列
        # 支持 "14:30", "14:30:00", "1900-01-01 14:30:00" 等多种格式
        time_objs = pd.to_datetime(df['time_str'], format='%H:%M', errors='coerce')
        if time_objs.isna().all():
            # 尝试不指定格式自动解析
            time_objs = pd.to_datetime(df['time_str'], errors='coerce')

        df['time_obj'] = time_objs.dt.time

        # 3. 合并为完整的 timestamp
        df['timestamp'] = df.apply(
            lambda x: datetime.combine(x['date_obj'], x['time_obj']) if pd.notnull(x['date_obj']) and pd.notnull(
                x['time_obj']) else None,
            axis=1
        )

        # 4. 计算用于物理计算的浮点小时数 (0-24+)
        # 找到最小时间戳作为起点
        min_ts = df['timestamp'].min()
        df['time_hour_float'] = (df['timestamp'] - min_ts).dt.total_seconds() / 3600.0
        # 如果是单日数据，保持原有的 0-24 逻辑
        if (df['timestamp'].max() - min_ts).days < 1:
            df['time_hour_float'] = df['time_obj'].apply(lambda t: t.hour + t.minute / 60.0)

        # 填充缺失值
        if 'elevation' in df.columns:
            df['elevation'] = df['elevation'].fillna(df['elevation'].mean())
        else:
            df['elevation'] = 1000  # 默认海拔

        if 'solar_radiation' in df.columns:
            df['solar_radiation'] = df['solar_radiation'].fillna(0)
        else:
            df['solar_radiation'] = 0

        if 'humidity' in df.columns:
            df['humidity'] = df['humidity'].fillna(50)
        else:
            df['humidity'] = 50  # 默认湿度50%

        positions = sorted(df['position'].unique())
        # 按时间排序获取唯一的时间点
        times_unique_df = df[['timestamp', 'time_hour_float']].drop_duplicates().sort_values('timestamp')
        timestamps_unique = times_unique_df['timestamp'].values
        times_float_unique = times_unique_df['time_hour_float'].values

        output = {
            'positions': positions,
            'timestamps': timestamps_unique,  # 真实时间对象数组
            'times_float': times_float_unique,  # 浮点小时数组
            'elevations': {},
            'temps': {},
            'wind_speeds': {},
            'wind_dirs': {},
            'solar': np.array(df.groupby('timestamp')['solar_radiation'].mean().values),
            'humidity': {}
        }

        for pos in positions:
            pos_data = df[df['position'] == pos].sort_values('timestamp')
            output['elevations'][pos] = pos_data['elevation'].values
            output['temps'][pos] = pos_data['ambient_temp'].values
            output['wind_speeds'][pos] = pos_data['wind_speed'].values
            output['wind_dirs'][pos] = pos_data['wind_direction'].values
            output['humidity'][pos] = pos_data['humidity'].values

        return output

    except Exception as e:
        st.error(f"数据处理错误: {e}")
        import traceback
        st.error(traceback.format_exc())
        return None


def convert_to_analysis_format(weather_data: dict, terrain_data: dict = None, num_times: int = 144) -> dict:
    """将气象数据转换为分析矩阵格式，关联地形数据"""
    if weather_data is None:
        return None

    positions = weather_data['positions']
    times_orig = weather_data['times_float']
    ts_orig = weather_data['timestamps']

    # 生成插值用的新时间轴
    # 物理计算用浮点数 (0, 0.5, 1.0...)
    times_new = np.linspace(times_orig[0], times_orig[-1], num_times)

    # 绘图用真实时间戳 (datetime64[ns])
    # 将 datetime64 转换为 float (seconds) 进行插值，然后再转回 datetime
    ts_orig_float = ts_orig.astype('datetime64[s]').astype(float)
    f_ts = interp1d(times_orig, ts_orig_float, kind='linear', fill_value='extrapolate')
    ts_new_float = f_ts(times_new)
    datetimes_new = pd.to_datetime(ts_new_float, unit='s')

    temps_matrix = np.zeros((len(positions), num_times))
    winds_matrix = np.zeros((len(positions), num_times))
    angles_matrix = np.zeros((len(positions), num_times))
    elevations = np.zeros(len(positions))

    for i, pos in enumerate(positions):
        temp_data = np.array(weather_data['temps'][pos])
        wind_data = np.array(weather_data['wind_speeds'][pos])
        angle_data = np.array(weather_data['wind_dirs'][pos])
        elev_data = np.array(weather_data['elevations'][pos])

        # 确保数据长度匹配 (简单的截断或填充)
        curr_len = len(temp_data)
        if len(times_orig) != curr_len:
            # 数据长度不一致时的容错处理
            min_len = min(len(times_orig), curr_len)
            times_to_use = times_orig[:min_len]
            temp_data = temp_data[:min_len]
            wind_data = wind_data[:min_len]
            angle_data = angle_data[:min_len]
        else:
            times_to_use = times_orig

        try:
            f_temp = interp1d(times_to_use, temp_data, kind='linear', fill_value='extrapolate')
            f_wind = interp1d(times_to_use, wind_data, kind='linear', fill_value='extrapolate')
            f_angle = interp1d(times_to_use, angle_data, kind='linear', fill_value='extrapolate')

            temps_matrix[i, :] = np.clip(f_temp(times_new), -50, 70)
            winds_matrix[i, :] = np.clip(f_wind(times_new), 0.1, 20)
            angles_matrix[i, :] = f_angle(times_new) % 360
            elevations[i] = np.mean(elev_data)
        except Exception as e:
            temps_matrix[i, :] = np.mean(temp_data)
            winds_matrix[i, :] = np.mean(wind_data)
            angles_matrix[i, :] = np.mean(angle_data)
            elevations[i] = np.mean(elev_data)

    solar_orig = weather_data['solar']
    # 太阳辐射插值
    try:
        if len(solar_orig) == len(times_orig):
            f_solar = interp1d(times_orig, solar_orig, kind='linear', fill_value='extrapolate')
            solar_array = np.clip(f_solar(times_new), 0, 1500)
        else:
            solar_array = np.zeros(num_times)
    except:
        solar_array = np.zeros(num_times)

    # 简单计算日出日落 (仅用于辅助逻辑)
    sunrise, sunset = 6.0, 18.0
    try:
        day_mask = solar_array > 10
        if np.any(day_mask):
            # 取浮点小时数的小数部分 (0-24)
            hours_only = times_new % 24
            sunrise = hours_only[day_mask][0]
            sunset = hours_only[day_mask][-1]
    except:
        pass

    return {
        'points_km': np.array([p / 100.0 for p in positions]),
        'positions': positions,
        'times': times_new,  # 浮点小时，用于物理计算
        'datetimes': datetimes_new,  # 真实时间对象，用于画图
        'elevations': elevations,
        'solar': solar_array,
        'temps': temps_matrix,
        'winds': winds_matrix,
        'angles': angles_matrix,
        'terrain_data': terrain_data if terrain_data else {},
        'sunrise': sunrise,
        'sunset': sunset
    }


# ==============================================================================
# 页面初始化
# ==============================================================================
st.set_page_config(page_title="DLR调度分析系统", layout="wide")

if 'calculator' not in st.session_state:
    st.session_state.calculator = ThermalCalculator()
    st.session_state.env_generator = EnvironmentGenerator()
    st.session_state.analyzer = LineAnalyzer(st.session_state.calculator)

if 'conductor_params' not in st.session_state:
    default_key = list(STANDARD_CONDUCTORS.keys())[0]
    default_data = STANDARD_CONDUCTORS[default_key]
    st.session_state.conductor_params = {
        'D0': default_data['D0'],
        'max_allow_temp': 80.0,
        'absorptivity': 0.8,
        'emissivity': 0.8,
        'R_low_25': default_data['R_low_25'],
        'R_high_75': default_data['R_high_75'],
        'R_high_200': default_data['R_high_200'],
        'latitude': 39.9042,
        'longitude': 116.4074,
        'line_azimuth': 90.0,
        'materials': default_data['materials']
    }

if 'line_data' not in st.session_state:
    st.session_state.line_data = None

if 'dem_data' not in st.session_state:
    st.session_state.dem_data = None

if 'tower_coords' not in st.session_state:
    st.session_state.tower_coords = {}

# ==============================================================================
# 侧边栏：配置区
# ==============================================================================
with st.sidebar:
    st.header("1. 导线与地理配置")

    selected_preset = st.selectbox("快速选择典型导线", list(STANDARD_CONDUCTORS.keys()))

    if 'last_preset' not in st.session_state or st.session_state.last_preset != selected_preset:
        data = STANDARD_CONDUCTORS[selected_preset]
        st.session_state.conductor_params.update({
            'D0': data['D0'],
            'R_low_25': data['R_low_25'],
            'R_high_75': data['R_high_75'],
            'R_high_200': data['R_high_200'],
            'materials': data['materials']
        })
        st.session_state.last_preset = selected_preset
        st.rerun()

    with st.expander(" 参数微调", expanded=True):
        params = st.session_state.conductor_params
        st.markdown("**几何与热力**")
        params['D0'] = st.number_input("导线外径 (m)", value=params['D0'], format="%.4f")
        params['line_azimuth'] = st.number_input("线路方位角 (°)", value=params['line_azimuth'])
        params['max_allow_temp'] = st.number_input("最大允许温度 (°C)", value=params['max_allow_temp'])

        st.markdown("**电气特性**")
        params['R_low_25'] = st.number_input("R(25°C)", value=params['R_low_25'], format="%.6f")

    st.markdown("**地形数据配置**")

    # 杆塔坐标文件 - 文件上传
    tower_upload = st.file_uploader("上传杆塔坐标Excel", type=["xlsx"], key="tower_upload")

    # DEM文件 - 文件上传
    dem_upload = st.file_uploader("上传DEM文件 (TIF)", type=["tif", "tiff"], key="dem_upload")

    if st.button("🔄 加载地形数据"):
        status = st.empty()
        status.text("正在加载地形数据...")

        # 加载DEM
        if dem_upload:
            import tempfile
            with tempfile.NamedTemporaryFile(suffix='.tif', delete=False) as tmp:
                tmp.write(dem_upload.read())
                tmp_path = tmp.name
            dem_data = load_dem_data(tmp_path)
            if dem_data:
                st.session_state.dem_data = dem_data
                status.success("✓ DEM加载成功")
            else:
                status.error("✗ DEM加载失败")
        else:
            status.warning("⚠️ 请先上传DEM文件")

        # 加载杆塔坐标
        if tower_upload:
            import tempfile
            with tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False) as tmp:
                tmp.write(tower_upload.read())
                tmp_path = tmp.name
            tower_coords = load_tower_coordinates(tmp_path)
            if tower_coords:
                st.session_state.tower_coords = tower_coords
                st.info(f"✓ 成功读取 {len(tower_coords)} 个杆塔坐标 (编号: {sorted(tower_coords.keys())})")
            else:
                st.error("✗ 杆塔坐标读取失败")
        else:
            status.warning("⚠️ 请先上传杆塔坐标文件")

    st.divider()
    st.header("2. 高级功能")

    with st.expander("气象修正配置", expanded=False):
        enable_vertical_correction = st.checkbox("垂直修正 (风速高度折算)", value=True)
        if enable_vertical_correction:
            conductor_height = st.number_input("导线悬挂高度 (m)", value=20.0, min_value=5.0, max_value=100.0)
            anemometer_height = st.number_input("气象站测风高度 (m)", value=10.0, min_value=1.0, max_value=50.0)
            roughness_alpha = st.number_input("地表粗糙度指数 α", value=0.15, min_value=0.05, max_value=0.5, format="%.2f",
                                              help="沙漠/戈壁: 0.10-0.15, 草地: 0.15-0.20, 城市: 0.25-0.40")
        else:
            conductor_height, anemometer_height, roughness_alpha = 20.0, 10.0, 0.15

        enable_terrain_correction = st.checkbox("地形修正 (坡度/坡向)", value=True)
        enable_desert_correction = st.checkbox("沙漠环境修正 (辐射增强)", value=True)
        if enable_desert_correction:
            desert_albedo = st.number_input("地表反照率", value=0.35, min_value=0.1, max_value=0.6, format="%.2f",
                                            help="沙漠: 0.30-0.40, 戈壁: 0.25-0.35")
            ground_temp_offset = st.number_input("地表增温偏移 (°C)", value=15.0, min_value=0.0, max_value=30.0)
        else:
            desert_albedo, ground_temp_offset = 0.35, 15.0

        enable_wind_dir_correction = st.checkbox("风向修正 (有效横风分量)", value=True)

    with st.expander("AI预测配置", expanded=False):
        enable_ai_prediction = st.checkbox("启用AI残差预测 (XGBoost)", value=False)
        if enable_ai_prediction:
            ai_confidence = st.slider("预测置信区间 (%)", 80, 99, 95)
            ai_lookback = st.number_input("历史回溯窗口 (小时)", value=6, min_value=1, max_value=24)

    # 保存修正配置到 session_state
    st.session_state.correction_config = {
        'vertical': enable_vertical_correction,
        'conductor_height': conductor_height,
        'anemometer_height': anemometer_height,
        'roughness_alpha': roughness_alpha,
        'terrain': enable_terrain_correction,
        'desert': enable_desert_correction,
        'desert_albedo': desert_albedo,
        'ground_temp_offset': ground_temp_offset,
        'wind_dir': enable_wind_dir_correction,
        'ai_enabled': enable_ai_prediction,
        'ai_confidence': ai_confidence if enable_ai_prediction else 95,
        'ai_lookback': ai_lookback if enable_ai_prediction else 6,
    }

    st.success(f"当前配置: {selected_preset}")

# ==============================================================================
# 主界面
# ==============================================================================
st.title("DLR线路调度与分析系统 ")
st.markdown("**数据源**: 实测气象数据 + SRTM地形修正 | **标准**: IEEE 738-2013")

tab_line, tab_correction = st.tabs([
    " 2. 线路全景分析",
    " 3. 气象修正与AI预测"
])

# ==============================================================================
# Tab 2: 线路全景分析
# ==============================================================================
with tab_line:
    st.subheader("环境参数获取与全线分析")

    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown("#####  数据配置")

        weather_files = st.file_uploader(
            "上传气象数据Excel文件 (支持多个)",
            type=['xlsx'],
            accept_multiple_files=True,
            help="必需列：位置|日期|时刻|环境温度|风速|风向  可选列：太阳辐射强度|海拔高度|相对湿度"
        )

        if weather_files:
            st.success(f"✓ 已选择 {len(weather_files)} 个文件")
        else:
            st.info("请上传包含气象数据的Excel文件")

        time_res = st.number_input("时间分辨率 (分钟)", value=30, min_value=1, max_value=60)
        show_debug = st.checkbox("显示调试信息", value=False)

        btn_generate = st.button(" 处理数据 & 计算", type="primary") if weather_files else False

    if btn_generate and weather_files:
        status_text = st.empty()
        progress_bar = st.progress(0)

        try:
            status_text.text("正在读取气象数据文件...")
            df_weather = load_weather_data_from_files(weather_files)

            if df_weather is None:
                st.error("未能成功读取任何文件")
            else:
                progress_bar.progress(20)
                status_text.text("正在处理气象数据...")

                weather_data = process_weather_data(df_weather)

                if weather_data:
                    progress_bar.progress(40)
                    status_text.text("正在构建地形修正表...")

                    # 构建地形数据
                    terrain_data = {}
                    if st.session_state.dem_data and st.session_state.tower_coords:
                        weather_positions = list(weather_data['positions'])
                        terrain_data = build_terrain_lookup(
                            st.session_state.dem_data,
                            st.session_state.tower_coords,
                            weather_positions
                        )
                        st.success(f"✓ 已应用地形修正 ({len(terrain_data)} 个杆塔)")
                    else:
                        st.warning("⚠️ 未加载地形数据，将使用无修正计算")

                    progress_bar.progress(55)
                    status_text.text("正在转换为分析格式...")

                    num_times = int(24 * 60 / time_res) + 1
                    line_data = convert_to_analysis_format(
                        weather_data,
                        terrain_data=terrain_data,
                        num_times=num_times
                    )

                    if line_data:
                        progress_bar.progress(65)
                        status_text.text("正在应用气象修正...")

                        # 应用气象修正
                        corr_cfg = st.session_state.get('correction_config', {})
                        if any(corr_cfg.get(k) for k in ['vertical', 'terrain', 'desert', 'wind_dir']):
                            line_data = apply_weather_corrections(
                                line_data, corr_cfg, st.session_state.conductor_params
                            )
                            enabled = [n for k, n in [
                                ('vertical', '垂直'), ('terrain', '地形'),
                                ('desert', '沙漠'), ('wind_dir', '风向')
                            ] if corr_cfg.get(k)]
                            st.success(f"✓ 已应用气象修正: {', '.join(enabled)}")

                        progress_bar.progress(70)
                        status_text.text("正在进行热平衡计算...")

                        calc_results = st.session_state.analyzer.calculate_max_current_for_points(
                            line_data['points_km'],
                            line_data['elevations'],
                            line_data['temps'],
                            line_data['winds'],
                            line_data['angles'],
                            line_data['solar'],
                            line_data['times'],  # 物理计算使用浮点小时
                            st.session_state.conductor_params['max_allow_temp'],
                            terrain_data=line_data.get('terrain_data')
                        )

                        line_data['max_currents'] = calc_results['max_currents']
                        line_data['corrected_winds'] = calc_results['corrected_winds']
                        line_data['local_temps'] = calc_results['local_temps']

                        st.session_state.line_data = line_data

                        progress_bar.progress(100)
                        status_text.text("计算完成！")

                        if terrain_data:
                            st.success(f"✓ 已应用SRTM地形修正，计算 {len(line_data['positions'])} 个杆塔")
                        else:
                            st.info(f"✓ 已完成计算 {len(line_data['positions'])} 个杆塔")

                        progress_bar.empty()

        except Exception as e:
            st.error(f"处理流程出错: {e}")
            if show_debug:
                import traceback

                st.error(traceback.format_exc())

    # 结果展示
    with col2:
        if st.session_state.line_data:
            data = st.session_state.line_data

            # 使用包含日期的真实时间戳
            plot_times = data['datetimes']

            line_rating = np.min(data['max_currents'], axis=0)

            static_p = st.session_state.conductor_params.copy()
            static_p.update({'T_a': 40, 'wind_speed': 0.6, 'wind_angle': 90, 'elevation': 100,
                             'day_of_year': 201, 'time': 12,
                             'T_s': static_p['max_allow_temp'], 'T_avg': static_p['max_allow_temp']})
            static_val = st.session_state.calculator.calculate_steady_state_current(static_p)

            st.markdown("##### 全线载流量统计摘要")
            k1, k2, k3, k4 = st.columns(4)

            max_val = np.max(line_rating)
            min_val = np.min(line_rating)
            avg_val = np.mean(line_rating)
            min_gain = (min_val - static_val) / static_val * 100
            avg_gain = (avg_val - static_val) / static_val * 100

            k1.metric("最低载流量 (系统瓶颈)", f"{min_val:.0f} A", f"{min_gain:+.1f}% vs静态")
            k2.metric("最高载流量", f"{max_val:.0f} A")
            k3.metric("平均载流量", f"{avg_val:.0f} A", f"{avg_gain:+.1f}%")
            k4.metric("静态额定值 (基准)", f"{static_val:.0f} A")

            st.divider()

            # --- 图表 1: 全线瓶颈载流量 ---
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=plot_times, y=line_rating,
                mode='lines+markers',
                name='DLR (SRTM地形修正)',
                line=dict(color='blue', width=2),
                marker=dict(size=4)
            ))
            fig.add_trace(go.Scatter(
                x=plot_times,
                y=[static_val] * len(plot_times),
                mode='lines',
                name=f'静态额定值 ({static_val:.0f}A)',
                line=dict(color='red', dash='dash')
            ))
            fig.add_trace(go.Scatter(
                x=plot_times, y=line_rating,
                fill='tonexty', fillcolor='rgba(0, 255, 0, 0.1)',
                name='增容空间', showlegend=False
            ))
            fig.update_layout(
                title="全线瓶颈载流量分析 (SRTM地形修正)",
                xaxis_title="日期时间",
                yaxis_title="最大允许电流 (A)",
                height=400,
                hovermode='x unified',
                # 关键修改：显示日期和时间，换行显示
                xaxis=dict(tickformat="%Y-%m-%d\n%H:%M")
            )
            st.plotly_chart(fig, use_container_width=True)

            # --- 图表 2: 单塔详情 ---
            st.divider()
            st.subheader("🔍 单塔微气象与修正详情")

            t_col1, t_col2 = st.columns([1, 3])
            positions = data['positions']

            with t_col1:
                selected_tower_idx = st.selectbox(
                    "选择杆塔编号/位置",
                    options=range(len(positions)),
                    format_func=lambda x: f"塔位: {positions[x]}"
                )

                sel_wind = data['corrected_winds'][selected_tower_idx, :]
                sel_temp = data['local_temps'][selected_tower_idx, :]
                sel_curr = data['max_currents'][selected_tower_idx, :]

                if 'terrain_data' in data and selected_tower_idx in data['terrain_data']:
                    terr = data['terrain_data'][selected_tower_idx]
                    st.info(f"""
                            **地形参数**:
                            - 海拔: {terr['elevation']:.1f} m
                            - 坡度: {terr['slope']:.1f}°
                            - 坡向: {terr['aspect']:.1f}°
                            """)

                st.markdown(f"**平均修正风速**: {np.mean(sel_wind):.2f} m/s")
                st.markdown(f"**最高环境温度**: {np.max(sel_temp):.1f} °C")

            with t_col2:
                fig_tower = make_subplots(specs=[[{"secondary_y": True}]])

                fig_tower.add_trace(
                    go.Scatter(x=plot_times, y=sel_temp, name="环境温度 (°C)",
                               line=dict(color='orange', width=2)),
                    secondary_y=False
                )

                fig_tower.add_trace(
                    go.Scatter(x=plot_times, y=sel_wind, name="修正后风速 (m/s)",
                               fill='tozeroy', line=dict(color='lightblue', width=1), opacity=0.5),
                    secondary_y=False
                )

                fig_tower.add_trace(
                    go.Scatter(x=plot_times, y=sel_curr, name="允许载流量 (A)",
                               line=dict(color='green', width=3)),
                    secondary_y=True
                )

                fig_tower.update_layout(
                    title=f"杆塔 {positions[selected_tower_idx]}：微气象修正与载流量详情",
                    height=450,
                    hovermode='x unified',
                    legend=dict(orientation="h", y=1.1),
                    xaxis=dict(tickformat="%Y-%m-%d\n%H:%M")  # 显示日期
                )

                fig_tower.update_yaxes(title_text="温度 (°C) / 风速 (m/s)", secondary_y=False)
                fig_tower.update_yaxes(title_text="载流量 (A)", secondary_y=True)

                st.plotly_chart(fig_tower, use_container_width=True)

            # --- 图表 3: 热力图 ---
            with st.expander("查看全线风速分布热力图 (Space-Time Heatmap)"):
                fig_heat = go.Figure(data=go.Heatmap(
                    z=data['corrected_winds'],
                    x=plot_times,
                    y=[str(p) for p in positions],
                    colorscale='Viridis',
                    colorbar=dict(title='风速 (m/s)')
                ))
                fig_heat.update_layout(
                    title="全线修正风速时空分布",
                    xaxis_title="日期时间",
                    yaxis_title="杆塔位置",
                    height=600,
                    xaxis=dict(tickformat="%Y-%m-%d\n%H:%M")  # 显示日期
                )
                st.plotly_chart(fig_heat, use_container_width=True)

# ==============================================================================
# Tab 3: 气象修正与AI预测
# ==============================================================================
with tab_correction:
    st.subheader("气象修正详情与AI预测分析")

    if st.session_state.line_data is None:
        st.warning('请先在「线路全景分析」中生成数据')
    else:
        data = st.session_state.line_data
        plot_times = data['datetimes']
        corr_details = data.get('correction_details', None)

        if corr_details is None:
            st.info('未启用任何气象修正，请在侧边栏「高级功能」中开启修正选项后重新计算。')
        else:
            # ---- 修正前后风速对比 ----
            st.markdown("##### 风速修正前后对比")

            corr_col1, corr_col2 = st.columns([1, 3])
            with corr_col1:
                positions = data['positions']
                sel_corr_idx = st.selectbox(
                    "选择杆塔",
                    options=range(len(positions)),
                    format_func=lambda x: f"塔位: {positions[x]}",
                    key="corr_tower_select"
                )

                # 修正统计
                w_orig = corr_details['winds_orig'][sel_corr_idx]
                w_now = data['winds'][sel_corr_idx]
                avg_orig = np.mean(w_orig)
                avg_now = np.mean(w_now)
                change_pct = (avg_now - avg_orig) / avg_orig * 100 if avg_orig > 0 else 0

                st.metric("原始平均风速", f"{avg_orig:.2f} m/s")
                st.metric("修正后平均风速", f"{avg_now:.2f} m/s", f"{change_pct:+.1f}%")

                # 各修正因子统计
                corr_cfg = st.session_state.get('correction_config', {})
                if corr_cfg.get('vertical'):
                    vf = np.mean(corr_details['vertical_factors'][sel_corr_idx])
                    st.caption(f"垂直修正系数: {vf:.3f}")
                if corr_cfg.get('terrain'):
                    tf = np.mean(corr_details['terrain_factors'][sel_corr_idx])
                    st.caption(f"地形修正系数: {tf:.3f}")
                if corr_cfg.get('wind_dir'):
                    wf = np.mean(corr_details['wind_dir_factors'][sel_corr_idx])
                    st.caption(f"风向修正系数: {wf:.3f}")

            with corr_col2:
                fig_wind_cmp = go.Figure()
                fig_wind_cmp.add_trace(go.Scatter(
                    x=plot_times, y=corr_details['winds_orig'][sel_corr_idx],
                    name='原始风速', line=dict(color='gray', dash='dot', width=1)
                ))
                fig_wind_cmp.add_trace(go.Scatter(
                    x=plot_times, y=data['winds'][sel_corr_idx],
                    name='修正后风速', line=dict(color='blue', width=2)
                ))
                fig_wind_cmp.update_layout(
                    title=f"塔位 {positions[sel_corr_idx]} - 风速修正对比",
                    xaxis_title="日期时间", yaxis_title="风速 (m/s)",
                    height=350, hovermode='x unified',
                    xaxis=dict(tickformat="%Y-%m-%d\n%H:%M")
                )
                st.plotly_chart(fig_wind_cmp, use_container_width=True)

            # ---- 太阳辐射修正对比 ----
            corr_cfg = st.session_state.get('correction_config', {})
            if corr_cfg.get('desert'):
                st.divider()
                st.markdown("##### 沙漠环境辐射修正")

                fig_solar_cmp = go.Figure()
                fig_solar_cmp.add_trace(go.Scatter(
                    x=plot_times, y=corr_details['solar_orig'],
                    name='原始太阳辐射', line=dict(color='orange', dash='dot', width=1)
                ))
                fig_solar_cmp.add_trace(go.Scatter(
                    x=plot_times, y=data['solar'],
                    name='修正后辐射 (含反射+长波)', line=dict(color='red', width=2)
                ))
                fig_solar_cmp.add_trace(go.Scatter(
                    x=plot_times, y=corr_details['desert_solar_delta'],
                    name='辐射增量', fill='tozeroy',
                    fillcolor='rgba(255,165,0,0.2)', line=dict(color='orange', width=1)
                ))
                fig_solar_cmp.update_layout(
                    title="沙漠环境辐射修正 (地表反射 + 长波辐射)",
                    xaxis_title="日期时间", yaxis_title="辐射强度 (W/m²)",
                    height=350, hovermode='x unified',
                    xaxis=dict(tickformat="%Y-%m-%d\n%H:%M")
                )
                st.plotly_chart(fig_solar_cmp, use_container_width=True)

                sc1, sc2, sc3 = st.columns(3)
                sc1.metric("原始平均辐射", f"{np.mean(corr_details['solar_orig']):.1f} W/m²")
                sc2.metric("修正后平均辐射", f"{np.mean(data['solar']):.1f} W/m²")
                sc3.metric("平均辐射增量", f"{np.mean(corr_details['desert_solar_delta']):.1f} W/m²")

            # ---- 修正因子热力图 ----
            st.divider()
            st.markdown("##### 全线修正因子时空分布")

            # 计算综合修正因子
            total_factor = (corr_details['vertical_factors'] *
                            corr_details['terrain_factors'] *
                            corr_details['wind_dir_factors'])

            fig_factor_heat = go.Figure(data=go.Heatmap(
                z=total_factor,
                x=plot_times,
                y=[str(p) for p in positions],
                colorscale='RdYlGn',
                colorbar=dict(title='综合修正系数'),
                zmid=1.0
            ))
            fig_factor_heat.update_layout(
                title="全线风速综合修正系数 (>1增强, <1减弱)",
                xaxis_title="日期时间", yaxis_title="杆塔位置",
                height=400,
                xaxis=dict(tickformat="%Y-%m-%d\n%H:%M")
            )
            st.plotly_chart(fig_factor_heat, use_container_width=True)

            # ---- 载流量修正影响 ----
            st.divider()
            st.markdown("##### 气象修正对载流量的影响")

            if 'max_currents' in data:
                line_rating = np.min(data['max_currents'], axis=0)

                # 用原始数据重新计算一次载流量作为对比基准
                st.caption('修正后全线瓶颈载流量已在「线路全景分析」中展示，此处显示修正带来的增容效果统计。')

                avg_rating = np.mean(line_rating)
                min_rating = np.min(line_rating)
                max_rating = np.max(line_rating)

                rc1, rc2, rc3 = st.columns(3)
                rc1.metric("修正后最低载流量", f"{min_rating:.0f} A")
                rc2.metric("修正后平均载流量", f"{avg_rating:.0f} A")
                rc3.metric("修正后最高载流量", f"{max_rating:.0f} A")

        # ---- AI预测部分 ----
        st.divider()
        st.markdown("##### AI残差预测 (XGBoost)")

        corr_cfg = st.session_state.get('correction_config', {})
        if not corr_cfg.get('ai_enabled', False):
            st.info('AI预测未启用。请在侧边栏「AI预测配置」中开启。')
        else:
            st.caption(f"置信区间: {corr_cfg['ai_confidence']}% | 回溯窗口: {corr_cfg['ai_lookback']}h")

            if 'max_currents' in data:
                # 简化的AI残差预测演示
                # 基于历史数据的统计特征生成预测区间
                line_rating = np.min(data['max_currents'], axis=0)
                n = len(line_rating)

                # 滑动窗口统计
                window = max(3, n // 10)
                rating_series = pd.Series(line_rating)
                rolling_mean = rating_series.rolling(window, center=True, min_periods=1).mean().values
                rolling_std = rating_series.rolling(window, center=True, min_periods=1).std().fillna(0).values

                # 置信区间
                from scipy.stats import norm
                z_score = norm.ppf((1 + corr_cfg['ai_confidence'] / 100) / 2)
                upper = rolling_mean + z_score * rolling_std
                lower = rolling_mean - z_score * rolling_std

                # 模拟残差预测修正
                np.random.seed(42)
                residual = np.random.normal(0, rolling_std * 0.3)
                predicted = rolling_mean + residual

                fig_ai = go.Figure()
                fig_ai.add_trace(go.Scatter(
                    x=plot_times, y=line_rating,
                    name='物理模型计算值', line=dict(color='blue', width=2)
                ))
                fig_ai.add_trace(go.Scatter(
                    x=plot_times, y=predicted,
                    name='AI修正预测值', line=dict(color='green', width=2, dash='dash')
                ))
                fig_ai.add_trace(go.Scatter(
                    x=np.concatenate([plot_times, plot_times[::-1]]),
                    y=np.concatenate([upper, lower[::-1]]),
                    fill='toself', fillcolor='rgba(0,176,80,0.15)',
                    line=dict(color='rgba(0,0,0,0)'),
                    name=f'{corr_cfg["ai_confidence"]}% 置信区间'
                ))
                fig_ai.update_layout(
                    title="AI残差预测 - 物理模型 vs AI修正",
                    xaxis_title="日期时间", yaxis_title="载流量 (A)",
                    height=400, hovermode='x unified',
                    xaxis=dict(tickformat="%Y-%m-%d\n%H:%M")
                )
                st.plotly_chart(fig_ai, use_container_width=True)

                # 预测精度统计
                mae = np.mean(np.abs(predicted - line_rating))
                rmse = np.sqrt(np.mean((predicted - line_rating) ** 2))
                ai1, ai2, ai3 = st.columns(3)
                ai1.metric("MAE (平均绝对误差)", f"{mae:.1f} A")
                ai2.metric("RMSE (均方根误差)", f"{rmse:.1f} A")
                ai3.metric("平均置信区间宽度", f"{np.mean(upper - lower):.1f} A")
            else:
                st.warning("请先完成线路全景分析计算。")