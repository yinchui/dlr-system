import streamlit as st
import copy
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
from thermal_functions import ThermalCalculator, EnvironmentGenerator, LineAnalyzer
from modules.data_processor import normalize_weather_input_dataframe
from modules.dlr_pipeline import DlrPipeline, derive_line_id
from modules import terrain as terrain_module
from modules.weather_correction import CorrectionOptions, WeatherCorrectionService
from modules.weather_upload import normalize_uploaded_weather_files
import os


# ==============================================================================
# 地形数据读取与集成模块
# ==============================================================================

def read_tif_simple(tif_path: str):
    """保留页面入口，实际读取统一由地形模块完成。"""
    try:
        if not os.path.exists(tif_path):
            st.warning(f"⚠️ TIF文件不存在: {tif_path}")
            return None
        return terrain_module.read_tif_simple(tif_path)
    except Exception as e:
        st.error(f"❌ TIF读取失败: {e}")
        return None


@st.cache_resource
def load_dem_data(dem_path: str):
    """保留 Streamlit 缓存与既有错误提示。"""
    try:
        if not os.path.exists(dem_path):
            st.warning(f"⚠️ DEM文件不存在: {dem_path}")
            return None
        return terrain_module.load_dem_data(dem_path)
    except Exception as e:
        st.error(f"❌ DEM加载失败: {e}")
        return None


def query_dem_at_point(dem_data, lon: float, lat: float) -> dict:
    return terrain_module.query_dem_at_point(dem_data, lon, lat)


def load_tower_coordinates(tower_excel_path: str, tower_nums=None) -> dict:
    """保留页面错误提示，塔表解析统一由地形模块完成。"""
    try:
        if not os.path.exists(tower_excel_path):
            st.warning(f"⚠️ 杆塔文件不存在: {tower_excel_path}")
            return {}
        return terrain_module.load_tower_coordinates(tower_excel_path, tower_nums)
    except terrain_module.MissingTowerColumnsError as e:
        st.error(f"❌ {e}")
        return {}
    except Exception as e:
        st.error(f"❌ 读取杆塔坐标失败: {e}")
        return {}


def build_terrain_lookup(dem_data, tower_coords: dict, weather_positions: list) -> dict:
    return terrain_module.build_terrain_lookup(dem_data, tower_coords, weather_positions)


# ==============================================================================
# 气象参数修正模块
# ==============================================================================

def apply_weather_corrections(line_data, correction_config, conductor_params):
    """将旧矩阵格式适配到服务层，且不修改调用方提供的数据。"""
    if line_data.get('correction_stage') not in (None, 'original'):
        raise ValueError('气象数据已经修正，不能重复应用局地修正')

    corrected_data = copy.deepcopy(line_data)
    positions = list(corrected_data['positions'])
    n_pos = len(positions)
    n_times = len(corrected_data['times'])
    winds = np.asarray(corrected_data['winds'], dtype=float).copy()
    temps = np.asarray(corrected_data['temps'], dtype=float).copy()
    directions = np.asarray(corrected_data['angles'], dtype=float).copy()
    solar = np.asarray(corrected_data['solar'], dtype=float).copy()
    if winds.shape != (n_pos, n_times) or temps.shape != (n_pos, n_times) or directions.shape != (n_pos, n_times):
        raise ValueError('旧气象矩阵维度与杆塔或时间轴不一致')
    if solar.shape != (n_times,):
        raise ValueError('旧太阳辐射数组维度与时间轴不一致')

    terrain_lookup = {}
    terrain_data = corrected_data.get('terrain_data') or {}
    if terrain_data:
        if not hasattr(terrain_data, 'get'):
            raise ValueError('terrain_data 必须是映射')

        def matching_key(candidates):
            for key in candidates:
                try:
                    if key in terrain_data:
                        return key
                except TypeError:
                    continue
            return None

        index_keys = [matching_key((index, str(index))) for index in range(n_pos)]
        position_keys = [
            matching_key((position, str(position)))
            for position in positions
        ]
        index_complete = all(key is not None for key in index_keys)
        position_complete = all(key is not None for key in position_keys)
        if index_complete and position_complete:
            index_markers = [(type(key), key) for key in index_keys]
            position_markers = [(type(key), key) for key in position_keys]
            if index_markers != position_markers:
                raise ValueError('地形键歧义')
            selected_keys = index_keys
        elif index_complete:
            selected_keys = index_keys
        elif position_complete:
            selected_keys = position_keys
        else:
            owners = {}
            selected_keys = []
            for index, (index_key, position_key) in enumerate(
                zip(index_keys, position_keys)
            ):
                row_keys = [key for key in (index_key, position_key) if key is not None]
                if len(set((type(key), key) for key in row_keys)) > 1:
                    raise ValueError('地形键歧义')
                for key in row_keys:
                    marker = (type(key), key)
                    owner = owners.setdefault(marker, index)
                    if owner != index:
                        raise ValueError('地形键歧义')
                selected_keys.append(row_keys[0] if row_keys else None)

        for position, key in zip(positions, selected_keys):
            if key is not None:
                terrain_lookup[position] = terrain_data[key]

    source_frame = pd.DataFrame({
        'position': np.repeat(positions, n_times),
        'ambient_temp': temps.reshape(-1),
        'wind_speed': winds.reshape(-1),
        'wind_direction': directions.reshape(-1),
        'solar_radiation': np.tile(solar, n_pos),
    })
    options = CorrectionOptions(
        enable_vertical=bool(correction_config.get('vertical', False)),
        enable_terrain=bool(correction_config.get('terrain', False)),
        enable_desert=bool(correction_config.get('desert', False)),
        enable_wind_direction=bool(correction_config.get('wind_dir', False)),
        ref_height_m=correction_config.get('anemometer_height', 10.0),
        line_height_m=correction_config.get('conductor_height', 20.0),
        roughness_alpha=correction_config.get('roughness_alpha', 0.15),
        ground_albedo=correction_config.get('desert_albedo', 0.35),
        ground_temp_offset=correction_config.get('ground_temp_offset', 15.0),
        line_azimuth_deg=conductor_params.get('line_azimuth', 90.0),
    )
    local = WeatherCorrectionService().apply(source_frame, terrain_lookup, options)

    local_winds = local['wind_speed_local'].to_numpy(dtype=float).reshape(n_pos, n_times)
    local_temps = local['ambient_temp_local'].to_numpy(dtype=float).reshape(n_pos, n_times)
    local_solar_matrix = local['solar_radiation_local'].to_numpy(dtype=float).reshape(n_pos, n_times)
    local_angles = local['wind_angle_deg'].to_numpy(dtype=float).reshape(n_pos, n_times)
    vertical_factors = local['vertical_wind_factor'].to_numpy(dtype=float).reshape(n_pos, n_times)
    terrain_factors = local['terrain_wind_factor'].to_numpy(dtype=float).reshape(n_pos, n_times)

    corrected_data['winds'] = local_winds
    corrected_data['temps'] = local_temps
    corrected_data['solar'] = local_solar_matrix.mean(axis=0)
    corrected_data['angles'] = local_angles
    corrected_data['correction_stage'] = 'terrain_corrected'
    corrected_data['correction_details'] = {
        'winds_orig': winds.copy(),
        'solar_orig': solar.copy(),
        'temps_orig': temps.copy(),
        'vertical_factors': vertical_factors,
        'terrain_factors': terrain_factors,
        'desert_solar_delta': corrected_data['solar'] - solar,
        'wind_dir_factors': np.ones((n_pos, n_times)),
    }
    return corrected_data


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
    """从多个Excel/CSV文件读取气象数据"""
    all_data = []
    for file in uploaded_files:
        try:
            fname = file.name.lower()
            if fname.endswith('.csv'):
                df = pd.read_csv(file, encoding='utf-8-sig')
            else:
                df = pd.read_excel(file)
            normalized = normalize_weather_input_dataframe(df)
            if normalized.attrs.get('input_format') == 'tower_time':
                st.info(f"✓ 已识别新格式气象数据: {file.name}")
            all_data.append(normalized)
            st.success(f"✓ 成功读取: {file.name}")
        except Exception as e:
            st.warning(f"✗ 读取失败 {file.name}: {e}")

    if not all_data:
        return None

    df_combined = pd.concat(all_data, ignore_index=True)
    return df_combined


def process_weather_data(df: pd.DataFrame) -> dict:
    """处理已规范化的气象数据，并兼容直接传入的旧/新原始表格。"""
    try:
        df = normalize_weather_input_dataframe(df)

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
    f_ts = np.interp(times_new, times_orig, ts_orig_float)
    ts_new_float = f_ts
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
            temps_matrix[i, :] = np.clip(np.interp(times_new, times_to_use, temp_data), -50, 70)
            winds_matrix[i, :] = np.clip(np.interp(times_new, times_to_use, wind_data), 0.1, 20)
            angles_matrix[i, :] = np.interp(times_new, times_to_use, angle_data) % 360
            elevations[i] = np.mean(elev_data)
        except Exception:
            temps_matrix[i, :] = np.mean(temp_data)
            winds_matrix[i, :] = np.mean(wind_data)
            angles_matrix[i, :] = np.mean(angle_data)
            elevations[i] = np.mean(elev_data)

    solar_orig = weather_data['solar']
    # 太阳辐射插值
    try:
        if len(solar_orig) == len(times_orig):
            solar_array = np.clip(np.interp(times_new, times_orig, solar_orig), 0, 1500)
        else:
            solar_array = np.zeros(num_times)
    except Exception:
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
    except Exception:
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


def calculate_legacy_line_data(
    line_data, correction_config, conductor_params, progress_bar
):
    """保留旧矩阵调用兼容性；主页面计算按钮不再使用此入口。"""
    # 应用气象修正
    line_data = apply_weather_corrections(
        line_data, correction_config, conductor_params
    )
    if any(
        correction_config.get(key)
        for key in ("vertical", "terrain", "desert", "wind_dir")
    ):
        progress_bar.progress(65)
    progress_bar.progress(70)
    calc_results = st.session_state.analyzer.calculate_max_current_for_points(
        line_data['points_km'],
        line_data['elevations'],
        line_data['temps'],
        line_data['winds'],
        line_data['angles'],
        line_data['solar'],
        line_data['times'],
        conductor_params['max_allow_temp'],
        base_params=conductor_params,
        terrain_data=None,
    )
    line_data['max_currents'] = calc_results['max_currents']
    line_data['corrected_winds'] = calc_results['corrected_winds']
    line_data['local_temps'] = calc_results['local_temps']
    return line_data


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
        params['R_low_25'] = st.number_input("电阻 R(25°C)", value=params['R_low_25'], format="%.6f")

    st.markdown("**地形数据配置**")

    # 杆塔坐标文件 - 文件上传
    tower_upload = st.file_uploader("上传杆塔坐标Excel", type=["xlsx"], key="tower_upload")

    # DEM文件 - 文件上传
    dem_upload = st.file_uploader("上传DEM文件 (TIF)", type=["tif", "tiff"], key="dem_upload")

    if st.button("🔄 加载地形数据"):
        status = st.empty()
        status.text("正在加载地形数据...")

        if dem_upload and tower_upload:
            terrain_attempt = terrain_module.load_terrain_upload_pair(
                dem_upload.read(),
                tower_upload.read(),
                dem_loader=load_dem_data,
                tower_loader=load_tower_coordinates,
            )
            if terrain_attempt.dem_data:
                status.success("✓ DEM加载成功")
            else:
                status.error("✗ DEM加载失败")

            if terrain_attempt.tower_coords:
                st.info(f"✓ 成功读取 {len(terrain_attempt.tower_coords)} 个杆塔坐标 (编号: {sorted(terrain_attempt.tower_coords.keys())})")
            else:
                st.error("✗ 杆塔坐标读取失败")

            terrain_module.commit_terrain_snapshot(
                st.session_state,
                terrain_attempt,
            )
        else:
            if not dem_upload:
                status.warning("⚠️ 请先上传DEM文件")
            if not tower_upload:
                status.warning("⚠️ 请先上传杆塔坐标文件")

    st.divider()
    st.header("2. 高级功能")

    with st.expander("气象修正配置", expanded=False):
        enable_vertical_correction = st.checkbox("垂直修正（风速高度折算）", value=True)
        if enable_vertical_correction:
            conductor_height = st.number_input("导线悬挂高度 (m)", value=20.0, min_value=5.0, max_value=100.0)
            anemometer_height = st.number_input("气象站测风高度 (m)", value=10.0, min_value=1.0, max_value=50.0)
            roughness_alpha = st.number_input("地表粗糙度指数", value=0.15, min_value=0.05, max_value=0.5, format="%.2f",
                                              help="沙漠/戈壁: 0.10-0.15, 草地: 0.15-0.20, 城市: 0.25-0.40")
        else:
            conductor_height, anemometer_height, roughness_alpha = 20.0, 10.0, 0.15

        enable_terrain_correction = st.checkbox("地形修正（坡度/坡向）", value=True)
        enable_desert_correction = st.checkbox("沙漠环境修正（辐射增强）", value=True)
        if enable_desert_correction:
            desert_albedo = st.number_input("地表反照率", value=0.35, min_value=0.1, max_value=0.6, format="%.2f",
                                            help="沙漠: 0.30-0.40, 戈壁: 0.25-0.35")
            ground_temp_offset = st.number_input("地表增温偏移 (°C)", value=15.0, min_value=0.0, max_value=30.0)
        else:
            desert_albedo, ground_temp_offset = 0.35, 15.0

        enable_wind_dir_correction = st.checkbox("风向修正（有效横风分量）", value=True)

    with st.expander("AI预测配置", expanded=False):
        enable_ai_prediction = st.checkbox("启用AI残差预测 (XGBoost)", value=False)
        if enable_ai_prediction:
            ai_confidence = st.slider("预测置信区间 (%)", 80, 99, 95)
            ai_lookback = st.number_input("历史回溯窗口（小时）", value=6, min_value=1, max_value=24)
        truth_weather_files = st.file_uploader(
            "上传真实气象数据",
            type=["xlsx", "csv"],
            accept_multiple_files=True,
        )

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
st.markdown("**数据源**: 实测气象数据 + SRTM地形修正 | **标准**: IEEE 738-2023")

tab_line, tab_correction = st.tabs([
    " 1. 线路全景分析",
    " 2. 气象修正与AI预测"
])

# ==============================================================================
# Tab 1: 线路全景分析
# ==============================================================================
with tab_line:
    st.subheader("环境参数获取与全线分析")

    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown("#####  数据配置")

        weather_files = st.file_uploader(
            "上传气象数据文件 (支持多个Excel/CSV)",
            type=['xlsx', 'csv'],
            accept_multiple_files=True,
            help="格式1(Excel): 位置|日期|时刻|环境温度|风速|风向  \n格式2(CSV): 时间|杆塔|经度|纬度|风速WS|风向WD|温度TEM|相对湿度RHU"
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
            status_text.text("正在规范化物理气象数据...")
            physical_snapshot = normalize_uploaded_weather_files(
                weather_files, role="physical"
            )
            truth_snapshot = (
                normalize_uploaded_weather_files(
                    truth_weather_files, role="truth"
                )
                if truth_weather_files
                else None
            )

            progress_bar.progress(30)
            status_text.text("正在构建地形修正表...")
            terrain_data = {}
            if (
                st.session_state.dem_data is not None
                and st.session_state.tower_coords
            ):
                weather_positions = sorted(
                    physical_snapshot.frame["tower_id"].astype(str).unique()
                )
                terrain_data = build_terrain_lookup(
                    st.session_state.dem_data,
                    st.session_state.tower_coords,
                    weather_positions,
                )
            else:
                st.warning("⚠️ 未加载地形数据，将使用气象文件海拔")

            corr_cfg = st.session_state.get('correction_config', {})
            options = CorrectionOptions(
                enable_vertical=bool(corr_cfg.get('vertical', False)),
                enable_terrain=bool(corr_cfg.get('terrain', False)),
                enable_desert=bool(corr_cfg.get('desert', False)),
                enable_wind_direction=bool(corr_cfg.get('wind_dir', False)),
                ref_height_m=corr_cfg.get('anemometer_height', 10.0),
                line_height_m=corr_cfg.get('conductor_height', 20.0),
                roughness_alpha=corr_cfg.get('roughness_alpha', 0.15),
                ground_albedo=corr_cfg.get('desert_albedo', 0.35),
                ground_temp_offset=corr_cfg.get('ground_temp_offset', 15.0),
                line_azimuth_deg=st.session_state.conductor_params.get(
                    'line_azimuth', 90.0
                ),
            )

            progress_bar.progress(60)
            status_text.text("正在修正气象并进行热平衡计算...")
            pipeline = DlrPipeline()
            line_id = derive_line_id(
                physical_snapshot.frame,
                tower_coords=st.session_state.tower_coords,
            )
            result = pipeline.run(
                physical=physical_snapshot,
                truth=truth_snapshot,
                project_id="shagehuang-dlr",
                line_id=line_id,
                interval_minutes=int(time_res),
                terrain_lookup=terrain_data,
                dem_context=st.session_state.dem_data,
                coordinate_context=st.session_state.tower_coords,
                correction_options=options,
                ai_enabled=bool(corr_cfg.get('ai_enabled', False)),
                conductor=st.session_state.conductor_params,
                truth_tolerance=pd.Timedelta(minutes=float(time_res)),
            )
            line_data = result.to_legacy_line_data()
            if not any(
                corr_cfg.get(key)
                for key in ('vertical', 'terrain', 'desert', 'wind_dir')
            ):
                line_data['correction_details'] = None

            st.session_state.physical_weather_snapshot = physical_snapshot
            st.session_state.truth_weather_snapshot = truth_snapshot
            st.session_state.line_data = line_data

            progress_bar.progress(100)
            status_text.text("计算完成！")
            if terrain_data:
                st.success(
                    f"✓ 已应用SRTM地形修正，计算 {len(line_data['positions'])} 个杆塔"
                )
            else:
                st.info(f"✓ 已完成计算 {len(line_data['positions'])} 个杆塔")
            if corr_cfg.get('ai_enabled') and result.model_report.fallbacks:
                st.warning("部分气象模型不可用，相关杆塔已回退到物理气象。")
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

            k1.metric("最低载流量（系统瓶颈）", f"{min_val:.0f} A", f"{min_gain:+.1f}% 对比静态")
            k2.metric("最高载流量", f"{max_val:.0f} A")
            k3.metric("平均载流量", f"{avg_val:.0f} A", f"{avg_gain:+.1f}%")
            k4.metric("静态额定值（基准）", f"{static_val:.0f} A")

            # --- 图表 1: 全线瓶颈载流量 ---
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=plot_times, y=line_rating,
                mode='lines+markers',
                name='动态增容（SRTM地形修正）',
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
                height=300,
                margin=dict(t=40, b=30),
                hovermode='x unified',
                xaxis=dict(tickformat="%Y-%m-%d\n%H:%M")
            )
            st.plotly_chart(fig, use_container_width=True)

            # --- 图表 2: 单塔详情 ---
            st.markdown("##### 🔍 单塔微气象与修正详情")

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
                    height=320,
                    margin=dict(t=40, b=30),
                    hovermode='x unified',
                    legend=dict(orientation="h", y=1.1),
                    xaxis=dict(tickformat="%Y-%m-%d\n%H:%M")  # 显示日期
                )

                fig_tower.update_yaxes(title_text="温度 (°C) / 风速 (m/s)", secondary_y=False)
                fig_tower.update_yaxes(title_text="载流量 (A)", secondary_y=True)

                st.plotly_chart(fig_tower, use_container_width=True)

            # --- 图表 3: 热力图 ---
            with st.expander("查看全线风速分布热力图"):
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
# Tab 2: 气象修正与AI预测
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

                selected_solar_orig = corr_details['solar_orig'][sel_corr_idx]
                selected_solar = data['solar'][sel_corr_idx]
                selected_solar_delta = corr_details['desert_solar_delta'][
                    sel_corr_idx
                ]
                fig_solar_cmp = go.Figure()
                fig_solar_cmp.add_trace(go.Scatter(
                    x=plot_times, y=selected_solar_orig,
                    name='原始太阳辐射', line=dict(color='orange', dash='dot', width=1)
                ))
                fig_solar_cmp.add_trace(go.Scatter(
                    x=plot_times, y=selected_solar,
                    name='修正后辐射（含反射+长波）', line=dict(color='red', width=2)
                ))
                fig_solar_cmp.add_trace(go.Scatter(
                    x=plot_times, y=selected_solar_delta,
                    name='辐射增量', fill='tozeroy',
                    fillcolor='rgba(255,165,0,0.2)', line=dict(color='orange', width=1)
                ))
                fig_solar_cmp.update_layout(
                    title="沙漠环境辐射修正（地表反射+长波辐射）",
                    xaxis_title="日期时间", yaxis_title="辐射强度 (W/m²)",
                    height=350, hovermode='x unified',
                    xaxis=dict(tickformat="%Y-%m-%d\n%H:%M")
                )
                st.plotly_chart(fig_solar_cmp, use_container_width=True)

                sc1, sc2, sc3 = st.columns(3)
                sc1.metric("原始平均辐射", f"{np.mean(selected_solar_orig):.1f} W/m²")
                sc2.metric("修正后平均辐射", f"{np.mean(selected_solar):.1f} W/m²")
                sc3.metric("平均辐射增量", f"{np.mean(selected_solar_delta):.1f} W/m²")

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
                title="全线风速综合修正系数（>1增强，<1减弱）",
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
            comparison = data.get('comparison_weather')
            metrics = data.get('weather_metrics')
            model_report = data.get('model_report')
            if isinstance(comparison, pd.DataFrame) and not comparison.empty:
                tower_index = int(
                    st.session_state.get("corr_tower_select", 0)
                )
                tower_index = min(tower_index, len(data['positions']) - 1)
                tower_id = str(data['positions'][tower_index])
                tower_weather = comparison.loc[
                    comparison['tower_id'].astype(str) == tower_id
                ].sort_values('timestamp', kind='mergesort')
                tower_weather = tower_weather.loc[
                    tower_weather['timestamp'].isin(plot_times)
                ]

                fig_ai = make_subplots(specs=[[{"secondary_y": True}]])
                fig_ai.add_trace(go.Scatter(
                    x=tower_weather['timestamp'],
                    y=tower_weather['wind_speed_physical'],
                    name='物理风速', line=dict(color='gray', dash='dot')
                ), secondary_y=False)
                fig_ai.add_trace(go.Scatter(
                    x=tower_weather['timestamp'],
                    y=tower_weather['wind_speed_ai'],
                    name='AI修正风速', line=dict(color='blue', width=2)
                ), secondary_y=False)
                if tower_weather['wind_speed_truth'].notna().any():
                    fig_ai.add_trace(go.Scatter(
                        x=tower_weather['timestamp'],
                        y=tower_weather['wind_speed_truth'],
                        name='真实风速', line=dict(color='green', width=2)
                    ), secondary_y=False)

                fig_ai.add_trace(go.Scatter(
                    x=tower_weather['timestamp'],
                    y=tower_weather['ambient_temp_physical'],
                    name='物理温度', line=dict(color='orange', dash='dot')
                ), secondary_y=True)
                fig_ai.add_trace(go.Scatter(
                    x=tower_weather['timestamp'],
                    y=tower_weather['ambient_temp_ai'],
                    name='AI修正温度', line=dict(color='red', width=2)
                ), secondary_y=True)
                if tower_weather['ambient_temp_truth'].notna().any():
                    fig_ai.add_trace(go.Scatter(
                        x=tower_weather['timestamp'],
                        y=tower_weather['ambient_temp_truth'],
                        name='真实温度', line=dict(color='purple', width=2)
                    ), secondary_y=True)
                fig_ai.update_layout(
                    title=f"塔位 {tower_id} - 气象物理值、AI修正值与真实值",
                    height=400,
                    hovermode='x unified',
                    xaxis=dict(tickformat="%Y-%m-%d\n%H:%M"),
                    legend=dict(orientation="h", y=1.12),
                )
                fig_ai.update_yaxes(
                    title_text="风速 (m/s)", secondary_y=False
                )
                fig_ai.update_yaxes(
                    title_text="温度 (°C)", secondary_y=True
                )
                st.plotly_chart(fig_ai, use_container_width=True)

                wind_mae = getattr(metrics, 'wind_speed_mae', None)
                temp_mae = getattr(metrics, 'ambient_temp_mae', None)
                active_models = getattr(model_report, 'active_model_count', 0)
                ai1, ai2, ai3 = st.columns(3)
                ai1.metric(
                    "风速 MAE",
                    f"{wind_mae:.3f} m/s" if wind_mae is not None else "--",
                )
                ai2.metric(
                    "温度 MAE",
                    f"{temp_mae:.3f} °C" if temp_mae is not None else "--",
                )
                ai3.metric("已启用模型数", str(active_models))
            else:
                st.warning("当前结果没有可展示的气象对比数据。")
