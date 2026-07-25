# DLR 修正链路与弧垂后验证实施计划

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 修复气象/地形修正、按塔 XGBoost 残差模型和 IEEE 738 DLR 计算漏洞，并增加与主 DLR 完全隔离的倾角弧垂后验证页面。

**Architecture:** 采用规范化长表和不可变阶段对象组织唯一数据流水线，所有地形与项目修正只在热模型上游执行一次。IEEE 738 由一个纯计算内核实现，AI 模型按项目/线路/杆塔/目标注册，弧垂服务只消费主流程发布的只读快照并独立持久化结果。

**Tech Stack:** Python 3.9+、Streamlit、Pandas、NumPy、SciPy、Rasterio、PyProj、XGBoost、scikit-learn、joblib、filelock、Plotly、pytest、streamlit.testing

---

## 执行前约束

- 仓库根目录：/Users/aa/Library/CloudStorage/GoogleDrive-wenhaozhu628@gmail.com/我的云端硬盘/项目/沙戈荒项目/界面/DLR动态增容评估系统/12.24
- 当前设计提交：c943f44。
- 当前工作区已有用户修改：dispatch_app_st.py、modules/data_processor.py、tests/modules/test_data_processor.py。执行时必须先阅读并保留这些修改，禁止 reset 或覆盖。
- 所有命令从仓库根目录运行，Python 命令统一使用 python3。
- 实施时按 @superpowers:test-driven-development 执行每个任务，完成前按 @superpowers:verification-before-completion 做证据化验收。
- 页面布局、已有控件和图表容器保持不变，只增加侧边栏真实气象上传入口和独立弧垂后验证页面。
- 每个任务单独提交；若 Git 因云盘锁阻塞，先确认没有活动 Git 进程，再处理明确的遗留锁，不能删除或重建索引。
- 小样本 full-fit 的生命周期规则固定为：没有现有模型时可以保存为 status=provisional 并在下次运行加载；已有 champion 时 full-fit 候选不能替换它，必须等冻结时间留出集证明改善。

### Task 0: 冻结现有用户气象格式兼容改动

**Files:**
- Verify only: dispatch_app_st.py
- Verify only: modules/data_processor.py
- Verify only: tests/modules/test_data_processor.py

**Step 1: 检查边界和测试**

~~~bash
git diff --check -- dispatch_app_st.py modules/data_processor.py tests/modules/test_data_processor.py
python3 -m pytest tests/modules/test_data_processor.py -q
~~~

Expected: 现有新旧格式测试通过，且只看到这三个已知文件的修改。

**Step 2: 阅读并确认**

确认当前改动只涉及逐文件气象规范化、新格式列识别和对应测试；若发现其他行为或冲突，先停下并保留原状，不得使用 reset、checkout 或 stash 覆盖。

**Step 3: 单独提交基线**

~~~bash
git add dispatch_app_st.py modules/data_processor.py tests/modules/test_data_processor.py
git commit -m "feat: preserve legacy and tower-time weather uploads"
~~~

Expected: 只提交上述三个文件，后续任务从干净边界开始。后续重叠文件只能用精确路径和精确补丁暂存，禁止 git add .。

### Task 1: 固定运行配置、依赖与机械参数目录

**Files:**
- Modify: requirements.txt
- Modify: config/config.py
- Modify: tests/config/test_config.py
- Create: tests/config/test_sag_config.py
- Modify: .gitignore

**Step 1: 写失败测试**

在 tests/config/test_config.py 增加：

~~~python
from config.config import (
    MODEL_DIR,
    PHYSICAL_BOUNDS,
    PROJECT_TIMEZONE,
    STANDARD_CONDUCTORS,
)


def test_runtime_and_physical_defaults_are_explicit():
    assert PROJECT_TIMEZONE == "Asia/Shanghai"
    assert PHYSICAL_BOUNDS["wind_speed"] == (0.0, 75.0)
    assert PHYSICAL_BOUNDS["ambient_temp"] == (-60.0, 70.0)
    assert MODEL_DIR.name == "models"
    assert STANDARD_CONDUCTORS["ACSR Drake (795 kcmil)"]["area_m2"] > 0
~~~

新建 tests/config/test_sag_config.py：

~~~python
from config.config import SAG_VALIDATION_DEFAULTS


def test_sag_defaults_use_si_units_and_safe_bounds():
    defaults = SAG_VALIDATION_DEFAULTS
    assert defaults["formula_version"] == "CN-patent-2025-v1"
    assert defaults["min_angle_deg"] > 0
    assert defaults["max_angle_deg"] < 90
    assert defaults["gravity_m_s2"] == 9.80665
    assert defaults["recovery_samples"] >= 2
~~~

**Step 2: 运行测试并确认失败**

Run:

~~~bash
python3 -m pytest tests/config/test_config.py tests/config/test_sag_config.py -q
~~~

Expected: FAIL，提示 PROJECT_TIMEZONE、PHYSICAL_BOUNDS 或 SAG_VALIDATION_DEFAULTS 尚未定义。

**Step 3: 最小实现**

在 config/config.py 中增加项目根目录、环境变量可覆盖路径、物理边界和机械参数：

~~~python
import os

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROJECT_TIMEZONE = "Asia/Shanghai"
MODEL_DIR = Path(os.getenv("DLR_MODEL_DIR", PROJECT_ROOT / "models"))
AUDIT_LOG_DIR = Path(os.getenv("DLR_AUDIT_LOG_DIR", PROJECT_ROOT / "runtime" / "logs"))
SAG_RESULT_DIR = Path(os.getenv("DLR_SAG_RESULT_DIR", PROJECT_ROOT / "runtime" / "results" / "sag"))

PHYSICAL_BOUNDS = {
    "wind_speed": (0.0, 75.0),
    "ambient_temp": (-60.0, 70.0),
}

SAG_VALIDATION_DEFAULTS = {
    "formula_version": "CN-patent-2025-v1",
    "gravity_m_s2": 9.80665,
    "min_angle_deg": 0.05,
    "max_angle_deg": 89.5,
    "reference_temp_c": 20.0,
    "reference_tension_n": 20000.0,
    "span_m": 300.0,
    "elastic_modulus_pa": 7.0e10,
    "area_m2": 6.75e-4,
    "thermal_expansion_per_c": 1.9e-5,
    "base_threshold_c": 5.0,
    "recovery_ratio": 0.6,
    "recovery_samples": 3,
    "recovery_alpha": 0.25,
}
~~~

为 STANDARD_CONDUCTORS 补充 area_m2、elastic_modulus_pa、thermal_expansion_per_c、rated_tensile_strength_n 和 mass_per_length_kg_m。requirements.txt 明确加入 joblib、xgboost、scikit-learn、rasterio、pyproj、filelock、pytest。将 models/、runtime/ 加入 .gitignore。

**Step 4: 运行测试并确认通过**

Run:

~~~bash
python3 -m pytest tests/config/test_config.py tests/config/test_sag_config.py -q
~~~

Expected: PASS。

**Step 5: 提交**

~~~bash
git add requirements.txt config/config.py tests/config/test_config.py tests/config/test_sag_config.py .gitignore
git commit -m "chore: define DLR runtime and mechanical defaults"
~~~

### Task 2: 建立规范化气象长表和质量报告

**Files:**
- Modify: modules/data_processor.py
- Modify: tests/modules/test_data_processor.py
- Modify: tests/fixtures/sample_data.py
- Create: modules/weather_upload.py
- Create: tests/modules/test_weather_upload.py

**Step 1: 写失败测试**

保留现有新格式测试，并增加：

~~~python
def test_canonical_weather_preserves_tower_id_timezone_and_source_role():
    raw = pd.DataFrame({
        "时间": ["2026-07-23 00:00"],
        "杆塔": ["001号"],
        "风速WS(m/s)": [2.0],
        "风向WD(°)": [359.0],
        "温度TEM(℃)": [20.0],
    })
    result = canonicalize_weather_frame(
        raw,
        role="physical",
        timezone="Asia/Shanghai",
        source_hash="abc",
    )
    assert result.frame.loc[0, "tower_id"] == "001"
    assert str(result.frame["timestamp"].dt.tz) == "Asia/Shanghai"
    assert result.frame.loc[0, "dataset_role"] == "physical"
    assert result.frame.loc[0, "source_file_hash"] == "abc"


def test_canonical_weather_drops_only_invalid_rows_and_reports_them():
    raw = make_weather_dataframe()
    raw.loc[0, "风速"] = -1
    result = canonicalize_weather_frame(raw, role="truth")
    assert len(result.frame) == len(raw) - 1
    assert result.report.dropped_rows == 1
    assert "wind_speed_out_of_range" in result.report.reasons
~~~

**Step 2: 运行测试并确认失败**

~~~bash
python3 -m pytest tests/modules/test_data_processor.py -q
~~~

Expected: FAIL，提示 canonicalize_weather_frame 未定义。

**Step 3: 最小实现**

在现有用户改动之上增加契约，不能删掉 normalize_weather_input_dataframe 对两种格式的支持：

~~~python
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


def normalize_tower_id(value) -> str:
    text = str(value).strip()
    match = re.search(r"(\d+)(?=\s*号|$)", text)
    if match:
        return match.group(1)
    raise ValueError("无法解析杆塔编号")


def canonicalize_weather_frame(
    df: pd.DataFrame,
    role: str,
    timezone: str = PROJECT_TIMEZONE,
    source_hash: str = "",
) -> CanonicalWeatherResult:
    normalized = normalize_weather_input_dataframe(df).copy()
    normalized["tower_id"] = normalized["position"].map(normalize_tower_id)
    normalized["timestamp"] = build_timezone_aware_timestamp(normalized, timezone)
    normalized["dataset_role"] = role
    normalized["source_file_hash"] = source_hash
    cleaned, report = validate_weather_rows(normalized)
    return CanonicalWeatherResult(cleaned.reset_index(drop=True), report)
~~~

必需字段、数值范围、单位和重复键统一校验。保留旧 wrapper，使当前页面仍能调用 normalize_weather_input_dataframe。

**Step 4: 运行测试并确认通过**

~~~bash
python3 -m pytest tests/modules/test_data_processor.py -q
~~~

Expected: PASS，现有旧格式与新格式测试均通过。

**Step 5: 提交**

~~~bash
git add modules/data_processor.py tests/modules/test_data_processor.py tests/fixtures/sample_data.py modules/weather_upload.py tests/modules/test_weather_upload.py
git commit -m "feat: normalize and freeze weather uploads"
~~~

**上传字节冻结补充步骤**

实现 UploadBlob 和 freeze_uploaded_file/freeze_uploaded_files，读取每个 Streamlit UploadedFile 的 bytes 一次并计算 SHA-256，再按文件分别解析 CSV/Excel。规范化结果只保存 bytes 哈希、文件名、规范化长表和 QC 事件，不保存 UploadedFile 对象或原始 bytes 引用。

~~~python
@dataclass(frozen=True)
class UploadBlob:
    name: str
    content: bytes
    sha256: str


def freeze_uploaded_file(uploaded_file) -> UploadBlob:
    content = uploaded_file.getvalue()
    return UploadBlob(
        name=uploaded_file.name,
        content=content,
        sha256=hashlib.sha256(content).hexdigest(),
    )
~~~

新增 test_freeze_uploaded_file_reads_bytes_once 和 test_mixed_files_are_normalized_before_concat；同一文件哈希不能同时作为 physical 和 truth 训练输入。

### Task 3: 实现逐塔重采样和真实值时间对齐

**Files:**
- Create: modules/weather_pipeline.py
- Create: tests/modules/test_weather_pipeline.py

**Step 1: 写失败测试**

~~~python
def test_resampling_never_uses_another_tower_values():
    source = make_two_tower_uneven_weather()
    result = resample_weather_by_tower(source, interval_minutes=30)
    tower_a = result[result["tower_id"] == "001"]
    assert tower_a["ambient_temp"].max() < 30


def test_wind_direction_uses_circular_interpolation():
    source = make_direction_wrap_weather(359.0, 1.0)
    result = resample_weather_by_tower(source, interval_minutes=30)
    middle = result.iloc[1]["wind_direction"]
    assert middle < 5 or middle > 355


def test_truth_alignment_uses_same_tower_and_no_future_sample():
    aligned, report = align_physical_and_truth(
        make_physical_truth_frames(),
        tolerance=pd.Timedelta("10min"),
    )
    assert (aligned["truth_timestamp"] <= aligned["timestamp"]).all()
    assert report.matched_rows > 0


def test_same_input_hash_is_rejected_for_training():
    with pytest.raises(ValueError, match="不能同时作为"):
        align_physical_and_truth(same_hash_physical(), same_hash_truth())
~~~

**Step 2: 运行测试并确认失败**

~~~bash
python3 -m pytest tests/modules/test_weather_pipeline.py -q
~~~

Expected: FAIL，提示 modules.weather_pipeline 不存在。

**Step 3: 最小实现**

实现按 tower_id 分组的重采样、圆周风向插值、测量高度统一和 backward merge_asof：

~~~python
@dataclass(frozen=True)
class AlignmentReport:
    physical_rows: int
    truth_rows: int
    matched_rows: int
    unmatched_rows: int
    coverage: float


def circular_interpolate(series: pd.Series) -> pd.Series:
    radians = np.deg2rad(series)
    sin_value = pd.Series(np.sin(radians), index=series.index).interpolate()
    cos_value = pd.Series(np.cos(radians), index=series.index).interpolate()
    return np.rad2deg(np.arctan2(sin_value, cos_value)) % 360


def align_physical_and_truth(physical, truth, tolerance):
    ensure_distinct_datasets(physical, truth)
    groups = []
    for tower_id, physical_tower in physical.groupby("tower_id", sort=False):
        truth_tower = truth[truth["tower_id"] == tower_id]
        groups.append(pd.merge_asof(
            physical_tower.sort_values("timestamp"),
            truth_tower.sort_values("timestamp").rename(
                columns={"timestamp": "truth_timestamp"}
            ),
            left_on="timestamp",
            right_on="truth_timestamp",
            direction="backward",
            tolerance=tolerance,
            suffixes=("_physical", "_truth"),
        ))
    return pd.concat(groups, ignore_index=True), build_alignment_report(groups)
~~~

禁止跨塔均值、跨塔插值和未来真值匹配。训练残差前将 truth 和 physical 折算到同一 measurement_height。

**Step 4: 运行测试并确认通过**

~~~bash
python3 -m pytest tests/modules/test_weather_pipeline.py -q
~~~

Expected: PASS。

**Step 5: 提交**

~~~bash
git add modules/weather_pipeline.py tests/modules/test_weather_pipeline.py
git commit -m "feat: align weather per tower without time leakage"
~~~

### Task 4: 修复 GeoTIFF、CRS 和杆塔地形查询

**Files:**
- Modify: modules/terrain.py
- Modify: tests/modules/test_terrain.py
- Modify: dispatch_app_st.py:17-230

**Step 1: 写失败测试**

~~~python
def test_geotiff_query_uses_crs_transform_and_affine(tmp_path):
    tif_path = write_test_geotiff(
        tmp_path,
        values=np.array([[100.0, 110.0], [120.0, 130.0]], dtype="float32"),
        crs="EPSG:4326",
        transform=from_origin(120.0, 50.0, 0.01, 0.01),
    )
    dem = load_dem_data(tif_path)
    result = query_dem_at_point(dem, lon=120.005, lat=49.995)
    assert result["elevation"] == pytest.approx(100.0)
    assert result["source"] == "measured"


def test_out_of_bounds_is_default_not_edge_pixel(tmp_path):
    dem = load_dem_data(write_test_geotiff(tmp_path))
    result = query_dem_at_point(dem, lon=0.0, lat=0.0)
    assert result["elevation"] == 1000.0
    assert result["source"] == "default"
    assert result["reason"] == "out_of_bounds"


def test_terrain_lookup_is_keyed_by_canonical_tower_id():
    result = build_terrain_lookup(None, {"001": {"lon": 120, "lat": 49}}, ["001"])
    assert list(result) == ["001"]
~~~

**Step 2: 运行测试并确认失败**

~~~bash
python3 -m pytest tests/modules/test_terrain.py -q
~~~

Expected: FAIL，当前实现使用硬编码经纬度范围、夹边并以数组索引为键。

**Step 3: 最小实现**

使用 rasterio 读取 CRS、transform、bounds、nodata，使用 rasterio.warp.transform 转换 WGS84 坐标。定义 DemGrid 和 TerrainSample 数据类。经纬度 CRS 下使用 pyproj.Geod 将像元角度换算为米后计算坡度；越界/nodata 返回带 reason 的默认样本。

dispatch_app_st.py 删除本地 DEM 数学实现，改为导入 modules.terrain 的兼容包装；页面上传控件和提示位置保持不变。主页面后续只能把冻结后的 UploadBlob 交给规范化服务，不能直接把 UploadedFile 放进 session_state。

**Step 4: 运行测试并确认通过**

~~~bash
python3 -m pytest tests/modules/test_terrain.py tests/modules/test_data_processor.py -q
~~~

Expected: PASS。

**Step 5: 提交**

~~~bash
git add modules/terrain.py tests/modules/test_terrain.py dispatch_app_st.py
git commit -m "fix: honor GeoTIFF coordinates in terrain lookup"
~~~

### Task 5: 建立不可变且只能执行一次的气象修正

**Files:**
- Modify: modules/weather_correction.py
- Modify: tests/modules/test_weather_correction.py
- Modify: dispatch_app_st.py:233-377

**Step 1: 写失败测试**

~~~python
def test_weather_correction_is_pure_and_single_stage():
    original = make_canonical_weather()
    before = original.copy(deep=True)
    corrected = WeatherCorrectionService().apply(
        original, make_terrain_lookup(), CorrectionOptions()
    )
    pd.testing.assert_frame_equal(original, before)
    assert set(corrected["correction_stage"]) == {"terrain_corrected"}
    with pytest.raises(ValueError, match="已经修正"):
        WeatherCorrectionService().apply(
            corrected, make_terrain_lookup(), CorrectionOptions()
        )


def test_wind_direction_is_not_multiplied_into_wind_speed():
    corrected = WeatherCorrectionService().apply(
        make_canonical_weather(wind_speed=4.0, wind_direction=0.0),
        terrain_lookup={},
        options=CorrectionOptions(enable_vertical=False, enable_terrain=False),
    )
    assert corrected.loc[0, "wind_speed_local"] == 4.0
    assert corrected.loc[0, "wind_angle_deg"] == 0.0
~~~

**Step 2: 运行测试并确认失败**

~~~bash
python3 -m pytest tests/modules/test_weather_correction.py -q
~~~

Expected: FAIL，当前实现允许重复修正并把风向系数乘入风速。

**Step 3: 最小实现**

修正服务只复制输入并输出：

~~~python
physical_columns = (
    "wind_speed_physical",
    "ambient_temp_physical",
    "solar_radiation_physical",
)
local_columns = (
    "wind_speed_local",
    "ambient_temp_local",
    "solar_radiation_local",
    "wind_angle_deg",
)
~~~

高度修正、温度递减、坡向风速和沙漠辐射只执行一次。风向控件启用时只计算导线夹角；关闭时传 90°，不修改风速。移除热模型会再次使用的 slope/aspect 参数。dispatch_app_st.py 的旧修正函数改成薄适配或删除，页面展示所需 correction_details 由新服务生成。

**Step 4: 运行测试并确认通过**

~~~bash
python3 -m pytest tests/modules/test_weather_correction.py tests/modules/test_weather_pipeline.py -q
~~~

Expected: PASS。

**Step 5: 提交**

~~~bash
git add modules/weather_correction.py tests/modules/test_weather_correction.py dispatch_app_st.py
git commit -m "fix: apply local weather corrections exactly once"
~~~

### Task 6: 冻结 IEEE 738 官方 Drake 金标准

**Files:**
- Create: tests/fixtures/ieee738_reference.py
- Create: tests/modules/test_thermal_engine_ieee738.py

**Step 1: 写标准 fixture 和失败测试**

~~~python
DRAKE_STEADY_PARAMS = {
    "D0": 0.02814,
    "R_low_25": 7.283e-5,
    "R_high_75": 8.688e-5,
    "R_high_200": 1.220e-4,
    "emissivity": 0.8,
    "absorptivity": 0.8,
    "T_a": 40.0,
    "T_s": 100.0,
    "wind_speed": 0.61,
    "wind_angle": 90.0,
    "elevation": 0.0,
    "latitude": 30.0,
    "line_azimuth": 90.0,
    "day_of_year": 161,
    "time": 11.0,
}


def test_ieee_738_drake_steady_reference():
    result = ThermalCalculator().calculate_heat_balance(DRAKE_STEADY_PARAMS)
    assert result.q_convection_natural == pytest.approx(42.42, abs=0.15)
    assert result.q_convection_low_re == pytest.approx(82.10, abs=0.20)
    assert result.q_convection_high_re == pytest.approx(77.06, abs=0.20)
    assert result.q_radiation == pytest.approx(39.11, abs=0.15)
    assert result.q_solar == pytest.approx(22.45, abs=0.25)
    assert result.resistance == pytest.approx(9.391e-5, rel=2e-3)
    assert result.current_a == pytest.approx(1025.0, abs=2.0)
~~~

注释中记录 IEEE 738-2023 式(9)使用 1.01；官方算例首行的 1.05 是印刷不一致，代入过程和结果均使用 1.01。

**Step 2: 运行并确认当前实现失败**

~~~bash
python3 -m pytest tests/modules/test_thermal_engine_ieee738.py -q
~~~

Expected: FAIL，当前实现没有 calculate_heat_balance，且现有公式结果约 649 A。

**Step 3: 仅提交失败的金标准测试**

本任务不修改生产代码，保留 RED 证据。

**Step 4: 提交**

~~~bash
git add tests/fixtures/ieee738_reference.py tests/modules/test_thermal_engine_ieee738.py
git commit -m "test: freeze IEEE 738 Drake reference values"
~~~

### Task 7: 修复 IEEE 738 稳态热平衡内核

**Files:**
- Modify: modules/thermal_engine.py:1-333
- Modify: tests/modules/test_thermal_engine_ieee738.py
- Modify: tests/modules/test_thermal_engine.py
- Verify: thermal_functions.py

**Step 1: 增加边界失败测试**

~~~python
def test_zero_wind_keeps_natural_convection():
    params = drake_params(wind_speed=0.0)
    result = ThermalCalculator().calculate_heat_balance(params)
    assert result.q_convection_natural > 0
    assert result.q_convection == result.q_convection_natural


@pytest.mark.parametrize(
    ("angle", "expected"),
    [(0.0, 0.388), (90.0, 1.0), (180.0, 0.388)],
)
def test_wind_angle_factor_is_normalized(angle, expected):
    assert ThermalCalculator.wind_angle_factor(angle) == pytest.approx(expected, abs=0.002)


def test_steady_calculation_does_not_mutate_params():
    params = drake_params()
    before = copy.deepcopy(params)
    ThermalCalculator().calculate_steady_state_current(params)
    assert params == before
~~~

**Step 2: 运行并确认失败**

~~~bash
python3 -m pytest tests/modules/test_thermal_engine_ieee738.py tests/modules/test_thermal_engine.py -q
~~~

Expected: FAIL，自然对流、风向归一、Drake 热项和输入不变性均未满足。

**Step 3: 最小实现**

在 modules/thermal_engine.py 定义 frozen HeatBalanceResult。实现 IEEE 738-2023：

~~~python
qcn = 3.645 * rho_f**0.5 * diameter_m**0.75 * delta_t**1.25
qc1 = k_angle * (1.01 + 1.35 * reynolds**0.52) * k_f * delta_t
qc2 = k_angle * 0.754 * reynolds**0.60 * k_f * delta_t
qc = max(qcn, qc1, qc2)
~~~

空气物性使用 film_temperature=(T_s+T_a)/2；移除热核中的湿度乘数、风沙系数、slope/aspect 修正和低风速早返回。按标准实现辐射、晴空太阳、实测太阳二选一、电阻插值和热容量。所有 public 方法复制输入，不写原字典。

thermal_functions.py 继续只导入同一 ThermalCalculator，新增测试确认旧入口和新入口数值一致。

**Step 4: 运行并确认通过**

~~~bash
python3 -m pytest tests/modules/test_thermal_engine_ieee738.py tests/modules/test_thermal_engine.py -q
~~~

Expected: PASS，Drake 结果约 1025 A。

**Step 5: 提交**

~~~bash
git add modules/thermal_engine.py tests/modules/test_thermal_engine_ieee738.py tests/modules/test_thermal_engine.py thermal_functions.py
git commit -m "fix: implement IEEE 738 steady heat balance"
~~~

### Task 8: 修复暂态积分、显式导线参数和线路分析器

**Files:**
- Modify: modules/thermal_engine.py:335-510
- Create: tests/modules/test_thermal_transient.py
- Create: tests/modules/test_line_analyzer.py

**Step 1: 写失败测试**

~~~python
def test_drake_step_response_uses_ten_second_steps():
    temperatures = ThermalCalculator().calculate_transient_temperature(
        params=drake_transient_params(),
        time_steps=[10.0, 10.0],
        initial_temp=100.0,
        current_profile=[1200.0, 1200.0],
    )
    assert temperatures[1] - temperatures[0] == pytest.approx(0.28, abs=0.03)
    assert temperatures[2] - temperatures[1] == pytest.approx(0.27, abs=0.03)


def test_line_analyzer_requires_and_uses_selected_conductor():
    first = analyzer.calculate_max_current_for_points(
        **weather_matrix(), base_params=drake_conductor()
    )
    second = analyzer.calculate_max_current_for_points(
        **weather_matrix(), base_params=jl_630_conductor()
    )
    assert not np.allclose(first["max_currents"], second["max_currents"])


def test_line_analyzer_rejects_terrain_reapplication():
    with pytest.raises(ValueError, match="上游"):
        analyzer.calculate_max_current_for_points(
            **weather_matrix(),
            base_params=drake_conductor(),
            terrain_data={0: {"slope": 10}},
        )
~~~

**Step 2: 运行并确认失败**

~~~bash
python3 -m pytest tests/modules/test_thermal_transient.py tests/modules/test_line_analyzer.py -q
~~~

Expected: FAIL，暂态步进和显式导线边界不满足。

**Step 3: 最小实现**

每个暂态子步重新计算 qc、qr、qs、R；外部区间大于 10 s 时拆成不超过 10 s 的子步。校验电流、天气和时间长度。实现 1000 A 稳态温度约 97.5°C、1025→1200 A 阶跃，以及 15 min/125°C 暂态额定值约 1312 A 的回归。

LineAnalyzer 新路径必须显式传 base_params，只接收已完成修正的 temps/winds/angles/solar/elevations，不再注入 terrain_data；返回键保持 max_currents、corrected_winds、local_temps，并新增 bottleneck_tower_ids 旁路元数据。

**Step 4: 运行并确认通过**

~~~bash
python3 -m pytest tests/modules/test_thermal_transient.py tests/modules/test_line_analyzer.py tests/modules/test_thermal_engine_ieee738.py -q
~~~

Expected: PASS。

**Step 5: 提交**

~~~bash
git add modules/thermal_engine.py tests/modules/test_thermal_transient.py tests/modules/test_line_analyzer.py
git commit -m "fix: implement IEEE 738 transient line analysis"
~~~

### Task 9: 实现按塔特征、训练和纯气象评价

**Files:**
- Modify: modules/ai_prediction.py
- Create: modules/ai_training.py
- Modify: tests/modules/test_ai_prediction.py
- Create: tests/modules/test_ai_training.py

**Step 1: 写失败测试**

~~~python
def test_features_never_lag_across_towers():
    frame = make_interleaved_two_tower_training_frame()
    features = FeatureBuilder().transform(frame, physical_col="wind_speed_local")
    first_rows = features.groupby("tower_id", sort=False).head(1)
    assert np.allclose(first_rows["lag_1"], first_rows["wind_speed_local"])


def test_training_metrics_compare_weather_to_truth_not_dlr():
    result = ResidualTrainer(estimator_factory=offset_factory).train_target(
        make_training_frame(), target="wind_speed"
    )
    assert set(result.metrics) == {
        "baseline_mae", "baseline_rmse", "corrected_mae", "corrected_rmse"
    }
    assert result.metrics["corrected_mae"] < result.metrics["baseline_mae"]


def test_single_sample_is_trained_without_rejection():
    result = ResidualTrainer(estimator_factory=constant_factory).train_target(
        make_one_row_training_frame(), target="ambient_temp"
    )
    assert result.metadata["evaluation_mode"] == "full_fit"
    assert result.metadata["sample_count"] == 1
~~~

**Step 2: 运行并确认失败**

~~~bash
python3 -m pytest tests/modules/test_ai_prediction.py tests/modules/test_ai_training.py -q
~~~

Expected: FAIL，当前只有推理接口且 lag 会跨塔。

**Step 3: 最小实现**

FeatureBuilder 按 tower_id、timestamp 稳定排序，使用 hour/day 周期、风向 sin/cos、物理天气、地形和物理 lag。ResidualTrainer 每塔每目标建立 XGBRegressor：

~~~python
def default_estimator():
    return XGBRegressor(
        objective="reg:squarederror",
        n_estimators=120,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42,
        n_jobs=1,
    )
~~~

能形成连续时间块时做冻结时间留出/滚动验证；否则全量拟合。单样本、常量目标和训练异常回退零/中位数残差估计器。指标只包含风速/温度的 MAE、RMSE。

预测残差使用该塔训练残差分位数/MAD 限制，最终风速和温度再按 PHYSICAL_BOUNDS 裁剪。

**Step 4: 运行并确认通过**

~~~bash
python3 -m pytest tests/modules/test_ai_prediction.py tests/modules/test_ai_training.py -q
~~~

Expected: PASS。

**Step 5: 提交**

~~~bash
git add modules/ai_prediction.py modules/ai_training.py tests/modules/test_ai_prediction.py tests/modules/test_ai_training.py
git commit -m "feat: train per-tower weather residual models"
~~~

### Task 10: 实现模型注册、原子晋级和逐塔回退

**Files:**
- Create: modules/model_registry.py
- Create: tests/modules/test_model_registry.py
- Modify: modules/ai_training.py

**Step 1: 写失败测试**

~~~python
def test_same_tower_on_different_lines_never_shares_model(tmp_path):
    registry = ModelRegistry(tmp_path)
    first = registry.path_for(ModelKey("p", "line-a", "001", "wind_speed"))
    second = registry.path_for(ModelKey("p", "line-b", "001", "wind_speed"))
    assert first != second


def test_full_fit_cannot_replace_existing_champion(tmp_path):
    registry = registry_with_champion(tmp_path, corrected_mae=1.0)
    candidate = candidate_bundle(evaluation_mode="full_fit", corrected_mae=0.1)
    decision = registry.promote(candidate)
    assert decision.promoted is False
    assert decision.reason == "full_fit_cannot_replace_champion"


def test_corrupt_model_falls_back_only_for_affected_tower(tmp_path):
    registry = registry_with_two_towers(tmp_path)
    corrupt(registry.path_for(key_for("001")))
    loaded = registry.load_many([key_for("001"), key_for("002")])
    assert loaded[key_for("001")].fallback_reason == "corrupt_model"
    assert loaded[key_for("002")].bundle is not None
~~~

**Step 2: 运行并确认失败**

~~~bash
python3 -m pytest tests/modules/test_model_registry.py -q
~~~

Expected: FAIL，模块不存在。

**Step 3: 最小实现**

定义 ModelKey、ModelMetadata、PromotionDecision。路径采用 project_id/line_id/tower_id/target。每个 key 使用 FileLock；候选 joblib 和 metadata 先写同目录临时文件，重新加载、校验 checksum 后用 os.replace 原子晋级。

晋级规则：

- 无现有模型时，候选气象误差优于物理基线即可成为 provisional active；
- 有现有模型时，必须在同一独立评估集上改善超过阈值；
- evaluation_mode=full_fit 不得覆盖已有冠军；
- DEM、坐标、导线、特征版本或修正配置哈希不兼容时拒绝加载；
- 缓存键包含模型版本和 checksum。

**Step 4: 运行并确认通过**

~~~bash
python3 -m pytest tests/modules/test_model_registry.py tests/modules/test_ai_training.py -q
~~~

Expected: PASS。

**Step 5: 提交**

~~~bash
git add modules/model_registry.py modules/ai_training.py tests/modules/test_model_registry.py
git commit -m "feat: register and promote weather models atomically"
~~~

### Task 11: 建立可测试的 DLR 编排并接入主页面

**Files:**
- Create: modules/dlr_pipeline.py
- Create: tests/integration/test_dlr_pipeline.py
- Modify: dispatch_app_st.py:405-604
- Modify: dispatch_app_st.py:653-768
- Modify: dispatch_app_st.py:792-925
- Modify: dispatch_app_st.py:1196-1275

**Step 1: 写失败集成测试**

~~~python
def test_pipeline_trains_missing_models_then_reuses_them(tmp_path):
    service = make_pipeline(model_root=tmp_path)
    first = service.run(
        physical=make_physical_weather(),
        truth=make_truth_weather(),
        ai_enabled=True,
        conductor=drake_conductor(),
    )
    second = service.run(
        physical=make_physical_weather(),
        truth=None,
        ai_enabled=True,
        conductor=drake_conductor(),
    )
    assert first.model_report.trained_targets
    assert second.model_report.loaded_targets == first.model_report.trained_targets


def test_pipeline_passes_selected_conductor_and_never_truth_to_dlr():
    spy = SpyThermalEngine()
    result = make_pipeline(thermal_engine=spy).run(
        physical=make_physical_weather(),
        truth=make_truth_weather(extreme_values=True),
        ai_enabled=False,
        conductor=jl_630_conductor(),
    )
    assert spy.last_conductor["D0"] == jl_630_conductor()["D0"]
    assert "truth" not in spy.last_weather_columns
    assert result.max_currents.shape[0] == 2
~~~

**Step 2: 运行并确认失败**

~~~bash
python3 -m pytest tests/integration/test_dlr_pipeline.py -q
~~~

Expected: FAIL，编排模块不存在。

**Step 3: 最小实现**

DlrPipeline 固定调用顺序：

~~~python
physical = canonicalize_and_resample(uploaded_physical)
terrain_corrected = correction_service.apply(physical, terrain, options)
model_report, final_weather = model_service.apply(
    terrain_corrected, truth=uploaded_truth, enabled=ai_enabled
)
dlr_result = line_analyzer.calculate_from_long_frame(
    final_weather, base_params=selected_conductor
)
~~~

dispatch_app_st.py 只做薄层：

- 在现有侧边栏增加 accept_multiple_files=True 的“上传真实气象数据”入口；
- 侧栏文件先通过 freeze_uploaded_files 读取为 bytes/hash，再分别写入 physical_weather_snapshot 和 truth_weather_snapshot；UploadedFile 对象不得进入 session_state；
- 现有“处理数据 & 计算”按钮触发 pipeline；
- 显式传 st.session_state.conductor_params；
- 不再向热核传 terrain_data；
- 暂态计算失败时由编排层使用同一导线、同一时刻的稳态额定值保守回退；Task 8 热核只显式报错，不提前吞异常；
- 标题中的 IEEE 738-2013 改为 IEEE 738-2023；
- 保留 line_data 兼容投影和现有图表形状；
- AI 区域删除随机残差和 DLR 自参照 MAE/RMSE，在同一图表容器展示温度/风速物理值、AI 修正值、真实值；保留三列指标容器并显示风速 MAE、温度 MAE、已启用模型数；
- 没有模型/真值时保持纯物理结果。

**Step 4: 运行针对性测试**

~~~bash
python3 -m pytest tests/integration/test_dlr_pipeline.py tests/modules/test_ai_training.py tests/modules/test_line_analyzer.py -q
~~~

Expected: PASS。

**Step 5: 运行页面烟测**

~~~bash
python3 -m streamlit run dispatch_app_st.py --server.headless true --server.port 8510
~~~

另一个终端运行：

~~~bash
curl --fail http://localhost:8510/_stcore/health
~~~

Expected: 输出 ok。停止服务器后继续。

**Step 6: 提交**

~~~bash
git add modules/dlr_pipeline.py tests/integration/test_dlr_pipeline.py dispatch_app_st.py
git commit -m "feat: run corrected weather models in the DLR pipeline"
~~~

> **Task 11 sealed 后端附加实施状态（2026-07-26）：** `docs/plans/2026-07-24-sealed-training-backend.md` 的 Task 1-5 已完成，生产持久化已收口到 sealed XGBoost，真实后端训练/复用验收通过。Task 6 的最终双阶段审查仍是进入 Task 12 前的强制门禁；本标记不表示 Task 12 已开始或完成。

### Task 12: 增加结构化审计和原子结果写入

**Files:**
- Create: utils/audit_log.py
- Create: tests/utils/test_audit_log.py
- Modify: modules/model_registry.py

**Step 1: 写失败测试**

~~~python
def test_audit_event_has_required_trace_fields(tmp_path):
    logger = JsonAuditLogger(tmp_path)
    logger.write(AuditEvent.example())
    payload = json.loads(next(tmp_path.glob("*.jsonl")).read_text().splitlines()[0])
    assert {
        "run_id", "result_id", "line_id", "tower_id", "stage",
        "input_hash", "config_hash", "source", "fallback_reason",
    } <= payload.keys()


def test_atomic_result_write_never_leaves_partial_target(tmp_path):
    target = write_result_atomic(tmp_path, "result-1", {"ok": True})
    assert json.loads(target.read_text()) == {"ok": True}
    assert list(tmp_path.glob("*.tmp")) == []
~~~

**Step 2: 运行并确认失败**

~~~bash
python3 -m pytest tests/utils/test_audit_log.py -q
~~~

Expected: FAIL，模块不存在。

**Step 3: 最小实现**

JsonAuditLogger 输出 JSONL，支持 datetime、NumPy 标量和 Enum 序列化。write_result_atomic 使用 tempfile.NamedTemporaryFile(dir=target.parent) 和 os.replace。日志失败返回 audit_persisted=False，但不能改变计算结果。禁止持久化原始上传 bytes。

将模型训练、加载、晋级、失效、裁剪和回退事件接到同一日志接口。

**Step 4: 运行并确认通过**

~~~bash
python3 -m pytest tests/utils/test_audit_log.py tests/modules/test_model_registry.py -q
~~~

Expected: PASS。

**Step 5: 提交**

~~~bash
git add utils/audit_log.py tests/utils/test_audit_log.py modules/model_registry.py
git commit -m "feat: add structured DLR audit logging"
~~~

### Task 13: 实现倾角输入、不可变快照和参数来源

**Files:**
- Create: modules/sag_validation.py
- Create: tests/fixtures/sag_data.py
- Create: tests/modules/test_sag_validation.py

**Step 1: 写失败测试**

~~~python
def test_angle_only_input_uses_selected_tower_and_snapshot_times():
    result = normalize_inclination_dataframe(
        pd.DataFrame({"倾角": [1.0, 1.1]}),
        selected_tower_id="001",
        snapshot=make_sag_snapshot(times=2),
    )
    assert result["tower_id"].tolist() == ["001", "001"]
    assert result["timestamp"].notna().all()


def test_parameter_priority_is_measured_then_derived_then_default():
    params = resolve_sag_parameters(
        inclination_row=make_inclination_row(),
        snapshot=make_sag_snapshot_with_adjacent_towers(),
        conductor=drake_conductor(),
    )
    assert params.span_m.source == ParameterSource.DERIVED
    assert params.area_m2.source == ParameterSource.MEASURED
    assert set(value.source.value for value in params.values()) <= {
        "measured", "derived", "default"
    }


def test_snapshot_does_not_alias_main_arrays():
    line_data = make_line_data()
    snapshot = build_sag_snapshot(line_data, make_conductor_params())
    line_data["max_currents"][0, 0] = -1
    assert snapshot.original_currents[0][0] >= 0
~~~

**Step 2: 运行并确认失败**

~~~bash
python3 -m pytest tests/modules/test_sag_validation.py -q
~~~

Expected: FAIL，模块不存在。

**Step 3: 最小实现**

在同一模块定义 frozen InclinationRecord、SagValidationSnapshot、ResolvedSagParameters、SourcedValue 和 ParameterSource。倾角列必需，塔号/时间可选；没有塔号使用页面选定塔，没有时间时优先映射等长快照时间，否则生成顺序索引。

参数顺序为 measured → derived → default。档距只由线路顺序中相邻杆塔坐标推导；单位荷载由质量乘重力；E/A/alpha 从导线目录读取。

重要边界：现有 calc_results["local_temps"] 是环境温度 T_a，绝不能作为 T_theor 或 T1。只有存在实际运行电流并调用 ThermalCalculator.calculate_steady_state_temperature 时才能得到 T_theor；否则使用参考温度并记录 default。I_recalc 通过可选热模型 callable 注入，弧垂模块不复制热模型。

**Step 4: 运行并确认通过**

~~~bash
python3 -m pytest tests/modules/test_sag_validation.py -q
~~~

Expected: PASS。

**Step 5: 提交**

~~~bash
git add modules/sag_validation.py tests/fixtures/sag_data.py tests/modules/test_sag_validation.py
git commit -m "feat: add sag validation contracts and parameter sources"
~~~

### Task 14: 实现专利公式、自适应阈值和逐塔状态机

**Files:**
- Modify: modules/sag_validation.py
- Modify: tests/modules/test_sag_validation.py

**Step 1: 写失败测试**

~~~python
def test_patent_tension_and_temperature_formula():
    h = horizontal_tension(weight_n_m=10, span_m=100, angle_deg=45)
    tm = infer_mean_temperature(
        current_tension_n=h,
        reference_tension_n=1000,
        elastic_modulus_pa=1e11,
        area_m2=1e-4,
        thermal_expansion_per_c=1e-5,
        reference_temp_c=20,
    )
    assert h == pytest.approx(500)
    assert tm == pytest.approx(25)


def test_checked_current_is_minimum_of_three_candidates():
    result = compute_derating(
        ambient_temp_c=20,
        theoretical_temp_c=60,
        measured_temp_c=80,
        original_current_a=1000,
        recalculated_current_a=900,
    )
    assert result.factor == pytest.approx((40 / 60) ** 0.5)
    assert result.checked_current_a == pytest.approx(min(1000, 900, 1000 * (40 / 60) ** 0.5))


def test_invalid_sample_does_not_advance_recovery_state():
    service = SagValidationService(config=test_state_config())
    results = service.validate_batch(make_risk_invalid_recovery_sequence())
    assert results[1].state == SagState.INVALID
    assert results[2].state != SagState.NORMAL
~~~

**Step 2: 运行并确认失败**

~~~bash
python3 -m pytest tests/modules/test_sag_validation.py -q
~~~

Expected: FAIL，公式和状态机尚未实现。

**Step 3: 最小实现**

实现纯函数：

~~~python
H = w * L / (2 * tan(theta))
Tm = T1 + (H1 - H) / (E * A * alpha)
delta_error = Tm - T_theor
k = sqrt(clamp((T_theor - Ta) / (Tm - Ta), 0.0, 1.0))
I_checked = min(I_orig, I_recalc, k * I_orig)
~~~

只有 Tm>Ttheor 且 delta_error 严格超过自适应阈值时触发。阈值随倾角 MAD、风速 MAD、档距和历史偏差单调增加。按 tower_id 独立维护 NORMAL/RISK/RECOVERY/INVALID；风险立即降额，恢复连续满足低阈值后按指数平滑慢升，异常点保持前一有效输出且不推进恢复计数。

所有角度、分母、平方根、温度和电流结果必须有限；无效样本产生 error_code/fallback_reason，不能中断同批其他记录。

**Step 4: 运行并确认通过**

~~~bash
python3 -m pytest tests/modules/test_sag_validation.py -q
~~~

Expected: PASS。

**Step 5: 提交**

~~~bash
git add modules/sag_validation.py tests/modules/test_sag_validation.py
git commit -m "feat: implement patented sag validation state machine"
~~~

### Task 15: 发布只读快照并增加独立弧垂页面

**Files:**
- Modify: modules/sag_validation.py
- Modify: dispatch_app_st.py:619-648
- Modify: dispatch_app_st.py:883-888
- Create: pages/弧垂后验证.py
- Create: tests/integration/test_sag_snapshot_bridge.py
- Create: tests/pages/test_sag_validation_page.py

**Step 1: 写失败测试**

~~~python
def test_completed_dlr_publishes_deep_copied_sag_snapshot():
    line_data = make_line_data()
    currents_before = line_data["max_currents"].copy()
    snapshot = build_sag_snapshot(line_data, make_conductor_params())
    run_sag_validation(snapshot, make_angle_only_frame())
    np.testing.assert_array_equal(line_data["max_currents"], currents_before)
    assert snapshot.source_run_id


def test_page_smoke_runs_without_main_snapshot():
    app = AppTest.from_file("pages/弧垂后验证.py")
    app.run(timeout=20)
    assert not app.exception
    assert len(app.file_uploader) == 1


def test_visible_projection_hides_parameter_sources():
    visible = build_visible_sag_result(make_backend_result())
    assert "parameter_sources" not in visible.columns
    assert "default" not in " ".join(map(str, visible.columns)).lower()
~~~

**Step 2: 运行并确认失败**

~~~bash
python3 -m pytest tests/integration/test_sag_snapshot_bridge.py tests/pages/test_sag_validation_page.py -q
~~~

Expected: FAIL，快照桥接和页面不存在。

**Step 3: 最小实现**

主页面只在完整 DLR 计算成功后发布 st.session_state["sag_validation_snapshot"]；失败或部分计算不能覆盖上一次完整快照。快照复制最终天气、塔序、坐标、导线参数、原始额定值和 run_id，不保存 line_data 引用。

pages/弧垂后验证.py 只做薄层：

- 一个 CSV/XLSX 倾角上传器；
- 无塔号时使用可选塔位选择；
- 调用 SagValidationService；
- 显示状态、反推温度、理论温度、误差和校验电流；
- 提供结果下载；
- 页面可见列白名单不含 parameter_sources/source/default/assumption；
- 后台 payload 保留参数来源、公式版本、input_hash、result_id；
- 页面只写 sag_ 前缀 session key。

使用 write_result_atomic 保存 runtime/results/sag/<result_id>.json，并写结构化审计事件。日志失败不影响页面数值。

**Step 4: 运行并确认通过**

~~~bash
python3 -m pytest tests/integration/test_sag_snapshot_bridge.py tests/pages/test_sag_validation_page.py tests/modules/test_sag_validation.py tests/utils/test_audit_log.py -q
~~~

Expected: PASS。

**Step 5: 提交**

~~~bash
git add modules/sag_validation.py dispatch_app_st.py pages/弧垂后验证.py tests/integration/test_sag_snapshot_bridge.py tests/pages/test_sag_validation_page.py
git commit -m "feat: add isolated sag post-validation page"
~~~

### Task 16: 端到端回归、文档和界面验收

**Files:**
- Create: tests/integration/test_weather_ai_dlr_e2e.py
- Create: tests/integration/test_sag_post_validation_e2e.py
- Modify: docs/TESTING.md
- Modify: README.md

**Step 1: 写端到端测试**

~~~python
def test_weather_truth_training_to_dlr_is_repeatable(tmp_path):
    service = make_real_pipeline(tmp_path)
    first = service.run(make_physical(), make_truth())
    second = service.run(make_physical(), truth=None)
    np.testing.assert_allclose(first.max_currents, second.max_currents)
    assert first.input_hash == second.input_hash


def test_sag_end_to_end_isolated_and_audited(tmp_path):
    line_data = make_line_data()
    before = copy.deepcopy(line_data)
    result = run_sag_e2e(
        snapshot=build_sag_snapshot(line_data, make_conductor_params()),
        inclination_bytes=angle_only_csv_bytes(),
        output_dir=tmp_path,
    )
    assert all(row.checked_current_a <= row.original_current_a for row in result.rows)
    assert line_data_deep_equal(line_data, before)
    assert result.result_path.exists()
    assert result.audit_events
~~~

**Step 2: 运行全部自动化测试**

~~~bash
python3 -m pytest -q
~~~

Expected: 全部 PASS，不得只运行新增测试。

**Step 3: 更新文档**

README.md 和 docs/TESTING.md 必须反映真实状态：

- 删除“9/9”和“XGBoost 已完成”等过期声明；
- 记录真实气象字段、模型目录和回退规则；
- 记录 IEEE Drake 回归和暂态测试命令；
- 记录弧垂倾角文件格式、独立性和后台来源日志；
- 不把默认机械参数宣传为实测值。

**Step 4: 启动页面并做浏览器验收**

~~~bash
python3 -m streamlit run dispatch_app_st.py --server.headless true --server.port 8510
~~~

使用浏览器检查桌面和移动视口：

- 主页面原有布局、控件和图表无重叠；
- 侧边栏新增真实气象上传入口；
- AI 区域无随机结果，指标单位为 °C 和 m/s；
- 弧垂后验证页只要求上传倾角；
- 页面结果不显示默认值来源标签；
- 上传真实样例后模型目录产生可复用 bundle；
- 浏览器控制台和 Streamlit 终端无异常。

**Step 5: 最终验证**

~~~bash
python3 -m pytest -q
python3 -m pytest tests/modules/test_thermal_engine_ieee738.py tests/modules/test_thermal_transient.py -q
python3 -m pytest tests/integration -q
git diff --check
~~~

Expected: 所有命令退出码为 0。

**Step 6: 提交**

~~~bash
git add tests/integration/test_weather_ai_dlr_e2e.py tests/integration/test_sag_post_validation_e2e.py README.md docs/TESTING.md
git commit -m "test: verify corrected DLR and sag workflows end to end"
~~~

## 完成定义

- IEEE 738 Drake 稳态约 1025 A，暂态官方回归通过；
- 地形、高度、沙漠和风向影响不会重复应用；
- 主 DLR 显式使用页面选定导线参数；
- 每塔风速/温度模型能自动训练、保存、加载和逐塔回退；
- AI 指标只比较修正气象与真实气象；
- 不同线路同塔号不会共享模型；
- 小样本可全量训练，但不能用训练误差替换已有冠军；
- 弧垂页只需倾角即可运行，参数来源只在后台记录；
- 弧垂运行前后主 DLR 数据深度相等；
- 原有页面布局不变，全部自动化和浏览器验收通过。
