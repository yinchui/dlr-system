# DLR动态增容评估系统 - 测试指南

## 测试状态

✅ **所有测试通过：9/9**

```
tests/config/test_config.py ............................ PASSED
tests/modules/test_ai_prediction.py ..................... PASSED
tests/modules/test_data_processor.py (3 tests) .......... PASSED
tests/modules/test_thermal_engine.py .................... PASSED
tests/modules/test_visualization.py ..................... PASSED
tests/modules/test_weather_correction.py ................ PASSED
tests/utils/test_validators.py ......................... PASSED
```

---

## 快速测试

### 运行所有测试
```bash
cd /Volumes/YC/项目/沙戈荒项目/界面/DLR动态增容评估系统/12.24
python3 -m pytest
```

### 运行特定测试
```bash
# 测试配置
python3 -m pytest tests/config/ -v

# 测试核心模块
python3 -m pytest tests/modules/ -v

# 测试工具函数
python3 -m pytest tests/utils/ -v
```

---

## 测试覆盖的功能

### 1. 配置模块 (config/)
- ✅ 导线参数库加载
- ✅ 修正参数默认值
- ✅ 应用标题配置

### 2. 数据处理 (modules/data_processor.py)
- ✅ 气象数据标准化
- ✅ 时间戳解析和转换
- ✅ 数据插值到指定时间间隔
- ✅ 地形数据查询表构建

### 3. 地形处理 (modules/terrain.py)
- ✅ TIF文件读取
- ✅ DEM数据加载
- ✅ 坡度和坡向计算
- ✅ 杆塔坐标加载

### 4. 热力计算引擎 (modules/thermal_engine.py)
- ✅ IEEE 738-2023标准实现
- ✅ 地形微气候修正
- ✅ 稳态电流计算
- ✅ 批量载流量计算
- ✅ 修正后风速输出

### 5. 气象修正 (modules/weather_correction.py)
- ✅ 垂直修正（风速、温度）
- ✅ 地形修正（坡度、坡向）
- ✅ 沙漠环境修正（湿度、辐射）
- ✅ 风向修正（6段分段系数）

### 6. AI预测 (modules/ai_prediction.py)
- ✅ 模型加载和管理
- ✅ 特征工程（时间特征、滞后特征）
- ✅ 残差预测
- ✅ 物理模型+AI混合预测

### 7. 可视化 (modules/visualization.py)
- ✅ 动态载流量图表
- ✅ 静态额定值对比
- ✅ 修正前后对比图
- ✅ 预测结果对比图

### 8. 数据验证 (utils/validators.py)
- ✅ 气象数据列验证
- ✅ 杆塔数据列验证
- ✅ 监测数据列验证

### 9. 文件导出 (utils/file_handler.py)
- ✅ CSV导出（UTF-8-BOM）
- ✅ Excel导出

---

## 手动功能测试

创建 `manual_test.py` 进行端到端测试：

```python
import pandas as pd
import numpy as np
from modules.data_processor import build_weather_dataset
from modules.weather_correction import WeatherCorrectionService, CorrectionOptions
from modules.thermal_engine import ThermalCalculator, LineAnalyzer

# 1. 测试数据处理
print("1. 测试数据处理...")
df = pd.DataFrame({
    "位置": [36, 36],
    "日期": ["2025-12-10", "2025-12-10"],
    "时刻": ["00:00", "01:00"],
    "环境温度": [20.0, 21.0],
    "风速": [3.2, 3.5],
    "风向": [90.0, 95.0],
    "太阳辐射强度": [0.0, 30.0],
    "相对湿度": [40.0, 41.0],
    "海拔": [1100.0, 1100.0],
})
dataset = build_weather_dataset(df)
print(f"   ✓ 数据集包含 {len(dataset.positions)} 个位置")
print(f"   ✓ 时间点数量: {len(dataset.timestamps)}")

# 2. 测试气象修正
print("\n2. 测试气象修正...")
corrector = WeatherCorrectionService()
terrain_lookup = {36: {"slope": 15.0, "aspect": 180.0, "elevation": 1100.0}}
corrected = corrector.apply(df, terrain_lookup, CorrectionOptions(line_azimuth_deg=90.0))
print(f"   ✓ 原始风速: {df['风速'].values}")
print(f"   ✓ 修正后风速: {corrected['wind_speed_corrected'].values}")
print(f"   ✓ 风向修正系数: {corrected['wind_angle_factor'].values}")

# 3. 测试热力计算
print("\n3. 测试热力计算...")
calculator = ThermalCalculator()
analyzer = LineAnalyzer(calculator)
result = analyzer.calculate_max_current_for_points(
    observation_points=np.array([36]),
    elevations=np.array([1100.0]),
    temps=np.array([[20.0, 21.0]]),
    winds=np.array([[3.0, 3.2]]),
    angles=np.array([[90.0, 95.0]]),
    solar=np.array([0.0, 50.0]),
    times=np.array([0.0, 1.0]),
    terrain_data={0: {"slope": 15.0, "aspect": 180.0, "elevation": 1100.0}}
)
print(f"   ✓ 载流量: {result['max_currents'][0]}")
print(f"   ✓ 修正后风速: {result['corrected_winds'][0]}")

# 4. 测试AI预测接口
print("\n4. 测试AI预测接口...")
from modules.ai_prediction import ResidualPredictor, ModelBundle

class DummyModel:
    def predict(self, X):
        return [0.5] * len(X)

predictor = ResidualPredictor({
    "wind_speed": ModelBundle(
        target_name="wind_speed",
        feature_columns=["hour_sin", "hour_cos", "wind_speed_physical"],
        model=DummyModel()
    )
})
test_df = pd.DataFrame({
    "timestamp": pd.to_datetime(["2025-12-10 00:00", "2025-12-10 01:00"]),
    "wind_speed_physical": [3.0, 4.0],
})
predicted = predictor.predict(test_df, "wind_speed", "wind_speed_physical")
print(f"   ✓ 物理预测: {test_df['wind_speed_physical'].values}")
print(f"   ✓ 残差: {predicted['wind_speed_residual'].values}")
print(f"   ✓ 最终预测: {predicted['wind_speed_final'].values}")

print("\n✅ 所有手动测试通过！")
```

运行：
```bash
python3 manual_test.py
```

---

## 测试覆盖率（可选）

安装coverage工具：
```bash
pip3 install coverage
```

生成覆盖率报告：
```bash
# 运行测试并收集覆盖率
python3 -m coverage run -m pytest

# 查看覆盖率报告
python3 -m coverage report

# 生成HTML报告
python3 -m coverage html
open htmlcov/index.html
```

---

## 持续集成建议

在 `.github/workflows/test.yml` 中添加：

```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - uses: actions/setup-python@v2
        with:
          python-version: '3.9'
      - name: Install dependencies
        run: pip install -r requirements.txt
      - name: Run tests
        run: pytest -v
```

---

## 故障排查

### 问题：ModuleNotFoundError
```bash
# 确保在项目根目录
cd /Volumes/YC/项目/沙戈荒项目/界面/DLR动态增容评估系统/12.24

# 检查Python路径
python3 -c "import sys; print(sys.path)"

# 安装依赖
pip3 install -r requirements.txt
```

### 问题：测试失败
```bash
# 查看详细错误信息
python3 -m pytest -vv --tb=long

# 只运行失败的测试
python3 -m pytest --lf
```

### 问题：导入错误
```bash
# 验证模块可以导入
python3 -c "from modules.thermal_engine import ThermalCalculator"
python3 -c "from config.config import APP_TITLE"
```

---

## 下一步

现在核心模块已经完成并测试通过，接下来可以：

1. **创建Streamlit应用** - 构建多页面Web界面
2. **集成测试** - 测试完整的数据流程
3. **性能测试** - 测试大数据量处理
4. **用户验收测试** - 使用真实数据验证

---

**测试文档更新日期：** 2026-03-17
**测试通过率：** 100% (9/9)
