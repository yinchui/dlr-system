# DLR动态增容评估系统

基于IEEE 738-2023标准的输电线路动态载流量评估系统，集成气象参数修正和AI预测功能。

## 项目状态

✅ **核心模块开发完成** - 6/6 任务完成
✅ **所有测试通过** - 9/9 测试通过
🚧 **Web界面开发中** - 准备创建Streamlit多页面应用

---

## 功能特性

### 1. 热力计算引擎
- ✅ IEEE 738-2023标准实现
- ✅ 稳态/暂态温度计算
- ✅ 对流、辐射、太阳热增益计算
- ✅ 沙漠环境优化（湿度、反照率修正）

### 2. 气象参数修正
- ✅ **垂直修正** - 风速对数律、温度递减率
- ✅ **地形修正** - 坡度、坡向、迎风/背风效应
- ✅ **沙漠环境修正** - 湿度、太阳辐射影响
- ✅ **风向修正** - 6段分段修正系数

### 3. AI智能预测
- ✅ XGBoost残差预测模型
- ✅ 物理模型 + AI混合预测
- ✅ 特征工程（时间特征、滞后特征）
- ✅ 模型加载和管理

### 4. 数据处理
- ✅ 气象数据标准化和插值
- ✅ DEM地形数据处理（SRTM）
- ✅ 杆塔坐标空间匹配
- ✅ 数据验证和质量控制

### 5. 可视化
- ✅ 动态载流量时间序列图
- ✅ 修正前后对比图
- ✅ AI预测结果对比图
- ✅ 交互式Plotly图表

---

## 快速开始

### 安装依赖

```bash
cd /Volumes/YC/项目/沙戈荒项目/界面/DLR动态增容评估系统/12.24
pip3 install -r requirements.txt
```

### 运行测试

```bash
# 运行所有测试
python3 -m pytest

# 查看详细信息
python3 -m pytest -v
```

### 使用示例

```python
import pandas as pd
import numpy as np
from modules.thermal_engine import ThermalCalculator, LineAnalyzer
from modules.weather_correction import WeatherCorrectionService, CorrectionOptions

# 1. 创建计算器
calculator = ThermalCalculator()
analyzer = LineAnalyzer(calculator)

# 2. 准备气象数据
temps = np.array([[20.0, 21.0, 22.0]])  # 环境温度
winds = np.array([[3.0, 3.2, 3.5]])     # 风速
angles = np.array([[90.0, 95.0, 100.0]]) # 风向
solar = np.array([0.0, 50.0, 100.0])    # 太阳辐射

# 3. 计算载流量
result = analyzer.calculate_max_current_for_points(
    observation_points=np.array([36]),
    elevations=np.array([1100.0]),
    temps=temps,
    winds=winds,
    angles=angles,
    solar=solar,
    times=np.array([0.0, 1.0, 2.0]),
    max_temp=80.0,
    terrain_data={0: {"slope": 15.0, "aspect": 180.0, "elevation": 1100.0}}
)

print(f"载流量: {result['max_currents']}")
print(f"修正后风速: {result['corrected_winds']}")
```

---

## 项目结构

```
DLR动态增容评估系统/12.24/
├── config/                      # 配置模块
│   ├── __init__.py
│   └── config.py               # 全局配置参数
├── modules/                     # 核心计算模块
│   ├── __init__.py
│   ├── data_processor.py       # 数据处理
│   ├── terrain.py              # 地形处理
│   ├── thermal_engine.py       # 热力计算引擎
│   ├── weather_correction.py   # 气象修正
│   ├── ai_prediction.py        # AI预测
│   └── visualization.py        # 可视化
├── utils/                       # 工具函数
│   ├── file_handler.py         # 文件导出
│   └── validators.py           # 数据验证
├── tests/                       # 测试套件
│   ├── config/
│   ├── modules/
│   ├── utils/
│   └── fixtures/
├── docs/                        # 文档
│   ├── plans/                  # 设计文档
│   └── TESTING.md              # 测试指南
├── thermal_functions.py         # 向后兼容shim
├── requirements.txt
├── pytest.ini
└── README.md
```

---

## 技术栈

- **Python**: 3.9+
- **科学计算**: NumPy, SciPy, Pandas
- **机器学习**: XGBoost, scikit-learn
- **可视化**: Plotly
- **测试**: pytest
- **Web框架**: Streamlit (待集成)

---

## 开发进度

### ✅ 已完成 (Phase 1: 核心模块)

- [x] Task 1: 配置和测试框架
- [x] Task 2: 数据处理管道
- [x] Task 3: 热力计算引擎
- [x] Task 4: 气象修正模块
- [x] Task 5: AI预测接口
- [x] Task 6: 验证器和可视化

### 🚧 进行中 (Phase 2: Web应用)

- [ ] Task 7: Streamlit应用框架
- [ ] Task 8: DLR计算页面
- [ ] Task 9: 气象修正页面
- [ ] Task 10: AI预测页面
- [ ] Task 11: 帮助页面

### 📋 计划中 (Phase 3: 部署)

- [ ] Streamlit Cloud部署
- [ ] 性能优化
- [ ] 用户文档
- [ ] 示例数据

---

## 测试

详细测试指南请参考 [docs/TESTING.md](docs/TESTING.md)

**当前测试状态：**
- ✅ 配置模块: 1 test
- ✅ 数据处理: 3 tests
- ✅ 热力引擎: 1 test
- ✅ 气象修正: 1 test
- ✅ AI预测: 1 test
- ✅ 可视化: 1 test
- ✅ 验证器: 1 test

**总计: 9/9 tests passed**

---

## Git提交历史

```
0569d1e docs: add comprehensive testing guide
2007c1c feat: add validators export helpers and charts
ed2bf2e feat: add ai residual prediction interface
dfdabb2 feat: add weather correction pipeline
a818603 refactor: isolate thermal engine
b89834c feat: extract weather and terrain data pipeline
700dd21 chore: bootstrap config and test harness
```

---

## 贡献指南

### 开发流程

1. 创建功能分支
2. 编写测试（TDD）
3. 实现功能
4. 运行测试确保通过
5. 提交代码

### 代码规范

- 遵循PEP 8
- 使用类型注解
- 编写文档字符串
- 保持函数简洁（<50行）

---

## 许可证

本项目用于沙戈荒大型能源基地输变电动态增容关键技术研究。

---

## 联系方式

如有问题或建议，请通过以下方式联系：

- 项目路径: `/Volumes/YC/项目/沙戈荒项目/界面/DLR动态增容评估系统/12.24`
- 文档: `docs/`
- 测试: `python3 -m pytest`

---

**最后更新**: 2026-03-17
**版本**: 1.0.0-alpha (核心模块完成)
