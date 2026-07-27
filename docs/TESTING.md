# DLR 动态增容评估系统测试指南

测试必须从仓库根目录运行。推荐 Python 3.11，并先安装 `requirements.txt`。

## 完整质量门禁

```bash
python3 -m pytest -q
python3 -m ruff check --target-version py311 .
python3 -m py_compile modules/*.py utils/*.py config/*.py dispatch_app_st.py thermal_functions.py pages/弧垂后验证.py
git diff --check
```

不要在最终验收时只运行新增测试；测试总数以当前 `pytest` 输出为准，不在文档中维护易失效的固定计数。

## IEEE 738 回归

稳态 Drake 金标准、低风速自然对流、风向归一、太阳热增益、电阻插值和输入不变性：

```bash
python3 -m pytest tests/modules/test_thermal_engine_ieee738.py -q
```

关键稳态参考值约为：

- 自然对流 `42.42 W/m`；
- 低/高雷诺数强制对流 `82.10/77.06 W/m`；
- 辐射 `39.11 W/m`；
- 太阳热增益 `22.45 W/m`；
- 额定电流 `1025 A`。

暂态 10 秒子步、125°C/15 分钟额定值、材料热容量和非法输入：

```bash
python3 -m pytest tests/modules/test_thermal_transient.py -q
```

## 气象、地形与 DLR 编排

```bash
python3 -m pytest \
  tests/modules/test_data_processor.py \
  tests/modules/test_weather_upload.py \
  tests/modules/test_weather_correction.py \
  tests/modules/test_terrain.py \
  tests/modules/test_line_analyzer.py \
  tests/integration/test_dlr_pipeline.py -q
```

这些测试验证：

- 新旧气象文件格式转换、时区、杆塔编号、物理边界和重复键；
- 上传字节只冻结一次，持久化内容只有 SHA-256 和规范化数据；
- DEM 坐标/CRS 查询及缺失覆盖，地形修正只执行一次；
- 风向只作为导线夹角，不乘入风速；
- 真实气象不会进入热核；
- 页面选择的导线参数显式传入 DLR；
- IEEE 738 标准回归保留未折减原值，稳态和成功暂态发布值统一乘 `0.8`；
- 暂态失败使用同一导线、同一气象时刻且已折减的稳态结果保守回退，不会折减两次；
- `safety_factor=0.8` 只作为后台发布元数据，不参与 `input_hash` 的热核输入语义；
- `DlrPipelineResult` 是不可变快照，数组视图只读。

## Sealed XGBoost 生命周期

生产模型只接受运行时合同验证通过的 sealed `xgboost.XGBRegressor`。模型按项目、线路、杆塔和目标隔离；模型文件、元数据、manifest、拒绝记录和 generation 都受原子写入及文件锁保护。

真实后端训练、保存和下次运行复用：

```bash
python3 -m pytest tests/integration/test_sealed_xgboost_lifecycle.py -q
python3 -m pytest tests/integration/test_weather_ai_dlr_e2e.py -q
```

训练、注册与攻击边界：

```bash
python3 -m pytest \
  tests/modules/test_ai_training.py \
  tests/modules/test_ai_prediction.py \
  tests/modules/test_model_registry.py -q
```

重点合同：

- 特征滞后不得跨杆塔；指标只比较修正气象与真实气象；
- 小样本全量训练可保存首个 provisional，但不能替换 champion；
- 时间留出候选必须在同一独立评价集上改善；
- XGBoost 类型、冻结参数、随机种子、依赖版本和实现哈希必须一致；
- 路径穿越、符号链接、非私有权限、checksum 或 manifest 不一致时失败关闭；
- 单塔模型损坏、不兼容、裁剪或预测失败只回退对应目标；
- 相同物理输入在首次训练和后续模型加载时产生相同 DLR 和 `input_hash`。
- `input_hash` 只表示有效热核数值路径；每塔非有效海拔、运行上下文和未使用导线字段不改变哈希，暂态窗口按实际气象范围裁剪并规范化负零。
- 暂态失败回退稳态后，哈希与未请求暂态的同一稳态路径一致；失败状态由 `transient_fallbacks` 和审计信息区分。

## 审计与弧垂后验证

```bash
python3 -m pytest \
  tests/utils/test_audit_log.py \
  tests/modules/test_sag_validation.py \
  tests/integration/test_sag_snapshot_bridge.py \
  tests/pages/test_sag_validation_page.py \
  tests/integration/test_sag_post_validation_e2e.py -q
```

覆盖范围：

- JSONL 并发写入、原子结果保存、bytes 拒绝及失败清理；
- 倾角 CSV/XLSX、空表、无效角度、时间匹配和杆塔选择；
- `H = wL/(2 tan θ)`、`Tm = T1 + (H1-H)/(EAα)` 和三候选最小电流；
- 自适应阈值以及 NORMAL/RISK/RECOVERY/INVALID 逐塔状态机；
- 无效点不推进恢复计数，风险快降、恢复慢升；
- 快照深冻结，失败发布不覆盖旧快照，且弧垂快照只接收已乘 `0.8` 的发布额定值；
- 可见结果不含参数来源、default 或 assumption 字段；
- 上传解析、规范化、行级异常和落盘失败均有审计，审计失败不改变数值；
- 后验证执行前后主 DLR 数据深度相等；
- 页面切换文件、杆塔或 DLR 快照时清除旧结果。

倾角最小样例：

```csv
倾角
1.00
1.08
```

可选字段为 `杆塔`/`tower_id` 和 `时间`/`timestamp`。机械默认参数只用于后台保守计算，不得在测试或界面中当作实测值。

## 浏览器验收

启动服务：

```bash
python3 -m streamlit run dispatch_app_st.py --server.headless true --server.port 8510
```

先确认健康检查：

```bash
curl --fail http://localhost:8510/_stcore/health
```

桌面和移动视口都要检查：

- 主页面原有布局、侧边栏、标签页和图表无重叠或截断；
- 全线摘要仅有最低、最高和平均三项动态额定值，图中没有静态基准、对比百分比或增容空间填充；
- 侧边栏“AI预测配置”中存在真实气象上传入口；
- AI 图表没有随机残差，指标单位为 `m/s` 和 `°C`；
- 确保气象文件包含每塔完整且无歧义的经纬度，或主流程已有完整杆塔坐标；上传真实气象后 `models/` 产生按线路/杆塔/目标隔离的 bundle，下一次无真实气象时可加载；
- 弧垂页面无主快照时正常启动；有快照时只需上传倾角即可运行；
- 弧垂结果不显示默认参数来源标签，下载按钮在普通重跑后仍存在；
- 浏览器控制台和 Streamlit 终端没有异常。

移动视口建议至少覆盖 `390 x 844`，桌面视口建议至少覆盖 `1440 x 900`。

## 运行产物

默认目录：

```text
models/                         XGBoost 模型、元数据和 manifest
runtime/logs/dlr-audit.jsonl   模型与弧垂结构化审计
runtime/results/sag/           弧垂后台 JSON 结果
```

自动化测试必须使用 `tmp_path` 或环境变量覆盖这些目录，不应在仓库中留下模型、日志、结果或原始上传文件。
