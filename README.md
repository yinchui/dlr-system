# DLR 动态增容评估系统

本系统面向沙戈荒输电线路，按 IEEE 738-2023 计算稳态和暂态动态线路额定值（DLR）。地形、高度、沙漠辐射和风向修正在热模型上游只执行一次；可选的按塔 XGBoost 残差模型只修正气象，真实气象不进入 DLR 热平衡。

系统同时提供独立的“弧垂后验证”页面。该页面消费主 DLR 发布的只读快照，按专利公式反推温度并校验电流，不会回写或改变主 DLR 结果。

## 运行

建议使用 Python 3.11：

```bash
python3 -m pip install -r requirements.txt
python3 -m streamlit run dispatch_app_st.py
```

主页面保持原有 DLR 分析界面。完成一次完整 DLR 计算后，可从 Streamlit 页面导航进入“弧垂后验证”。

## 气象数据

物理气象和真实气象均支持多个 CSV/XLSX 文件，系统会转换为按 `tower_id + timestamp` 组织的规范化长表。支持以下两类表头：

| 格式 | 必需字段 | 可选字段 |
| --- | --- | --- |
| 旧格式 | `位置`、`日期`、`时刻`、`环境温度`、`风速`、`风向` | `太阳辐射强度`、`相对湿度`、`海拔` |
| 杆塔时序格式 | `时间`、`杆塔`、`风速WS(m/s)`、`风向WD(...)`、`温度TEM(...)` | `相对湿度RHU(...)`、`经度`、`纬度` |

关键规则：

- 项目时区固定为 `Asia/Shanghai`；杆塔编号会规范化，例如 `001号` 变为 `001`。
- 风速、温度、时间、重复键和非有限值会经过质量校验；无效行不会进入热模型。
- 杆塔时序格式未提供太阳辐射或海拔时，当前兼容层分别使用 `0 W/m²` 和 `1000 m`。需要工程精度时应上传实测值或加载 DEM。
- 真实气象使用同一字段格式，但必须是独立数据集；与物理气象内容相同或文件哈希相同时会拒绝训练。
- 真实气象只用于残差训练及 MAE/RMSE 评价。评价比较“修正气象”和“真实气象”，不使用 DLR 作为训练指标。

## 气象修正与 DLR

计算顺序固定为：规范化物理气象 -> 地形/项目修正 -> 可选 AI 残差修正 -> IEEE 738 热平衡。热核显式使用页面选择的导线参数，不会再次应用坡度、坡向、湿度或风沙乘数。

IEEE 738 内核包括：

- 自然对流与两条强制对流相关式取最大值；
- 导线轴向夹角归一化；
- 辐射、晴空或实测太阳热增益、电阻插值；
- 不超过 10 秒子步的暂态积分和保守稳态回退。

官方 Drake 稳态回归的额定电流约为 `1025 A`。这只是标准算例，不代表当前线路的固定额定值。

IEEE 738 稳态和暂态热核始终保留未折减的标准计算值。`DlrPipeline` 在后台发布边界对所有稳态和成功暂态 DLR 统一乘以固定安全系数 `0.8`；暂态失败时直接复用已折减的稳态结果，不会再次折减。该系数只是后台发布策略，页面不提供修改控件；页面、下载、瓶颈统计和弧垂后验证快照只使用折减后的额定值。主页仅展示动态额定值的最低、最高和平均统计，不计算或展示静态基准及其对比。

每次结果的 `input_hash` 只绑定规范化后的有效热核数值路径：最终气象、稳态实际导线参数，以及成功暂态实际采用的热容量、初温和裁剪后窗口。暂态请求失败并回退稳态时，哈希与同一稳态计算一致；失败原因另存于结果和审计信息。

## XGBoost 模型生命周期

启用 AI 后，模型按 `project/line/tower/target` 隔离，目标为 `wind_speed` 和 `ambient_temp`：

- 没有可用模型且上传真实气象时，系统先尝试训练；数据量少时直接全量拟合，不设置最小样本门槛。
- 首个全量拟合模型可作为 `provisional` 保存并在下次运行加载；全量拟合候选不能覆盖已有 champion。
- 能形成独立时间留出时，候选只有在相同评价集上改善后才可晋级。
- 缺失、损坏、不兼容、质量不达标或预测异常只影响对应杆塔和目标，其余模型继续使用；失败目标回退到物理气象。
- 跨运行持久化要求线路具有完整且无歧义的杆塔坐标。坐标不完整时，模型只在当前运行使用，避免不同线路误共享。

本地开发默认模型目录：

```text
models/<project_id>/<line_id>/<tower_id>/<target>/
```

可通过 `DLR_MODEL_DIR` 覆盖。当 `DLR_SUPABASE_URL` 和 `DLR_SUPABASE_SECRET_KEY` 都未配置时，系统使用本地 `ModelRegistry`，该模式仅用于本地开发；只配置其中一项会明确报配置错误。

生产部署使用 Streamlit 服务端 Secrets 连接 Supabase：

```toml
DLR_SUPABASE_URL = "https://<project-ref>.supabase.co"
DLR_SUPABASE_SECRET_KEY = "<server-side-secret-key>"
DLR_SUPABASE_MODEL_BUCKET = "dlr-models"
```

`DLR_SUPABASE_MODEL_BUCKET` 可省略，默认为私有 bucket `dlr-models`。其中以禁止 upsert 的不可变 generation 对象保存 `model.joblib`；PostgreSQL 的 `dlr_model_generations`、`dlr_model_heads` 和 `dlr_model_rejections` 分别保存 generation 元数据、当前激活 head 和拒绝指纹。凭据只供服务端使用，页面不提供配置控件。

生产持久化仅接受经过运行时合同校验的 sealed `xgboost.XGBRegressor`。远端现有模型下载、完整性校验或兼容性校验失败时，受影响的杆塔/目标回退到物理修正气象并继续计算 DLR。新候选模型上传或 CAS 激活失败时，未激活候选不会进入推理；如果已加载旧 champion 则继续使用旧 champion，否则回退到物理修正气象。一次 DLR 运行中的首个明确传输故障会停止该运行后续的模型查询、训练发布和远端写入，避免超时按杆塔数放大；下一次独立运行会重新探测 Supabase。依赖、导线、DEM、坐标或修正配置不兼容时同样拒绝加载。

## 弧垂后验证

倾角文件支持 CSV/XLSX，至少包含一行和一个倾角列：

```csv
倾角
1.00
1.08
```

也可提供 `杆塔`/`tower_id` 和 `时间`/`timestamp`。未提供杆塔时使用页面选择值；未提供时间且行数与 DLR 快照一致时使用快照时间，否则按顺序索引匹配。

后验证遵循以下隔离边界：

- 只读取完整 DLR 运行发布的深冻结快照；失败或部分 DLR 运行不会覆盖上一次完整快照。
- 结果只显示状态、反推温度、理论温度、温差和校验电流等白名单字段。
- 参数按“上传实测 -> 可推导 -> 后台默认”解析。默认机械参数是保守计算假设，不是实测值，也不会在结果表中标记为实测。
- 结果原子保存到 `runtime/results/sag/`，审计写入 `runtime/logs/dlr-audit.jsonl`；后台保留参数来源、公式版本、输入哈希和错误码，不保存原始上传字节。
- 后验证运行前后，主页面 `line_data`、额定电流矩阵和导线参数保持不变。

可通过 `DLR_SAG_RESULT_DIR` 和 `DLR_AUDIT_LOG_DIR` 覆盖运行目录。

## 测试

完整测试和关键回归命令：

```bash
python3 -m pytest -q
python3 -m pytest tests/modules/test_thermal_engine_ieee738.py tests/modules/test_thermal_transient.py -q
python3 -m pytest tests/integration/test_weather_ai_dlr_e2e.py tests/integration/test_sag_post_validation_e2e.py -q
python3 -m ruff check --target-version py311 .
```

更详细的验证矩阵和浏览器验收项见 [docs/TESTING.md](docs/TESTING.md)。

## 主要目录

```text
config/                 运行目录、物理边界、导线和机械默认参数
modules/                气象、地形、AI、IEEE 738、DLR 编排和弧垂后验证
pages/                  独立 Streamlit 页面
tests/                  单元、集成、页面和端到端测试
utils/audit_log.py      JSONL 审计和原子结果写入
runtime/                运行日志及弧垂结果（Git 忽略）
models/                 按线路/杆塔/目标隔离的模型（Git 忽略）
```

本项目用于沙戈荒大型能源基地输变电动态增容关键技术研究。
