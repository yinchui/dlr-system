# 封闭生产训练后端设计

## 1. 背景

Task 11 已经证明，依靠静态反射去完整描述任意 Python trainer、factory、闭包、模块全局和对象状态无法可靠收口。每补充一种可执行语义，仍可通过动态属性访问、别名拓扑、隐藏状态或第三方随机默认值构造相同合同、不同模型的反例。

本设计取消“任意训练代码都可以进入持久模型生命周期”的假设。生产系统只允许一个受支持、参数冻结且可直接验证的 XGBoost 后端进入模型注册、拒绝缓存和后续复用。任意自定义 trainer/factory 仍可用于纯单元训练或故障注入，但不得发布模型、写确定性拒绝 sidecar 或作为页面 AI 结果来源。

## 2. 目标与非目标

### 目标

- 同一个生产训练合同只对应一个确定性 estimator 类型、参数集、随机种子和依赖版本。
- XGBoost、Python、NumPy、Pandas 和 joblib 的相关版本变化会使旧模型和旧拒绝记录失效。
- estimator 的真实参数与元数据一致，不能声明 `random_seed=42` 而实际使用 `None`。
- XGBoost 缺失、构造失败、拟合失败或预测失败时继续输出物理 DLR，并允许环境恢复后重试。
- 小样本不设下限；能形成独立时间留出时使用留出评估，否则全量训练。
- 模型晋级仍只比较修正气象与真实气象的 MAE/RMSE，不使用 DLR 指标。

### 非目标

- 不提供任意 scikit-learn estimator 或用户自定义 Python 插件的持久化支持。
- 不尝试证明任意 Python callable、闭包或运行时全局状态的完整确定性。
- 不改变页面 UI、地形/沙漠修正、真实气象上传或弧垂后验证设计。

## 3. 方案选择

### 采用：封闭生产后端

生产模型仅允许：

1. `XGBRegressor`，使用系统声明的完整冻结参数，包括 `random_state=42` 和 `n_jobs=1`；
2. 数据本身为常量残差时使用内部 `ConstantResidualEstimator`，该路径属于确定性数据回退；
3. XGBoost 操作性失败时使用临时常量结果完成本次物理回退，但该结果不晋级、不写确定性拒绝缓存。

未采用的方案：

- 运行时 estimator 原型证明：仍需解释第三方实现、隐藏状态和动态依赖，复杂度高且存在漏判面。
- 继续扩展 callable 反射器：终审已用多类反例证明不可可靠收口。

## 4. 核心合同

新增冻结 `SealedEstimatorSpec`，只由系统代码构造，至少包含：

- `schema_version`；
- `backend_id`，固定为受支持的 XGBoost 后端标识；
- estimator 的导入路径；
- 完整、排序且严格 JSON 化的 estimator 参数；
- 真实随机种子；
- Python 和相关 distribution 的规范名称与版本；
- 后端实现文件或受控模块的 SHA-256 摘要；
- 特征版本和训练策略版本。

distribution 版本使用 `importlib.metadata.packages_distributions()` 将 import 根名映射到真实 distribution 名；映射缺失、歧义或版本不可读时生产合同构造失败，按操作性故障处理。

`ResidualTrainer` 默认创建封闭 XGBoost spec。构造出的每个实际 estimator 在拟合前必须通过 attestation：

- 类型与 spec 完全一致；
- `get_params(deep=False)` 与 spec 参数完全一致；
- `random_state` 等随机参数存在且与 spec 一致；
- distribution 和实现摘要仍与 spec 一致。

任一不一致都产生 `TrainingContractError`，且不得写确定性拒绝记录。

## 5. 自定义注入边界

`DlrPipeline(trainer=...)` 保留为兼容和测试入口，但只有系统默认 `ResidualTrainer` 的封闭 spec 可以训练并发布生产模型。

自定义 trainer、`ResidualTrainer(estimator_factory=...)` 或其他非封闭后端：

- 可以在模块单元测试中直接训练，验证数据处理和回退逻辑；
- 在 DLR pipeline 中不允许发布模型、不允许写拒绝 sidecar、不允许启用 AI 气象；
- 对每个塔/目标记录 `unsupported_training_backend` 操作性回退；
- 物理气象和 DLR 继续运行。

测试不再通过自定义 trainer 验证持久模型生命周期。模型保存、加载、晋级和拒绝缓存测试使用真实封闭 XGBoost 后端；调用次数通过注册表文件、报告和稳定输入变化验证，不通过替换训练实现验证。

## 6. 模型加载与失效

当前运行在加载模型前为每个 target/cadence 派生预期的封闭训练合同哈希。`ModelRegistry.load()` 除现有 DEM、CRS、坐标、导线、特征和修正配置外，还核对：

- `training_contract_hash`；
- estimator backend id；
- 必要 dependency versions。

不匹配时只使对应塔/目标模型失效，记录明确原因；有真实值时重新训练，没有真实值时回退物理气象。旧拒绝 sidecar 的 fingerprint 已包含训练合同，因此后端或依赖变化会自然允许重试。

旧 metadata 缺少新 sealed backend 字段时按 legacy 模型读取，但不能作为当前生产合同兼容模型继续使用；它会以合同不兼容失效，而不是损坏其他模型。

## 7. 数据流

1. pipeline 规范化物理气象和真实气象。
2. 启用 AI 时创建当前 cadence 的默认 `ResidualTrainer` 和封闭 spec。
3. registry 使用当前 spec 的 scoped contract 加载每塔/目标模型。
4. 缺失或失效且有真实值时，准备训练数据并构建 attempt。
5. 在查拒绝缓存前合同已经固定；训练中每个 estimator 创建后再次 attestation。
6. 只有 `sealed_trained` 或确定性 `data_fallback` 候选可以进入质量准入。
7. 操作性失败仅记录回退，不写确定性 sidecar。
8. 最终气象进入 IEEE 738 DLR；真实气象永不进入热计算输入。

## 8. 错误处理

- XGBoost distribution 缺失或版本不可读：`xgboost_unavailable`，物理回退，可重试。
- estimator 参数、类型或随机种子不匹配：`training_contract_mismatch`，物理回退，可重试。
- 非封闭 trainer/factory：`unsupported_training_backend`，物理回退，不持久化。
- 单塔/单目标失败不影响其他塔、其他目标和主 DLR。
- 正常气象质量拒绝继续写确定性 sidecar，避免相同输入重复训练。

## 9. 测试策略

严格 TDD，重点覆盖：

- 默认 XGBoost spec 的完整参数、真实 `random_state=42` 和 distribution 映射；
- estimator 实例参数或类型偏离 spec 时在拟合前拒绝；
- XGBoost 版本变化使模型和拒绝 sidecar 失效；
- 裸 `RandomForestRegressor`、自定义 factory 和自定义 trainer 不能进入 pipeline 持久生命周期；
- 非封闭后端逐塔/目标回退但物理 DLR 正常；
- 真实 XGBoost 的首次训练、保存、下次加载和气象 MAE/RMSE 晋级；
- 常量残差为确定性数据回退；操作性回退不写 sidecar；
- 旧模型只影响自身，不破坏 generation、ledger 或其他塔模型；
- 全量回归、Ruff、编译和页面验收保持通过。

## 10. 兼容与迁移

- 页面调用 `DlrPipeline()` 无需修改，自动使用封闭生产后端。
- 现有模型文件保留；合同不兼容时按逐模型失效处理，不做破坏性删除。
- 测试中的自定义 trainer/factory 迁移到非持久单元测试或改用真实 XGBoost 场景。
- 本设计只收口 Task 11 的训练后端边界；Task 12 及后续计划保持不变。
