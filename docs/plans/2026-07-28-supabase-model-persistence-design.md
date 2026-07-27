# Supabase XGBoost 模型持久化设计

## 1. 背景

当前 `ModelRegistry` 将训练后的 XGBoost 模型、元数据和 manifest 写入本地
`models/`。这一实现可以在单个进程内安全复用模型，但 Streamlit Community
Cloud 的容器文件系统会随休眠、重启、重新部署或容器迁移而丢失，不能作为跨
运行的模型仓库。

本设计把 Supabase 项目 `ciapxhuldarsupmvrgwu` 作为远端权威存储，同时保留
现有本地注册表作为单次 DLR 运行的受控缓存。页面 UI、训练准入规则、气象修正、
IEEE 738 热计算和弧垂后验证均不改变。

## 2. 目标与非目标

### 目标

- 应用重启后可按项目、线路、杆塔和气象目标加载已训练模型。
- 模型文件位于私有 Supabase Storage bucket，数据库保存可查询的版本和生效指针。
- 继续校验 SHA-256、`ModelMetadata`、sealed XGBoost 合同和运行时兼容性。
- 模型发布采用“不可变对象先写入，数据库指针最后切换”的顺序。
- 同一候选的确定性拒绝记录跨容器保存，避免重复训练。
- Supabase 暂时不可用时继续输出物理气象和 DLR，不使用未持久化候选。
- 密钥只进入 Streamlit Cloud Secrets，不进入 Git、日志、页面或模型元数据。

### 非目标

- 不迁移审计日志、弧垂结果或用户上传文件。
- 不改变气象训练指标；仍只比较修正气象与真实气象的 MAE/RMSE。
- 不开放浏览器直传模型，不为匿名用户创建 Storage 或数据库写策略。
- 不提供模型管理 UI。

## 3. 方案比较

### 采用：Storage 文件 + PostgreSQL 元数据

`model.joblib` 存入私有 bucket，PostgreSQL 保存模型元数据、校验哈希、当前生效
generation 和拒绝记录。数据库行小、模型可独立下载，后续增加线路或模型版本时
仍易管理。

### 备选：模型直接存 PostgreSQL `bytea`

少量模型可以放进 `bytea`，且模型与指针可在单个数据库事务中提交。但通过
PostgREST 传输二进制需要额外编码，数据库备份和行膨胀也会随模型版本增加。本期
不采用。

### 不采用：继续使用本地目录或写回 GitHub

本地目录不具备跨容器持久性；运行时写 GitHub 会引入密钥、提交冲突和重新部署
循环，均不适合作为模型仓库。

## 4. Supabase 资源

创建私有 bucket：

```text
dlr-models
```

对象路径：

```text
<project_id>/<line_id>/<tower_id>/<target>/<generation_uuid>/model.joblib
```

创建三张表：

1. `dlr_model_generations`
   - generation UUID、四元 scope、模型版本、Storage path、SHA-256；
   - 完整 `ModelMetadata` JSONB、模型状态、创建时间；
   - generation 永不覆盖。
2. `dlr_model_heads`
   - 四元 scope 为主键；
   - 指向当前 generation，并保存递增 revision。
3. `dlr_model_rejections`
   - 四元 scope 与 attempt fingerprint 唯一；
   - 保存 champion context、拒绝原因、attempt JSONB 和创建时间。

表和 Storage 均不向 `anon` 或 `authenticated` 开放。应用仅使用服务端 secret key。

## 5. 组件边界

新增 `SupabaseModelStore`，只负责 Supabase I/O：

- 查询当前 head 和 generation；
- 下载、上传不可变模型对象；
- 调用数据库 RPC 原子切换 head；
- 查询和写入拒绝记录。

新增 `SupabaseModelRegistry`，复用现有 `ModelRegistry` 的训练准入、模型合同和
文件校验：

- 每次页面计算创建一个临时本地注册表；
- `load` 前从远端下载当前 generation 并导入临时注册表；
- `promote` 前重新读取远端 head，再由本地注册表执行原有质量决策；
- 本地晋级成功后上传模型，再以 compare-and-swap RPC 切换远端 head；
- RPC 成功前不向 pipeline 报告晋级成功；
- 单次运行内缓存已下载 generation，避免重复请求。

`DlrPipeline` 保留原有模型操作，并在其外增加可选运行边界和可用性查询：

```text
begin_pipeline_run
  -> load_many -> model_operations_available
  -> build_attempt -> was_rejected -> model_operations_available
  -> promote -> load
end_pipeline_run
```

普通本地 `ModelRegistry` 不实现这些可选钩子，行为不变；Supabase registry 用运行边界限制传输故障的影响范围。

## 6. 发布与并发

发布顺序：

1. 读取当前远端 head，记录 expected generation UUID。
2. 使用现有 `ModelRegistry` 在临时目录重新校验候选和 champion，并做晋级决策。
3. 将本地已校验的 active artifact 以 `upsert=false` 上传到新的不可变 Storage path。
4. 调用 `activate_dlr_model_generation` RPC。
5. RPC 在事务内锁定 scope head，核对 expected generation，插入 generation 并切换 head。
6. CAS 冲突时本次晋级返回 `remote_head_conflict`，不覆盖另一个运行刚发布的模型。

Storage 和 PostgreSQL 无法处于同一事务，因此对象必须先上传、head 最后可见。
CAS 失败可能留下不可见 orphan 对象，但不会被应用加载；后续可按数据库引用进行
清理。模型量很小，本期不增加定时清理任务。

## 7. 加载与校验

加载远端模型时依次验证：

1. head scope 与请求的 `ModelKey` 完全一致；
2. generation metadata 可由 `ModelMetadata.from_dict()` 解析；
3. metadata scope、Storage path 和数据库 scope 一致；
4. 下载字节的 SHA-256 与 generation 和 metadata 一致；
5. joblib 内容是 `ModelBundle`；
6. sealed backend、训练合同、特征、导线、地形、坐标、修正配置和依赖版本继续由
   现有 `ModelRegistry.load()` 校验。

任何单模型损坏只使对应杆塔/目标回退，不影响其他模型。

## 8. 故障处理

- Supabase 查询或下载失败：抛出操作性 I/O 错误，由 pipeline 对应 key 回退物理气象。
- 一次 DLR 运行内首个明确传输故障打开运行级熔断，停止后续远端模型 I/O 和训练发布；运行结束后复位，下一次运行重新探测。
- 响应合同损坏、checksum 错误和 Storage 单对象 404 不打开熔断，仍按模型 key 隔离。
- 远端没有 head：返回 `model_not_found`；有真实值时按原流程训练。
- Storage 上传失败：远端 head 不改变，本地候选不作为本次 AI 来源。
- RPC CAS 冲突：返回 `remote_head_conflict`，保留远端胜出版本。
- RPC 响应超时：按 generation UUID 查询提交结果；已成为 head 则按成功处理。
- 拒绝记录写失败：候选仍不晋级，但报告持久化失败，不能假装已成功缓存拒绝。
- Supabase 配置只提供 URL 或只提供 key：启动时明确报配置错误，不静默降级。
- 未配置 Supabase：本地开发仍可使用现有 `ModelRegistry`；生产部署必须配置两项 secret。

## 9. 配置和安全

Streamlit Cloud Secrets 使用：

```toml
DLR_SUPABASE_URL = "https://ciapxhuldarsupmvrgwu.supabase.co"
DLR_SUPABASE_SECRET_KEY = "<server-side secret>"
DLR_SUPABASE_MODEL_BUCKET = "dlr-models"
```

代码不得打印或显示 secret。bucket 为 private；数据库启用 RLS 且不创建客户端策略。
SQL RPC 固定 `search_path`，仅授予 `service_role` 执行权限。

## 10. 测试策略

- 用内存 fake store 覆盖首次发布、重启后加载、拒绝缓存和 CAS 冲突。
- 用真实 joblib 字节验证 checksum、metadata scope 和损坏隔离。
- 验证远端失败不激活候选，物理 DLR 仍可输出。
- 验证读侧和写侧传输故障在同一次 DLR 运行内不会按模型数量放大，下一次运行仍会重试。
- 验证配置工厂在完整 secrets 时选择 Supabase，缺一项时报错，均缺失时本地回退。
- 保留全部现有 `ModelRegistry`、XGBoost 生命周期、DLR 和弧垂测试。
- 在真实 Supabase 项目执行一个最小 round trip，确认表、bucket、上传、下载和 head 查询。
