# 第五组 Joint Verification Decision 实现报告

版本说明：下文保留 v1 初次实现记录。当前联合决策已按用户授权升级为 v2；现行 RACA 规则、非劣保证、搜索上限及重跑说明见 [v2 修改说明](05_joint_decision_v2_changes.md)。入口现已支持版本变化时自动归档旧结果并启动新运行，来源一致时断点恢复，PyCharm 不需要添加参数。其他实验环节保持原定义。

本次仅完成完整代码实现及必要的静态检查，尚未运行实验、测试或绘图。正式结果将由用户在PyCharm执行后产生。

## 交付文件与运行方式

本组源码均在 `experiments/05_joint_decision/`，原有入口占位文件已替换，其余文件新增：

| 文件 | 职责 |
| --- | --- |
| `run_joint_decision.py` | PyCharm 入口，默认完整正式实验，支持 `--plots-only` 与 `--force` |
| `config_joint_decision.py` | 完整参数、版本、网格、论文附件哈希与协议校验 |
| `joint_experiment.py` | 共同输入复用、五方法执行、独立评价、逐 trial 提交、导出与绘图调用 |
| `joint_inputs.py` | 标签隔离的数据适配、完整 N 候选域 GT、共同资源成本 |
| `historical_state.py` | 独立合成预历史、异常状态、verification age 与 risk |
| `joint_objective.py` | 不依赖优化器的统一 J 重算及约束检查 |
| `joint_methods.py` | RACA-HIM、DVA-P-Adapted、RF-TS、Cost-First、Random |
| `joint_storage.py` | 原子压缩 checkpoint、来源校验、CSV 导出、配对统计 |
| `plot_joint_decision.py` | 只读取已保存 CSV，独立重绘 J1/J2 |
| `static_check_joint_decision.py` | 仅解析源码与编译代码对象，不导入或执行项目代码 |
| `validate_joint_decision.py` | 用户之后手动执行的运行时校验；本次未执行 |
| `__init__.py` | 包标识 |

报告文件为本报告和 `reports/05_joint_decision_baseline_adaptation.md`。没有修改 Dataset、公共模块、前四组代码或结果，也没有实现第六组。现有 `experiments/run_all.py` 未改动；第五组使用专属入口运行。

在配置好项目现有 `requirements.txt` 依赖的 PyCharm Python 环境中，直接右击 `experiments/05_joint_decision/run_joint_decision.py`，选择 Run。入口根据自身位置定位项目根目录，关闭项目 bytecode 写入，并加载合法的数字目录包名；不需要改 Working Directory、PYTHONPATH、命令行参数或绝对路径，也不依赖 Desktop 附件在运行时仍然存在。

以下命令供用户之后自行执行；本次未执行前三条：

```powershell
python -B experiments/05_joint_decision/run_joint_decision.py
python -B experiments/05_joint_decision/run_joint_decision.py --plots-only
python -B experiments/05_joint_decision/run_joint_decision.py --force
```

`plot_joint_decision.py` 也可在 PyCharm 独立右击运行。它只导入本组配置和公共绘图样式，不导入数据加载、GT、优化器或实验驱动。

## 论文依据与冲突处理

已经读取 Prompt、最新版 TNSE Markdown、Dataset README/config/loader/validation report、已有实现，以及 DVA-P 原论文有关公式与算法。TMC PDF 只用于视觉参考，未移植其委员会协议。源码配置保存四个附件的 SHA256；最新稿件 SHA256 为 `61de825be020a3196d3f175fedfa7a82c6c87ddc6ad58caa3a5e0aaf4e9ac2c3`。

最新版稿件优先于 Prompt 的简略措辞：

1. GT admission 使用固定 `ceil(2K/3)`，而非简单半数。可用候选只计一次自支持；同伴统计排除对角线。
2. 最终 GT 排序为 `(history, -peer_support_count, -mean_support_similarity, temporary_id)`。
3. 联合问题要求非空解 `1 <= selected_count <= B`。空任务没有有效 J=0。
4. RACA-HIM 不在首次非正边际增益时停止；生成到预算或无可行 pair，再选择 J 最大的非空前缀。
5. 成本记号在稿件为 `d_gi`，代码统一为毫秒 `costs[g,i]`，与 Prompt 的 `c_gi` 同义。

## 数据边界与共同输入

通过 `dataset.load_dataset.load_group` 按逻辑配置加载现有基础库，固定 cr=3%、d=30、H=0.5。适配器只访问返回字典的 `X`，随后释放该字典；没有读取标签、prototype、cosine_to_prototype 或 sample_id 来决定历史、GT、目标或分配。

临时身份为独立随机流产生的 `permutation(N)`，与源样本 ID 无关。资源 API 使用内部行位置，优化器只看到临时 ID。两者不会混作索引。描述符、风险和成本具有独立数组指纹；全 Dataset 文件快照和校验清单用于来源标识，不进入决策。

每个 `(N, seed)` 只加载一次描述符、生成一次历史、构建一次 GT、取得一次基础资源快照。改变预算不会重抽模型；改变 sigma 不会重抽历史或 GT。每种 sigma 的成本矩阵缓存并在五方法之间共享。只有各方法自己选择的目标和 assignment 不同。

dataset_seed 就是正式 seed 2026–2055。其他流通过 `SeedSequence([trial_seed, stream_code])` 派生，stream_code 分别为 history=5101、resource=5102、temporary-ID/GT tie=5103、Random=5104。没有使用 Python `hash()`。同一 `(N, seed)` 跨两图和所有参数面板共享这些状态。

## 历史初始化

正式决策轮为 t=1；20 轮预历史记为 -19 至 0，避免把正式单轮实验改成 200 轮长期实验。每模型每轮以 10% 概率产生观测；条件于观测，normal/anomalous/timeout-or-invalid 概率为 95%/3%/2%。这是公开的合成假设，不是数据集提供的真实历史。

初值 h=0.10；正常、异常、无效更新分别为 `(1-rho)*h`、`h+rho*(1-h)`、`h+rho*eta*(1-h)`，rho=eta=0.5。只有正常/异常这两类有效状态判定刷新 last_verified；无效事件不刷新。未有效验证过者 age=1，其余 age=`min((1-last_verified)/20,1)`，risk=`0.5*age+0.5*h`。配置值显式传入历史生成器，未依赖另一个实验的默认值。

历史函数只接收 N、独立 seed 和公开设置，API 不接收任何当前真实损坏信息。五方法拿到同一份风险。此实验测量对可观测风险分值的覆盖，不支持“历史风险已预测当前损坏”的结论。

## GT 复用与候选域

第四组的 `ground_truth_methods.py` 包装器限制为 100 候选且统一 h0，不能直接用于本组。该包装器使用的底层内核是 `experiments/01_detection/proposed_detection.py::build_ground_truth`，已有固定 quorum 与 consensus 排序参数，符合最新稿件。

本组沿用该内核和公共 `AlgorithmInput`：候选域为整个 N，历史升序、临时 ID 平局，从 K=50 开始，每次保留旧成员、增加 25，首次获得 M=25 即返回，否则扩大到 N 后明确失败。未使用第四组按标签采样的候选污染压力池，也没有读取原型构造参考。

为避免原内核在候选域耗尽时分配完整 N×N 相似度矩阵，K<=256 的每个实际步骤直接调用该内核；更大 K 用 128 行分块调用公共 `cosine_pairwise`，在完整同序 peer 向量上累计相同支持数和支持相似度均值，再复用原 quorum 和最终排序函数。没有减少比较域或改变 admission/扩展规则。每次实际 K、门槛、通过数和逻辑比较次数保存到 checkpoint。分块与整矩阵同属 float64 计算，但不同 BLAS 分块可能带来末位舍入差别；阈值没有加宽，尚待用户运行校验。

GT 成功后 target pool 为完整域减去 GT，大小 N-25，五方法共用这次 GT。GT 失败时保留该配置五条共同 `gt_unavailable` 记录，不重抽 seed，不补入未通过门槛的候选。

## 资源成本复用

复用第二组 `overhead_resources.shared_resource_profiles` 的固定 5000 节点、1 轮基础快照及其 N-prefix，避免在改变 N 时改变 round 资源抽样。实际成本调用原 Dataset 的 `resource_costs`，没有另设分布，也没有复用第二组 Fixed-Workload Mode。

成本矩阵维度为 `(25, N-25)`。原接口将 4 ms 计算、50 Mbps 带宽和 10 ms 基础时延施加其已有确定性 sigma 变换；float32 描述符载荷为 `4*d=120` bytes。记录 compute/transfer/network 三分量和 total 指纹，检查 total 与分量之和一致。`total_ms` 已含 network 一次，不再追加。sigma=0 保留 Dataset 同质公式，单任务总成本约 14.0192 ms，以原 API 的浮点结果为准。

ROUND_ID=1、外生失败率=0。800 ms 在本组配置显式覆盖 Dataset 原 400 ms 默认值，Dataset 本身没有被修改。GT CPU 时间、读取时间、算法求解时间和绘图时间不进入 verification makespan。

## 五方法与统一评价

| 方法 | 实际决策规则 |
| --- | --- |
| RACA-HIM | 每步对全部合法 target–GT pair 计算精确 marginal J，构造全部可行前缀，返回最大 J 非空前缀 |
| DVA-P-Adapted | 保留原文零初值虚队列、slot 目标、priority/add/单元素替换；风险给遍历顺序，新增 GT 映射与共同硬约束在适配报告详述 |
| RF-TS | 先按风险降序，再对当前目标选择合法最小 `L_g+c_gi` 的 verifier |
| Cost-First | 在所有未选 pair 中选择最小 marginal makespan，平局按成本与临时 ID，不用风险排名 |
| Random | 固定独立 seed，将初始合法 pair 均匀随机排列，按当前容量逐个接受至预算或无可行选择 |

DVA-P 原文没有独立可信 GT 的 assignment 变量，所以本组明确命名为 Adapted；不能将其视为原文多轮完整复现。其真实保留步骤、初值退化、资源映射以及不具备的长期保证，见适配报告。图例显示 DVA-P，原始记录保留 DVA-P-Adapted。

每个方法返回 assignment 后，都由同一个 `evaluate_joint_objective` 从 ID、risk 和共同成本重新计算：

`risk_reward=sum(selected risk)`；`normalized_risk_reward=risk_reward/B`；`L_g=sum(costs of jobs assigned to g)`；`Tver=max_g L_g`；`normalized_delay_penalty=0.5*Tver/800`；`J=normalized_risk_reward-normalized_delay_penalty`。

独立 evaluator 检查唯一 target、唯一 assignment、GT membership、自验证、预算、时限、合法成本和非空性。非法解不被偷偷删任务后当作合法解；J 记缺失并保存失败原因和 proposed assignment。合法少选不是失败，completed_count 只等于实际可行调度数量；没有把预算上限当成实际完成数。负 J 不截断。

## 网格、结果、恢复与统计

J1 的五预算面板固定 sigma=.50；J2 的五 sigma 面板固定 beta=.10。两图共用的 beta=.10/sigma=.50 仅运行一次。每个 N 只有九个不同参数 pair，总计 81 个不同配置、2430 个 trial；完整运行后应有 12150 条方法记录。此数字是计划网格，当前尚未产生结果。

每个 trial 的五结果、逐任务 target/verifier/risk/cost/累计负载/makespan、全部 verifier 最终负载、共享输入指纹、GT trace 和方法诊断原子写入 `logs/checkpoints/<run_id>/*.json.gz`。主 CSV 还包含最终 verifier load JSON、实际预算占比和未用预算；assignment CSV 可联合 raw 中 B 重算 J。汇总时逐文件流式展开逐任务数据，避免将所有 assignment 同时放进内存。

正常入口按 checkpoint 跳过已完成配置。恢复校验代码、Dataset、参数、Python/NumPy/SciPy/BLAS 数值环境的 fingerprint，及 checkpoint 内容与 manifest 哈希。中断发生在 checkpoint 提交和 manifest 更新之间时，可恢复已完整写好的 trial。来源不兼容时拒绝静默继承，须由用户显式 `--force` 创建新 run_id，重新跑完整网格。旧 checkpoint 保留在旧 run_id 下；只覆盖第五组正式导出，不改前四组。

完整运行后生成：

```text
results/05_joint_decision/
  raw/joint_runs.csv
  raw/assignment_diagnostics.csv
  summary/joint_summary.csv
  summary/feasibility_summary.csv
  figures/Fig_J1_joint_utility_budget.pdf
  figures/Fig_J1_joint_utility_budget.png
  figures/Fig_J2_joint_utility_heterogeneity.pdf
  figures/Fig_J2_joint_utility_heterogeneity.png
  logs/run_manifest.json
  logs/figure_manifest.json
  logs/execution.log
  logs/execution_report.md
  logs/checkpoints/<run_id>/*.json.gz
```

共同 GT 失败、无目标、无合法 verifier、零预算、deadline 不可行、方法异常、非法 assignment 均有独立状态；数据加载或资源接口自身出错会使运行 manifest 失败并保留已有 checkpoint，修复后恢复。方法异常作为该方法失败记录保留，正常重跑不会有选择地替换这些 seed。

主图每个配置采用“五方法均可行”的共同 seed 集合。各方法独立有效数/失败率另存；summary 同时包含 n_total、n_valid、n_failed、n_paired。均值、样本标准差(ddof=1)、双侧 Student-t 95% CI 仅对同一配对集合计算。n_paired<2 时均值和 CI 全部缺失，不补零、插值或复制相邻点。完整有效时为 30 个 seed，存在失败时明确披露实际样本数。J1/J2 重合 summary 从同一结果复制并再次校验一致。

## Figure 代码

复用公共 `plotting_style.py` 的颜色、marker、字体和第二至四组的完整连续轴外框：7.16×1.50 inches，五个约 82.71 pt 宽、64.26 pt 高的轴，间距约 14.89 pt，上方共享 legend。没有沿用第一组 Accuracy 的断轴。RACA-HIM 蓝、DVA-P 棕、RF-TS 青、Cost-First 灰、Random 深灰；全部实线。

每条曲线使用全部九个真实 N 配置，缺失值保留 gap；不平滑、不增加噪声、没有 CI 阴影或误差带。每张 Figure 内五子图共用实际数据确定的纵轴范围，保留负 J。输出矢量 PDF 与 600 dpi PNG。读取前校验 summary 的 manifest 身份、文件哈希、完整网格、配对数量和两图重合统计。重绘只读已有统计，不重新优化。

不同预算的 J 使用不同 B 分母，主要在同一面板比较方法；不能把跨预算的纵轴变化无条件解释为性能提升。若整张 Figure 均无有效配对均值，绘图明确报错，原始失败记录和汇总保留，不能伪造曲线。

## 校验与未执行事项

实际执行范围仅为源码读取、AST 解析和 `compile(AST, ..., 'exec')` 的非执行式语法检查；没有执行生成的代码对象、没有导入实验模块、没有运行单元测试或绘图。本组静态检查入口同时检查相对模块文件存在、禁止直接访问标签字段以及 plot-only 的导入边界。

用户可之后自行运行：

```powershell
python -B experiments/05_joint_decision/static_check_joint_decision.py
python -B experiments/05_joint_decision/validate_joint_decision.py
python -B experiments/05_joint_decision/validate_joint_decision.py --check-dataset
```

第二条是单独的校验程序，不是 Quick Mode，不产生论文结果；默认包含八类人工不变量校验。第三条另加同 seed 两次读取描述符的只读复现检查。本次最终对 12 个 Python 源文件的 AST/非执行式编译检查完成；其余两条运行时校验没有执行。源码内还提供配置、历史、资源、GT 分区、assignment、保存诊断和 summary 的运行时断言。验证覆盖关系如下；这些断言和运行时校验的结果尚未知：

| Prompt 检查项 | 实现位置 |
| --- | --- |
| 1–7 数据网格、shape、标签隔离、复现 | config、joint_inputs、static_check、validate |
| 8–14 GT 大小、完整候选域、失败、共同池 | joint_inputs、joint_experiment、validate |
| 15–22 历史参数、标签隔离、risk范围、共同状态 | historical_state、显式 HistoryConfig、validate |
| 23–28 sigma、成本合法性、索引和 network 去重 | ResourceCostMatrix、共同 Dataset API、JointProblem、validate |
| 29–44 唯一 assignment、并行负载、约束、算法差异、J一致性、负值 | joint_methods、joint_objective、checkpoint 复核、validate |
| 45–58 图网格、样式、负轴、来源、独立重绘、重合配置 | plot_joint_decision、公共样式、summary 校验、配置去重 |

尚未验证的事项包括依赖环境实际导入、各算法数值行为、运行耗时、GT 成功率、完整输出数量、断点的实际中断恢复、最终图的字体布局和视觉效果。它们需要用户按上述说明执行后确认。当前没有实验数据、胜负结论或已经渲染成功的正式 Figure；不能将静态检查表述为运行时测试通过。
