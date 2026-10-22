# 第二组最近运行的只读审计

本报告核验用户已经保存的 `results/02_overhead_core_v3`，没有执行正式实验、算法试跑、测试套件或绘图，也没有改写任何 `results` 文件。机器可读证据、逐文件 SHA256、消息及操作汇总见同目录 `02_overhead_latest_run_audit.json`。

## 完成情况及图片位置

已保存运行版本为 `overhead-core-v3`，fingerprint 为 `399018002e9ad72399fa967fe7220bf4a8e8e2dc6a513e379bd59d3e70773760`。45 个 checkpoint 全部完整，对应 9 个 N、5 个 d、seed=2026；每点四方法均为 success，合计 180 条记录。每种方法在全部配置中实际完成 13,410 个目标；逐点核验 selected、attempted、completed 均等于 B，没有删掉部分 B 目标。

以下文件确实存在且非空；PDF 的文件头、startxref 与 EOF 标记正确，PNG 的全部 chunk CRC 与 IEND 检查通过，PNG 为 4296×900 像素。本审计没有重新生成或进行视觉检查：

| 文件绝对路径 | 字节数 |
|---|---:|
| `E:/Code/Paper_Code/Paper_Yan/20261009_TNSE/results/02_overhead_core_v3/figures/Fig_O1_time_overhead.pdf` | 41,284 |
| `E:/Code/Paper_Code/Paper_Yan/20261009_TNSE/results/02_overhead_core_v3/figures/Fig_O1_time_overhead.png` | 328,424 |
| `E:/Code/Paper_Code/Paper_Yan/20261009_TNSE/results/02_overhead_core_v3/figures/Fig_O2_communication_overhead.pdf` | 44,362 |
| `E:/Code/Paper_Code/Paper_Yan/20261009_TNSE/results/02_overhead_core_v3/figures/Fig_O2_communication_overhead.png` | 285,698 |

这些图片位于 **core-v3 新目录**，不是旧版 `results/02_overhead/figures`。路径错位可以解释在旧目录找不到新版图片；该判断并不等于本审计已检查图片视觉版面。

## 为什么运行很快

最近 `run_manifest.json` 明确记录 `executed_configurations_this_invocation=0`、`resumed_configurations=45`、`elapsed_wall_seconds=1.0267065002117306`：最近这次调用复用了全部 checkpoint，没有重新执行算法。`progress.jsonl` 保存 45 个首次执行配置，逐配置墙钟之和为 **8.16047089966014 秒**，最短 0.0632434000726789 秒、最长 0.3121469998732209 秒；checkpoint 自身 wall_seconds 之和与此一致。该累计值只对应配置主体，不能当作包含数据检查、保存和绘图的完整入口耗时。

运行快另有两个可核验原因。正式工作量已从 30 个 seed 改为一个 seed，并取消附加 native-budget 实验及大体积逐事件导出；全部 45 点的 GT 均在初始 K=50 时已获得 M 个参考，因此没有需要实际执行的扩展轮。扩展条件仍在代码中，不能为了增加运行时间强行触发扩展。

程序墙钟与 O1 不同。墙钟是运行主机处理算法和仿真事件的真实时间；O1 为实测准备/决策/历史 CPU 加上必要传输和验证内核的模拟时延。传播时延直接累加事件，不以 sleep 等待。现有 O1 的单点范围：

| 方法 | 最小 total_time_ms | 最大 total_time_ms |
|---|---:|---:|
| RACA-HIM | 217.0532 | 893.9139 |
| DVA-P-Adapted | 214.9215 | 875.0819 |
| I-EDI-Adapted | 214.9360 | 875.3346 |
| CTCV-Adapted | 312.3428 | 732.1751 |

本轮不添加人为等待、重复协议执行、虚构失败或多 seed 平均来延长耗时。

## 已保存账本与实际核心步骤

所有 checkpoint 的通信字节及消息数与各阶段汇总守恒，total_time_ms 与准备时间、decision_cpu_ms、verification_time_ms、history_cpu_ms 之和一致。180 条 compact raw CSV 的对应字段逐项与 checkpoint 核对一致。审计读取前后的 raw、日志、checkpoint 和图像 SHA256 保持相同。

| 方法 | 完成目标总数 | 账本核验 |
|---|---:|---|
| RACA-HIM | 13,410 | descriptor_cosine 操作 13,410 次、comparisons 13,410；逐点均为 B，目标响应负载逐点为 4dB 字节。 |
| DVA-P-Adapted | 13,410 | 验证 hash 操作 13,410 次，逐点等于 B。 |
| I-EDI-Adapted | 13,410 | 验证 hash 操作 13,410 次，逐点等于 B；有限模 tag 判定仍保留。 |
| CTCV-Adapted | 13,410 | 验证阶段 hash 操作 17,813 次，其中 13,410 为目标响应、4,403 为实际正确挑战；逐点与 B + correct_challenges_computed 一致，ell=1000B、collector_count=2。 |

RACA-HIM 的历史排序、多数支持、动态阈值、必要扩展判断、联合贪心分配、单 verifier 的实际 cosine 判定及历史更新仍在。GT 的实测 CPU 为 0.17179991118609905–0.6784999277442694 ms。决策 CPU 为 1.8370000179857016–22.878600051626563 ms，45 点合计 461.8543004617095 ms；结合源码可确认它执行贪心分配，而非直接填写一条成本曲线。由于 compact 账本不保存每次贪心扫描，本审计不声称仅凭 CPU 数值已经独立证明每个内部比较。

`raca_core.py` 的可信协调器一次收集和增量复用避免了重复上传、重复 normalization 和已有 pair 重算；多数支持及扩展后的阈值仍重算。这是公开的部署适配，不能将其开销解释为完整去中心化全连接交换的测量。

## 固定工作量的科学范围

以下限制必须与速度解释同时保留，不能宣称四方法的完整原生选择过程全部运行：

- `overhead_baselines.py` 中 DVA-P 的 fixed_workload 分支直接调用中立 `_fixed_plan`，旁路原生 Eq.21 优先判断、增补/替换选择。
- I-EDI 的 fixed_workload 分支同样调用 `_fixed_plan`；本版没有执行 O²RS subset sweep，源码明确记录 source_selection_passes_executed=0。
- RACA-HIM 保留 marginal-gain 分配计算，但固定 B 覆盖了原生最佳 prefix 选择，并关闭原生硬 deadline 截断。800 ms 仍作为及时完成诊断。
- CTCV 保留实际分块/重排请求重建、目标 hash/modulo、k+1 collectors、packed CR/R 与最终协同判定；没有测量完整链广播、共识和认证部署。

现有诊断 `all_core_steps_executed=true` 应理解为**固定工作量协议全部完成**。它不能证明 DVA/I-EDI 的原生 subset selector、RACA 最佳 prefix、原论文长期保证或完整部署都已执行。固定 B 适配用于比较相同执行任务量，应在图表和论文说明中明确；若未来恢复原生选择成本，需要另行定义测量范围和新版来源身份，不能复用旧数值冒称新机制。

## 源码一致性

保存元数据中的 20 个 computation source 文件已逐个核验，当前 SHA256 均与此次用户运行保存值一致，包括 `overhead_core_simulator.py`、`raca_core.py`、`overhead_baselines.py`、`overhead_simulator.py`、费用模型及依赖。审计时完整 source inventory 中仅 `run_overhead.py` 的来源 hash 与保存值不同，它属于正在修订的入口/输出说明；没有计算源码差异。逐文件值及这项差异的前后 hash 均记录于 JSON。

本轮结论是：现有固定 B 的执行项完整、当前缓存恢复可以解释极短调用时间，而原生 subset selector 的范围确实受固定工作量适配限制。此次审计不改计算、结果值或绘图文件，也不人为延长执行。
