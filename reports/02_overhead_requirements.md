# 第二组实验需求与验收边界

本次代码工作以用户当前请求、已读取的“损坏识别性能”chat 最新要求、`Prompt - Codex.txt`、TNSE 最新稿件及对比论文为依据。附件正文用于明确实验方法；其中与用户已经表达的操作偏好冲突的步骤，以用户的最新要求为准。

## 已确认的优先级

- 本轮只编写第二组 `02_overhead`；用户自行运行实验。本轮不执行实验、Quick Mode、pytest，也不生成虚构结果或示意曲线。
- 不提供 Quick Mode；PyCharm 直接运行入口与正常命令行入口都执行完整正式网格。中断后的继续运行采用阶段 checkpoint。
- Baseline 的参考构建、损坏判定及定位遵循原论文的 hash / proof / digest 机制，不使用相似度，不调用 Proposed 的 Ground-Truth 作为自己的原生参考构建。
- 仅 Proposed 使用 TNSE 的余弦相似度与多数支持规则。
- 直接复用 `experiments/common/plotting_style.py` 的配色；Proposed 蓝色、DVA-P 棕色、I-EDI 橙色、CTCV 绿色。算法曲线均为实线，保留一致 marker，不画置信区间阴影。
- 采用 TMC 风格的紧凑横向子图、共享图例及 serif 字体。第一组检测图的柱状图和断轴要求不迁移到第二组；O1 / O2 使用五张横向折线子图。

## 正式配置

| 配置 | 值 |
| --- | --- |
| N | 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000 |
| d 子图 | 10, 20, 30, 40, 50 |
| seeds | 2026 至 2055，共 30 次独立重复 |
| corruption rate / H | 3.0% / 0.5 |
| K / M | 50 / 25 |
| 相似度区间 | [0.997, 1.0]，仅 Proposed 使用 |
| alpha / lambda | 0.5 / 0.5 |
| h0 / Amax | 0.10 / 20 |
| rho / eta | 0.5 / 0.5 |
| beta_B / sigma / p_fail | 0.10 / 0.5 / 0.0 |
| deadline | 800 ms，约束验证阶段的 makespan |
| 单轮资源索引 | round_id = 1；实际行索引在 0 至 N-1 |

O1 / O2 共用一次配置执行得到的时间、字节、消息及完成数。原始数据集保持只读，不修改其仍可能记录 400 ms 的配置。

## 必须保持的算法语义

1. Proposed 从全验证域按 `(历史异常状态, 临时实例 ID)` 排序构造候选集合；候选集合不足时逐步扩大并重新比较。阈值为 `ceil((actual_K + 1)/2)`，自支持只加一次，只有有效的其他候选响应计入支持数。
2. 初始状态不使用当前标签。未知最近成功验证轮次时 `A=1`；`h=0.10`，因而初始 `r=0.55`。
3. 每个被选择目标只对应一个合法 verifier；GT 与目标互斥。预算为 `ceil(0.10 * (N - 25))`。
4. Proposed 贪心使用 `ΔJ = r/B - lambda * ΔT/deadline`。持续生成至预算耗尽或不存在可行 pair；不能在首次非正边际增益处终止；从所有非空前缀中选择原目标值最大的前缀。
5. 同一 verifier 的任务串行，不同 verifier 并行；验证 makespan 为每个 verifier 累计任务成本的最大值。GT 与验证阶段有前后依赖，但 GT 时间不占用 TNSE 的 800 ms 验证阶段 deadline。
6. GT 失败、deadline 不可行、预算未用满及未完成任务均保留状态和数量，不删除配置、不把未验证目标视为正常。
7. Fixed-workload 与 native-budget 的指标不能混入同一图。固定任务模式须明确哪些 baseline 选择机制被绕过；native 模式保留诊断结果。

## 资源与消息核算

资源成本来自 `dataset.load_dataset.resource_profiles()` 与 `resource_costs()`，四种方法共享配置、seed、资源状态及实例。`transfer_ms` 已包含一次 `4*d` 字节描述符传输；补充控制与 proof 传输时使用同一 pair 的每字节传输成本，不能重复计入描述符。每条单程逻辑消息的基础传播时延单独核算。

每条真实逻辑发送记录 sender、receiver、phase、type、payload/control/security 字节以及模拟时间。总字节和总消息数必须由事件表直接汇总，分项不重叠。链路速度改变只影响传输时间，不改变报文字节数。64-byte request / 32-byte result 是公开仿真假设，不宣称是原论文指定的编码；proof / digest 优先按实际生成的结构计量。

模拟系统时间与 `perf_counter()` 的实际 Python CPU 时间分别记录。文件加载、checkpoint 写入和绘图不计入决策 CPU 时间；未实际执行的密码学机制只能称为适配或成本假设。若主图使用 hybrid 定义，需明确 `GT_sim + decision_CPU + verification_sim` 及避免重复计费的边界。

## 可复现与恢复

checkpoint 保存稳定的配置签名、实现版本、实例 ID、工作负载 ID、协议阶段和原始结果。恢复时拒绝配置、ID 或版本不一致；不采用标签决定跳过或恢复阶段。CPU runtime 是实测值，不参与确定性结果比较；模拟时间、通信事件、任务及预测证据保持同 seed 可复现。

统计独立单位是 seed；mean、sample std 与 95% CI 基于 30 个 seed-level 数据，不将同一 seed 内的数千任务作为独立重复。失败与完成数分开汇总，避免只保留成功运行制造较低开销。

## 本轮交付状态的表述

代码与独立 pytest 用例可供用户运行。本轮只进行静态检查；不得表述为“测试通过”“正式实验完成”“图已生成”或报告尚未计算的任务完成数。正式结果、Figure PDF / PNG 及图形视觉验收应由实际完整运行产生，并记录其有效配置与状态。
