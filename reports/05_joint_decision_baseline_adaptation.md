# 第五组联合决策：方法定义与 DVA-P 适配说明

版本说明：下文的 RACA-HIM 节描述 v1 保留的贪心子程序。当前 v2 的候选择优和局部改进见 [v2 修改说明](05_joint_decision_v2_changes.md)。RF-TS、DVA-P 与其余基线的决策定义没有被弱化或改变。

本文档说明源码定义，不报告实验结果。此次未运行任何方法、测试或绘图。

## 共同输入与独立外部评价

五种方法接受同一 `JointProblem`：GT verifier 临时 ID、target 临时 ID、可见风险、同一资源成本矩阵、预算 B、λ=0.5 和 800 ms 验证时限。矩阵下标为 `[verifier, target]`，所有元素直接来自共同 Dataset 资源接口；不会按方法修改成本，也不会重复累加 network 项。决策 API 不接收 binary_label、subtype_label、prototype、cosine_to_prototype 或 sample_id。

每种方法自行产生 `(target_id, verifier_id)` assignment。`joint_objective.evaluate_joint_objective` 随后从 ID 和共同成本独立重建负载、makespan、被选风险和

\[
J=\sum_{i\in S}r_i/B-0.5\max_g L_g/800.
\]

最终 CSV 使用独立评价值，不使用调度器报告的内部目标。约束检查拒绝重复 target、非 GT verifier、自验证、超预算、超时以及不可用 pair；不会截断或修复非法方案。最终解遵守最新稿件的 `1 <= |S| <= B`，因此合法负 J 不截零，空任务不冒充可行 J=0。所有可行调度任务在共同 `p_fail=0` 的成本模型中完成，`completed_count` 等于实际被调度数量。该值不是模型正确性或检测成功数量。时限重算允许 1e-9 ms 数值舍入容差，优化器始终按原始 `<=800` 检查。

## RACA-HIM

实现依据为最新 TNSE 稿件 Section IV-C，而非 Prompt 中可能被误读为普通风险排序的表述。每轮枚举所有未选 target 与合法 GT verifier 的 pair，按

\[
\Delta J_{gi}=r_i/B-\lambda[\max(T,L_g+c_{gi})-T]/T_{max}
\]

选择最大者。精确平局时按临时 target ID、verifier ID 升序。NumPy 矩阵每个单元对应原始贪心循环的一个 pair，不截取 top-risk 候选，不随机跳过 pair。

构造持续到 B 个任务或再无可行 pair，即使遇到零或负边际增益也继续。记录所有生成的非空前缀的 J，返回最大者，J 精确相同时取最短前缀。这样既允许少于 B 个任务，也满足稿件非空约束。内部 prefix J 仅用于选解，最终仍由独立 evaluator 重算。

## DVA-P 原文依据

阅读来源：用户提供的 Zhao 等人论文 *Long-Term Over One-Off: Heterogeneity-Oriented Dynamic Verification Assignment for Edge Data Integrity*，IEEE TMC 23(5), 2024，DOI 10.1109/TMC.2023.3294180。已直接提取并阅读与实现有关的 PDF 第 3–8 页（印刷页 4603–4608）。以下页码均为印刷页。

| 原文位置 | 本实现保留的机制 |
| --- | --- |
| p.4603，式 (1)、(6)–(7) | 每个 edge endpoint 的二元 inspection 决策及通信/处理资源成本意识 |
| p.4604，式 (12a) | 随总 selected delay 单调下降的 `2/(1+exp(sum delay))` 内部目标方向 |
| p.4605，式 (13) | 每个 target endpoint 的虚拟时延队列，初值 Q_i(0)=0；单轮后按原式更新 |
| p.4605，式 (21) | slot drift-plus-penalty，而非 RACA-HIM 的 marginal J |
| p.4607，Algorithm 1 lines 3–8 | 前轮 selected 集合与其 unselected complement 的 priority judgement；不可行时从空集开始 |
| p.4607，Algorithm 1 lines 9–13 | source objective 非正增量时加入一个 endpoint |
| p.4607，Algorithm 1 lines 14–23 | 枚举当前 selected endpoint 的单元素替换，接受最大非负目标下降 |
| p.4607，Algorithm 1 line 26 | 一次真正的 slot queue update，不制造长期训练记录 |

原论文输出的是 App vendor 是否检查某个服务器自身缓存 replica 的二元子集。它没有“target i 由独立可信 GT g 验证”的 x_gi 变量，也没有本组的 risk 奖励、B 个 target 的预算或 GT 串行累计负载。因此不能声称它原生解决本组联合模型，也不能把本组共享 GT 协议算作它自己的协议。第二组已有的 `choose_dvap` 使用 earliest-completion assignment adapter；本组不复用该适配器，避免将 DVA-P 与 RF-TS 的 assignment 混同。

## DVA-P-Adapted 的明确扩展

本组保留 target endpoint 为原式索引 i，并给每个 i 提供共同 GT 池中的合法 pair 选项。每次 Algorithm 1 加入或单元素替换时，先把新 target 映射到**当前容量合法且 intrinsic pair cost `c_gi` 最小**的 GT，平局取 verifier ID 最小者；随后仅对映射后的 endpoint 子集使用 DVA-P 自身式 (21) 判断 add/swap。`L_g` 只用于容量可行性检查，不加入 assignment 排序指标，因而该适配器与 RF-TS 的 `argmin(L_g+c_gi)` 不同，也没有使用 RACA-HIM 的 marginal J。原式 subset 目标不参与新增 g 自由度的排序，避免把原生 endpoint 目标方向扩张成对最慢 verifier 的人为偏好。

共同实验增加以下公开适配条件：

1. 候选 endpoint 按共同可见 risk 降序遍历，同风险时按临时 ID。risk 只提供中立遍历优先级，不替代 DVA-P 的 add/swap 目标；后续 source replacement 允许改变最终子集。
2. 每次加入与替换均检查共同 B 和 GT 串行累计负载，所有 GT 可以并行；成本采用共同 `c_gi`。替换只移除一个旧 target，保留其他 assignment，再为新 target 寻找合法的最小 intrinsic cost GT。枚举每个旧 target 的移除选项后，以映射 endpoint 成本计算原式 replacement 改善，接受最大的非负目标下降；平局按被移除 target ID、分配 verifier ID。并不强制其他方法复用 RACA-HIM 选择的 targets。
3. 式 (21) 的 `C_i` 使用当前 target 被选 pair 的共同 `c_gi`。其 exponent 中的 delay sum 是所有 selected pair cost 之和，属于 DVA-P 自己的内部目标；本组外部 J 的 delay 始终使用 `max_g L_g`，二者不混用。
4. 内部物理时间统一转为秒，C=0.8 s，Λ=1；Λ与外部 λ=0.5 是不同参数。式 (21) 的常数 Ω 不影响选择，被省略。除 sigmoid 项外保留 `Σ_i Q_i(d_i C_i-C)` 的定义。
5. 这是一轮独立决策，采用原生 Q_i(0)=0 和前轮 selected 为空。合成异常历史不能用来伪造 DVA-P 队列。初始 complement 包含全部 targets；正式配置 B<|V|，所以该 complement 不满足新增预算，priority judgement 回退空集。代码也提供小规模 complement 可行时的完整初始化分支。
6. 加入可行但不改善原式目标的情形才进入 replacement；新增预算或时限使加入不可行时，也尝试原式单元素替换。这是将原始 add/swap 机制推广到共同硬约束的明确规则。

实际计算的内部目标为

\[
F(S)=\Lambda\frac{2}{1+\exp(\sum_{i\in S}c_{g(i)i}/1000)}
+\sum_i Q_i\left(\mathbf{1}_{i\in S}c_{g(i)i}/1000-0.8\right).
\]

**零队列下的重要限制：** 原式 sigmoid 明确随总 selected delay 下降。Q=0 时，source add/swap 只按**已经由中立 assignment adapter 映射的** selected endpoint 总成本比较，不会自动变成 makespan 最小化；source replacement 仍可能以牺牲 risk 为代价增加该总成本。每次新增 GT 选择本身始终采用最小合法 intrinsic pair cost，不受这项反向成本激励影响。这里严格保留了原式 subset 目标方向，没有为了让基线更好或更差而改成本、反转目标或额外添加 J。该现象是原生单轮零队列与共同硬约束下的结果，不能解释为原论文长期方法的普遍性能结论。

这一 assignment adapter 是本组新增规则，不能称为 DVA-P 原论文的 verifier assignment。它在每次 add/swap 的当前容量条件下重新映射，而非永久绑定一个最低成本 GT；永久绑定会在该 GT 已满、其他 GT 仍有容量时不必要地拒绝 target。动态最小合法 intrinsic cost 保留资源利用机会，同时把负载仅作为硬约束。该扩展不是唯一可能的映射，结果解释必须保留本适配定义。

为避免 exponent 上溢，显示值按 `2*exp(-s)/(1+exp(-s))` 计算。零队列决策比较利用严格单调性直接比较总 delay，和最小化原式数学等价；不会因浮点 sigmoid 饱和而把不同代价误判成相同。式 (13) 仍在本轮结束执行一次；合法 pair cost 不超过 C，所以该轮零初值 queue 仍为零。这是约束下的自然退化，不补造虚拟积压。

不能直接移植或声称的内容：原论文 AV challenge-response 和 Shannon channel 物理层、原生 replica 完整性协议、未来资源统计未知条件下的多 slot 轨迹、前轮未验证补集长期优先性、非零队列稳态调节、长期平均目标、Theorem 5 性能界以及 Λ 的长期 trade-off。本组用统一 Dataset 描述符成本与可信 GT 池隔离联合决策能力，故图例虽显示 DVA-P，CSV/manifest 使用 `DVA-P-Adapted` 并保留适配版本。此实现应被引用为**具有上述明确映射的单轮适配**，不是原生完整复现；其他合理 pair 映射可能得到不同结果。

## 三个结构化对照

| 方法 | 目标选择 | verifier assignment | 是否使用内部 marginal J |
| --- | --- | --- | --- |
| RF-TS | 按 risk 降序、临时 ID 平局 | 对当前 target 选最小 `L_g+c_gi` 的合法 GT；verifier ID 平局 | 否 |
| Cost-First | 在所有未选 pair 中选最小 makespan 增量 | 与目标同时确定 pair；增量平局依次按 pair cost、target ID、verifier ID | 否 |
| Random | 对初始合法 target–GT pairs 生成一次均匀随机排列 | 依次接受 target 未选且当前 GT 累计负载合法的 pair，直到 B 或排列耗尽 | 否 |

RF-TS 对暂时不可安排的高风险 target 记录跳过并继续后续目标。Cost-First 完全不使用 risk 排名，但外部 J 仍正常计入所选 target 的风险。Random 不读取 risk 排名，也不刻意选择好或坏的资源；一次排列中每个 pair 的相对顺序均匀。被容量拒绝的 pair 不会因后续负载增加而重新可行，因此单次遍历具有明确停止依据。Random 使用独立、稳定派生的 seed。三个 baseline 均受相同预算/时限约束，可以合法少选；没有额外外生失败事件。

## 合成历史与解释范围

`build_historical_state(n, seed, config)` 仅接收数量、独立 seed 和公开参数。以正式 t=1 为当前轮，预历史轮为 -19 至 0；负轮编号仅保留 t=1 的正式实验设定，不表示执行长期实验。每模型每轮以 0.10 概率被历史观测；条件于发生事件，normal/anomalous/timeout-or-invalid 概率分别为 0.95/0.03/0.02。h 初值 0.10，更新使用 ρ=0.5、η=0.5，age 尺度 20，risk=0.5 age+0.5 h。仅正常或异常的有效状态判定刷新 last_verified；invalid 只更新 h。未被有效验证者 age=1。

所有方法读取同一次历史生成的风险，预算或 sigma 改变不会重新生成历史。历史数组和 `JointProblem` 数组使用只读 bytes backing，避免方法误改共享数值。这些合成风险与当前 corruption 标签独立，只允许说明对可观测风险的覆盖，不支持风险准确预测当前损坏的结论。

## 交付验证边界

本次仅完成完整代码实现及必要的静态检查，尚未运行实验、测试或绘图。正式结果将由用户在PyCharm执行后产生。尚未获得有效率、耗时、各方法 J、统计显著性或 Figure 渲染结果，也不预设 RACA-HIM 优于任何基线。
