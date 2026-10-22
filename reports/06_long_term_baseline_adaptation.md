# 第六组长期动态验证：基线适配说明

> 本文件记录旧五方法适配运行。用户后续指定 EDI-V 后，已新增随机 challenge 的 I-EDI 与真实 VMHT 证明的 DVA-P+EDI-V；最新代码边界与独立结果目录见 [06_long_term_native_protocols.md](06_long_term_native_protocols.md)。下述旧固定模数、SHA256 wrapper、Age 排序和 shortfall 修正不适用于新 native 分支。

适配版本：`long-term-native-feedback-adapters-v1`。本文件记录实际源码和公开假设；本次没有运行算法、实验、测试或绘图，因此不包含实测性能结论。

后续原文一致性复核明确：I-EDI 当前固定模 tag 未实现 Algorithm 2 的完整随机 challenge；DVA-P 当前 SHA256 wrapper 也不是原实验的 EDI-V/EDI-S 完整证明协议。两者均未使用 RACA-HIM 余弦判定，但不能称为“完全原文”。三方法正式图、明确 Adapted 标签及输入缺口详见 `06_long_term_baseline_fidelity_and_figure_scope.md`。原五方法实验与消融数据保留。

## 论文与可复用实现

依据为本次提供的最新 TNSE Markdown、I-EDI 原论文 *Intelligent Edge Data Integrity Verification With Dynamic Unreliable Data Replica Selection*（TIFS 2025，DOI 10.1109/TIFS.2025.3638138）及 DVA-P 原论文 *Long-Term Over One-Off: Heterogeneity-Oriented Dynamic Verification Assignment for Edge Data Integrity*（TMC 2024，DOI 10.1109/TMC.2023.3294180）。本次重新从用户提供的 PDF 提取文本至 `tmp/long_term_sources/`；旧文本仅作交叉阅读。`TMC.pdf` 的视觉规范与公共绘图样式用于图形协调，不作为 RACA-HIM 或基线机制的替代定义。

| 附件 | SHA256 |
| --- | --- |
| 最新 TNSE Markdown | `61de825be020a3196d3f175fedfa7a82c6c87ddc6ad58caa3a5e0aaf4e9ac2c3` |
| I-EDI PDF | `c07a2116294b725384ea21575a68cfd81e534c06d9727a367dc155ddc35e3256` |
| DVA-P PDF | `8066c88e06d528f81da2058137f6ccd5ce870056bb29f9486d6dfbcc61ddb11a` |
| 视觉参考 TMC.pdf | `320cf6280cd643841e4ac1ace1f20ed7d51b8ffa95c26cd956d4bc34f134ddcb` |

RACA-HIM GT 复用第一组 `proposed_detection.build_ground_truth` 的三分之二 admission 与 history-consensus 排序。联合决策复用第五组 `joint_methods.raca_him_greedy` 的 manuscript marginal-J / best-nonempty-prefix 核心，以及第五组共同问题数据结构；不调用第五组后续额外 v2 refinement。I-EDI 的二元选择核心复用第一组 `iedi_adapted._o2rs_exclusion_greedy`；不复用第一组仅为静态检测设计的“多数完整 hash 估计 anchor”，因为本组有明确的每模型 t=0 可信登记。

## 共同输入与协议差异

五方法共享每个场景唯一的动态 trace、onset/recovery、normal/corrupted 描述符、节点失败矩阵、资源 profile、βB=0.10、200 轮和 800 ms hard verification deadline。各方法独立持有 h、Age、有效验证时间、反馈计数和适用队列。其他方法的结果不成为当前方法的公共观测，真实事件位置及未来失败值不进入调度。

RACA-HIM、Age-Only、Random 每轮协同确认 M=25 个 GT，GT 身份不作为 target。I-EDI、DVA-P 使用自己的 t=0 tag 作为内容参考；另由不读取内容/标签的中立 adapter 按 temporary ID 顺序逐轮轮换 25 个 processor，当前 processor 身份从本轮 target pool 排除。轮换避免固定 25 个模型在整个实验中永远不能作为 target。processor 不被宣称为通过 RACA-HIM admission 的可信内容 GT，其职责是计算目标的 tag/digest 比较。

所有方法正常时均以 N−M=975 为 target 域、B=98 为预算上限。不同参考机制导致不同当前合法目标池；这属于公开协议差异，未按真实事件剔除或优先任何模型。GT-based 方法已在候选交互时确认其最终 verifier 本轮可响应；原生基线的 processor 不预探测，其 target/verifier 任一端的外生失败都在执行时暴露。因此不能仅由相同 p_fail=0.05 推断五方法实际任务成功概率完全相同。

每次失败仍消耗预留完整 pair 模拟时间与一次预算，不免费补选。相同 pair 同轮成本完全一致；串行负载按 verifier 累计、并行 makespan 取最大值。deadline 约束 target 验证阶段；GT 另记录实际候选请求数、失败响应数和 peer 比较次数，不将候选观察计为额外 target 完整验证。当前未给 GT 构建另加毫秒成本模型，故该 800 ms 不应解读为包含 GT 预处理的端到端整轮时延上界。

## I-EDI-Adapted：保留 O²RS 与真实有限模 tag

### 保留的原文部分

原论文 Eq. (4) 使用 `VR_i=(1+SR_i−ER_i)/2`，由 QoS 与 VR 形成 Eq. (8) 的可靠性表示。长期 L-URS 选择以虚拟数量队列处理最小巡检数量约束，Eq. (20) 的确定性目标为：

`Ω + Υ Σ_i d_i R_i + Q(γ−Σ_i d_i)`，其中 `Ω=γ²/2`。

实际复用的 Algorithm 1 核心先将当前 target 域全部选中，逐候选尝试把 d_i 从 1 改为 0；仅在新目标严格小于已存 minimum 时接受排除。遵循原文伪代码 line 10，Eq. (12) `Q←max(Q+γ−selected_count,0)` 放在候选扫描内部，且不在队列变动后偷偷重算已存 minimum。该实现不是将 I-EDI 改名为 RACA-HIM risk ranking，也没有替换成另一个闭式阈值解。

初始 Q=0，跨真实轮次保留。固定 Υ=1.0，γ=`min(B,max(1,ceil(0.10|target_pool|)))`，正式通常为 98。原文最小数量约束与本组共同 hard 最大预算具有不同方向，适配限制如下。

### 可见历史 QoS 与可靠性代理

TNSE-SimSimple-v1 没有原论文 WS-DREAM 缓存吞吐量、服务响应时间及 retrieval-error 日志。本组只用本方法实际过去验证反馈建立代理，不将当前真实损坏状态转换为 QoS 降级。

* `SR_i=(Normal_count_i+0.5)/(Normal_count_i+Corrupted_count_i+1)`，表示本方法既往有效完整性判定中 Normal 的平滑比例。
* `ER_i=(failed_count_i+0.5)/(attempted_count_i+1)`，表示既往实际交互失败的平滑比例。
* 一次有效任务的 timeliness 为 `max(1−finish_ms/800,0)`；失败任务为 0。finish_ms 包含该 verifier 上此前排队负载，属于当前实际任务反馈。
* 轮次 t 对历史观测 `(u,x)` 使用原文长期加权思想的公开参数化：`w(u,t)=0.5+0.5 exp((u−t)/20)`，QoS 为本方法历史 timeliness 的加权均值。无历史时 QoS=1.0，SR=ER=0.5，VR=0.5。
* 可靠性 `R_i=0.5 QoS_i+0.5 VR_i`；QoS 固定归一化区间为 [0,800] ms，不从未验证模型的本轮真实表现估计 min/max。

这些代理与原文对缓存服务的观测语义不同，不能据此宣称原文所有实验结论仍适用。其信息来源严格限于自己的过去尝试，未被该方法验证的模型没有虚构观测。

### 遍历、hard budget 与 pair adapter

原文未规定候选遍历次序。本组固定按历史可靠性降序、Age 升序、temporary ID 升序遍历传入原生 exclusion 核心，使排除尝试先作用于较可靠的模型。原生选择是在全选状态进行排除，该遍历不等同于“直接取最低风险/可靠性前 B”；报告和诊断保留 source selected count、接受排除数、初末目标和 stored minimum。

若原文核心给出超过 B 的 source-selected 集合，只在该已选集合中按历史可靠性升序、Age 降序、temporary ID 升序保留前 B 个，不将 source-excluded 模型重新补入；在资源不足时也不填补不属于原生选择的目标。为这些目标使用**预计完成时间最小**的可行 processor，中立 adapter 只考虑 `current_load_g+pair_cost_gi` 与 hard deadline，不使用 RACA-HIM marginal-J。

原生扫描完成后，以 `source_selected_count−actually_dispatched_count` 向 Q 加入一次公开的容量短缺修正，防止被本组 hard cap / 资源约束阻止的原文选中任务被视为已完成巡检。该项是本组适配扩展，不是原文 Eq. (12) 本身。原生扫描队列变化和短缺修正分别保存；不因希望获得特定曲线而重新初始化队列。

### trusted enrollment 与 Eq. (39)

每个模型在事件发生前的 t=0 正常初始化阶段，按自己当时可见的原始 `<f4` little-endian 字节执行一次 SHA256 登记，只保存其独立 digest。此后 expected tag 不更新，也不回读 `normal_X`、prototype 或事件表。

实际 tag 按原论文实验路径 Eq. (39) 计算：`int(SHA256(current_descriptor_bytes),16) mod 7919`；与本模型自己的登记 digest 所得模 tag 精确比较。没有添加 cosine 容差，没有人为更改 tag 输入或制造 miss。模碰撞若发生可能导致漏检，代码保留其真实结果。

这是具有可信初始登记条件的 tag 适配，不假定现实部署天然存在可信原始副本。它保留论文 Eq. (39) 的确定性有限模路径，没有完整复现随机 challenge、nonce 安全证明、多副本密码学交互或真实攻击者。不能继承原论文针对伪造、外包、串谋、重放和自适应攻击的完整安全保证。

## DVA-P-Adapted：异构二元 subset 与长期 delay queue

### 原文作用域

DVA-P 原文以每个 replica/server 的二元巡检决定 d_i 为变量，优化长期验证数量与 delay queue；它并未提供本组“每个 target 另绑定一个 peer verifier”的独立决策变量。因此不能把某种新 peer assignment 称为 DVA-P 原生完整机制，也不能将其简化为 Random/Round-Robin。

本组保留原文 Eq. (21) 的确定性漂移目标（常数 Ω 不影响比较）、Algorithm 1 的 previous-selected/complement priority 初始化及逐候选 add/replace、Eq. (13) 的每目标长期 delay queue。毫秒资源成本在其原生目标和 queue 中转换为秒，避免 `exp(sum_ms)` 数值溢出。固定控制参数 Λ=1.0，初始所有 Q_i=0。

使用的决策值为：

`Λ·2/(1+exp(Σ_i d_i c_i)) + Σ_i Q_i(d_i c_i−C)`，`C=0.8` 秒。

用等价 `z=exp(−Σ d_i c_i)` 计算 sigmoid 项，提高稳定性；Q 全零时按其解析单调关系比较 summed endpoint cost，避免浮点饱和产生伪平局。此值是 DVA-P 原文的 subset surrogate，**不能解释为本组 makespan 的直接最小化目标**。

### 本组的 target / verifier / capacity adapters

按 Age 降序、temporary ID 升序遍历候选是公开的中立 target priority adapter，不使用 RACA-HIM h/risk 做 target ranking。原文 add/replace 是否接纳仍由上式原生目标决定，不能将全部 Age 前 B 目标直接冒称 DVA-P 结果。

previous-selected 与其在当前合法域的 complement 均按原文思路构造；只有完整 complement 本身满足共同 B 与累计 verifier deadline，才允许以它完成 priority 初始化，否则使用原文 empty fallback 继续 add/replace。hard cap 不在初始化阶段被绕过，也不把 oversized complement 静默截断后冒充原文完整 complement。

每次原文 add/replace 尝试需要的 pair 分配由中立 adapter 补上：从当前累计负载不超过 800 ms 的 processor 中选择 **intrinsic pair cost 最小**者，用 temporary ID 处理同成本平局。replacement 先移除旧任务的相应 verifier 负载，再检查新任务的可行 processor。该 adapter 使用计算、传输和网络成本，不读取损坏标签，不引入 RACA-HIM J，也没有任意概率或 Round-Robin assignment。

这与 I-EDI/消融方法的“最低预计完成时间” adapter 略有不同：DVA-P adapter 保留原生 endpoint cost 对 subset 目标的含义，只以累计负载筛除不可行 pair；区别明确记录在逐轮诊断中。

### 长期队列及本组 hard deadline 的影响

跨轮更新 `Q_i←max(Q_i+consumed_i−0.8,0)`，selected 任务包括失败尝试的完整 pair 成本，未 selected 的 consumed_i=0。队列不使用真实损坏或其他方法结果，也不人为制造 backlog。

本组还要求更强的每轮 verifier accumulated-load hard deadline。由此每个合法单任务 `consumed_i≤0.8`，从 Q_i(0)=0 出发，按该更新规则 **DVA-P 队列可以始终为零**。这是原文长期平均 delay 约束在本组更强 hard 约束下可能成为冗余约束的结构性结果，并非证实已运行观测。本实现保留 queue 更新与记录，不添加虚构延迟、额外噪声或人为 backlog 使其“看上去动态”。即使 queue 为零，当前轮异构 pair 成本、processor capacity 和原文 add/replace 仍影响决策；不能据此把实验宣称为对原文活跃 queue-stability 优势的充分验证。

### 验证 wrapper 与安全边界

DVA-P 主要解决 assignment/subset，原文可与不同 EDI 协议集成。本 Dataset 没有完整密码学证明对象，本组采用同样 t=0 合法可信登记来源的中立 SHA256 wrapper：当前目标 `<f4` 字节 digest 与本模型自己的登记 digest 精确比较。它不使用 RACA-HIM GT/cosine，不查询当前标签或未来正常描述符。

wrapper 用于真实计算完整性判定，不冒充已复现 Merkle tree、同态 tag、挑战应答、密码学通信负载或原论文底层 EDI 协议安全保证。DVA-P 在本组的名称因此始终存为 `DVA-P-Adapted`，图例仅显示简写 `DVA-P`。

## Age-Only 与 Random

Age-Only 是本研究定义的 RACA-HIM 目标选择消融：只按自己的 Age 降序和 temporary ID 处理 target priority；h 仅供相同 GT 协议及独立历史更新，不能进入 target ranking。Random 用独立随机流均匀打乱当前合法 target pool。二者均按当前预计完成时间最低的可行 GT verifier 分配，满足相同预算/deadline，执行与 RACA-HIM 相同的 GT-based cosine 判定，不故意选择昂贵 verifier 削弱基线。

## 结果应如何解释

正式 Figure 的 CDR 是全部 200 轮事件固定分母上的首次正确发现比例，I-EDI 的信任登记与 finite-mod tag、DVA-P 的 exact digest、RACA-HIM/消融方法的相似度参考具有不同检测能力。相同 descriptor 可能被不同检测器给出不同判断，真实轻度损坏也可能仍处于 cosine 允许区间。

因此结果是公开适配与初始信任假设下的长期端到端发现效果，不是仅比较相同分类器下调度效率，也不是原论文密码学安全的复现实证。基线因为可信初始 tag 在部分场景更准确属于可能的真实结果；方法曲线接近、Age-Only 更均衡、DVA-P 队列不增长均不能作为修改 Dataset、seed、event duration、判定或输出数值的理由。

本次仅完成完整代码实现及必要的静态检查，尚未运行实验、测试或绘图；正式结果由用户在PyCharm中执行后产生。
