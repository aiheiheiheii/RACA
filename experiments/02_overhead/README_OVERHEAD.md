# 第二组：时间与通信开销（overhead-core-v4）

当前时间实验保持原v4代码及已保存结果；通信实验增加独立的判决等价渐进编码适配。已有完整v4结果时，直接运行入口默认只回放通信并更新O2，原时间图/CSV/raw/checkpoint逐字节保留。需要明确只更新通信时运行：

```powershell
.venv/Scripts/python.exe -B experiments/02_overhead/run_overhead.py --communication-only
```

没有完整原结果时才先计算原v4时间；`--force`明确重测完整实验。已有通信清单时，`--plots-only`只重绘O2。通信独立摘要为 `summary/communication_encoded_summary.csv`；原 `communication_overhead_summary.csv`保留原未压缩账本，不再作为新的O2来源。

在PyCharm选择项目 `.venv` 解释器，直接右击 `run_overhead.py` 运行，也可从项目根目录执行：

```powershell
.venv/Scripts/python.exe -B experiments/02_overhead/run_overhead.py
```

默认输出到 `results/02_overhead/`，最终图片位于 `results/02_overhead/figures/`。v4改变规模扫描的部署fixture、计算计时及GT通知方式，必须重新计算；不复用v2/v3数值或checkpoint。正式运行将按v4生成该目录的结果文件，如需保留其中旧版文件，应先备份或通过 `--output-dir` 指定其他目录。

默认9个N×5个d×1个seed，共45配置、180方法记录。N=1000..5000每500，d=10/20/30/40/50，seed=2026，cr=3%、H=.5。四方法执行同一固定B个目标，B=ceil(.1(N−25))。没有多轮平均、native-budget附加运行或Quick模式；两图共享原工作量，通信采用独立的传输回放。

## 规模扫描与计时修正

旧资源生成器先抽N个节点再抽轮次变量，导致同seed在不同N得到不同轮次冲击。v4在第二组单独生成最大5000节点固定资源域，各N只取前缀；节点资源与轮次变量不随N重抽，sigma=.5的异质性和原公式保留。

描述符也使用同一最大域的嵌套部署：首1000节点，之后每批新增500。离线构造保持原加载器的损坏比例及80/20 subtype配额，原始float32行不变。匿名身份在每个到达批次内独立打乱，后续批次编号更高，已有身份不变；历史相同时参考排序优先较早到达批次，这是v4的受控部署假设。首批抽B(1000)=98目标，之后每批抽新增预算50；仅用匿名身份与双方参考排除集，不读取内容/标签选目标。GT仍可在全N域必要扩展；参考集不变时目标严格嵌套。

各seed/d仅在最大域标定一次cosine、SHA256、scalar comparison/modulo与vector操作的真实主机单位成本，四方法、全部N共享这套标尺和节点资源因子。每类一个预先规定的计时batch和一次预热，不属于额外实验轮次。单位成本、batch政策、输入hash与snapshot SHA256保存进manifest并纳入运行身份；断点恢复复用核验后的标定，不能半途更换标尺。

准备/决策/历史更新CPU实测，准备通信与验证完成时延按实际消息、操作和依赖模拟。准备CPU不重复计入模拟CPU，验证CPU采用共享标定；实测验证CPU另外保留为诊断。文件读取、离线fixture、标定、checkpoint、统计和绘图不计入在线时间。旧固定毫秒/吞吐字段仅供旧模拟器兼容，v4验证CPU由标定决定。

## RACA-HIM保留与简化

保留历史排序、多数支持GT、必要候选扩展、风险计算、greedy联合分配、单GT原始描述符验证及历史更新。K=50、M=25，必要时每25扩至成功或穷尽全域，不固定缩小候选域。

GT采用可信协调器适配：每个新增候选只上传一次4d字节原始描述符，归一化和已比较pair缓存，扩展仅算新增比较，但重新判断多数阈值。参与验证者的GT通知并入后续batch assignment固定64字节控制头，无单独membership消息；分配仍等待GT判断完成。

greedy重用矩阵缓冲区，避免每次迭代分配多份数组，分配规则、tie规则和prefix结果不变。实际目标cosine缓存重复参考的归一化，历史更新向量化。独立测试与原表达式逐位核对，不删除风险计算或必要目标任务。

保存的v4原始时间实验使用batch dispatch/result，每target请求64字节，RACA收到完整4d描述符，batch assignment保留每目标4字节身份。CTCV使用自身SC最终结果。该原始逻辑和账本继续保留；新增通信适配的实际字节见下一节。

原始payload大小为RACA描述符4d字节、DVA完整SHA256 32字节、I-EDI默认7919模标签2字节。原始账本是通信编码回放的核对来源，不直接给新编码协议指定排名。

## 渐进通信编码

RACA原float32输入不修改，仅对其验证所需的方向进行int16量化；scale为实际携带的上舍入float32。首次发送高8位及6字节mode/dim/scale头，接收方按高位中点重建方向，并用严格角度误差界计算原cosine上下界。只有整个区间确定原包含端点tau判决时才结束；否则发送精化请求与低8位，仍不明确再请求完整原float32。GT也按收到的候选包、必要精化、多数支持及扩展处理，最终参考集与原记录核对。

这是保二元判决的渐进传输适配，不是描述符逐位无损编码，也不是对原文实现的逐步复刻。完整raw回退逐位还原；哈希基线的输入、proof、判决及已有compact通信不变。精化、回退、控制头、任务身份、bitmap均按真实序列化长度收费，不乘人为折扣。

包采用4字节version/opcode/body-length应用层头，sender/receiver与轮次通过原独立link/配置上下文绑定；同原统计范围，不额外计TCP/TLS/链路层头。低位包须与同一目标的初包绑定，不提供独立密码认证。消息账本写入 `logs/raca_encoded_packet_ledger.csv`，每条有端点、阶段、实际长度和包hash；独立来源在 `logs/communication_manifest.json`。

时间图仍衡量原v4未压缩传输；通信图衡量新增编码传输的实际应用层字节。编码CPU、精化往返时延没有加入冻结的旧时间图，不能把两者解释为同一新压缩协议的端到端性能。修改论文时应披露这两个实验的传输设置和迁移假设。代码与核验说明见 `reports/02_overhead_communication_revision.md`。

## 基线与统计边界

CTCV保留ell=1000B、k+1 collectors、producer proof fanout、packed CR/R、AV.Inquiry与四Flag/成员位定位。20字节请求携带seed/块数/长度/cyclic start，在目标实际重建排列、执行原字节重排和SHA256/modulo；仅缓存同一请求与正确参考的重复挑战，不缓存目标响应/判定。基线不读相似度、标签或RACA的GT。详细迁移说明保留于 `reports/02_overhead_baseline_adaptation.md`。

四方法必须完成同B才形成可比较点。800ms仅是独立deadline诊断，不截断固定工作量；success表示完整获得最终结果，超时数单独记录。固定B适配旁路原生动态subset/预算选择及RACA best-prefix选择，不用于评价原生selector优势。这是披露的关键步骤协议适配，不等同完整原部署/密码系统端到端性能。

每点一次配对观测，summary的value及兼容mean列保存同一原值，std/CI未定义。不会累积取最大、平滑或修改观测满足预定趋势/排名。

## 输出与恢复

只生成两张逻辑图，各保存PDF与600dpi PNG：

```text
results/02_overhead/figures/Fig_O1_time_overhead.pdf/png
results/02_overhead/figures/Fig_O2_communication_overhead.pdf/png
```

画布7.16×1.50英寸，子图绘图区82.70578×64.26252pt，与01_detection一致。沿用第一组字体/间距，RACA-HIM图例、全部实线、无阴影；纵轴为Time Overhead (ms)、Communication Overhead (KB/MB)，完整显示且从0开始。

保存紧凑raw/overhead_runs.csv、protocol_diagnostics.csv、两份summary、checkpoint及清单。core_phase_counts_json包含阶段精确字节/消息/操作；diagnostics保留资源/部署hash、参考集、目标与标定。没有逐消息/任务大CSV。

完整原v4计算恢复同版本、计算源、入口AST、配置、数据、环境、标定snapshot与网格匹配的checkpoint。通信回放另核对原任务分配、原工作量和冻结文件hash；旧版checkpoint不参与。Windows原子写入/访问错误重试及目录锁保留，O2缺失或冻结文件变化会报错，不宣告交付成功。通信回放不会重新测量算法时间。

```powershell
# 完成本版后，只重绘
.venv/Scripts/python.exe -B experiments/02_overhead/run_overhead.py --plots-only
```

`--force`重算当前目录；`--output-dir`选择独立目录。原v4代码阶段的d30内存核验见 `reports/02_overhead_v4_revision.md`。本次已在用户完整v4结果上仅回放45点通信并更新O2：各点RACA通信减少70.13%–74.15%，全部GT/目标判决相同，三个基线字节不变，原时间图/CSV/raw/checkpoint保留；详见 `reports/02_overhead_communication_revision.md` 和对应validation.json。这里的排序属于上述明确传输适配，不代表原作者协议或其他压缩设置的普遍排序。
