# 第二组通信编码修订

本次保留已完成v4的所有时间观测、原raw、checkpoint、原统计及O1 PDF/PNG，只增加独立通信回放、编码账本和O2图。原文件预先SHA256冻结于 `02_overhead_time_frozen.json`，执行时再次检查；1350个checkpoint中仅manifest对应的45个用于回放，其余旧文件不参与计算。

## 实际压缩方式

RACA需要验证的是描述符方向间的cosine。编码器对原float32行进行同样的float64归一化，采用上舍入float32 scale量化到signed int16，先发高8位和6字节mode/dim/scale头。高位区间中点为high×256+127.5，量化舍入误差加中点误差不超过128scale；接收者用每维误差、范数与保守浮点界转换为角度界，再计算cosine上下界。

只有整个区间已经确定原tau=[.997,1]的包含端点判决时才接受。否则真实发送请求和低8位；仍不明确时请求原始float32。收到raw才允许使用其准确内容。这不是逐位无损描述符编码，而是保持本组二元判决的渐进传输适配；没有改原阈值、原输入或哈希基线。

GT多数支持采用上下界短路：每候选L=1+已证实支持数，U=L+未认证邻居数；L达到原多数阈值才接受，U低于阈值才拒绝。最早M个接受者之前所有未知候选必须处理，避免改变历史/匿名身份排序。只有接受者及未知者总数小于M才证明需要扩展；扩展后重算阈值。无需补齐已经不影响最终成员的全部候选低位，完整支持计数不作等价声明。选出的GT及每目标判决、历史状态hash与原结果分别核对，原X仅用于发送端编码、本地参考和决定后的审计。

## 字节与比较口径

每条消息实际序列化：4字节version/opcode/body-length帧；query body为维度/精度两字节；assignment用varint身份和zigzag delta保留队列原顺序；result用带任务数量的bitmap。所有初包、精化请求、低位、raw请求及raw回复都按实际len计账，没有通信折扣或预设曲线排名。

统计应用层字节，sender/receiver/round由原独立link上下文绑定，不额外计TCP/TLS/链路层/认证开销；与旧实验相同，不模拟恶意发送者伪造编码或信道攻击。低位包自身不提供身份认证，须与同目标初包绑定。三个基线保留原v4已公开的compact适配账本，不宣称有原作者源代码或所有方法使用同一新增编码层。因此比较的是这些具体适配设置。

时间图仍属于原v4未压缩传输，通信图属于新增渐进编码传输。编码CPU、额外精化往返和该协议端到端时延未测入冻结时间图，不能将两图解释为同一新压缩协议的端到端性能；论文应披露此区别。

## 来源、文件和运行

原 `raw/overhead_runs.csv`、`summary/time_overhead_summary.csv`、`summary/communication_overhead_summary.csv`、原run_manifest及checkpoint不改。新增：

- `raw/communication_encoded_runs.csv`：原字节、编码字节、节省量、精化/回退及判决核验。
- `summary/communication_encoded_summary.csv`：O2专用单观测摘要。
- `logs/raca_encoded_packet_ledger.csv`：端点、phase/tier、真实帧长及包SHA256。
- `logs/communication_manifest.json`：原来源、独立codec/policy/source指纹、冻结文件和输出hash。
- `logs/communication_plot_style.json`：独立O2图形来源。

直接运行入口时，有完整v4结果则默认只回放通信。显式命令为：

```powershell
.venv/Scripts/python.exe -B experiments/02_overhead/run_overhead.py --communication-only
```

O2继续保存在 `results/02_overhead/figures/Fig_O2_communication_overhead.pdf/png`。已有编码清单时 `--plots-only`只重绘O2；`--force`才重测完整原v4时间。对原计时代码的改变会拒绝回放，不自动混合旧时间与新算法。

## 检查

完整第二组测试652项通过（9.37秒，无警告），包括codec边界/极端值、必要回退、早序未知GT、必要扩展、真实len与roundtrip、源/输入变动拒绝、输出失败不宣告完成、完整重绘生命周期及只更新O2路由。随后仅回放原45点通信并生成O2，未运行新的四方法时间实验。

完整回放中，RACA各点通信减少70.13%–74.15%，总计3,045,255→859,373字节（减少71.78%）。45/45点编码字节低于三个保留的基线；五个d面板均随N递增。原GT成员、13410个目标的判决和更新历史hash全部一致，三个基线的135条字节观测未改变。验证阶段精化2215次、完整raw回退5次；GT精化918次、raw回退0次，均收费。

| N=5000 | d=10 | d=20 | d=30 | d=40 | d=50 |
|---|---:|---:|---:|---:|---:|
|RACA原字节|61,458|83,378|105,298|127,218|149,138|
|RACA编码字节|16,677|23,268|29,521|35,879|42,885|
|保留的I-EDI字节|39,983|40,983|41,983|42,983|43,983|

原run_manifest、raw/overhead_runs.csv、时间summary、原通信summary及O1的PDF/PNG全部SHA256与冻结值一致；45活跃checkpoint完整SHA及1350个archive路径/size/mtime检查通过。新独立通信指纹为 `cf3f21b38097bd583e3d58a8304f7232d2755cfdf288e34dc56d1a6bb4557098`。细项见 `02_overhead_communication_validation.json`。

O2已核验PDF/PNG存在且非空，PNG为4296×900，与第一组7.16×1.50英寸、600dpi一致。目视确认五子图完整、字体/标签/图例无裁切，全部9个真实N点保留，RACA在各面板下方；相近曲线按原值显示，没有移动点或扩大差距。新图已在 `results/02_overhead/figures/`，图形来源与冻结时间来源各有清单。
