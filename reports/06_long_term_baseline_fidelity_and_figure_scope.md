# 第六组基线原文一致性及三方法正式图

> 后续修订：用户已选择 DVA-P+EDI-V，现已新增独立 native 协议代码；最新说明见 [06_long_term_native_protocols.md](06_long_term_native_protocols.md)。下文保留为选择 EDI-V 之前的旧适配结果审计记录；其中“只改绘图、不需重新计算”仅适用于旧 Adapted 曲线，新协议必须重新计算。按用户最新要求，所有重绘统一输出到已有 `results/06_long_term/figures/`，旧图备份也平铺在同一目录，不新建目录。

本次按用户后续要求复核基线判定机制、判断图形类型，并将 Figure L1 的显示范围收窄到 RACA-HIM 与两种已发表基线的场景适配。未运行实验、测试或绘图；原 CSV、checkpoint、run manifest 及现有 PDF/PNG 均未改写。

## 1. I-EDI、DVA-P 是否用了余弦判定

没有。`long_term_methods.py` 的协议分派为：

| 方法 | 执行协议 | 当前实际判定 | 是否使用 RACA-HIM GT/cosine |
| --- | --- | --- | --- |
| RACA-HIM | cosine | 目标与当前协同 GT 的余弦值是否在 [0.997,1.0] | 是 |
| I-EDI-Adapted | iedi_tag | 当前 `<f4` 描述符 SHA256 取模 7919，与该模型 t=0 登记 tag 比较 | 否 |
| DVA-P-Adapted | sha256 | 当前 `<f4` 描述符完整 SHA256，与该模型 t=0 登记 digest 比较 | 否 |

`dynamic_simulator.py` 的 cosine、iedi_tag、sha256 是独立执行分支。两种 hash 基线不调用协同 GT，不读取真实损坏标签，也不借用 cosine 输出。

## 2. 不能将现有实现称为“完全原文”

原文核查来源为用户提供的 I-EDI、DVA-P PDF，抽取文本位于 `tmp/long_term_sources/iedi.txt` 与 `dvap.txt`，论文 SHA256 见本组 baseline adaptation 报告。

### I-EDI

原论文 PDF 第 8 页，Algorithm 2 / Eq.(39) 定义 `hash(s_i) mod eta_i` 并比较 vendor 正确 tag 与服务器响应 tag；第 9 页实验使用 SHA256、7919。当前实现保留实际有限模计算，但 Algorithm 2 line 4 和正文还要求逐模型、逐轮随机产生 challenge 参数 eta_i；当前固定 7919 没有实现这一完整交互。

选择阶段保留 O²RS Algorithm 1、Eq.(4) 的 VR 形式、Eq.(8) 的组合形式，以及 Eq.(12) / Eq.(20) 的队列与目标。原文 PDF 第 4 页 Eq.(1)–(3) 的服务 QoS 属性及第 9 页 QWS 吞吐量/响应时间，在当前模型描述符数据集中并不存在；现代码使用自己的过去验证 timeliness、平滑 SR/ER 代理。另外，共同 hard 最大预算、cap 截断、shortfall 修正和 peer processor 映射也属于公开适配。

原文 Eq.(39) 的 `H(m) mod eta` 与第 9 页 Theorem 6 的 `H(eta || m)` 不是同一个定义；不能混合二者后把当前实现说成完整 nonce 安全协议复现。

### DVA-P

DVA-P 本身主要定义长期巡检子集/任务选择及资源调度。当前保留原文 PDF 第 7 页 Algorithm 1 的 priority、add/replace，以及第 5 页 Eq.(13) 的 queue 和 Eq.(21) 的目标；但新增的 peer verifier 映射、Age 遍历、共同 B 上限和累计 processor 容量属于当前场景的适配。

原文 PDF 第 9 页 Section V-B1 将 DVA-P 分别与 EDI-V 或 EDI-S 集成：EDI-V 使用 modified Merkle hash tree，EDI-S 使用 elliptic-curve integrity proofs。当前单纯 SHA256 descriptor wrapper 没有复现其中任一完整底层协议。因此不能把它称为 DVA-P 原论文端到端验证的完全复现。

### 严格复现的实际边界

仅凭现有描述符，无法恢复原缓存服务的 QoS/检索日志、原数据块、挑战消息和 Merkle/椭圆曲线证明对象。若要求原协议完整复现，必须先确定 DVA-P 的 EDI-V 或 EDI-S 底层组合、取得相应协议原文和输入，另行实现并运行；原文 QWS/QoS、资源与当前模拟环境的差异仍需披露。

本次没有为了宣称“完全原文”而改名旧结果或补造输入。现有结果继续保留 `I-EDI-Adapted`、`DVA-P-Adapted` 的内部方法身份；新图例也显示 `I-EDI (Adapted)`、`DVA-P (Adapted)`。它们不能作为完整原生协议的性能结论。

余弦相似度本身是通用数学工具。论文可强调面向合法模型异构的协同参考构建、相似性判定及风险/资源联合决策，而不能把通用余弦公式本身表述为首次提出。

## 3. Figure L1 保留折线图

已查看用户现有结果 PNG。横轴为连续验证轮次，研究重点是累计发现过程、增长速度与进入平台的时间；五个事件比例面板应继续使用折线图。每条曲线保留 0、20、…、200 的 11 个真实统计点，不插值生成测量点、不平滑。

改为每面板 3×11=33 根分组柱，会明显增加五面板的密度；只画第 200 轮三根柱则丢失动态发现过程。终点柱状图回答的是“最终发现了多少”，与本组“长期如何演化”的问题不同，因此没有切换到条形/柱状图。

## 4. Age-Only、Random 的处理

修改仅位于 `experiments/06_long_term/plot_long_term.py`：

- `DISPLAY_METHODS` 固定为 RACA-HIM、I-EDI-Adapted、DVA-P-Adapted。
- 曲线、图例、逐显示点证据与样式只包含这三种方法，图例列数为 3。
- 读取与校验仍完整认证原五方法 summary，不把 750 条原始轨迹篡称为一次新三方法运行。
- Figure manifest 分别记录 `source_methods`、`display_methods`、`omitted_methods`、选择原因和比较范围。
- 原 Age-Only/Random 的 CSV、checkpoint 与算法实现保留以供复核，不删除不利于主方法的证据。当前主图范围是与已发表适配基线的比较，不能据此声称主方法优于被省略的基础对照。
- 用户重绘时，若现有 Figure 的方法集合不同，先将旧 PDF、PNG 和原 figure manifest 按字节校验归档到 `figures/archive/`，再生成新版；不改动任何仿真输出。

未修改 config、方法、资源或指标代码，所以不会改变计算协议/源码指纹，也不需要重新计算已完成的三种方法。

## 5. 用户运行与本次校验

在 PyCharm 直接右击 `experiments/06_long_term/plot_long_term.py` 运行即可读取已有 summary 并生成三方法 Figure L1。也可自行执行 `run_long_term.py --plots-only`。不需要 `--force` 或正式实验重跑。

本次只执行本组 10 个 Python 文件的 AST/代码对象编译及静态边界检查，没有导入实验模块；并只读核查来源身份。绘图入口没有实际执行，所以当前磁盘中的原五方法图仍保持原样，三方法图将由用户重绘后产生。
