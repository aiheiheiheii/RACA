# 第三组阈值敏感性实验：实现说明

## 当前交付状态

初次交付完成代码实现及静态检查，未执行实验、测试和绘图。用户随后完成 1650 条正式实验记录，实际运行报告为 `03_threshold_report.md`。
2026-10-13 后续版式修订：按用户要求统一前三组子图尺寸，仅从已保存 CSV 重绘 S1；没有重新执行实验。排版核查见 `03_threshold_layout_revision.json`。

## 附件及聊天依据

已阅读桌面附件 `Prompt - Codex.txt`、`Latest version of the paper.md`，并阅读“损坏识别性能”与“Overhead”聊天的用户要求。
TMC.pdf 第 9 页 Fig.3 和第 10 页 Fig.8 已抽取并查看：参考统一配色、五子图、共享图例和紧凑布局。
初次第三组遵循附件对折线图独立尺寸的要求。用户随后明确要求与第一、第二组子图宽高一致，因此当前复用其 7.16 × 1.50-inch 画布和物理绘图区尺寸；第三组仍使用完整连续纵轴。
既有聊天中关于其他实验性能趋势与压缩通信的要求不转化为本组的指标调整规则。
最新稿件 IV-A/IV-D 的文字及公式与第一组 GT/单 verifier 判定一致；稿件未提供单独的 Algorithm Pseudocode 段落，按实际公式和既有源码实现复用。
本实验的相似度区间 `[tau_lower, 1]` 是判定容许区间；统计均值的 95% CI 是另外计算的种子层面统计区间，两者不混淆。

## 创建的源码及兼容性

`experiments/03_threshold/` 包含：

- `config_threshold.py`：完整正式参数及静态协议检查。
- `threshold_experiment.py`：每个 (cr, seed) 的 11 个独立阈值分支、GT/预测诊断。
- `threshold_metrics.py`：原始记录校验及种子层面统计。
- `threshold_storage.py`：源码/数据/CSV 身份、原子保存和只读重绘校验。
- `run_threshold.py`：PyCharm 完整运行入口、归档与运行报告。
- `plot_threshold.py`：PyCharm 独立重绘入口及 S1。
- `__init__.py`、`README_THRESHOLD.md`：包及使用说明。

没有修改公共模块、第一组检测代码、第二组 Overhead 代码、原始 dataset、第一组或第二组结果。
没有修改 `run_all.py`；本组正式执行入口为 `run_threshold.py`。
第一组原默认阈值 0.997 与第二组行为保持原有源码。

## 数据及直接复用的算法

复用 `common.dataset_adapter.load_evaluation_data()`、`AlgorithmInput`、`EvaluationData`；其内部使用原始 `load_group()`，不查找虚构的物理配置文件。
固定 N=3000、d=30、H=0.5；cr=1%–5%，seed=2026..2055；每个 (cr, seed) 只加载一次。
流 1101 固定参考/评估池为 100/2900，流 1102 固定临时 ID；历史统一初始化 0.10。
各阈值共享描述符、标签、池、模型 ID、候选排序和初始历史，并对这些输入保存指纹、检查不变性。

复用 `experiments.01_detection.proposed_detection.detect_proposed()`，由其调用原 `build_ground_truth()`：

1. 按历史和临时 ID 排序选候选。
2. 使用传入的 tau 形成候选余弦支持。
3. 按实际 K 的 `ceil((K+1)/2)` 判断多数；K=50/75/100 的门槛为 26/38/51。
4. 选排序最先的 M=25 个通过者；不足则在同一参考池扩展，耗尽后失败，不重新抽样。
5. 对所有固定评估目标按第一组的循环规则分配单个 GT verifier，用相同 tau 判定正常/损坏。

每次分支使用 `replace(BASE_CONFIG, tau_lower=tau)`，重新执行 GT 决策；不继承 GT、预测或异常分数。
算法只收到无标签的 `AlgorithmInput`；标签仅在结果返回后计算指标和 GT purity，prototype 等字段不进入检测。
固定 100 参考池和单 verifier 循环分配属于第一组静态评估协议，不声称是整个系统候选域上限，也不执行资源预算或长期调度。

## 指标与失败处理

调用既有 `common.metrics.binary_metrics()`，Positive=Corrupted=1。
Accuracy=(TP+TN)/total；Precision=TP/(TP+FP)；Recall=TP/(TP+FN)；F1=2TP/(2TP+FP+FN)；FPR=FP/(FP+TN)；FNR=FN/(FN+TP)。
零分母采用 NaN 并记录具体指标；失败的 Accuracy、F1、FPR、FNR 不填 0 或 1。
GT 不可用记录 `gt_failed`；算法/验证异常记录 `error`。保留种子、失败原因、扩展轨迹以及所有网格记录。
已确认的 GT 成功但预测异常，与 GT 构建未知分别记录，不将未知错误称为已知 GT 失败。

统计按 (cr, tau, metric) 聚合四主指标，共 220 行；GT 诊断 55 行，记录成功、失败、未知及完整预测次数。
30 种子等权计算有限值均值、样本 std 与双侧 Student-t 95% CI；没有把 2900 个目标当作 2900 次独立重复。
有效数小于 2 时保留统计缺失及 `statistics_reason`；保存 `n_valid/n_failed/n_complete/n_undefined/n_trials`。
这些指标是成功且该指标有定义的种子上的条件性结果；合成 bank 的逻辑抽样不等于真实部署独立性。

## 文件来源、运行和重绘

正式运行自动产生 150 个 trial、1650 条 threshold-run；所有默认参数见 `README_THRESHOLD.md`。
manifest 保存数据集逐文件 SHA256、源码逐文件 SHA256、实际 RACA-HIM 实现身份、完整参数、环境、时刻、状态和 CSV 身份。
每个 trial 后保存原始记录；中断保留部分数据并禁止其进入论文 Figure。
每次新正式运行自动归档本组旧结果，保存既有来源身份。

右击 `run_threshold.py` 即执行完整实验；右击 `plot_threshold.py` 或使用 `run_threshold.py --plots-only` 仅重绘。
两入口自动定位项目根目录，无需设置 Working Directory 或 PYTHONPATH。
重绘只读取 CSV/manifest，并核对完整网格、哈希、失败记录和统计；不导入检测器、不加载数据集、不执行算法。
重绘来源单独保存至 `figure_provenance.json`，不重写原始 CSV 或运行记录。

## S1 图表规则

只有 `Fig_S1_threshold_sensitivity` 一张正式 Figure，五个 cr 横向子图，四条指标曲线。
复用 `common.plotting_style.METRIC_STYLES`：Accuracy 蓝圆、F1 橙方、FPR 绿三角、FNR 红菱形，全部实线。
7.16 × 1.50-inch 画布、兼容 serif、紧凑共享图例、底部 (a)–(e) 标题；完整 0–1 纵轴与细灰色默认 tau 参考线。
子图绘图区 82.7057809082 × 64.26252 pt，间距 14.8870400826 pt；左/下边距 33.5/31 pt；图例距绘图区顶边 3 pt。刻度/图例字号 5.8288 pt，标签/题注字号 6.5574 pt，与前两组一致。
11 个真实 x 点全部绘制，少量显示刻度文本；0.9995 保持真实非均匀坐标。缺失留断点；无 CI 阴影、平滑、插值或改数值。
运行后输出 PDF 矢量图与 PNG 600 dpi，图中不预留论文外部 caption 空白。

## 静态检查与待执行工作

源码语法、AST、导入路径、完整参数和第一组阈值传递的静态检查结果见 `03_threshold_static_checks.json`。
该检查仅解析和编译源码及核对文件，不导入本组模块、不执行算法、校验函数、测试或绘图。
原始 dataset、第一/第二组源码及结果的逐文件哈希对比也保存在该静态检查记录中。
以上静态记录描述初次交付；用户已完成正式运行。后续读取结果核对指标、统计及来源，并按最新尺寸要求仅重绘 S1，完成 PNG 与 PDF 渲染视觉检查。
本次排版修订对照前两组保存的几何元数据核对尺寸，差值均为零；16 个受保护文件（本组原始/统计 CSV、原运行 manifest、前两组绘图源码与图文件、公共样式）的 SHA256 未改变。没有重新运行算法或自动测试套件。
