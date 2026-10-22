# 第四组实验实现报告

> 版本说明：下文记录最初交付的 v1 严格多数实现。用户运行后，已明确授权重设计方法；第四组当前默认版本为 `ground-truth-supermajority-v2`，采用三分之二门槛及一致性平局排序。最新定义、论文修订与兼容范围见 [方法修订](E:/Code/Paper_Code/Paper_Yan/20261009_TNSE/reports/04_ground_truth_method_revision.md)，旧结果诊断见 [只读审计](E:/Code/Paper_Code/Paper_Yan/20261009_TNSE/reports/04_ground_truth_result_audit.md)。v1 的结果和以下历史检查不能作为 v2 的运行验证。

> 最新输出策略：第四组全部正式输出固定到 `results/04_ground_truth/`，已移除时间戳运行目录与自动发现逻辑。下文整体归档描述属于旧实现，已由 [固定输出说明](E:/Code/Paper_Code/Paper_Yan/20261009_TNSE/reports/04_ground_truth_fixed_output.md) 取代。

日期：2026-10-14（Asia/Shanghai）。实现版本：`ground-truth-reliability-v1`。

**本次仅完成完整代码实现及必要的静态检查，尚未运行实验、测试和绘图。正式结果由用户在PyCharm中执行后产生。**

已按用户请求采用附件 Prompt 的实验规格，并先核对最新 TNSE 稿件 Section IV-A / V。TMC 仅作为论文绘图参考；未引入设备信誉公式、委员会加权投票、安全协议、验证预算、资源调度、联合决策或长期动态状态。

## 文件与入口

第四组实现目录为 `experiments/04_ground_truth/`，共 11 个 Python 文件：

| 文件 | 用途 |
|---|---|
| `run_ground_truth.py` | 默认执行完整正式实验、统计、生成正式图及来源信息；支持 `--plots-only` |
| `config_ground_truth.py` | 固定参数网格、1000 个 seed、方法配置、附件内容哈希 |
| `ground_truth_experiment.py` | 单 seed 加载一次 Dataset、共享场景、流式保存原始记录 |
| `candidate_sampler.py` | 受控 100 候选池、标签隔离、固定顺序、共享失败随机数 |
| `ground_truth_methods.py` | 四种方法；RACA-HIM / Fixed-K 包装现有 GT 内核 |
| `ground_truth_metrics.py` | 后置评价、所有 seed 的二元统计、Wilson 95% CI |
| `ground_truth_storage.py` | 原子保存、内容哈希、完整性及只读重绘来源核验 |
| `plot_ground_truth.py` | 仅根据保存的 CSV 和 manifest 重绘五子图 |
| `test_ground_truth.py` | 必要数值与行为校验，已编写但未执行 |
| `static_check_ground_truth.py` | 仅 AST / 源码一致性检查，不导入项目算法 |
| `__init__.py` | 第四组包标识 |

在 PyCharm 中直接右击 [run_ground_truth.py](E:/Code/Paper_Code/Paper_Yan/20261009_TNSE/experiments/04_ground_truth/run_ground_truth.py)，选择 Run。无需参数、Working Directory、环境变量或 PYTHONPATH；入口在导入之前通过 `Path(__file__).resolve().parents[2]` 定位项目根目录，并禁止 bytecode 写入。沿用项目 `requirements.txt` 的依赖和用户已有 Python 环境。

只读重绘可直接右击 [plot_ground_truth.py](E:/Code/Paper_Code/Paper_Yan/20261009_TNSE/experiments/04_ground_truth/plot_ground_truth.py)，或者由用户之后执行：

```powershell
python -B experiments/04_ground_truth/run_ground_truth.py --plots-only
```

该路径只校验保存的 raw CSV / Summary / 场景审计 / manifest，再绘制保存的 Summary 均值；不会加载 Dataset、计算 cosine 或执行 GT。重新累计 raw CSV 的统计仅用于核对保存的 Summary，不会重新运行算法。

`experiments/run_all.py` 保持现状。第四组的正式入口是以上独立文件。

## 固定正式协议

| 参数 | 值 |
|---|---|
| N / global cr / d / H | 3000 / 3.0% / 30 / 0.5 |
| q_C | 0%、5%、…、50%，共 11 点 |
| p_fail | 0%、5%、10%、15%、20%，共 5 点 |
| cosine interval | [0.997, 1.0]，inclusive，无放宽容差 |
| 初始 K / M / ΔK / 最大池 | 50 / 25 / 25 / 100 |
| 初始历史异常状态 | 所有候选 h=0.10 |
| 重复与种子 | 1000 次，2026…3025 |
| 场景与记录 | 11000 个独立候选池，55000 个失败场景，220000 条方法记录 |

global cr 是逻辑 Dataset 组的损坏百分比（约 90 个损坏模型）；q_C 是离线受控 100 候选池的损坏比例。通过既有 `draw_candidate_set(K=100, use_severe_only=False, use_common_only=False)` 在全局 3% 数据中富集至候选 q_C=50%，不将整个网络改为 50% 损坏。

四类样本均保留在可抽样集合中；没有强行保证每个小候选池包含每个 subtype。q_C=0% 当然没有损坏样本，其余 subtype 数量取决于原始随机抽样，不为得到曲线改变配额。

Dataset 加载由已有 `dataset.load_dataset.load_group()` 完成，基础库在其现有缓存中复用；每个 seed 只加载一个逻辑组。未重新生成、物理导出或修改 Dataset。

## 候选、失败与算法边界

完整池沿用 loader 已随机化的顺序。算法可见的 ID 为该随机顺序上的临时 0…99，不携带原始 `sample_id`、标签或 subtype 排列。所有历史评分相同，因此固定随机顺序用于平局处理。

真正参与第一轮 admission 的集合只有前 50 个；失败才揭示前 75 / 100 个，保持前缀嵌套。100 行 observations 和相似度缓存是完整实验域及数值缓存，不代表初始 K=100。每次扩展都根据当前 K 重新计算门槛与 admission，绝不复用旧 K 的录取结果。

每个 seed/q_C 生成一次 100 个 uniforms；五档 p_fail 采用 `available = uniform >= p_fail`。同一候选的失败事件随 p_fail 增大嵌套，同一 p_fail 下四方法共享同一掩码；候选在 trial 内的状态固定。随机流为 Dataset 1147、失败 4401、Random-M 4402，所有 entropy 和派生 Random-M seed 写入场景审计。

方法函数仅接收 `AlgorithmInput`、availability、Random-M seed 和可选数值缓存，不接收完整 Scene、标签、原型或来源行号。真实标签用于离线场景控制与后置指标；source indices / subtype / failure uniforms 写入审计而不反馈给方法。

| 方法 | 选择规则 | 扩展 |
|---|---|---|
| RACA-HIM | 现有候选 peer cosine 多数 admission，再按历史 / 临时 ID 选 M | 50→75→100 |
| Fixed-K Majority | 与 RACA-HIM 同一内核、阈值、响应处理与 Top-M，仅禁止扩展 | 只 K=50 |
| Top-M | 无多数过滤，按历史与固定平局顺序取前 M 个可用候选 | 可用数量不足 M 才扩展 |
| Random-M | 无历史优先与多数过滤，从当前可用候选无放回随机选 M | 可用数量不足 M 才扩展 |

多数门槛为 `ceil((K+1)/2)`：K=50/75/100 时分别 26/38/51。peer supports 排除对角；仅自身可用时获得一次自支持。失败候选仍占 K 的位置，不能给 peer 支持、不能自支持、不能入 GT。未改为“响应者半数”。

现有内核未采用 early stopping，最新稿件也没有要求必须实现该优化。因此本组保留完整多数判定，不宣称节省 early-stopping 比较。每池一次 100×100 精确 float64 cosine 缓存供所有方法 / p_fail 复用；逐方法 `pairwise_comparison_count` 是各访问 K 的有效有向 peer 关系累计，物理缓存点积另列：10000（含对角）/ 9900（有向 peer）每池，只有第一条池记录记新增预算，避免累计时重复计入。Top-M / Random-M 本身不读取缓存或执行 similarity。

## 复用与兼容范围

实际 GT 调用来源是 `experiments.01_detection.proposed_detection.build_ground_truth`，也就是第三组经 `detect_proposed` 调用的同一内核。未另写另一套 RACA-HIM admission 算法。

本次修改现有模块两处：

1. `experiments/01_detection/proposed_detection.py`：新增 keyword-only `responder_available=None`、`pairwise_cache=None`、`allow_expansion=True`；`GroundTruthResult` 尾部新增默认逻辑比较计数字段。省略参数时保留原多数判断、排序、扩展、GT 失败文案和 `p_fail=0` 限制；第四组使用外生掩码而不放宽第一组配置限制。
2. `experiments/common/plotting_style.py`：新增独立 `GT_METHOD_COLORS` / `GT_METHOD_MARKERS` / `GT_METHOD_LINESTYLES`，原 `METHOD_*` 映射未改。

未修改前三组其他算法规则或结果文件。Group02 继承的 result 类型使用默认附加字段，静态上兼容其 keyword 构造；其 source inventory 包括 GT 文件，故新源码内容会改变后续来源哈希。旧 checkpoint 是否允许续跑由第二组既有身份规则决定，不改写旧 manifest；旧结果保持其原身份。默认调用的数值回归仍需用户运行验证。

最新稿件规定失败时扩展至当前验证域覆盖，本组根据 Prompt 将受控 GT 实验域限定为 100 候选，而非偷偷扩展至全网络 3000 个模型。稿件关于严格正常可响应多数的假设在高污染 / 失败压力点可能不成立；只记录真实形成与可靠率。Prompt 与最新稿件在历史、peer 支持、多数、自支持与 Top-M 主规则上无实质冲突。

## 原始记录、统计与来源

可靠成功要求同时满足：形成恰好 M=25 个不同 GT、全部有效响应、真实标签全部 Normal。形成成功与可靠成功分别记录。GT 不足、耗尽候选池、污染或响应失效均为可靠失败；不存在通过未录取候选补齐、oracle 参考、重抽候选、修改标签或删除失败 seed。

raw CSV 包含 Prompt 指定字段，另保存 K75 / K100 的真实损坏比例、全池与各前缀失败比例、入选临时 ID、各步 threshold / passing_count / responding_count、随机 seed、三种共享状态指纹、run ID、代码内容版本和独立物理缓存预算。单位：`global_cr` / `global_cr_realized` 用百分数 3.0；所有 q_C、prefix corruption、p_fail 及可靠率用 0…1 比例。

Summary 按 method/q_C/p_fail 聚合，每点必须恰好全部 1000 个唯一 seed：包含 `n_total`、`n_reliable_success`、`reliable_gt_rate`、`n_gt_formed`、`gt_formation_rate`、`mean`、样本 `std(ddof=1)`、Wilson 95% CI。0 次和全部成功均保留非退化的 Wilson 区间。

Diagnostics 的条件性字段明确命名，例如 `formed_gt_conditional_normal_fraction`；条件分母为零时 CSV 用 `nan`，不替代为 0。整体可靠率的分母始终包含未形成 GT 的试验。

正式运行生成：

```text
results/04_ground_truth/
    raw/ground_truth_runs.csv
    summary/ground_truth_summary.csv
    summary/ground_truth_diagnostics.csv
    figures/Fig_G1_ground_truth_reliability.pdf
    figures/Fig_G1_ground_truth_reliability.png
    logs/run_manifest.json
    logs/candidate_scenarios.jsonl
    logs/figure_provenance.json
    logs/ground_truth_run_report.md
```

候选审计逐池保存 source indices、离线标签 / subtype、uniform、prefix composition、descriptor / cosine fingerprint 与分层 pairwise cosine 分布，支持诊断曲线重合、高异构正常模型不支持及 mild corruption 混入；没有额外正式 Figure。

Manifest 记录 Dataset 名称和全文件哈希、源码 inventory / content version、附件内容哈希、真实 GT 调用来源、完整网格 / seeds / 方法配置、随机流 / 失败规则、开始 / 结束 / elapsed、成功 / 失败计数、输出文件身份。来源检验不要求历史保存的源码等于当前机器文件，因此重绘仍保留原实验来源及新绘图源码身份。

原始结果流式写入 `.partial` 文件；计算异常中断保留文件和非 complete manifest，拒绝当作正式图输入。不提供 Quick 或部分 seed 入口，也不将部分结果混入正式目录。默认重新正式运行时，已有第四组目录先保存至 `results/archive/04_ground_truth_<run_id>/`，仅移动已解析且在 results 内的第四组目录。前三组结果不被移动或覆盖。

原子文件替换针对 Windows 短暂共享锁作最多 6 次有限重试，未引入额外存储框架；持续 I/O 失败仍明确中止并保留失败状态。

计算完成后若绘图失败，完整 CSV 仍保留，manifest 分别记录计算 complete 与 figures failed；之后可只读重绘。独立绘图入口也更新图文件身份及状态。

## Figure 配置

图名：`Fig_G1_ground_truth_reliability`。按第三组连续 0…1 五子图布局复用公共几何，未采用第一组断轴。

| 配置 | 实际定义 |
|---|---|
| 总图大小 | 7.16 × 1.50 英寸 |
| 单面板绘图区 | 82.7057809082 × 64.26252 pt（约 1.14869 × 0.89254 英寸） |
| 水平面板间隔 | 14.8870400826 pt |
| axes 左 / 底 | 33.5 / 31.0 pt |
| caption 中心位置 | 底部 5.5 pt |
| legend 底边间隔 | axes 顶部上方 3.0 pt |
| 字体 | 公共 serif 优先 Times New Roman，其次 Liberation Serif / DejaVu Serif |
| 标签字号 / tick 与 legend | 6.5574 / 5.8288 pt，沿用第三组 |
| 线宽 / marker size | 公共 1.2 / 3.2 pt |
| 输出 | 矢量 PDF；600 dpi PNG；固定 page size，无 tight 裁剪 |

五面板分别固定 p_fail=0%、5%、10%、15%、20%；横轴 `Candidate Corruption Ratio (%)`，保留 11 个真实 marker 点，显示 0/10/20/30/40/50 的 tick 字样。共享纵轴 `Reliable GT Construction Rate` 为 0…1，双行文字只是排版；四条真实均值折线均为实线，无平滑、插值、CI 带、阴影、抖动或人为区分重合曲线。

配色 / marker：RACA-HIM 蓝 `#1F77B4` / o；Fixed-K Majority 紫 `#9467BD` / s；Top-M 粉 `#E377C2` / ^；Random-M 棕 `#A6761D` / D。RACA-HIM 沿用前三组蓝色，其余方法采用独立稳定 GT 映射，不占用旧算法配色。

## 本次检查与尚待验证

本次执行了标准库 AST 解析 / 源码一致性检查，覆盖所有第四组文件、共享 GT 内核、公共样式及 adapter；没有导入执行这些项目模块。另以标准库只读核对 Dataset 原有 `MANIFEST.sha256` 的 30 个列出文件：无缺失、无哈希不一致。PDF 仅提取参考页文字，未生成实验图或示意图。

静态检查证据：固定参数与源码网格、direct-run main / guard、GT 可选接口、label-free method 参数、plot / storage 无 GT / Dataset / cosine / smoothing 调用均通过。这些检查不等同于数值、运行环境、回归、测试或渲染验证。

用户之后可以自行执行：

```powershell
python -B experiments/04_ground_truth/static_check_ground_truth.py
python -B -m pytest experiments/04_ground_truth/test_ground_truth.py
```

测试覆盖 Dataset 可复现与四类、q=0 / 50 以及所有 11 点精确池组成、无标签输入、固定随机排列、失败嵌套、三种多数门槛与自支持一次、不可用行列、扩展重新录取、Fixed-K 消融、Top-M / Random-M、可靠指标与 1000 seed 总分母、Wilson 端点、绘图源码不执行 GT。测试只作必要独立行为校验，不启动正式 55000 场景或写论文图。

仍待用户实际验证：第一 / 三组默认 GT 数值回归与第二组继承接口；cache / uncached 决策一致；完整 220000 行正式执行与逐点 1000 分母；环境依赖；实际字体边界、共享 legend / caption / 横轴标签排版和 PDF / PNG 视觉效果。当前未发现阻止交付的静态实现缺口，但不能声称上述运行检查已经通过。

若真实结果显示 Top-M 与 Random-M 接近、RACA-HIM 在 50% 污染边界下降、正常异构难以获得多数或 mild corruption 混入，应按保存的原始审计解释；不修改实验值或算法规则使曲线符合预期。

**没有在本会话执行任何实验、测试套件、Quick Mode、配置试跑、GT 算法试跑或绘图；没有生成实验数值或 Figure。**
