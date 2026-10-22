# 第四组 GT v2 调整说明

> 最新输出要求：已移除独立时间戳运行目录，CSV、原始记录、日志、PDF / PNG 全部统一到 `results/04_ground_truth/` 的相应子目录。来源记录保留 run ID，详见 [固定输出说明](E:/Code/Paper_Code/Paper_Yan/20261009_TNSE/reports/04_ground_truth_fixed_output.md)。

本次根据用户明确选择“重新设计 GT 构建规则，并同步修改论文方法”，实现 `ground-truth-supermajority-v2`。新版代码已完成静态检查；尚未运行新版实验、测试或绘图。只读分析的是用户已保存的 v1 结果，未生成 v2 数值。

## 原始图回升的原因

保存的 220000 条记录未发现指标、响应、门槛、停止过程或 Summary 一致性错误。p_fail=0 时，q_C=30% 的可靠率 0.516、污染 484 次、扩展 18 次；q_C=50% 的可靠率 0.905、污染 86 次、扩展 885 次。较高污染更频繁触发 75 / 100 候选判定，减少被录取的 mild_corrupt，因而出现回升。详见 [结果审计](E:/Code/Paper_Code/Paper_Yan/20261009_TNSE/reports/04_ground_truth_result_audit.md)。

Top-M / Random-M 的快速下降也与原指标一致：一次必须选出 25 个全部正常的模型，随机池中含 5 个损坏时，纯净概率已约为 0.23；不是两条曲线被人为设置为失败。

## 算法与论文修订

新版 admission 门槛 θ_K=⌈2K/3⌉：K=50/75/100 时为 34/50/67。GT 选择按历史升序、支持数降序、支持同伴平均 cosine 降序、临时 ID 升序；平均数排除自身及无效支持。首次达到 M=25 即停止，不足仍按 50→75→100 扩展。Fixed-K 使用同一新规则且禁止扩展，Top-M / Random-M 保留原定义。

新规则增加候选接纳要求，并在同等历史状态下优先选一致支持更强的参考。三分之二为新版实验执行前固定的保守协议选择，没有运行参数搜索或挑选成功率最优值。它可能降低可形成性，也不能普遍保证曲线单调或真实标签全部纯净。可检测异常子集与所有损坏标签的保证范围已在定理中明确区分。

[方法及证明修订说明](E:/Code/Paper_Code/Paper_Yan/20261009_TNSE/reports/04_ground_truth_method_revision.md) 和 [完整修订稿副本](E:/Code/Paper_Code/Paper_Yan/20261009_TNSE/reports/Latest_version_of_paper_GT_v2.md) 已保存。桌面原稿 `C:/Users/MengKe/Desktop/Tutu/TNSE/Latest version of the paper.md` 已同步更新 Section IV-A、定理 1 / 证明、GT 复杂度及历史更新中的检验名称，其他内容保留。原稿备份在同目录 `Latest version of the paper.before_GT_v2_20261014.md`。

## 代码变动与保存证据

- 共享 `proposed_detection.py` 新增参数 `support_quorum`、`selection_policy` 及共享 threshold / ranking helpers；其默认参数保留前三组历史规则。
- 第四组 `ground_truth_methods.py` 使用新版门槛与排序；trace 保存选择支持数和支持均值。
- `config_ground_truth.py` 明确新旧算法版本、固定比例与论文副本 SHA；`protocol_metadata(version)` 保留 v1 的原始协议身份。
- 新 `ground_truth_diagnostics.py` 在方法返回后统计每个访问 K 的候选 / 响应 / admission subtype、支持分布、定理前提和最终 K 分解；诊断不反馈给算法。
- `ground_truth_experiment.py` 保存新诊断 JSONL、raw 中的选中 subtype 和诊断 hash、新增两个 Summary，并自动报告实际向上台阶的分解。
- `ground_truth_metrics.py` / `ground_truth_storage.py` 允许在保存的算法版本下复算统计；重绘旧 v1 时保留旧值，新 v2 则校验全部额外诊断来源。
- `plot_ground_truth.py` 明确显示所绘结果的算法版本，避免将旧图重绘当作新算法结果；公共图形样式没有调整。
- 新 `audit_ground_truth_results.py` 只分析已经保存的文件，可独立在 PyCharm 运行；默认输出到 reports，原 results 不改写。
- 新 `test_ground_truth_v2.py` 提供必要新规则、旧版兼容、支持排序、响应失败、诊断及不可检测异常反例校验，已编写但未执行。原测试中一个方法扩展用例按新门槛调整响应设置，其余旧默认内核校验保留。

新版正式运行会额外生成：

```text
logs/ground_truth_admission.jsonl
summary/ground_truth_final_k_summary.csv
summary/ground_truth_admission_summary.csv
```

两个新 Summary 各 660 行（method×q_C×p_fail×K），未访问的 K 记录为零访问；最终 K 贡献仍使用每点全 1000 seed 分母。无条件失败率与条件性均值严格区分，无有效条件的项留空，并记录有效分母。

## 用户后续运行与版本一致性

直接在 PyCharm 右击 [run_ground_truth.py](E:/Code/Paper_Code/Paper_Yan/20261009_TNSE/experiments/04_ground_truth/run_ground_truth.py) 进行完整新版实验。入口仍无需参数或 Working Directory 设置，正式输出固定为 `results/04_ground_truth/`。重跑按同名文件保存，不重命名整个目录；本次源码调整未移动、覆盖或重新贴标签到任何现有结果。

`--plots-only` 及独立绘图入口均读取固定目录。manifest 不是 complete 时拒绝重绘；若该目录仍保存完整 v1 结果，重绘会保留 v1 数据并显示其版本。得到 v2 图必须执行一次新版完整实验。

用户可自行运行以下校验，本次未执行：

```powershell
python -B -m pytest experiments/04_ground_truth/test_ground_truth.py experiments/04_ground_truth/test_ground_truth_v2.py
```

第一、二、三组的现有结果仍来自旧 GT 规则。若论文全部采用新版方法，应同步切换并重跑相关组，不能将旧结果作为新版方法证据；本次没有擅自改变前三组默认调用或执行这些实验。

本次仅执行源码 / AST / 接口静态检查、既存文件的只读审计以及论文文本修订；新版数值效果、回归与实际绘图视觉效果仍待用户验证。
