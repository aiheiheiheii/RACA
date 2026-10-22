# 第三组：Similarity Threshold Sensitivity

本组只比较 RACA-HIM 的不同相似度下界。初次交付仅做静态检查；用户已完成正式实验。后续图表调整通过仅重绘入口读取这些已保存结果。

## 在 PyCharm 运行

使用已安装项目 `requirements.txt` 依赖的 Python 解释器，右击本目录的 `run_threshold.py`，选择 **Run 'run_threshold'**。
入口自动从文件位置查找项目根目录，默认完成全部正式实验，无需 Working Directory、PYTHONPATH、环境变量或命令行参数。
没有 Quick Mode，也没有先跑少数配置的前置步骤。

正式运行将自动写入：

```text
results/03_threshold/
  raw/threshold_runs.csv
  summary/threshold_summary.csv
  summary/gt_success_summary.csv
  figures/Fig_S1_threshold_sensitivity.pdf
  figures/Fig_S1_threshold_sensitivity.png
  logs/run_manifest.json
  logs/figure_provenance.json
reports/03_threshold_report.md
```

再次正式运行前，会将第三组已有 raw/summary/figures/logs 归档到 `results/03_threshold/archive/<新运行身份>/`。
第一组、第二组与 dataset 的代码和结果不在本组写入范围内。
运行采用 `.run.lock` 防止同一输出目录的并发写入；正常退出会删除锁。如果强制终止后遗留锁，确认没有运行中的第三组进程，再删除该锁。
中断时保留已完成 trial 的原始记录并标记 `failed/interrupted`，不会将部分网格当作正式 Figure 的输入；重新运行会开始完整协议。

## 仅重绘

在 PyCharm 中独立右击 `plot_threshold.py`。
也可使用以下命令：

```text
python -B experiments/03_threshold/run_threshold.py --plots-only
```

重绘只读保存的 CSV 和 manifest，验证 1650 条记录的完整网格、来源身份、文件 SHA256、统计与原始记录的一致性。
统计复核只计算已经保存的混淆计数和种子层面的指标，不调用 RACA-HIM，不加载描述符，不改变 CSV。
缺失均值保留缺口，部分种子失效时保留有效数和失败数。
新图的来源记录位于 `logs/figure_provenance.json`，包含本次绘图代码、公共风格代码和源 CSV 的哈希。
`run_manifest.json` 记录原始正式运行的状态；重绘不会改写原始运行记录。

## 正式参数

| 参数 | 值 |
|---|---|
| N / d / H | 3000 / 30 / 0.5 |
| cr (%) | 1 / 2 / 3 / 4 / 5 |
| tau 下界 | 0.990 / 0.991 / 0.992 / 0.993 / 0.994 / 0.995 / 0.996 / 0.997 / 0.998 / 0.999 / 0.9995 |
| tau 上界 / 默认下界 | 1.0 / 0.997 |
| K / M / 扩展步长 | 50 / 25 / 25 |
| 初始历史 / 通信失败率 | 0.10 / 0.0 |
| reference / evaluation | 100 / 2900 |
| seeds | 2026..2055，共 30 |
| trial / threshold-run | 150 / 1650 |

通过 `common.dataset_adapter.load_evaluation_data()` 调用原始 `dataset.load_dataset.load_group()`。
每个 (cr, seed) 加载一次；11 个阈值复用不可变 X、池划分、模型 ID、候选排序和初始历史。
算法输入 `AlgorithmInput` 不包含标签、prototype、cosine_to_prototype 或 sample_id。

## 算法与统计

直接复用第一组 `proposed_detection.detect_proposed()` 与 `build_ground_truth()`。
每次以 `dataclasses.replace(BASE_CONFIG, tau_lower=tau)` 建立新配置；构建 GT 与目标相似度判定使用同一配置。
候选按历史及 seed 生成的临时 ID 排序，K 扩展为 50、75、100，多数门槛对应 26、38、51。
每个目标只由一个 GT verifier 判定，保留第一组的循环分配规则。
固定参考池及循环分配属于第一组静态检测协议；本组不运行稿件的资源预算调度或长期更新。

Corrupted=1。复用 `common.metrics.binary_metrics()` 计算 TP/TN/FP/FN 和 Accuracy、Precision、Recall、F1、FPR、FNR。
零分母产生 NaN 并明确记录 `undefined_metrics`；GT 失败记录 `gt_failed`，没有预测和检测指标，不重新抽样。
异常记录 `error`；已确认的 GT 结果可与完整检测成功与否分别统计，未知 GT 结果保留 unknown。

四个主指标按 (cr, tau, metric) 保存 220 行统计；GT 诊断保存 55 行。
均值、样本标准差和双侧 Student-t 95% CI 只使用成功且指标有定义的种子，各种子等权。
CI 不裁切为 0–1；图只显示 0–1 均值。0 个有效种子时全部统计缺失，1 个有效种子时 std/CI 缺失。
同时保存 `n_valid/n_failed/n_complete/n_undefined/n_trials`，不得将条件性均值称作全部尝试的整体准确率。
合成数据来自固定 descriptor banks；30 个 seed 是逻辑抽样重复，不能解释为 30 个独立真实部署。

## 运行时校验与后续核查

- `validate_protocol()` 检查正式参数网格。
- `paired_input_fingerprints()` 对 X、参考/目标池、临时 ID、历史和候选排序计算哈希，并检查不可变性。
- `validate_detection_result()` 检查 GT 支持诊断、实际多数门槛、池归属、GT/target 不重叠、全部目标的二元预测和单 verifier 分配。
- `validate_run_records()` 检查网格、无重复、跨阈值输入一致、所有混淆公式及缺失原因。
- `load_plot_sources()` 检查来源和 CSV 身份，并复核已保存统计；完全不调用检测算法。

这些是正式运行和读取结果时的校验能力；仅重绘也会核对保存记录和统计。未通过自动测试套件或重新运行实验来调整图表。
可复现性复核方式：完整运行两次，比较归档和新 `threshold_runs.csv` 的配对输入哈希、GT、状态及指标；忽略不同 `run_id`。
源码或数据集改变时，应保留不同版本身份，不能将旧结果重新标注为新算法。

## 绘图约定

唯一 Figure 为 S1：五张横向子图，每张四条实线，复用公共 `METRIC_STYLES` 的蓝/橙/绿/红及圆/方/三角/菱形。
按 2026-10-13 用户的后续要求，画布与前两组统一为 7.16 × 1.50 inches；子图绘图区宽高为 82.7057809082 × 64.26252 pt，间距 14.8870400826 pt。
复用公共检测图形的物理包络、边距、字号、图例及题注位置，保持 serif 字体。全部 11 个真实 x 坐标参与绘图，包括非均匀的最终 0.9995 点。
纵轴统一完整 0–1；0.997 使用细灰色参考线；无断轴、阴影、平滑或插值。
现有结果已按统一布局重绘并完成 PDF/PNG 视觉检查；尺寸及数据不变性记录见 `reports/03_threshold_layout_revision.json`。
