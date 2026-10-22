# 本次修改（hash-baselines-v4）

## 2026-10-12：紧凑版面与RACA-HIM命名

- 仅修改排版和名称，不改变检测算法、参数、种子、实际预测、指标或hash-baselines-v4版本。
- D1/D2整图尺寸从7.16×2.15英寸改为7.16×1.50英寸，保留矢量PDF和600dpi PNG输出。
- 子图绘图区采用TMC Fig.3原始矢量坐标的物理尺寸：宽82.7057809pt；上段10.0304169pt；下段52.5121816pt；断轴间隙1.7199215pt；总高64.2625200pt。相邻子图间隔14.8870401pt。共享图例仍距轴顶3pt，标题中心距底5.5pt，避免为论文外部图注保留图片内白边。
- 本方法对外名称统一为RACA-HIM，完整标题为Risk-Aware Collaborative Assurance of Heterogeneous Inference Models in Edge Networks。当前注册、后续CSV/NPZ/报告/图例均采用此名，保留detect_proposed内部API。
- 重绘旧Proposed结果只映射内存中的方法名称，保留原CSV/NPZ、预测值、summary数值、来源版本及manifest。图形metadata明确记录saved_method_names与method_name_aliases。相同配置的两别名重复会拒绝，避免静默覆盖。
- 已有第二组代码也统一新方法名称及旧统计表显示别名，不改变第二组图尺寸或算法。
- 审计兼容新旧名称和实际保存的图形尺寸。仅完成AST和源码静态检查；遵照用户自行运行要求，未运行Quick、测试、检测算法、完整实验或绘图。

已有完整v4结果仅需运行：

```powershell
python -B experiments/01_detection/plot_detection.py
```

## 前次v4算法修正记录

共享图例框距子图顶边3pt。只读核验v3完整结果：三baseline在1949/1950 trial同混淆矩阵，CTCV仅一次多判对1个Normal，均值点差0.0017544个百分点。详见01_detection_baseline_overlap_analysis.md。

原文复核修正：I-EDI η恢复7919；CTCV按CR成员位定位，非空byteblock数按min(targetcount,rawbytecount)；Dyna按至少1个已展开异常块判模型。三者仍只用真实hash/digest，不引入相似度、不预设差距。

本次静态修改与读取已保存结果，没有运行测试、新算法/实验或出图。完整入口生成v4；plot可重绘已有v3/v4，metadata保留实际数值版本。
