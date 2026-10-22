# 保存结果重合核验与原文复现修正

保存数据版本：hash-baselines-v3；用户已完成65配置×30seed=1950trial、7800个四方法记录。以下只读取已有CSV和NPZ，没有重跑算法、运行测试、改变dataset/结果或重绘。

## 真实记录

| 核验项 | 数量 |
|---|---:|
| 三个baseline混淆矩阵/Accuracy都一致的trial | 1949/1950 |
| I-EDI与Dyna完全一致的trial | 1950/1950 |
| I-EDI预测Normal的目标 | 0/5,655,000 |
| Dyna预测Normal的目标 | 0/5,655,000 |
| CTCV预测Normal的目标 | 1/5,655,000 |
| 三baseline Accuracy均值完全一致的配置 | 64/65 |

唯一例外：N=2000、cr=4%、seed=2051。I/Dyna：TP74、TN0、FP1826、FN0；CTCV：TP74、TN1、FP1825、FN0。eval位置1171、临时ID1001534、真实label0，CTCV实际模碰撞使它判Normal；完整重排hash并不相等。所有CTCV run都是Flag2。

对应30seed均值：I/Dyna Accuracy=.04014035087719299，CTCV=.04015789473684211，差1/(1900×30)=0.000017543859649，亦即0.001754386个百分点。断轴低段0..0.18内，这个柱高差在600dpi图约0.06像素，难以辨认。不是绘图复制相同列。

NPZ直接核对例：N1000/cr1/seed2026、N3000/cr3/seed2026、N5000/cr5/seed2055，以及唯一N2000/cr4/seed2051。前三例三个预测数组相同，末例CTCV仅1个位置不同。

## 为什么不同算法也会同Accuracy

这批合法个性化descriptor字节也不同，而三个基线按同内容副本验证。I-EDI没有真实tag匹配；Dyna每目标完整root对100参考都不相同，且全部5块都满足不一致quorum；CTCV只有一次有限mod碰撞。遂几乎所有目标都判Corrupted。

当全部预测Corrupted时，TN=FN=0、TP=真实Corrupted数量，因此Accuracy=目标实际损坏比例。O²RS审计顺序、证明组合、树展开可以不同，但在这个退化条件下二元输出相同。算法不同不意味着该指标必然不同，更不能规定它们必须有“轻微差距”。

## v4有原文依据的修正

1. **I-EDI η=7919**：原PDF第9页/印刷13197 V-A Experimental Setup明确该值。原实现2^256让hash mod等同完整hash比对，未保留实验中的有限模碰撞；现恢复原设置。Algorithm2与Theorem6定义不同，实验按Eq39+V-A路径执行。
2. **CTCV成员位**：原PDF第6–7页将σ_i聚合为CR，SC只有CR而无individual σ_i映射；定位应检查CR[r_i]，不是r_i==σ_i。原源码更强的individual比较已改为CR成员位，仍按4Flag处理组合。
3. **CTCV块数**：原Algorithm1 n节点对应n块；compact30float32仅120bytes，不能划900..4900个非空byteblocks。默认块数改为min(n,rawbytecount)，即120，且长度限制/1:1映射破坏明确为适配。原ℓ=1000n保持，参数不依据结果选择。
4. **Dyna block-final**：原PDF第5页III-C Step3.3和第6页树展开最终定位的是异常block。现root quorum只触发展开，至少1个块满足严格68/100才判模型损坏。旧记录都5块满足，因此这一修正本身不会使已保存的v3数据出现不同Accuracy。

v4没有人为扰动、定制误报率、label/prototype参考选择或虚构timeout/collusion。原模型身份/更新版本/认证状态不在descriptor数据中，不能借造这些状态获得差距。原机制可能产生真实有限mod/成员位碰撞，但其数量仍需用户完整运行确定。

## 图例间距与运行

共享图例改为lower-center定位，其框下缘距子图顶边3pt，替代原固定upper-anchor造成的较大空白。

只更新已有v3图排版：运行plot_detection.py，保存的结果版本仍标v3。生成修正算法的新数值：运行run_detection.py，完整65×30×4，记录hash-baselines-v4。两者均由用户自行执行，本次未出图或运行新实验。

源文件：I-EDI、CTCV、DynaEDI+的2025原PDF位于C:/Users/MengKe/Desktop/Tutu/TNSE/参考文献/对比实验/；精确文件名和其他适配假设见01_detection_baseline_adaptation.md。
