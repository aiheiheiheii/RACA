# 第一组实验：合法异构与模型损坏识别精度

实现版本：hash-baselines-v4；正式完整参数网格。
实际执行 1950 个配对配置/seed，共 7800 个方法记录；耗时 2457.74 秒。

## 四方法实现

- Proposed：最新TNSE稿件的多数支持GT和单GT目标判定；K=50、M=25，必要时按25扩至100。只有Proposed使用cosine区间[0.997,1]。
- I-EDI-Adapted：原可靠性公式、O²RS排除贪心和虚拟队列；批次覆盖全部目标；原始内容SHA256后实际计算hash(s) mod η并比较正确tag。默认η=7919，依据原文V-A实验设置；有限模碰撞由实际hash决定。
- CTCV-Adapted：原始字节分块重排、SHA256后mod ℓ、原组合位向量与k+1收集者四Flag/VR流程；默认ℓ=1000×共同目标数；原SC只保留CR位向量，定位使用CR[actual_tag]成员位。紧凑描述符以min(目标数,原始字节数)确定最大非空块数；不看结果选参数。
- DynaEDI+-Adapted：原始块SHA256、nonce/版本公共子树、Szudzik可逆树；完整root严格不一致数>ceil(2R/3)触发展开；至少一个块达到该quorum才将模型判损坏，R=100时门槛68；只比较子树和叶digest。
- I-EDI/CTCV缺少vendor原内容，独立按参考池完整hash重复数、历史和临时ID选一个代理原内容锚点；没有相似度、原型或标签参与选择。Dyna使用全部100参考见证。
- 三个基线将当前同family模型按同内容副本验证；合法个性化产生不同字节时可能误报。本实验测量这种迁移设置，不能解释为原协议在各模型拥有独立正确原内容时的性能。
- 固定100参考池及N−100共同目标，算法输入仅描述符、临时ID、历史和划分；真值只在预测后计算指标。完整输入SHA/目标指纹对所有方法相同。
- 静态p_fail=0、版本0与中立QoS/时限；本地SHA承诺不等于BLS或真实网络认证。详细依据见reports/01_detection_baseline_adaptation.md。

## 参数与统计

- D1：N=1000..5000每500，五子图固定cr=1..5%；D2：cr=1..5%每0.5%，五子图固定N=1000,2000,3000,4000,5000。
- 两图配置去重后65个点，seed=2026..2055，完整1950个数据trial；四方法应产生7800个方法记录。重叠点只运行一次。
- Accuracy=(TP+TN)/(TP+TN+FP+FN)，Corrupted=1。每点为定义有效的成功seed均值；失败不伪报Normal，保留状态/原因和有效n。
- CSV保留均值、标准差、95% Student-t CI及其他指标/四类错误记录；图仅绘制Accuracy均值，不附加阴影或误差带。
- 仅D1/D2绘图；默认柱状图，TMC Fig.3双区间断轴、两侧双斜线、隐藏相邻横框线；纵轴为Accuracy的0..1数值。默认下段0..0.18、上段0.94..1.02。
- 图内不显示Mean detection accuracy；子图标题更靠近横轴，共享图例框距子图顶边3pt。若真实均值进入默认省略区间，则扩大显示区间或退回完整轴，不隐藏真实结果。
- 配色Proposed蓝、I-EDI橙、CTCV绿、DynaEDI+红；可选折线全部实线。95%CI刻画固定合成库重复抽样，不代表真实部署总体。

## 实际执行状态

| 状态 | 方法记录数 |
|---|---:|
| success | 7800 |
| gt_unavailable | 0 |
| unsupported | 0 |
| failure | 0 |

## 实际检测精度

以下为各配置Accuracy均值的等权汇总；未把不同配置作为独立重复构造总体CI。

| 方法 | 配置平均Accuracy (%) | 成功记录 | 不可用/失败记录 |
|---|---:|---:|---:|
| Proposed | 96.336863 | 1950 | 0 |
| I-EDI-Adapted | 3.015487 | 1950 | 0 |
| CTCV-Adapted | 3.007834 | 1950 | 0 |
| DynaEDI+-Adapted | 3.003284 | 1950 | 0 |

## 输出

- 结果根目录：`E:\Code\Paper_Code\Paper_Yan\20261009_TNSE\results\01_detection`。
- raw/detection_runs.csv及raw/predictions/*.npz：真实预测、状态、共同目标/参考临时ID、GT及hash协议诊断JSON；无pickle对象。
- raw/subtype_metrics.csv及summary/subtype_summary.csv：保留离线错误诊断，不绘制额外Figure。
- summary/detection_summary.csv：每配置seed级均值、标准差、CI和状态统计。
- figures/Fig_D1_detection_vs_N.pdf/png及Fig_D2_detection_vs_cr.pdf/png：两张检测精度图，矢量PDF和600dpi PNG。
- logs/plot_style.json记录实际断轴区间和样式；run_manifest记录完整参数、代码SHA和环境。
- Dataset manifest：{'verified': 30, 'missing': [], 'mismatch': []}；完整运行前后dataset树SHA256相同。
- 入口不运行Quick、测试套件或重复预试跑；历史测试/结果不代表本版通过运行时验证。
