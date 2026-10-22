# 第一组实验图形检查

结果：PASS。正式图和 Quick 图均来自保存的 summary CSV，没有重算算法或修改原始指标。

- D1 与 D2 正式 PDF 均为单页矢量图，7.16 × 2.15 inches，五个横向子图。每子图有9个真实实验点、两条有效方法曲线和95% seed-level Student-t CI阴影。
- D1 五图采用相同N轴；D2五图采用相同cr轴。主图统一85%–100%纵轴；显示范围未夸大小幅差异，没有断轴或曲线平滑。
- 共用图例位于曲线上方，不遮挡坐标或数据；字体为Times New Roman，蓝色Proposed、紫色Single-Reference与公共配置一致。其余三种不适用发表方法没有曲线。
- 正式内部诊断图为5个横向子图，显示Proposed的Accuracy/F1/FPR/FNR，纵轴0%–100%，不作为新增正式主Figure。
- 六张PNG（正式3、Quick3）可打开，均为4296 × 1290像素、600 dpi。Quick图明确标记QUICK subset，只含3个子图及3个点，存于独立quick目录，不替代正式图。
- 正式三张PDF使用Poppler以200 dpi重新渲染，并逐张检查：全部文字在页内、无重叠/截断、曲线完整、图例与子图标题可读，与PNG一致。D1/D2 y轴为Corruption Detection Accuracy (%)。
- 原始Student-t区间完整保存在CSV；图中仅将阴影显示限制在物理指标范围0%–100%，未更改数值。正常性可分假设不足造成的波动和方法交叉原样保留。

输出位置：

- `results/01_detection/figures/Fig_D1_detection_vs_N.pdf` / `.png`
- `results/01_detection/figures/Fig_D2_detection_vs_cr.pdf` / `.png`
- `results/01_detection/figures/Fig_D_diagnostic_metrics.pdf` / `.png`

重新绘图只需直接运行 `experiments/01_detection/plot_detection.py`。数据和文件格式独立复核见 `reports/01_detection_result_audit.md`。
