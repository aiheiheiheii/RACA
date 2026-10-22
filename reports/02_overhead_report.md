# 第二组核心步骤 Overhead 实际运行报告

实现版本：`overhead-core-v4`。运行身份：`c289bb7ca63f9e0d7b5cf40871f36d94870b80ddd87bec4247a51e3aaaf0487b`。
配置记录状态：`complete`；已保存 45/45 个配置，尚未运行 0 个配置。此状态描述记录是否齐全，不表示全部协议成功。

数据集运行前后完全一致：True。
完整网格为 N=[1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000]、d=[10, 20, 30, 40, 50]、seed=[2026]，cr=3%、H=.5。
默认每个 N/d 仅执行一次配对试验；每点四方法执行相同 B=ceil(.1(N−25)) 个固定目标。
45 个配置共 180 条方法记录；不运行 native_budget，不做多轮平均。
验证阶段 800 ms 只作 deadline 诊断；所有 B 任务执行到最终结果，超时任务计数单独保存。

## 实际工作量

| 方法 | 模式 | 已记录 trials | selected 总数 | attempted 总数 | completed 总数 | completed 未知 trials | 状态 |
|---|---|---:|---:|---:|---:|---:|---|
| RACA-HIM | fixed_workload | 45 | 13410 | 13410 | 13410 | 0 | {"success": 45} |
| DVA-P-Adapted | fixed_workload | 45 | 13410 | 13410 | 13410 | 0 | {"success": 45} |
| I-EDI-Adapted | fixed_workload | 45 | 13410 | 13410 | 13410 | 0 | {"success": 45} |
| CTCV-Adapted | fixed_workload | 45 | 13410 | 13410 | 13410 | 0 | {"success": 45} |

计数总数只加已观测记录；异常中断后的未知完成数不能解释为零。core_phase_counts_json 保存各阶段精确消息、字节和操作汇总。

## 时间、通信和可比性

O1 = 实测核心准备 CPU + 模拟准备通信时延 + 实测核心决策 CPU + 模拟核心验证完成时延 + 实测必要历史更新 CPU。
数据加载、离线状态构造和文件输出不属于核心在线时间；各方法采用同一记账边界。
同一主 verifier 的完整任务串行，其他主 verifier 并行；CTCV 按批次执行原有块核验和聚合流程。
O2 等于核心逻辑消息精确字节总数：gt_bytes 和 verification_bytes 只计 payload，control_bytes 计两阶段控制字段，三项互斥且相加为 total_bytes。
网络带宽改变传输时间，不改变消息字节数。KB=1024 bytes；MB=1024² bytes。

验证 CPU 单位费用由每个 seed/d 的完整5000实例人口校准一次；四方法及各N共同复用同一不可变标尺。
续跑只复用严格核验代码、配置、数据、环境、驱动契约和snapshot身份后的原校准，不跨N重新计时或改变方法倍率。
校准snapshot SHA256：`4913dd4d5e5eadf7cde31b8b093cbdd7970323c74b64861502c84f9a2f624ea6`；本次复用已有标尺：False。

进入主图的完整配对点：45/45。
被排除配对 seed 的原因：`{}`。
只有四方法都得到同一 B 个目标最终结果、目标指纹一致且时间/字节有效才进入主图；超时不会改变共同任务量。
value 与兼容列 mean 都保存单次原始观测，std/CI 为未定义；本组没有重复测量，不能作为均值或置信区间解释。
固定任务模式不评估原生 subset/prefix 的选择优势，比较的是同一任务量下的核心流程开销。

## 方法与适配限制

RACA-HIM 保留多数支持、必要扩展、风险/联合分配、描述符验证及历史更新；GT采用一次候选上传和协调器增量比较的公开适配。
各方法共用批量assignment/result控制规则；CTCV完整分块排列由20字节seed请求重建，保留实际hash/modulo、k+1 collectors及完整CR/reports。
DVA-P、I-EDI、CTCV 采用同一核心在线边界，保留各自分配、标签/块核验和必要消息。
本组采用公开适配流程及中立 vendor 内容代理；数据集没有 authenticated vendor 原内容。
全链广播、区块链共识、签名和认证封装未模拟；实际SHA256/cosine等微内核校准后按共享节点资源倍率收费，并非完整密码学部署实测。
完整来源和不能继承的安全/长期保证见 reports/02_overhead_baseline_adaptation.md。

## 保存结果

输出目录：`E:\Code\Paper_Code\Paper_Yan\20261009_TNSE\results\02_overhead`。

- raw/overhead_runs.csv：180 条单次核心步骤观测及精确阶段汇总。
- raw/protocol_diagnostics.csv：状态、deadline 诊断与核心协议计数。
- summary/time_overhead_summary.csv、communication_overhead_summary.csv：单次配对观测。
- checkpoints/：紧凑原子保存配置；按新版本代码/配置/数据/环境指纹续跑，不复用 v2 结果。
- figures/Fig_O1_time_overhead 与 Fig_O2_communication_overhead：仅两幅图，各保存 PDF/PNG。
不导出逐消息、逐CPU操作或逐任务大体积轨迹。

本次实际生成的 Figure：`["Fig_O1_time_overhead", "Fig_O2_communication_overhead"]`。
没有有效配对样本时不补画曲线；旧同名文件不证明当前运行生成了图片。图片需要人工打开后确认裁剪和可读性；代码未冒充视觉检查。
