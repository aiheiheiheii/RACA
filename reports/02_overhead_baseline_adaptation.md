# 第二组核心步骤适配说明（overhead-core-v3）

最新用户要求改为单次固定工作量实验，仅比较关键步骤的O1/O2。默认45个N/d点、seed=2026、四方法共180条观测，不做30seed平均，不运行native-budget附加模式。旧v2完整协议轨迹不混入本版结果。

## 共同输入和费用边界

算法只接收原始float32 X、临时身份和可见历史。标签、prototype、sample provenance不进入参考构建、选择或判定；common、dataset和第一组代码不修改。基线对原始字节做真实hash/proof，不能借用RACA-HIM的相似度或GT。

各方法使用同一资源profile、相同B=ceil(.1(N−M))个固定目标和中立控制编码。各方参考身份独立确定，目标从排除各参考身份的共同域抽取。基线独立对M个内容盲选参考的raw SHA256频数/历史/临时ID选择代理原内容，不共享另一方法的hash、预测或证明。

所有方法统一批量dispatch；普通验证使用每verifier一个bitmap结果，CTCV保留其自身SC聚合结果。assignment header为64字节，目标行号每个4字节；普通result header为32字节，加ceil(本批任务数/8)字节。descriptor、digest、modulo tag、packed proof、异常身份取实际序列化长度，没有人为倍率。

## RACA-HIM

依据最新TNSE稿件Section IV–V。保留历史排序、K=50/M=25多数支持、每25个必要扩展、动态多数阈值、风险、greedy联合分配、cosine判定及历史更新。固定B实验保留greedy的风险/时延增益和分配，但不选择较短prefix或因800ms截断任务。

稿件规定候选间支持计算，没有规定每份描述符都必须向所有候选重复网络复制。核心版本明确采用可信协调器：新增候选各请求/上传一次，缓存原始描述符、归一化和已有pair。余弦对称计算同时更新双方支持；扩展只计算新pair，重新应用新的多数阈值。GT身份通知保留；每个被验证目标的原始4d字节也保留。

这是公开的协调器部署适配，不声称已经测量去中心化全连接交换的完整协议。GT选择和支持计数通过原内核等值测试；不能通过固定更小候选池或取消必要扩展降低费用。

## DVA-P-Adapted

依据 Zhao等2024《Long-Term Over One-Off: Heterogeneity-Oriented Dynamic Verification Assignment for Edge Data Integrity》，PDF页7 Section IV-A Algorithm1及Eqs.13/21。源原生增补/替换、漂移加惩罚及队列更新仍在适配模块中，但固定任务实验绕过源subset选择；只比较相同B任务的中立预计完成时间分配和执行流程，不能验证原长期动态选择优势。

原文没有唯一指定本场景的密码插件，执行采用公开的真实raw SHA256与独立正确摘要精确比较。32字节digest、正确摘要分发、challenge、核心结果均保留。不是完整PDP/bilinear网络部署，也不继承原安全和长期保证。

## I-EDI-Adapted

依据 Zhao等2025《Intelligent Edge Data Integrity Verification With Dynamic Unreliable Data Replica Selection》的Eqs.4/8/12/20、Algorithm1 O²RS、Algorithm2/Eq39及Section V-A。原O²RS/队列实现保留，但固定任务实验不运行native subset sweep，不能声称测量了动态选择优势。

保留实际SHA256(raw bytes) mod7919、2字节tag、独立代理正确tag及真实精确判定。full SHA只作诊断，不推翻有限模碰撞；相似度不进入该方法。未知长期历史用中立冷启动条件，不制造有利历史。

## CTCV-Adapted

依据 Zhao等2025《Collusion-Resistant and Time-Aware Co-Verification for Edge Data Integrity》：PDF页5 Algorithms1/2与Eqs.4a/4b；页6 AV.Inquiry、CR与Algorithm3；页9 Section V-C的ell=1000n。

保留原始字节非空分块/重排、真实SHA256 mod ell、k+1收集者、producer fanout、AV.Inquiry correct combination、每份完整packed CR/R、SC四Flag及CR[actual_tag]成员位定位。ell=1000B、k=1没有因期望趋势而缩小。目标响应全部真实计算。

原版本每目标重复发送完整u32边界/排列，形成B×min(B,4d)项。关键步骤transport使用真实20字节 `<QIII>` 请求：uint64 seed、uint32 requested blocks、payload字节数、cyclic start。各目标实际解包、重建完整边界/排列，再执行原字节重排、hash和modulo。目标重建开销仍计入其operations；AV设置partition也计入。AV仅缓存同一request/anchor/ell的重复cyclic正确挑战，不缓存目标响应或verdict。

请求→真实response→producer/collector fanout→全部响应结束→AV.Inquiry→collector请求/packing→CR/R到达→SC核验→最终结果的依赖保留；不同主节点并行，同节点CPU及完整任务串行。producer自己为collector的本地操作不虚构网络发送。

原完整链广播、共识、签名和认证封装没有模拟；这里的逻辑transport是明确适配。完整广播本身可能产生更高增长阶，不能预设原论文曲线必然与其他方法一致。新版趋势仅由实际消息数量/长度产生，不拟合曲线或修改值。

## 时间、通信及完成定义

准备阶段所有方法统一采用实测核心kernel CPU＋模拟必要准备消息；风险/分配CPU实测，验证kernel/通信按同一公开资源模型，必要历史更新CPU加一次。实际验证CPU单独诊断，不与模拟验证CPU重复相加。网络等待不会记成CPU，文件加载/输出不属于在线费用。

verification cosine采用dataset计算模型；实际hash对应200MiB/s、vector处理1000MiB/s、scalar/modulo 1µs并按节点资源factor调整。这些是公开仿真假设，不是密码学实测。controller/SC为逻辑端点；链路镜像参与节点资源，controller–SC镜像node0，独立全双工不模拟共享链路争用。

O2来自每条真实逻辑发送的精确累加；逐事件对象不持久化，阶段/消息类型的count、payload、control、total汇总保存。三类字节互斥相加，无内存copy或CPU当网络费用。

本组是完整同任务量的开销观测：四方法全部B目标有最终结果且数据有效才绘图。800ms与CTCV响应限时保留为deadline诊断，不能删除超时观察来减小曲线。success表示协议最终完成，within_deadline_count单独表示及时完成。

数据缺少authenticated vendor original，代理anchor不保证可信，合法异构模型也可能被exact-hash判异常。本实验既不确认原协议在各自正确原内容下的准确率，也不评价长期学习、安全或完整模型传输。仅一次seed不提供均值稳定性或置信区间证据。

本次执行了独立代码验证与两个真实数据角点内存检查；完整45点和最终图由用户运行入口生成。实现不会强制RACA-HIM最低或强制CTCV曲线与其他方法相同。
