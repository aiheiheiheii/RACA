# 四方法一致性验证适配说明（hash-baselines-v4）

用户最新要求：基线不能采用相似度度量，应保留原文哈希/摘要比对；移除Single-Reference；只绘制检测精度D1/D2，断轴仿照TMC Fig.3。本版据此移除三个基线所有cosine、距离、容差、向量归一化及相似度参考选择。RACA-HIM仍按TNSE定义使用cosine。

此前descriptor-adapted-v2的相似度基线实现与本版不同，旧数值不能改标为哈希基线结果。本次没有执行Quick/正式实验/测试/新图，仅修改并静态检查源码。

## 共同输入和内容假设

100参考池由独立seed随机划分，剩余N−100模型为四方法共同目标；临时ID和来源/标签分离。算法只收到原始float32描述符、历史、临时ID及划分，不收到prototype、prototype相似度、sample_id或真实标签。成功时必须预测全部共同目标。

基线把同family的模型迁移为“应具有同一内容的副本”。哈希输入仅固定little-endian float32/C-order序列化，保留原始值和位模式，不normalize、不round、不quantize、不加容差。合法个性化造成不同内容时，真实哈希不一致会造成误报；不会通过相似度补救，也不预设误报数或Accuracy。

数据没有独立可信vendor原始内容，因此I-EDI和CTCV独立调用exact_hash_anchor：参考池中完整SHA256重复数最多的内容作为代理原内容，同频按较低历史异常状态及较小临时ID。全部不同hash时按上述中立平局选择。这不是可信真值认证，也不使用RACA-HIM的GT。Dyna比较全部100参考见证。此假设必须随本组结果说明，不能推论这些原协议在每个个性化模型拥有独立正确原内容时也低精度。

## I-EDI-Adapted

原文Sec.III Eq.(4)/(8)的VR和R、Eq.(12)/(20)虚拟队列目标、Sec.IV Algorithm1 O²RS排除贪心保留。静态数据无QoS与真实历史，固定QoS1、SR=ER=.5、β=.5，R=.75；Q初始0、Υ1、γ=min(ceil(.1×目标数),剩余数)。γ为最少审计数量，不是最大预算。每候选末按原line10更新队列，接受严格改善的排除，多批覆盖全部目标且各目标一次。

Algorithm2/Eq.(39)真实计算`int(SHA256(raw content),16) mod η`并与代理原内容正确tag精确比较。原PDF第9页/印刷13197的V-A Experimental Setup明确β=.5、C=1、Υ=1、η=7919，本版恢复该η默认值，保留实际有限模碰撞。上一版2^256使模运算等同完整SHA比较，是此前复现遗漏。原Algorithm2随机挑战与Theorem6的nonce-prefixed hash定义不同，本版按Eq39及V-A实验路径，不混用其安全保证。O²RS只决定审计顺序，不靠小子集提高Accuracy。

诊断保留参考完整hash/ID、正确tag、tag流SHA、相等目标ID、hash与mod-tag碰撞计数、批次及队列。达到计算上限未完整覆盖则failure，无子集Accuracy。

## CTCV-Adapted

依据原Sec.III Algorithms1–3、Eqs.(4a)–(9)。原文Algorithm1按n个副本划n块；本版默认以min(共同目标数,原始字节数)确定非空块数量并shuffle，30维float32的120字节上限对应120块，S.Response按target临时ID确定循环重排，生成真实块摘要及完整重排内容SHA256。Eq.(4b)的`r_i=hash(P_i) mod ℓ`保留；代理原内容按同一个target的块序构造正确σ_i。

紧凑120字节descriptor不足以划出900..4900个非空块，因此保留能表示的最大非空块数；`ctcv_block_count=0`表示此长度限制策略，正值为显式预设override。此上限使原node↔block的1:1映射不能完整保留，报告明确为输入长度适配；不是根据结果调参。上一版固定5块进一步缩小了CR可用位置种类，已修正。原实验采用ℓ=1000n，本组n映射为共同目标数量；默认`ctcv_proof_vector_length=0`表示按该公式计算。每collector实际核验完整响应并将r_i位置置1生成原ℓ长组合位向量，CR由正确σ_i对应位置置1产生。k+1=2个参考collector各自构建组合，按原四Flag逻辑比较/定位。原inq.message及SC只持有聚合CR而非每target的σ_i映射，原PDF第6–7页Eq5前按r_ji对应的CR位是否为1检查；故Flag2/3定位谓词是`CR[actual_tag]==0`。上一版按`actual_tag!=individual_sigma_i`定位，额外使用聚合时丢失的关联，已移除。逐target正确tag仅用于AV构造CR及辅助碰撞统计。response_age统一1，T1；无实测latency或串谋注入。

组合与模哈希碰撞可能遮蔽不一致，本版保留实际位向量结果，不使用更强的完整hash辅助信息逆改SC判定。辅助完整hash差异仅用于解释碰撞。Flag3若无真实追加报告和唯一quorum则failure，不伪造投票。保存实际mod标签、稀疏组合置位位置/摘要、collector及Flag/VR证据，不把百万维向量重复写入JSON。

本地SHA承诺用于响应一致性检查，不提供原密码学认证或BFT安全保证。原哈希和位向量决定二元结果，未通过相似度适配。

## DynaEDI+-Adapted

原III-A Step1.1至少ceil(2n/3)确认rt，III-B Step2.2版本匹配公共子树与digest响应，III-C Step3.3严格不一致数>ceil(2n/3)，IV Eq.(5)–(7)盐化叶哈希与Szudzik配对/逆展开保留。

同descriptor family、5个原始float32连续特征块、版本0。叶摘要为真实`SHA256(raw_block_bytes || rt)`，父节点用任意精度Szudzik函数构造，可用整数平方根逆展开。比较目标完整root与100个已承诺witness root的严格相等/不等，ACK门槛67，root不一致68用于筛选展开候选。原III-C Step3.3最终在block层面判损坏，本版模型级汇总采用“至少1个块达到严格68门槛才判Corrupted”。root不一致不再直接成为最终标签。只用实际digest计票，不做cosine或距离判断。

子树和叶定位同样依据真实digest不一致quorum，沿真实unpair结果递归展开；不再有比例距离预算。若root候选未形成任一块quorum，最终判Normal并记录未定位root候选；至少1个块满足门槛才Corrupted。Counter仅压缩同值digest的真实见证次数，与逐见证精确比较完全等价。

SHA256本地响应承诺绑定family/ID/nonce/版本/root，但不提供BLS身份认证。本组不模拟动态更新、修复或恶意认证攻击。记录全部witness ID/indices、逐目标digest不一致数、nonce、示例真实root/展开叶、定位与承诺摘要；verifier_ids=None，避免虚报单判定者。

## RACA-HIM和输出

RACA-HIM按最新TNSE稿件构建多数支持GT，K50/M25，必要时按25扩至100；各target只有一个GT verifier，循环分配，不使用TMC委员会投票。它是本组唯一采用cosine的算法。新方法名只改变对外身份，不改变检测步骤。

Single-Reference的函数、注册、配色和当前测试断言均移除；历史结果仍保留原身份。参数网格未缩小：65配置×30seed=1950数据trial、4方法7800记录、260个方法汇总点。

只绘制D1/D2 Accuracy。TMC Fig.3的分组柱状图、原四色、上下段比例、双侧双斜断轴及相邻框线隐藏已写入公共样式。共享图例框与子图顶边间距固定3pt。纵轴使用Accuracy的0..1刻度，上段默认.94..1.02、下段0..18%；图中Mean annotation移除，子图标题间距收紧。真实均值进入省略区间时自动调整显示段或完整轴，未隐藏数据。日志只登记D1/D2；诊断/subtype CSV可离线保留，额外图不会再生成。

重绘允许保存版本hash-baselines-v3/v4，严格核对CSV与manifest并记录实际数据版本；旧v3重绘不会得到v4预测。当前算法完整入口生成v4，旧cosine结果拒绝。本次只读分析了用户已保存的v3结果，未执行v4算法/测试/实验或出图。三方法重合原因及逐项修正见01_detection_baseline_overlap_analysis.md，未宣称v4已有数值或性能优越性。

## 原文依据

- TNSE：`C:/Users/MengKe/Desktop/Tutu/TNSE/Latest version of the paper.md`。
- I-EDI：`C:/Users/MengKe/Desktop/Tutu/TNSE/参考文献/对比实验/Zhao 等 - 2025 - Intelligent Edge Data Integrity Verification With Dynamic Unreliable Data Replica Selection.pdf`。
- CTCV：`C:/Users/MengKe/Desktop/Tutu/TNSE/参考文献/对比实验/Zhao 等 - 2025 - Collusion-Resistant and Time-Aware Co-Verification for Edge Data Integrity.pdf`。
- DynaEDI+：`C:/Users/MengKe/Desktop/Tutu/TNSE/参考文献/对比实验/Yang 等 - 2025 - DynaEDI+ Reliable and Decentralized Integrity Verification for Dynamic Edge Data.pdf`。
- TMC Fig.3（PDF第9页）仅用于图形：`C:/Users/MengKe/Desktop/TMC.pdf`。
