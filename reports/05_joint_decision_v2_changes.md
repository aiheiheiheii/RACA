# Joint Verification Decision v2 修改说明

本次按用户授权增强联合决策阶段。GT 构建、历史风险、描述符、资源成本、预算、deadline、统一 J 评价及两图参数网格保持原定义。没有修改 Dataset、已有结果或前四组实验，也没有运行实验、运行时校验或绘图；仅进行源码静态检查。

## 代码变化

1. `joint_methods.py`：原来的完整 marginal-J 贪心及最佳非空前缀原样保留为 `raca_him_greedy`，用于候选生成和消融。
2. RF-TS 的风险降序、最小 `L_g+c_gi` 分配、临时 ID 平局、预算/时限和停止规则均保留。仅将原构造逻辑提取为共用函数；RACA 从相同可见输入独立构造风险优先候选，不读取已保存的基线结果。
3. 新的 `raca_him` 保留三个初始候选：原贪心最佳非空前缀、完整风险优先方案、该风险优先路径的最佳非空前缀。使用原 `evaluate_joint_objective` 评价后，选最优起点。
4. 对最优起点进行最多 25 次严格增 J 的可行局部改进：单目标 verifier 重分配、预算允许时添加一个目标、非空条件下移除一个目标、将一个已选目标替换为一个未选目标并分配合法 verifier。每次实际接受前均重新调用统一 evaluator，要求 J 至少增加超过 1e-12；该阈值用于拒绝舍入级变化，没有给 J 加分。
5. 局部搜索后再次独立评价全部原始候选和改进方案，返回真实 J 最大者。原 RF 完整方案始终保留，避免局部搜索损失原始起点。

所有重分配、添加和非空移除的单移动均被考察。替换采用公开的固定候选池，每侧最多 64 个目标：进入池由高风险与高 singleton J 各取一半，重复位置按风险顺序补齐；退出池按移除后的 J 排序。它是增强阶段的有界搜索，不改变原 marginal-J 贪心遍历全部 target–verifier pair 的规则。达到 25 次上限或候选池没有改进，不代表全局最优或完整交换邻域最优。

## 论文中需要改写的算法与性质

可将联合决策阶段描述为“候选构造、可行局部改进和统一目标择优”，原贪心仍是候选构造子程序。

设原 marginal-J 贪心解为 X_g，风险优先完整解为 X_r，其最佳非空前缀为 X_p，从最佳初始候选获得的局部改进解为 X_l，则

\[
X^*=\arg\max_{X\in\{X_g,X_r,X_p,X_l\}\cap\mathcal F}J(X),
\]

其中 F 为原预算、时限、唯一分配、合法 GT 和非空约束定义的可行集合。

在风险优先方案可行且使用相同输入与同一 J 定义时，直接由候选包含关系得到

\[
J(X^*)\geq J(X_r)=J_{\mathrm{RF\text{-}TS}},
\qquad J(X^*)\geq J(X_g).
\]

若局部搜索找到严格更优的可行方案，或原贪心/风险优先前缀已优于 RF 完整方案，则有严格提升。不存在对所有配置严格 `>` 的保证；单目标唯一可行解、同质资源及其他已最优情形可以相等。

原稿定理 2 的“返回值是原贪心路径最佳非空前缀”仍适用于保留的 `raca_him_greedy` 子程序。新版整体返回值可能不再是该路径的前缀，整体性质应改写为可行性、候选非劣性与每次接受移动的严格单调改进。不能将新版曲线高于 RF-TS 单独作为“原 marginal-J 贪心优于 RF-TS”的证据；新版显式包含风险优先候选。

求解开销也需要更新：除了原 O(BVM) 贪心，还包括风险优先构造及局部搜索。令 L<=25、P<=64，一次搜索的主要移动评分数上界分别为 BG、VG、B 和 P^2 G，并有候选排序及独立评价开销。现有 `algorithm_seconds` 会将内部风险优先构造和局部搜索的时间全部计入 RACA-HIM，不转嫁给基线。

## 可复核诊断与版本

新诊断记录原贪心 J、RF 完整方案 J、RF 最佳前缀 J、最终 J、最终候选来源、局部搜索起点、各类接受移动次数、每步改进前后 J、候选池限制、评分规模和停止原因。原贪心的 prefix 诊断保留在 `legacy_greedy_diagnostics`。最终正式 CSV 仍由实验驱动中的统一 evaluator 独立重算。

配置版本改为 `joint-decision-incumbent-refinement-v2`，RACA 方法版本为 `raca-him-candidate-refinement-v2`。checkpoint 的代码/参数 fingerprint 不会复用 v1；绘图也会拒绝将旧 CSV 冒充 v2。

PyCharm 直接右击入口即可运行，无需添加参数。启动时自动比较完整来源身份：一致时继续断点；代码、参数、Dataset 或数值环境变化时，先归档旧导出，再生成新 run_id 重算完整实验。等价命令：

```powershell
python -B experiments/05_joint_decision/run_joint_decision.py
```

来源变化时，旧 raw/summary/figures 及非 checkpoint 日志复制到 `results/05_joint_decision/archive/<old_run_id>_<unique_id>/`，保留旧 manifest 原字节，并逐文件校验复制前、复制后与源文件的 SHA256。旧 checkpoint 留在 `logs/checkpoints/<old_run_id>/`，归档清单明确记录其位置。只有完整归档成功，才激活新的运行；归档失败保留旧 active manifest。运行中断后来源一致时仍正常恢复，不会每次都重新计算。

`--force` 仍可用于显式重跑来源相同的配置，也会先归档。当前旧结果在本次代码修复中没有被实际归档或覆盖；归档与新实验将在用户下次运行时执行。仅执行 plots-only 不能把旧结果变成 v2 结果。

## 校验状态

### Windows 写入中断修复

针对运行到第 319 个 trial 后出现的 `run_manifest.json` 原子替换 WinError 5，JSON、checkpoint、CSV 的原子替换现使用有界退避重试：只重试访问/共享错误（PermissionError、WinError 5/32/33），总等待最多约 8.55 秒；不会重放算法，也不会先删目标文件。若占用持续，保留原目标与完整临时文件，错误中给出临时文件位置。序列化失败后的清理异常不会掩盖原错误，异常状态清单再次保存失败也不再覆盖最初异常。

普通控制台日志改为标准输出，避免 PyCharm 将正常 `INFO Committed` 显示为红色；真实异常仍正常抛出。

本次只读核验确认当前失败运行 `4fd8cf35385a464f9fa622a6e4ecf943` 已保存 319 个完整 checkpoint，文件哈希、payload 摘要和记录身份均一致。对此次明确的 I/O 补丁提供一次严格兼容续跑：算法、数据、协议、数值环境及其他源文件必须完全相同，且旧源码版本及三个修改文件的原哈希必须命中已审计前版本。恢复时保留原 run_id、code_version、source_sha256 和 run_fingerprint，另外在 `runtime_patch_history` 中记录当前 I/O 补丁的完整源码身份。后续只有源码与该补丁记录精确相同时继续恢复，其他变化仍按新运行归档，不能反复借此放行任意算法修改。

因此用户可直接右击入口继续运行，不需要通过 `--force` 重算已有结果。此修复没有实际启动实验、更新 active manifest 或改动已保存 checkpoint。

`validate_joint_decision.py` 已同步提供用户后续运行的校验：旧贪心与字面全 pair oracle 一致、RF-TS 与独立 v1 标量参考一致、新 RACA 对两种起点及所有 RF 非空前缀不劣、可行性、不可变输入、负 J 非空、真实局部改进案例和最优情形允许相等。

新增 I/O 故障注入校验可通过 `python -B experiments/05_joint_decision/validate_joint_decision.py --check-io-only` 单独执行，不运行调度算法或读取正式数据；本次未执行。

本次未执行上述运行时校验，也未用旧 seed 试跑或生成新版结果。当前只能报告实现与静态检查完成，不能声称新版曲线已经高于 RF-TS。
