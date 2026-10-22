# 第二组静态诊断

本文件只记录代码审查发现和修复，不记录尚未运行的数值异常。

1. **GT身份间接泄漏到基线**：初稿把Proposed GT节点作为共同verifier池，虽没有传相似度，却仍让基线间接使用相似度筛选结果。改为基线独立按临时身份选择中立M个参考；共同固定任务从双方参考集合的补集交集中选取。
2. **deadline范围**：原数据集默认400ms，第二组显式800ms；GT准备和decisionCPU计入总开销，但不误用于verification-stage deadline。CTCV额外记录实际response到达时限和最终协同结果deadline。
3. **证明时序**：修正摘要被错误归到前置控制消息的风险。生成顺序为raw reorder/hash/modulo → proof发送 → receiver比较；SC必须等packed reports/CR到达才能执行Flag核验，最终结果消息在核验后发送。
4. **CTCV收集者身份**：选择阶段按seed确定k+1 collectors，执行阶段传递同一集合，避免执行函数默认2026导致两阶段身份不一致。
5. **重复计字节/时间**：每条消息只计一次实际payload和control；descriptor基础transfer不再重复添加。CPU测量与模拟操作分开，风险CPU标明已包含在decisionCPU中。
6. **工作量掩盖**：native预算和fixed强制工作量分开保存；fixed未安排目标单列，迟到任务与失败保留。只有四方法完整共同B的seed进入paired主图。
7. **异常证据丢失**：方法级异常隔离保留已经产生的通信/运行时/任务轨迹，其他方法继续；无法观察的完成数和时间填缺失而不是0。汇总已知计数并单列unknown trial数。
8. **续跑混入不同实现**：checkpoint必须匹配实现、全部相关代码SHA、配置、dataset快照和运行环境。--plots-only核对保存summary/manifest版本，不运行算法。
9. **CTCV主节点空闲身份**：改为实际logical producer接收/范围校验/转发真实proof，再由k+1 collectors核验，避免主队列仅作为虚构的并行资源。接收、校验、下一跳fanout顺序严格记录；本地collector复制不计网络消息。

未验证：这些修复经过源代码审查和静态语法检查，尚未由运行测试确认。是否存在GT反复扩张、实际deadline失败、基线重合、Proposed开销较大等现象，必须在用户正式运行后从事件和完成数判断。
