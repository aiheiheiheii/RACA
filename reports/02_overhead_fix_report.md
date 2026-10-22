# 第二组无图片问题修复：overhead-events-v2

用户运行的v1清单记录1350配置/10800 method-mode行，所有行status=failure，消息数为0。原始traceback直接指向EventSimulator.pair内的float(v)：resource_costs返回shape=(1,)的数组，当前NumPy2.5不能将非0维数组直接转为Python标量。这是执行错误，不是算法实际超时或绘图漏画。

## 修复

- 单pair结果显式np.asarray(v).item()后转float，不修改dataset资源接口、资源值或deadline。
- CTCV改为依赖图+按ready time推进的事件队列。不同primary的网络请求并行，只有同一节点的实际CPU操作排队；每个primary自身仍整任务串行。移除了按Python任务遍历顺序预留未来collectorCPU引起的伪全局屏障。
- Proof接收/范围核验后才能forward；AV.Inquiry供给CR到SC后SC请求collectors；所有报告到达后才执行SC核验和最终verdict。迟到/不完整任务仍保留，未放宽主图配对筛选。
- 所有方法/模式共有执行异常时，保存第一个配置证据后停止并报execution_failed，避免重复同一错误且称记录完成。
- 版本提升到overhead-events-v2；新版本/代码指纹拒绝复用旧失败checkpoint。无需手动删除旧文件，普通完整运行会重算。
- 图例、注册和新结果统一RACA-HIM。绘图采用第一组物理常量：画布7.16×1.50in，panel绘图区82.7057809082×64.26252pt。保持第二组连续轴，标题自动分两行。

## 验证

93项第二组测试全部通过（NumPy2.5.4，用户项目.venv）。覆盖单pair数组提取、两primary同为collectors仍并行、交换独立任务顺序不改变makespan、节点CPU无重叠、消息/报告依赖、字节守恒、共同工作量、seed级CI、checkpoint和45个N/d数据加载等。绘图测试只在内存中验证几何/图例/文字边界，没有保存测试图或作为论文数据。

针对性真实数据验证使用seed2026，仅在内存执行两个配置，不写正式raw/summary/checkpoint/figures：

| 配置 | Fixed预算 | RACA-HIM完成 | DVA-P完成 | I-EDI完成 | CTCV完成 |
|---|---:|---:|---:|---:|---:|
| N1000,d10 | 98 | 98 | 98 | 98 | 98 |
| N5000,d50 | 498 | 498 | 498 | 498 | 498 |

四方法两配置均为fixed success。Native的I-EDI仍保留自己的97/497选择数；N5000,d50的DVA-P Native实际超时，完成464/498，未隐藏或改判success。这说明合法协议失败仍会被记录，修复没有通过放宽deadline制造图。

完整配置细节见02_overhead_fix_validation.json。两配置验证不能代替完整1350配置或30seed统计，没有承诺所有seed都满足deadline或RACA-HIM开销最低。

## 接下来运行

使用PyCharm直接右击experiments/02_overhead/run_overhead.py完整运行。旧v1数据全部失败，--plots-only无法产生真实曲线。新运行完成后O1/O2保存于results/02_overhead/figures，实际有效seed数和失败原因仍如实记录；本次没有代跑完整网格或生成论文图片，也未更改第一组代码、第一组结果或dataset。
