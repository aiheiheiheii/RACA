# 第二组静态检查

2026-10-12：34个 experiments/tests Python文件通过 AST parse 和 compile 语法检查，其中第二组9个Python文件。compile只编译语法树，没有执行任何项目模块或生成字节码。详细文件SHA256记录见02_overhead_static_checks.json。

第二组独立测试文件含32个test函数，另有N/d、配置、方法、checkpoint等参数化用例。**这不是32项测试通过**：未运行pytest、数据加载实验、完整配置或绘图。

静态复核确认：

- 入口自定位项目；--plots-only不导入模拟器；没有Quick。
- 正式9N×5d×30seeds，两种模式四方法，固定800ms验证阶段deadline。
- 无标签/原型/来源ID的算法输入边界；基线独立中立参考身份和实际raw hash。
- Proposed greedy继续生成非正增益之后的可行前缀，再选最大J。
- 单节点任务串行、不同主verifier并行；消息、operation、CTCV报告和最终结果依赖可追踪。
- 字节按消息事件求和，payload/control三分项互斥；实测CPU与模拟时间分列。
- 配对seed统计、真实失败/部分工作量/未知计数，原子checkpoint与兼容身份检查。
- 第一组、common、dataset未编辑；第二组结果尚未生成。

实际运行仍可能发现错误或资源瓶颈。PyCharm入口可用性、测试断言、算法数值结果、完整运行耗时、实际断点恢复和PDF/PNG视觉质量需要用户运行后确认；本报告没有将静态通过等同于运行通过。
