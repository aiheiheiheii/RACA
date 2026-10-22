# 第二组 Overhead 中断现场审计

当前运行指纹：`297794eadd5fc6ded0cc9badabddd11b4b7b64399f37ce46fa1e5c554229d180`；版本：`overhead-events-v2`。

- 磁盘共有 1350 个 checkpoint；当前指纹 877 个，其中有效 877 个；其他指纹 473 个。
- 当前运行覆盖连续配置 1–877，共 7016 条方法/模式结果；当前指纹完整配对 fixed-workload seed 为 847。
- gzip/JSON、配置身份、八方法模式、行字段、消息字节分区、消息 ID/上下文/时序、任务完成/超时计数和时间分量核对发现 0 个异常配置。
- 当前结果累计消息 15,602,413 条、任务 1,540,429 条、runtime 事件 10,251,057 条。
- manifest 正式文件计数 876，待替换临时文件计数 877；progress 已记录配置 877。
- 审计开始时源文件相对中断 manifest 变更 0 个；dataset 文件变更 0 个；当前 Python/依赖环境与原运行一致：True。

## 结论

第 877 个配置的 checkpoint 已在 manifest 替换失败前成功保存。877 个当前运行 checkpoint 可作为安全续跑依据，剩余旧身份文件不能混入当前结果。原程序在 `finally` 中开始物化 raw 文件，因此当前 raw 文件并非一致提交的一组结果，必须从匹配身份的 checkpoint 重建。

记录状态汇总：`{"success": 6720, "deadline_failure": 295, "partial_workload": 1}`。

原子替换失败发生在持久化阶段，不等于算法执行失败。普通源码全指纹续跑会因 runner IO 修复触发身份变化，需要在严格核对未变更算法、配置、数据和环境的基础上兼容既有运行，不能盲目忽略指纹。

本审计只读 checkpoint、raw、logs、源码和 dataset，并仅写入本审计 JSON/Markdown；未执行完整网格、未修改结果、未生成图片。逐文件压缩文件 SHA256、结果/事件计数及每个 N/d 点的 seed 清单见同名 JSON。
