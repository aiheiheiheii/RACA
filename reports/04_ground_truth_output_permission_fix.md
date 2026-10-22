# 第四组输出目录 WinError 5 修复

> 此文记录历史隔离输出策略。用户之后要求移除时间戳结果目录；当前代码固定使用 `results/04_ground_truth/`，不再执行本文描述的新运行目录分配或发现。当前规则见 [固定输出说明](E:/Code/Paper_Code/Paper_Yan/20261009_TNSE/reports/04_ground_truth_fixed_output.md)。

> 后续图片位置要求：原始结果和日志仍按本文策略分运行保存；正式 PDF / PNG 已固定到 `results/04_ground_truth/figures/`，并附源运行 provenance。详见 [图片位置与评估](E:/Code/Paper_Code/Paper_Yan/20261009_TNSE/reports/04_ground_truth_figure_location_and_assessment.md)。

用户报错位于 `_prepare_output()` 的 `root.rename(archive)`，发生在 GT 实验开始之前。Windows 拒绝对整个已存在目录重命名；具体原因可能是目录 / 文件占用或目录权限，仅凭 traceback 无法确定占用程序。

修复删除了“先整体重命名已有目录”的运行依赖：

- 若 `results/04_ground_truth/` 尚为空，首次运行仍写该目录。
- 若已存在结果，新运行使用 `results/04_ground_truth_run_<run_id>/`；旧目录及其原文件全部原地保留。
- 不复制、移动、删除或覆盖旧结果，不解除系统锁，也不修改文件权限。
- 新目录名包含唯一运行 ID，禁止与现有目录合并，路径仍限制在项目 results 下的第四组命名范围。
- `run_experiment()` 返回实际 `result_root`，入口后续绘图与 manifest 更新使用同一实际路径。
- 默认 `--plots-only` 和独立绘图入口按保存 manifest 的 `started_at` 自动定位最新运行；若最新运行未完成则明确报错。显式 `--result-root` 始终优先，可指定旧结果重绘。

此修复只涉及输出目录准备和路径传播；三分之二支持门槛、排序规则、候选采样、失败模型与指标均未改动。

用户可再次在 PyCharm 直接右击 `run_ground_truth.py` 运行，无需设置管理员权限或修改 Working Directory。控制台会打印旧结果保留位置和本次实际输出目录。

本次仅完成代码修改、AST 与静态接口检查，并编写供用户后续执行的回归校验；未运行实验、测试、GT 或绘图。真实 Windows 环境中的完整运行仍待用户执行验证。
