"""Paired full-coverage experiment, persisted evidence, and seed-level statistics."""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time
import traceback

import numpy as np
from scipy.stats import t
from threadpoolctl import threadpool_limits

from experiments.common.config import (
    DATASET_ROOT, PROJECT_ROOT, DetectionConfig, SEEDS, configurations, DETECTION_IMPLEMENTATION_VERSION,
)
from experiments.common.dataset_adapter import load_evaluation_data
from experiments.common.metrics import binary_metrics, subtype_metrics
from experiments.common.method_names import RACA_HIM, normalize_method_rows
from experiments.common.reproducibility import (
    dataset_snapshot, environment_metadata, sha256_array, verify_manifest, write_csv,
)
from .detection_baselines import METHODS, IMPLEMENTED_METHODS, METHOD_FUNCTIONS
from .proposed_detection import DetectionResult

METRICS = ("accuracy", "precision", "recall", "f1", "fpr", "fnr")
COUNTS = ("TP", "TN", "FP", "FN")
STATUSES = ("success", "gt_unavailable", "unsupported", "failure")


def run_configuration(N, cr, seed, config, dataset_root=DATASET_ROOT):
    """Return deterministic rows and per-target evidence for one paired trial.

    All algorithms receive only observations. Labels are read in this function
    after each prediction, including the explicitly post-hoc GT contamination audit.
    Failed methods retain the same target fingerprint, with no predictions.
    """
    prepared = load_evaluation_data(N, cr, seed, config, dataset_root=dataset_root)
    data = prepared.observations
    indices = data.evaluation_indices
    true = prepared.binary_label[indices]
    subtypes = prepared.subtype_label[indices]
    targets = data.model_ids[indices]
    target_hash = sha256_array(targets)
    base = {
        "implementation_version": DETECTION_IMPLEMENTATION_VERSION,
        "seed": int(seed), "N": int(N), "cr": float(cr), "d": config.dim, "H": config.H,
        "tau_lower": config.tau_lower, "tau_upper": config.tau_upper,
        "K": config.K, "M": config.M, "h0": config.h0, "p_fail": config.p_fail,
        "reference_pool_size": len(data.reference_indices),
        "iedi_budget_fraction": config.iedi_budget_fraction, "iedi_control_V": config.iedi_control_V,
        "iedi_tag_modulus": config.iedi_tag_modulus,
        "baseline_block_count": config.baseline_block_count,
        "ctcv_collusion_bound": config.ctcv_collusion_bound, "ctcv_time_limit": config.ctcv_time_limit,
        "ctcv_proof_vector_length": config.ctcv_proof_vector_length,
        "ctcv_block_count": config.ctcv_block_count,
        "dyna_quorum_fraction": config.dyna_quorum_fraction,
        "target_count": len(indices), "target_ids_sha256": target_hash,
        "descriptor_sha256": sha256_array(data.X),
        "realized_cr_percent": 100 * float(prepared.binary_label.mean()),
        "evaluation_cr_percent": 100 * float(true.mean()),
    }
    arrays = {
        "evaluation_indices": indices, "target_ids": targets, "binary_label": true,
        "subtype_label": subtypes, "reference_indices": data.reference_indices,
        "reference_ids": data.model_ids[data.reference_indices],
    }
    rows, subtype_rows = [], []
    for method in METHODS:
        try:
            result = METHOD_FUNCTIONS[method](data, config)
        except Exception as exc:
            result = DetectionResult(method, "failure", None, targets, None, None, None,
                                     f"{type(exc).__name__}: {exc}",
                                     diagnostics={"exception_traceback": traceback.format_exc()})
        if not np.array_equal(result.target_ids, targets):
            raise AssertionError(f"{method} changed the common evaluation targets")
        if result.method != method or result.status not in STATUSES:
            raise AssertionError(f"{method} returned an invalid method/status record")
        diagnostics_json = json.dumps(result.diagnostics, ensure_ascii=False,
                                      separators=(",", ":"), default=_numpy_json)
        arrays[f"{method}_diagnostics_json"] = np.asarray(diagnostics_json)
        row = {**base, "method": method, "status": result.status,
               "evaluated_count": 0,
               "gt_success": result.gt_success if result.ground_truth is not None else None,
               "reference_ready": result.gt_success if result.gt_success is not None else result.status == "success",
               "diagnostics_json": diagnostics_json,
               "gt_size": 0, "gt_final_K": np.nan, "gt_threshold": np.nan,
               "gt_corrupt_count_posthoc": np.nan, "gt_corrupt_fraction_posthoc": np.nan,
               "gt_ids_json": "[]", "gt_expansion_trace": "[]", "reason": result.reason,
               "undefined_metrics": "all", **{name: np.nan for name in METRICS + COUNTS}}
        if result.ground_truth is not None:
            gt = result.ground_truth
            selected = gt.selected_indices
            row.update({"gt_size": len(selected), "gt_final_K": len(gt.candidate_indices),
                        "gt_threshold": gt.threshold,
                        "gt_expansion_trace": json.dumps([asdict(x) for x in gt.expansion_trace]),
                        "gt_ids_json": json.dumps(data.model_ids[selected].tolist())})
            # Evaluation-only diagnostic, never fed back into the GT or detector.
            if gt.success:
                corrupt_count = int(prepared.binary_label[selected].sum())
                row["gt_corrupt_count_posthoc"] = corrupt_count
                row["gt_corrupt_fraction_posthoc"] = corrupt_count / len(selected)
                arrays[f"{method}_gt_indices"] = selected
        if result.predictions is not None:
            if (result.status != "success" or result.predictions.shape != true.shape
                    or not np.isin(result.predictions, (0, 1)).all()):
                raise AssertionError(f"{method} returned incomplete predictions")
            row.update(binary_metrics(true, result.predictions))
            row["evaluated_count"] = len(true)
            arrays[f"{method}_predictions"] = result.predictions
            if result.verifier_ids is not None:
                if result.verifier_ids.shape != targets.shape:
                    raise AssertionError(f"{method} returned incomplete single-reference assignments")
                arrays[f"{method}_verifier_ids"] = result.verifier_ids
            for diagnostic in subtype_metrics(true, result.predictions, subtypes):
                diagnostic = dict(diagnostic)
                diagnostic["subtype_name"] = diagnostic.pop("name")
                subtype_rows.append({**base, "method": method, "status": result.status,
                                     **diagnostic})
        else:
            if result.status == "success":
                raise AssertionError(f"{method} reported success without predictions")
            # Retain absent/failed method subtypes too; no invented error rate.
            for code, name in enumerate(("common_normal", "high_personalized_normal",
                                         "severe_corrupt", "mild_corrupt")):
                subtype_rows.append({**base, "method": method, "status": result.status,
                                     "subtype": code, "subtype_name": name,
                                     "count": int(np.count_nonzero(subtypes == code)),
                                     "errors": np.nan, "error_rate": np.nan,
                                     **{key: np.nan for key in METRICS + COUNTS},
                                     "undefined_metrics": "all"})
        rows.append(row)
    return rows, subtype_rows, arrays


def _numpy_json(value):
    """Persist protocol evidence as Unicode JSON, never pickle/object arrays."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Unsupported diagnostic type: {type(value).__name__}")


def summarize_runs(rows):
    """Student-t 95% intervals of seed-level means, never instance-level CIs."""
    rows = normalize_method_rows(rows)
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["N"], row["cr"], row["method"])].append(row)
    summaries = []
    for (N, cr, method), trials in sorted(grouped.items(), key=lambda x: (x[0][0], x[0][1], METHODS.index(x[0][2]))):
        versions = {row.get("implementation_version", "unrecorded_legacy") for row in trials}
        if len(versions) != 1:
            raise ValueError("Cannot combine metrics from different detector versions")
        counts = Counter(row["status"] for row in trials)
        valid = [row for row in trials if row["status"] == "success"]
        attempted = [row for row in trials if row["gt_success"] is not None]
        summary = {"N": N, "cr": cr, "method": method, "run_count": len(trials),
                   "implementation_version": next(iter(versions)),
                   **{f"{status}_runs": counts[status] for status in STATUSES},
                   "gt_attempts": len(attempted),
                   "gt_success_rate": (sum(bool(row["gt_success"]) for row in attempted) / len(attempted)
                                       if attempted else np.nan),
                   "prediction_total": sum(row["evaluated_count"] for row in trials),
                   "statistics_scope": "successful seed-level trials; failures retained separately"}
        for metric in METRICS + COUNTS:
            values = np.array([row[metric] for row in valid], dtype=float)
            values = values[np.isfinite(values)]
            n = len(values)
            mean = float(values.mean()) if n else np.nan
            std = float(values.std(ddof=1)) if n > 1 else np.nan
            half = float(t.ppf(.975, n - 1) * std / np.sqrt(n)) if n > 1 else np.nan
            summary.update({f"{metric}_n": n, f"{metric}_mean": mean, f"{metric}_std": std,
                            f"{metric}_ci_low": mean - half, f"{metric}_ci_high": mean + half})
        summaries.append(summary)
    return summaries


def summarize_subtypes(rows):
    """Seed-level subtype errors, separately retaining failures and absent strata."""
    rows = normalize_method_rows(rows)
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["N"], row["cr"], row["method"], row["subtype"], row["subtype_name"])].append(row)
    summaries = []
    for (N, cr, method, code, name), trials in sorted(grouped.items()):
        values = np.array([row["error_rate"] for row in trials if row["status"] == "success"], dtype=float)
        values = values[np.isfinite(values)]
        n = len(values)
        mean = float(values.mean()) if n else np.nan
        std = float(values.std(ddof=1)) if n > 1 else np.nan
        half = float(t.ppf(.975, n - 1) * std / np.sqrt(n)) if n > 1 else np.nan
        summaries.append({"N": N, "cr": cr, "method": method, "subtype": code, "subtype_name": name,
                          "run_count": len(trials), "success_runs": sum(r["status"] == "success" for r in trials),
                          "error_rate_n": n, "error_rate_mean": mean, "error_rate_std": std,
                          "error_rate_ci_low": mean - half, "error_rate_ci_high": mean + half,
                          "statistics_scope": "mean of defined seed-level subtype error rates"})
    return summaries


def _json_write(path, payload):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def generate_report(output, rows, summaries, manifest, elapsed):
    """Describe only this run's four detectors and measured accuracy."""
    from experiments.common.plotting_style import DETECTION_FIGURE_WIDTH, DETECTION_FIGURE_HEIGHT
    rows = normalize_method_rows(rows)
    summaries = normalize_method_rows(summaries)
    output = Path(output).resolve()
    statuses = Counter(row["status"] for row in rows)
    lines = ["# 第一组实验：合法异构与模型损坏识别精度", "",
             "实现版本：hash-baselines-v4；正式完整参数网格。",
             f"实际执行 {len(rows)//len(METHODS)} 个配对配置/seed，共 {len(rows)} 个方法记录；耗时 {elapsed:.2f} 秒。", "",
             "## 四方法实现", "",
             f"- {RACA_HIM}：最新TNSE稿件的多数支持GT和单GT目标判定；K=50、M=25，必要时按25扩至100。只有{RACA_HIM}使用cosine区间[0.997,1]。",
             "- I-EDI-Adapted：原可靠性公式、O²RS排除贪心和虚拟队列；批次覆盖全部目标；原始内容SHA256后实际计算hash(s) mod η并比较正确tag。默认η=7919，依据原文V-A实验设置；有限模碰撞由实际hash决定。",
             "- CTCV-Adapted：原始字节分块重排、SHA256后mod ℓ、原组合位向量与k+1收集者四Flag/VR流程；默认ℓ=1000×共同目标数；原SC只保留CR位向量，定位使用CR[actual_tag]成员位。紧凑描述符以min(目标数,原始字节数)确定最大非空块数；不看结果选参数。",
             "- DynaEDI+-Adapted：原始块SHA256、nonce/版本公共子树、Szudzik可逆树；完整root严格不一致数>ceil(2R/3)触发展开；至少一个块达到该quorum才将模型判损坏，R=100时门槛68；只比较子树和叶digest。",
             "- I-EDI/CTCV缺少vendor原内容，独立按参考池完整hash重复数、历史和临时ID选一个代理原内容锚点；没有相似度、原型或标签参与选择。Dyna使用全部100参考见证。",
             "- 三个基线将当前同family模型按同内容副本验证；合法个性化产生不同字节时可能误报。本实验测量这种迁移设置，不能解释为原协议在各模型拥有独立正确原内容时的性能。",
             "- 固定100参考池及N−100共同目标，算法输入仅描述符、临时ID、历史和划分；真值只在预测后计算指标。完整输入SHA/目标指纹对所有方法相同。",
             "- 静态p_fail=0、版本0与中立QoS/时限；本地SHA承诺不等于BLS或真实网络认证。详细依据见reports/01_detection_baseline_adaptation.md。", "",
             "## 参数与统计", "",
             "- D1：N=1000..5000每500，五子图固定cr=1..5%；D2：cr=1..5%每0.5%，五子图固定N=1000,2000,3000,4000,5000。",
             "- 两图配置去重后65个点，seed=2026..2055，完整1950个数据trial；四方法应产生7800个方法记录。重叠点只运行一次。",
             "- Accuracy=(TP+TN)/(TP+TN+FP+FN)，Corrupted=1。每点为定义有效的成功seed均值；失败不伪报Normal，保留状态/原因和有效n。",
             "- CSV保留均值、标准差、95% Student-t CI及其他指标/四类错误记录；图仅绘制Accuracy均值，不附加阴影或误差带。",
             "- 仅D1/D2绘图；默认柱状图，TMC Fig.3双区间断轴、两侧双斜线、隐藏相邻横框线；纵轴为Accuracy的0..1数值。默认下段0..0.18、上段0.94..1.02。",
             "- 图内不显示Mean detection accuracy；子图标题更靠近横轴，共享图例框距子图顶边3pt。若真实均值进入默认省略区间，则扩大显示区间或退回完整轴，不隐藏真实结果。",
             f"- 图幅{DETECTION_FIGURE_WIDTH:g}×{DETECTION_FIGURE_HEIGHT:.2f}英寸；子图绘图区采用TMC Fig.3的原始点尺寸，上下外部白边收紧；logs/plot_style.json记录各段的实际物理尺寸。",
             f"- 配色{RACA_HIM}蓝、I-EDI橙、CTCV绿、DynaEDI+红；可选折线全部实线。95%CI刻画固定合成库重复抽样，不代表真实部署总体。", "",
             "## 实际执行状态", "", "| 状态 | 方法记录数 |", "|---|---:|",
             *[f"| {status} | {statuses[status]} |" for status in STATUSES], "",
             "## 实际检测精度", "",
             "以下为各配置Accuracy均值的等权汇总；未把不同配置作为独立重复构造总体CI。", "",
             "| 方法 | 配置平均Accuracy (%) | 成功记录 | 不可用/失败记录 |",
             "|---|---:|---:|---:|"]
    for method in METHODS:
        values = np.asarray([float(row["accuracy_mean"]) for row in summaries if row["method"] == method])
        values = values[np.isfinite(values)]
        accuracy = 100 * float(values.mean()) if len(values) else np.nan
        count = sum(row["method"] == method and row["status"] == "success" for row in rows)
        total = sum(row["method"] == method for row in rows)
        lines.append(f"| {method} | {accuracy:.6f} | {count} | {total-count} |")
    lines.extend(["", "## 输出", "",
                  f"- 结果根目录：`{output}`。",
                  "- raw/detection_runs.csv及raw/predictions/*.npz：真实预测、状态、共同目标/参考临时ID、GT及hash协议诊断JSON；无pickle对象。",
                  "- raw/subtype_metrics.csv及summary/subtype_summary.csv：保留离线错误诊断，不绘制额外Figure。",
                  "- summary/detection_summary.csv：每配置seed级均值、标准差、CI和状态统计。",
                  "- figures/Fig_D1_detection_vs_N.pdf/png及Fig_D2_detection_vs_cr.pdf/png：两张检测精度图，矢量PDF和600dpi PNG。",
                  "- logs/plot_style.json记录实际断轴区间和样式；run_manifest记录完整参数、代码SHA和环境。",
                  f"- Dataset manifest：{manifest}；完整运行前后dataset树SHA256相同。",
                  "- 入口不运行Quick、测试套件或重复预试跑；历史测试/结果不代表本版通过运行时验证。", ""])
    report = output / "report.md"
    report.write_text("\n".join(lines), encoding="utf-8")
    destination = PROJECT_ROOT / "reports" / "01_detection_report.md"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(report.read_text(encoding="utf-8"), encoding="utf-8")


def run_experiment(output_dir=None, plot_kind="bar"):
    if plot_kind not in ("bar", "line"):
        raise ValueError("plot_kind must be bar or line")
    output = Path(output_dir or PROJECT_ROOT / "results" / "01_detection").resolve()
    for child in ("raw/predictions", "summary", "figures", "logs"):
        (output / child).mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    print(f"TNSE Experiment Group 01\nDataset: {DATASET_ROOT}\nMode: FULL", flush=True)
    unsupported_methods = ()
    print("Implemented methods: " + ", ".join(IMPLEMENTED_METHODS), flush=True)
    snapshot = dataset_snapshot(DATASET_ROOT)
    manifest = verify_manifest(DATASET_ROOT)
    # Missing or altered source files must fail before producing scientific results.
    if manifest.get("missing") or manifest.get("mismatch") or manifest.get("mismatches"):
        raise RuntimeError(f"Dataset manifest verification failed: {manifest}")
    seeds = tuple(SEEDS)
    grid = list(configurations())
    config = DetectionConfig()
    _json_write(output / "logs" / "dataset_before.json", snapshot)
    _json_write(output / "logs" / "run_manifest.json", {
        "mode": "FULL", "config": asdict(config), "seeds": seeds,
        "implementation_version": DETECTION_IMPLEMENTATION_VERSION, "plot_kind": plot_kind,
        "source_sha256": {path.relative_to(PROJECT_ROOT).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
                          for path in sorted((PROJECT_ROOT / "experiments").rglob("*.py"))},
        "configurations": grid, "methods": METHODS, "implemented_methods": IMPLEMENTED_METHODS,
        "unsupported_methods": unsupported_methods, "all_methods_implemented": True,
        "comparison_complete": None,
        "preflight_experiment_runs": 0, "repeatability_check": "not_executed_by_entry_point",
        "environment": environment_metadata(), "dataset_manifest": manifest,
        "statistics": "95% Student-t interval from seed-level metrics; NaN for undefined denominators",
        "figure_statistics": "seed-level mean only; confidence intervals remain in CSV, not shaded in plots",
    })
    all_rows, all_subtype = [], []
    log = output / "logs" / "progress.log"
    log.write_text("", encoding="utf-8")
    _json_write(output / "logs" / "completion.json", {
        "completed": False, "mode": "FULL", "comparison_complete": False,
        "implemented_methods": IMPLEMENTED_METHODS, "unsupported_methods": unsupported_methods,
    })
    try:
        with threadpool_limits(limits=1):
            total = len(grid) * len(seeds)
            completed = 0
            print(f"Running {len(grid)} unique D1/D2 configurations x {len(seeds)} seeds...", flush=True)
            for N, cr in grid:
                for seed in seeds:
                    rows, subtype, evidence = run_configuration(N, cr, seed, config)
                    all_rows.extend(rows); all_subtype.extend(subtype)
                    np.savez_compressed(output / "raw" / "predictions" / f"N_{N}_cr_{cr:.1f}_seed_{seed}.npz", **evidence)
                    completed += 1
                message = f"[{completed}/{total}] N={N}, cr={cr:g}% completed"
                print(message, flush=True)
                with log.open("a", encoding="utf-8") as stream:
                    stream.write(message + "\n")
        print("Saving raw metrics and computing seed-level confidence intervals...", flush=True)
        write_csv(output / "raw" / "detection_runs.csv", all_rows)
        write_csv(output / "raw" / "subtype_metrics.csv", all_subtype)
        summaries = summarize_runs(all_rows)
        write_csv(output / "summary" / "detection_summary.csv", summaries)
        write_csv(output / "summary" / "subtype_summary.csv", summarize_subtypes(all_subtype))
        print("Generating figures from saved results...", flush=True)
        from .plot_detection import plot_results
        plot_metadata = plot_results(output, plot_kind=plot_kind)
        _json_write(output / "logs" / "plot_style.json", plot_metadata)
        after = dataset_snapshot(DATASET_ROOT)
        _json_write(output / "logs" / "dataset_after.json", after)
        if snapshot != after:
            raise AssertionError("Dataset changed during the experiment")
        elapsed = time.perf_counter() - started
        generate_report(output, all_rows, summaries, manifest, elapsed)
        statuses = Counter(row["status"] for row in all_rows)
        _json_write(output / "logs" / "completion.json", {
            "completed": True, "mode": "FULL", "elapsed_seconds": elapsed,
            "data_trials": total, "method_rows": len(all_rows), "statuses": dict(statuses),
            "dataset_unchanged": True, "figures_generated": True,
            "figure_names": ["Fig_D1_detection_vs_N", "Fig_D2_detection_vs_cr"],
            "comparison_complete": not unsupported_methods and not any(statuses[status] for status in ("unsupported", "failure", "gt_unavailable")),
            "implemented_methods": IMPLEMENTED_METHODS, "unsupported_methods": unsupported_methods,
        })
        print(f"FULL computations finished. Statuses: {dict(statuses)}\nResults: {output}", flush=True)
        print("Four-method comparison complete: " + str(not any(
            statuses[status] for status in ("unsupported", "failure", "gt_unavailable"))), flush=True)
        if statuses["failure"]:
            raise RuntimeError(f"Experiment finished with {statuses['failure']} algorithm failures; see raw reasons")
    except Exception:
        _json_write(output / "logs" / "completion.json", {
            "completed": False, "mode": "FULL", "comparison_complete": False,
            "implemented_methods": IMPLEMENTED_METHODS, "unsupported_methods": unsupported_methods,
            "method_rows_computed": len(all_rows), "error": traceback.format_exc(),
        })
        if all_rows:
            write_csv(output / "raw" / "detection_runs.csv", all_rows)
            write_csv(output / "raw" / "subtype_metrics.csv", all_subtype)
        (output / "logs" / "error.log").write_text(traceback.format_exc(), encoding="utf-8")
        raise
