"""Post-construction label evaluation and unconditional binary statistics."""
from __future__ import annotations

from collections import defaultdict
import math
from statistics import NormalDist

import numpy as np

from .config_ground_truth import (
    METHODS, Q_VALUES, FAILURE_VALUES, SEEDS, GT_SIZE_M, IMPLEMENTATION_VERSION,
)


def evaluate_ground_truth(result, labels, available, M=GT_SIZE_M) -> dict:
    """Labels enter here only after a method has returned its final selection.

    Formation and reliability are distinct. Every unsuccessful method returns
    an empty GT; no passing-but-insufficient set is substituted as the final GT.
    """
    truth = np.asarray(labels)
    response = np.asarray(available)
    selected = np.asarray(result.selected_indices)
    if truth.ndim != 1 or not np.all(np.isin(truth, (0, 1))):
        raise ValueError("Evaluation requires a one-dimensional 0/1 label vector")
    if response.shape != truth.shape or response.dtype != np.bool_:
        raise ValueError("Availability must be a boolean vector matching the labels")
    if selected.ndim != 1 or not np.issubdtype(selected.dtype, np.integer):
        raise ValueError("Selected indices must be a one-dimensional integer vector")
    if np.any((selected < 0) | (selected >= len(truth))):
        raise ValueError("Selected GT row is outside the candidate pool")
    unique = len(np.unique(selected)) == len(selected)
    formed = bool(result.success and len(selected) == M and unique)
    responsive = bool(len(selected) > 0 and np.all(response[selected]))
    corrupt = int(np.count_nonzero(truth[selected]))
    normal = int(len(selected) - corrupt)
    reliable = bool(formed and responsive and corrupt == 0)
    if not formed:
        status, reason = "gt_unavailable", result.reason or "GT has fewer than M unique models"
    elif not responsive:
        status, reason = "gt_invalid_response", "GT includes an unavailable candidate"
    elif corrupt:
        status, reason = "gt_contaminated", "Formed GT includes corrupted candidates"
    else:
        status, reason = "reliable_success", ""
    return {
        "gt_success": int(formed), "gt_size": len(selected),
        "gt_normal_count": normal, "gt_corrupt_count": corrupt,
        "gt_all_responsive": int(responsive), "reliable_success": int(reliable),
        "status": status, "failure_reason": reason,
    }


def wilson_interval(successes: int, total: int) -> tuple[float, float]:
    """Two-sided Wilson score interval, including 0/total and total/total."""
    if total < 1 or not 0 <= successes <= total:
        raise ValueError("Wilson counts require total>0 and 0<=successes<=total")
    z = NormalDist().inv_cdf(0.975)
    p = successes / total
    denominator = 1.0 + z * z / total
    center = (p + z * z / (2.0 * total)) / denominator
    half = z * math.sqrt(p * (1.0 - p) / total + z * z / (4.0 * total * total)) / denominator
    return (0.0 if successes == 0 else max(0.0, center - half),
            1.0 if successes == total else min(1.0, center + half))


class SummaryAccumulator:
    """Keep small seed sets and sums while raw records are streamed to disk."""

    def __init__(self, algorithm_version=IMPLEMENTATION_VERSION) -> None:
        self.algorithm_version = algorithm_version
        self.groups = defaultdict(lambda: {
            "seeds": set(), "success": 0, "formed": 0, "contaminated": 0,
            "normal": 0, "size": 0, "final_K": 0, "expansions": 0,
            "responsive": 0, "comparisons": 0, "unavailable": 0,
        })

    def add(self, row: dict) -> None:
        key = (row["method"], float(row["q_C_requested"]), float(row["p_fail_requested"]))
        if key[0] not in METHODS or key[1] not in Q_VALUES or key[2] not in FAILURE_VALUES:
            raise ValueError(f"Record outside the formal grid: {key}")
        seed = int(row["seed"])
        stats = self.groups[key]
        if seed not in SEEDS or seed in stats["seeds"]:
            raise ValueError(f"Unknown or duplicate seed at {key}: {seed}")
        success, formed = int(row["reliable_success"]), int(row["gt_success"])
        if success not in (0, 1) or formed not in (0, 1) or success > formed:
            raise ValueError("Invalid binary formation/reliability outcomes")
        stats["seeds"].add(seed)
        stats["success"] += success
        stats["formed"] += formed
        stats["contaminated"] += int(formed and int(row["gt_corrupt_count"]) > 0)
        stats["unavailable"] += 1 - formed
        for field, column in (("final_K", "final_K"), ("expansions", "candidate_expansion_count"),
                              ("responsive", "responsive_count"), ("comparisons", "pairwise_comparison_count")):
            stats[field] += int(row[column])
        if formed:
            stats["normal"] += int(row["gt_normal_count"])
            stats["size"] += int(row["gt_size"])

    def finish(self, *, require_complete: bool = True) -> tuple[list[dict], list[dict]]:
        expected = {(m, q, p) for m in METHODS for q in Q_VALUES for p in FAILURE_VALUES}
        if require_complete and set(self.groups) != expected:
            raise ValueError("Summary requires every declared method/q_C/p_fail group")
        summary, diagnostics = [], []
        for method in METHODS:
            for q in Q_VALUES:
                for p in FAILURE_VALUES:
                    key = (method, q, p)
                    if key not in self.groups:
                        continue
                    stats = self.groups[key]
                    if require_complete and stats["seeds"] != set(SEEDS):
                        raise ValueError(f"Summary requires exactly all 1000 seeds: {key}")
                    n = len(stats["seeds"])
                    rate = stats["success"] / n
                    lo, hi = wilson_interval(stats["success"], n)
                    common = {"method": method, "q_C": q, "p_fail": p, "n_total": n,
                              "algorithm_version": self.algorithm_version}
                    summary.append({
                        **common, "n_reliable_success": stats["success"],
                        "reliable_gt_rate": rate, "n_gt_formed": stats["formed"],
                        "gt_formation_rate": stats["formed"] / n, "mean": rate,
                        "std": math.sqrt(rate * (1.0 - rate) * n / (n - 1)) if n > 1 else math.nan,
                        "ci95_lower": lo, "ci95_upper": hi, "ci_method": "Wilson 95%",
                    })
                    diagnostics.append({
                        **common, "gt_formation_rate": stats["formed"] / n,
                        "n_gt_formed": stats["formed"], "n_gt_insufficient": stats["unavailable"],
                        "n_gt_contaminated": stats["contaminated"],
                        "unconditional_gt_contamination_rate": stats["contaminated"] / n,
                        "formed_gt_conditional_contamination_rate": (
                            stats["contaminated"] / stats["formed"] if stats["formed"] else math.nan),
                        "formed_gt_conditional_normal_fraction": (
                            stats["normal"] / stats["size"] if stats["size"] else math.nan),
                        "mean_final_K": stats["final_K"] / n,
                        "mean_candidate_expansions": stats["expansions"] / n,
                        "mean_responsive_candidate_count": stats["responsive"] / n,
                        "mean_pairwise_comparison_count": stats["comparisons"] / n,
                    })
        return summary, diagnostics


def summarize_runs(rows, *, require_complete=True, algorithm_version=IMPLEMENTATION_VERSION) -> tuple[list[dict], list[dict]]:
    accumulator = SummaryAccumulator(algorithm_version=algorithm_version)
    for row in rows:
        accumulator.add(row)
    return accumulator.finish(require_complete=require_complete)
