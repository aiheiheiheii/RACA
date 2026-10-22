"""Full paired Group 04 experiment; invoked only by the user's run entry."""
from __future__ import annotations

import csv
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import time
import uuid

import numpy as np
from threadpoolctl import threadpool_limits

from experiments.common.reproducibility import (
    dataset_snapshot, environment_metadata, sha256_array, verify_manifest,
)
from experiments.common.similarity import cosine_pairwise, within_interval
from .candidate_sampler import load_trial_group, draw_scenario, FAILURE_STREAM, RANDOM_METHOD_STREAM
from .config_ground_truth import (
    PROJECT_ROOT, DATASET_ROOT, RESULT_ROOT, DATASET_NAME, EXPERIMENT,
    IMPLEMENTATION_VERSION, N, GLOBAL_CR_PERCENT, D, H, TAU_LOWER, TAU_UPPER,
    INITIAL_K, GT_SIZE_M, MAX_CANDIDATE_POOL_SIZE, METHODS, SEEDS, Q_VALUES,
    FAILURE_VALUES, EXPECTED_RUNS, EXPECTED_SCENARIOS, EXPECTED_POOLS,
    GT_KERNEL_SOURCE, protocol_metadata, validate_protocol,
    DIAGNOSTICS_VERSION,
)
from .ground_truth_methods import run_method
from .ground_truth_metrics import evaluate_ground_truth, SummaryAccumulator
from .ground_truth_diagnostics import diagnose_construction, DiagnosticAccumulator
from .ground_truth_storage import atomic_json, atomic_csv, file_identity, source_identity, replace_file

TIMEZONE = timezone(timedelta(hours=8), "Asia/Shanghai")
RAW_FIELDS = (
    "experiment", "method", "seed", "N", "global_cr", "global_cr_realized", "d", "H",
    "q_C_requested", "q_C_realized", "initial_K_realized_cr", "K75_realized_cr", "K100_realized_cr",
    "p_fail_requested", "p_fail_realized", "initial_K_p_fail_realized", "final_K_p_fail_realized",
    "tau_lower", "tau_upper", "initial_K", "final_K", "M", "max_candidate_pool",
    "candidate_pool_size", "responsive_count", "pool_responsive_count", "candidate_expansion_count",
    "gt_success", "gt_size", "gt_normal_count", "gt_corrupt_count", "gt_all_responsive", "reliable_success",
    "pairwise_comparison_count", "similarity_cache_dot_products_new",
    "similarity_cache_peer_pairs_new", "similarity_cache_owner", "selected_temporary_ids",
    "gt_trace", "status", "failure_reason", "candidate_fingerprint", "availability_fingerprint",
    "failure_uniform_fingerprint", "random_method_seed", "dataset_fingerprint", "algorithm_version",
    "code_version", "run_id", "diagnostics_version", "admission_diagnostic_sha256",
    "gt_common_normal_count", "gt_high_heterogeneity_normal_count", "gt_severe_corrupt_count", "gt_mild_corrupt_count",
)


def _now() -> str:
    return datetime.now(TIMEZONE).isoformat()


def _prepare_output(result_root: Path) -> Path:
    """Use the single fixed Group 04 output directory without moving it."""
    root = result_root.resolve()
    fixed = (PROJECT_ROOT / "results" / "04_ground_truth").resolve()
    if root != fixed or root.parent != (PROJECT_ROOT / "results").resolve():
        raise ValueError(f"Group 04 output must use the fixed directory: {fixed}")
    if root.exists() and not root.is_dir():
        raise NotADirectoryError(f"Result output is not a directory: {root}")
    for name in ("raw", "summary", "figures", "logs"):
        (root / name).mkdir(parents=True, exist_ok=True)
    print(f"Group 04 output for this run: {root}", flush=True)
    return root


def _cosine_diagnostics(similarity: np.ndarray, labels: np.ndarray, subtypes: np.ndarray) -> dict:
    """Posthoc pair distributions; these diagnostics never enter selection."""
    upper_i, upper_j = np.triu_indices(len(labels), k=1)
    values = similarity[upper_i, upper_j]
    supports = within_interval(values, TAU_LOWER, TAU_UPPER)
    a, b = labels[upper_i], labels[upper_j]
    masks = {"normal_normal": (a == 0) & (b == 0),
             "normal_corrupt": a != b, "corrupt_corrupt": (a == 1) & (b == 1)}
    for first in range(4):
        for second in range(first, 4):
            left, right = subtypes[upper_i], subtypes[upper_j]
            masks[f"subtype_{first}_{second}"] = (
                ((left == first) & (right == second)) | ((left == second) & (right == first)))
    result = {}
    for name, mask in masks.items():
        selected = values[mask]
        result[name] = {"n_pairs": len(selected),
                        "min": float(selected.min()) if len(selected) else None,
                        "max": float(selected.max()) if len(selected) else None,
                        "mean": float(selected.mean()) if len(selected) else None,
                        "interval_support_fraction": float(supports[mask].mean()) if len(selected) else None}
    return result


def _scenario_record(scene, similarity: np.ndarray, run_id: str) -> dict:
    return {
        "run_id": run_id, "seed": scene.seed, "q_C_requested": scene.q_C_requested,
        "q_C_realized": scene.q_C_realized, "candidate_fingerprint": scene.candidate_fingerprint,
        "source_indices": scene.source_indices.tolist(),
        "temporary_ids": scene.algorithm_input.model_ids.tolist(),
        "evaluation_labels": scene.evaluation_labels.tolist(),
        "evaluation_subtypes": scene.evaluation_subtypes.tolist(),
        "prefix_corruption_ratios": scene.prefix_corruption_ratios,
        "failure_uniform": scene.failure_uniform.tolist(),
        "failure_uniform_fingerprint": sha256_array(scene.failure_uniform),
        "availability_fingerprints": {str(p): sha256_array(scene.availability(p)) for p in FAILURE_VALUES},
        "descriptor_fingerprint": sha256_array(scene.algorithm_input.X),
        "cosine_cache_fingerprint": sha256_array(similarity),
        "random_method_seed": scene.random_method_seed, "random_streams": scene.random_streams,
        "subtype_counts": {str(i): int(np.count_nonzero(scene.evaluation_subtypes == i)) for i in range(4)},
        "pairwise_cosine_posthoc": _cosine_diagnostics(
            similarity, scene.evaluation_labels, scene.evaluation_subtypes),
        "physical_cache_dot_product_count": len(similarity) ** 2,
        "physical_cache_directed_peer_count": len(similarity) * (len(similarity) - 1),
    }


def _runtime_report(result_root: Path, manifest: dict, summary: list[dict], final_k_rows: list[dict]) -> Path:
    """Report observations only after a full user-executed run has completed."""
    path = result_root / "logs" / "ground_truth_run_report.md"
    lines = ["# Group 04 execution report", "", f"Run ID: `{manifest['run_id']}`",
             f"Algorithm version: `{IMPLEMENTATION_VERSION}`", f"Completed at: {manifest['completed_at']}",
             f"Elapsed computation and saved-evidence time: {manifest['elapsed_seconds']:.3f} seconds",
             "", "All rates below include unformed and contaminated GT trials in the denominator.",
             "Synthetic descriptors; these are simulation results, not measurements of deployed models.",
             "", "| Method | q_C (%) | p_fail (%) | Reliable / total | Formation rate | Wilson 95% CI |",
             "|---|---:|---:|---:|---:|---|" ]
    for row in summary:
        lines.append(f"| {row['method']} | {100*row['q_C']:g} | {100*row['p_fail']:g} | "
                     f"{row['n_reliable_success']} / {row['n_total']} | {row['gt_formation_rate']:.4f} | "
                     f"[{row['ci95_lower']:.4f}, {row['ci95_upper']:.4f}] |")
    lines.extend(["", "The full pool is controlled at q_C=0..50% despite global corruption of 3%.",
                  "High legitimate heterogeneity and mild corruption remain in the data. Normal labels do not guarantee pairwise support.",
                  "50% pollution is outside an unconditional guarantee; the similarity interval, labels and failed-trial counts were not adjusted. The revised quorum is fixed before the new experiment.",
                  "With equal initial history, Top-M and Random-M can have similar aggregate reliability.",
                  "Logical peer checks are reported separately from the shared 100x100 physical cosine-cache work.",
                  "Use the saved pool audit and GT traces to diagnose overlapping curves, failures and contamination."])
    lines.extend(["", "## Revised reference admission", "",
                  "The revised GT rule uses theta_K=ceil(2K/3), giving 34/50/67 at K=50/75/100.",
                  "Only quorum-admitted candidates enter selection. Rank ascending history, descending peer support, "
                  "descending mean supporting-peer cosine, then ascending temporary ID.",
                  "This is a conservative assurance/availability tradeoff; it does not guarantee monotone reliability or oracle label purity.",
                  "The original v1 results remain a separate protocol and must not be relabeled as revised results.",
                  "", "## Reliability changes and final-K decomposition", "",
                  "| p_fail (%) | q_C (%) | Final K | Reliable | Contaminated | Insufficient | Trials ending at K |",
                  "|---:|---:|---:|---:|---:|---:|---:|"])
    for row in final_k_rows:
        if row["method"] == "RACA-HIM":
            lines.append(f"| {100*row['p_fail']:g} | {100*row['q_C']:g} | {row['K']} | "
                         f"{row['n_reliable']} | {row['n_contaminated']} | {row['n_insufficient']} | {row['n_at_final_K']} |")
    for p in FAILURE_VALUES:
        points = sorted((row for row in summary if row["method"] == "RACA-HIM" and row["p_fail"] == p),
                        key=lambda row: row["q_C"])
        for before, after in zip(points, points[1:]):
            if after["reliable_gt_rate"] > before["reliable_gt_rate"]:
                lines.append(f"Observed upward step at p_fail={100*p:g}%: q_C={100*before['q_C']:g}% "
                             f"to {100*after['q_C']:g}%, reliability={before['reliable_gt_rate']:.4f} "
                             f"to {after['reliable_gt_rate']:.4f}. Inspect contamination and final-K contributions; no monotonic adjustment is applied.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def run_experiment(result_root=RESULT_ROOT, dataset_root=DATASET_ROOT) -> dict:
    """Run exactly the full predeclared protocol; no seed/grid override."""
    validate_protocol()
    result_root, dataset_root = Path(result_root).resolve(), Path(dataset_root).resolve()
    checks = verify_manifest(dataset_root)
    if checks["missing"] or checks["mismatch"]:
        raise ValueError(f"Dataset manifest verification failed: {checks}")
    snapshot = dataset_snapshot(dataset_root)
    fingerprint = hashlib.sha256(json.dumps(snapshot, sort_keys=True).encode("utf-8")).hexdigest()
    code = source_identity()
    run_id = datetime.now(TIMEZONE).strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8]
    result_root = _prepare_output(result_root)
    manifest_path = result_root / "logs" / "run_manifest.json"
    metadata = {
        "run_id": run_id, "status": "running", "started_at": _now(), "timezone": "Asia/Shanghai",
        "dataset_name": DATASET_NAME, "dataset_fingerprint": fingerprint,
        "dataset_files_sha256": snapshot, "dataset_manifest_check": checks,
        "algorithm_version": IMPLEMENTATION_VERSION, **code, "environment": environment_metadata(),
        "diagnostics_version": DIAGNOSTICS_VERSION,
        "protocol": protocol_metadata(), "methods": list(METHODS), "gt_kernel_source": GT_KERNEL_SOURCE,
        "gt_kernel_optional_arguments": ["responder_available", "pairwise_cache", "allow_expansion", "support_quorum", "selection_policy"],
        "failure_rng": f"SeedSequence([seed, {FAILURE_STREAM}, q_index])",
        "random_method_rng": f"SeedSequence([seed, {RANDOM_METHOD_STREAM}, q_index])",
        "result_root": str(result_root),
        "output_policy": "fixed results/04_ground_truth; completed reruns replace named output files atomically",
        "counts": {}, "files": {},
        "comparison_count_definition": "sum of effective responsive directed peer relationships over visited K values; no early stopping",
        "cache_budget_definition": "10000 scalar dot products including self; 9900 directed peers per independent pool, once across all methods/p_fail",
    }
    metadata["environment"]["rng"] = (
        "numpy.random.PCG64; dataset loader owns sampling streams; Group 04 uses "
        "SeedSequence([seed,4401,q_index]) and a uint64 seed derived from [seed,4402,q_index]")
    atomic_json(manifest_path, metadata)
    partial_raw = result_root / "raw" / "ground_truth_runs.partial.csv"
    partial_scenarios = result_root / "logs" / "candidate_scenarios.partial.jsonl"
    partial_admissions = result_root / "logs" / "ground_truth_admission.partial.jsonl"
    accumulator = SummaryAccumulator()
    admission_accumulator = DiagnosticAccumulator()
    counts = {"candidate_pools": 0, "scenarios": 0, "method_runs": 0,
              "gt_formed": 0, "reliable_success": 0, "gt_unavailable": 0, "gt_contaminated": 0}
    started = time.perf_counter()
    try:
        with (partial_raw.open("w", encoding="utf-8", newline="") as raw_stream,
              partial_scenarios.open("w", encoding="utf-8") as audit_stream,
              partial_admissions.open("w", encoding="utf-8") as admission_stream,
              threadpool_limits(limits=1)):
            writer = csv.DictWriter(raw_stream, fieldnames=RAW_FIELDS, lineterminator="\n")
            writer.writeheader()
            for seed_index, seed in enumerate(SEEDS):
                group = load_trial_group(seed, dataset_root)
                for q in Q_VALUES:
                    scene = draw_scenario(group, seed, q)
                    # Numerical cache only: no admission decision is cached.
                    similarities = cosine_pairwise(scene.algorithm_input.X)
                    similarities.setflags(write=False)
                    audit_stream.write(json.dumps(_scenario_record(scene, similarities, run_id),
                                                  ensure_ascii=False, allow_nan=False) + "\n")
                    counts["candidate_pools"] += 1
                    prefix = scene.prefix_corruption_ratios
                    uniform_hash = sha256_array(scene.failure_uniform)
                    for p_index, p in enumerate(FAILURE_VALUES):
                        available = scene.availability(p)
                        availability_hash = sha256_array(available)
                        counts["scenarios"] += 1
                        for method_index, method in enumerate(METHODS):
                            result = run_method(method, scene.algorithm_input, available,
                                                scene.random_method_seed, pairwise_cache=similarities)
                            outcome = evaluate_ground_truth(result, scene.evaluation_labels, available)
                            # Labels enter only after the method has returned.
                            admission = diagnose_construction(result, scene.evaluation_labels,
                                                             scene.evaluation_subtypes, available, similarities)
                            admission_json = json.dumps(admission, sort_keys=True, separators=(",", ":"), allow_nan=False)
                            admission_hash = hashlib.sha256(admission_json.encode("utf-8")).hexdigest()
                            selected_counts = admission["selected_subtype_counts"]
                            owns_cache = p_index == 0 and method_index == 0
                            row = {
                                "experiment": EXPERIMENT, "method": method, "seed": seed,
                                "N": N, "global_cr": GLOBAL_CR_PERCENT,
                                "global_cr_realized": 100.0 * float(np.mean(group["binary_label"])),
                                "d": D, "H": H, "q_C_requested": q, "q_C_realized": scene.q_C_realized,
                                "initial_K_realized_cr": prefix[50], "K75_realized_cr": prefix[75],
                                "K100_realized_cr": prefix[100], "p_fail_requested": p,
                                "p_fail_realized": float(np.mean(~available)),
                                "initial_K_p_fail_realized": float(np.mean(~available[:INITIAL_K])),
                                "final_K_p_fail_realized": float(np.mean(~available[:result.final_K])),
                                "tau_lower": TAU_LOWER, "tau_upper": TAU_UPPER,
                                "initial_K": INITIAL_K, "final_K": result.final_K,
                                "M": GT_SIZE_M, "max_candidate_pool": MAX_CANDIDATE_POOL_SIZE,
                                "candidate_pool_size": len(available),
                                "responsive_count": int(np.count_nonzero(available[:result.final_K])),
                                "pool_responsive_count": int(np.count_nonzero(available)),
                                "candidate_expansion_count": result.expansion_count, **outcome,
                                "pairwise_comparison_count": result.pairwise_comparison_count,
                                "similarity_cache_dot_products_new": len(available)**2 if owns_cache else 0,
                                "similarity_cache_peer_pairs_new": len(available)*(len(available)-1) if owns_cache else 0,
                                "similarity_cache_owner": int(owns_cache),
                                "selected_temporary_ids": json.dumps(scene.algorithm_input.model_ids[result.selected_indices].tolist()),
                                "gt_trace": json.dumps(result.trace, separators=(",", ":")),
                                "candidate_fingerprint": scene.candidate_fingerprint,
                                "availability_fingerprint": availability_hash, "failure_uniform_fingerprint": uniform_hash,
                                "random_method_seed": scene.random_method_seed,
                                "dataset_fingerprint": fingerprint, "algorithm_version": IMPLEMENTATION_VERSION,
                                "code_version": code["code_version"], "run_id": run_id,
                                "diagnostics_version": DIAGNOSTICS_VERSION,
                                "admission_diagnostic_sha256": admission_hash,
                                "gt_common_normal_count": selected_counts["common_normal"],
                                "gt_high_heterogeneity_normal_count": selected_counts["high_heterogeneity_normal"],
                                "gt_severe_corrupt_count": selected_counts["severe_corrupt"],
                                "gt_mild_corrupt_count": selected_counts["mild_corrupt"],
                            }
                            writer.writerow(row)
                            accumulator.add(row)
                            admission_accumulator.add(row, admission)
                            admission_stream.write(json.dumps({
                                "run_id": run_id, "seed": seed, "q_C": q, "p_fail": p, "method": method,
                                "algorithm_version": IMPLEMENTATION_VERSION, "diagnostics_version": DIAGNOSTICS_VERSION,
                                "admission_diagnostic_sha256": admission_hash, "diagnostic": admission,
                            }, separators=(",", ":"), allow_nan=False) + "\n")
                            counts["method_runs"] += 1
                            counts["gt_formed"] += outcome["gt_success"]
                            counts["reliable_success"] += outcome["reliable_success"]
                            counts["gt_unavailable"] += int(not outcome["gt_success"])
                            counts["gt_contaminated"] += int(outcome["status"] == "gt_contaminated")
                raw_stream.flush()
                audit_stream.flush()
                admission_stream.flush()
                if seed_index == 0 or (seed_index + 1) % 25 == 0:
                    metadata.update(counts=dict(counts), last_completed_seed=seed, elapsed_seconds=time.perf_counter()-started)
                    atomic_json(manifest_path, metadata)
                    print(f"Group 04: {seed_index+1}/{len(SEEDS)} seeds, "
                          f"{counts['method_runs']}/{EXPECTED_RUNS} method records", flush=True)
        if (counts["candidate_pools"], counts["scenarios"], counts["method_runs"]) != (EXPECTED_POOLS, EXPECTED_SCENARIOS, EXPECTED_RUNS):
            raise ValueError("Run did not complete the formal grid")
        if dataset_snapshot(dataset_root) != snapshot:
            raise RuntimeError("Dataset changed during the run; results cannot be certified")
        if source_identity()["code_version"] != code["code_version"]:
            raise RuntimeError("Source files changed during execution; start a fresh full run")
        summary, diagnostics = accumulator.finish()
        final_k_rows, admission_rows = admission_accumulator.finish()
        for row in summary + diagnostics + final_k_rows + admission_rows:
            row.update(run_id=run_id, dataset_fingerprint=fingerprint, code_version=code["code_version"])
        for row in final_k_rows + admission_rows:
            row.update(algorithm_version=IMPLEMENTATION_VERSION, diagnostics_version=DIAGNOSTICS_VERSION)
        raw_path = partial_raw.with_name("ground_truth_runs.csv")
        scenario_path = partial_scenarios.with_name("candidate_scenarios.jsonl")
        replace_file(partial_raw, raw_path)
        replace_file(partial_scenarios, scenario_path)
        admission_path = partial_admissions.with_name("ground_truth_admission.jsonl")
        replace_file(partial_admissions, admission_path)
        summary_path = result_root / "summary" / "ground_truth_summary.csv"
        diagnostic_path = result_root / "summary" / "ground_truth_diagnostics.csv"
        atomic_csv(summary_path, summary)
        atomic_csv(diagnostic_path, diagnostics)
        final_k_path = result_root / "summary" / "ground_truth_final_k_summary.csv"
        admission_summary_path = result_root / "summary" / "ground_truth_admission_summary.csv"
        atomic_csv(final_k_path, final_k_rows)
        atomic_csv(admission_summary_path, admission_rows)
        metadata.update(status="complete", completed_at=_now(), elapsed_seconds=time.perf_counter()-started,
                        counts=counts, dataset_unchanged=True, source_unchanged=True,
                        physical_cache_dot_products=EXPECTED_POOLS*MAX_CANDIDATE_POOL_SIZE**2,
                        physical_cache_directed_peer_pairs=EXPECTED_POOLS*MAX_CANDIDATE_POOL_SIZE*(MAX_CANDIDATE_POOL_SIZE-1),
                        figures_status="pending")
        metadata["files"] = {name: file_identity(path, result_root) for name, path in (
            ("raw", raw_path), ("summary", summary_path), ("diagnostics", diagnostic_path), ("scenarios", scenario_path),
            ("admission_runs", admission_path), ("final_k_summary", final_k_path), ("admission_summary", admission_summary_path))}
        report = _runtime_report(result_root, metadata, summary, final_k_rows)
        metadata["files"]["report"] = file_identity(report, result_root)
        atomic_json(manifest_path, metadata)
        return metadata
    except BaseException as exc:
        metadata.update(status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                        stopped_at=_now(), elapsed_seconds=time.perf_counter()-started, counts=counts,
                        error={"type": type(exc).__name__, "message": str(exc)})
        atomic_json(manifest_path, metadata)
        raise
