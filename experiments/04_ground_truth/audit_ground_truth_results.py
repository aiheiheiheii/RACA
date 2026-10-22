"""Audit saved Group 04 evidence without importing or executing experiment code.

Right-click Run in PyCharm. Only Python's standard library is used. The input
directory is read-only; all audit artifacts are written to ``reports/``.
Saved source identities describe the original run, even after code changes.
This audit cannot reconstruct peer support decisions from aggregate cosines.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone, timedelta
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
from statistics import NormalDist
import struct


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULTS = PROJECT_ROOT / "results" / "04_ground_truth"
DEFAULT_REPORTS = PROJECT_ROOT / "reports"
AUDIT_VERSION = "saved-evidence-audit-v1"
SUBTYPES = ("common_normal", "high_heterogeneity_normal", "severe_corrupt", "mild_corrupt")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def array_fingerprint(values, dtype: str, code: str) -> str:
    """Reproduce the saved one-dimensional dtype/shape/bytes hash, no NumPy."""
    digest = hashlib.sha256()
    digest.update(dtype.encode("ascii"))
    digest.update(repr((len(values),)).encode("ascii"))
    if dtype == "|b1":
        digest.update(bytes(int(value) for value in values))
    else:
        endian = dtype[0]
        digest.update(struct.pack(f"{endian}{len(values)}{code}", *values))
    return digest.hexdigest()


def close(a, b) -> bool:
    a, b = float(a), float(b)
    return (math.isnan(a) and math.isnan(b)) or math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-12)


def wilson(success: int, total: int) -> tuple[float, float]:
    z = NormalDist().inv_cdf(0.975)
    p = success / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return (0.0 if not success else max(0.0, center - half),
            1.0 if success == total else min(1.0, center + half))


def admission_threshold(protocol: dict, candidate_size: int) -> int:
    """Interpret the saved policy, never the current project configuration.

    Legacy runs use strict majority including one self vote. Revised runs can
    save quorum_fraction as a rational mapping, [numerator, denominator], a
    string such as "2/3", or a number. Rational metadata avoids rounding.
    """
    value = protocol.get("quorum_fraction")
    if value is None:
        return (candidate_size + 2) // 2
    if isinstance(value, dict):
        fraction = Fraction(int(value["numerator"]), int(value["denominator"]))
    elif isinstance(value, (list, tuple)):
        require(len(value) == 2, "Saved quorum_fraction needs two rational components.")
        fraction = Fraction(int(value[0]), int(value[1]))
    else:
        fraction = Fraction(str(value)).limit_denominator(1000000)
    require(Fraction(1, 2) < fraction <= 1, "Saved quorum_fraction must lie in (1/2, 1].")
    return (candidate_size * fraction.numerator + fraction.denominator - 1) // fraction.denominator


def load_evidence(root: Path) -> tuple[dict, dict[str, Path], dict[str, str]]:
    manifest_path = root / "logs" / "run_manifest.json"
    manifest_hash = sha256_file(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    require(manifest.get("status") == "complete", "Audit requires a complete saved run.")
    for inventory, fingerprint in (("source_sha256", "code_version"),
                                   ("dataset_files_sha256", "dataset_fingerprint")):
        saved = manifest.get(inventory)
        require(isinstance(saved, dict) and bool(saved), f"Missing {inventory} inventory.")
        digest = hashlib.sha256(json.dumps(saved, sort_keys=True).encode("utf-8")).hexdigest()
        require(digest == manifest.get(fingerprint), f"Manifest {fingerprint} is inconsistent.")
    paths, hashes = {}, {"manifest": manifest_hash}
    for name in ("raw", "summary", "diagnostics", "scenarios"):
        identity = manifest.get("files", {}).get(name)
        require(isinstance(identity, dict), f"Missing saved {name} identity.")
        path = (root / identity["path"]).resolve()
        require(path.is_relative_to(root) and path.is_file(), f"Invalid saved path for {name}.")
        digest = sha256_file(path)
        require(digest == identity["sha256"] and path.stat().st_size == identity["size_bytes"],
                f"Saved {name} differs from the run manifest: {path}")
        paths[name], hashes[name] = path, digest
    return manifest, paths, hashes


class Checks:
    def __init__(self) -> None:
        self.counts = Counter()
        self.failures = Counter()
        self.examples: list[str] = []

    def check(self, name: str, condition: bool, context="") -> None:
        self.counts[name] += 1
        if not condition:
            self.failures[name] += 1
            if len(self.examples) < 16:
                self.examples.append(f"{name}: {context}")


def read_scenarios(path: Path, manifest: dict, checks: Checks):
    protocol = manifest["protocol"]
    seeds, qs, ps = set(protocol["seeds"]), protocol["q_values"], protocol["failure_values"]
    size, n, h0 = protocol["max_candidate_pool"], protocol["N"], protocol["h0"]
    scenes, pairs, pool_conditions = {}, defaultdict(Counter), defaultdict(Counter)
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            row = json.loads(line)
            key = (int(row["seed"]), float(row["q_C_requested"]))
            require(key[0] in seeds and key[1] in qs and key not in scenes,
                    f"Unknown or duplicate candidate scene on JSONL line {line_number}.")
            require(row.get("run_id") == manifest["run_id"], "Mixed candidate scene run identity.")
            labels, subtypes = tuple(row["evaluation_labels"]), tuple(row["evaluation_subtypes"])
            uniforms, ids, source = (tuple(row[name]) for name in
                                     ("failure_uniform", "temporary_ids", "source_indices"))
            require(all(len(values) == size for values in (labels, subtypes, uniforms, ids, source)),
                    f"Candidate vector length mismatch at {key}.")
            require(set(labels) <= {0, 1} and set(subtypes) <= {0, 1, 2, 3},
                    f"Invalid evaluation labels at {key}.")
            require(ids == tuple(range(size)) and len(set(source)) == size
                    and all(isinstance(value, int) and 0 <= value < n for value in source),
                    f"Invalid temporary/source mapping at {key}.")
            require(all(isinstance(value, (int, float)) and math.isfinite(value)
                        and 0 <= value < 1 for value in uniforms), f"Invalid uniforms at {key}.")
            checks.check("scene_label_subtype", all(label == int(subtype >= 2)
                         for label, subtype in zip(labels, subtypes)), key)
            checks.check("scene_full_pool_q", close(sum(labels) / size, key[1])
                         and close(row["q_C_realized"], key[1]), key)
            counts = Counter(subtypes)
            checks.check("scene_subtype_counts", all(row["subtype_counts"][str(t)] == counts[t]
                         for t in range(4)), key)
            for k in protocol["K_values"]:
                checks.check("scene_prefix_q", close(row["prefix_corruption_ratios"][str(k)],
                             sum(labels[:k]) / k), (key, k))
            # The original run's native ndarray byte order is recovered from
            # its saved uniform hash; both byte orders remain stdlib-only.
            byte_orders = [endian for endian in ("<", ">") if
                           array_fingerprint(uniforms, endian + "f8", "d")
                           == row["failure_uniform_fingerprint"]]
            require(bool(byte_orders), f"Failure uniform fingerprint mismatch at {key}.")
            endian = byte_orders[0]
            identity = {
                "seed": key[0], "q_grid_index": qs.index(key[1]),
                "descriptor_sha256": row["descriptor_fingerprint"],
                "temporary_id_sha256": array_fingerprint(ids, endian + "i8", "q"),
                "history_sha256": array_fingerprint((h0,) * size, endian + "f8", "d"),
                "reference_order_sha256": array_fingerprint(ids, endian + "i8", "q"),
                "source_row_sha256": array_fingerprint(source, endian + "i8", "q"),
            }
            candidate_hash = hashlib.sha256(json.dumps(identity, sort_keys=True,
                                                       separators=(",", ":")).encode("utf-8")).hexdigest()
            require(candidate_hash == row["candidate_fingerprint"],
                    f"Candidate fingerprint/mapping mismatch at {key}.")
            availability = {p: tuple(value >= p for value in uniforms) for p in ps}
            for p, available in availability.items():
                require(array_fingerprint(available, "|b1", "?")
                        == row["availability_fingerprints"][str(p)],
                        f"Availability fingerprint mismatch at {key}/{p}.")
            posthoc = row["pairwise_cosine_posthoc"]
            for name, stats in posthoc.items():
                count = int(stats["n_pairs"])
                fraction = stats["interval_support_fraction"]
                require(count >= 0 and ((count == 0 and fraction is None)
                        or (count > 0 and fraction is not None and 0 <= fraction <= 1)),
                        f"Invalid saved pairwise aggregate at {key}/{name}.")
                normal_count, corrupt_count = size - sum(labels), sum(labels)
                if name == "normal_normal":
                    expected_pairs = math.comb(normal_count, 2)
                elif name == "normal_corrupt":
                    expected_pairs = normal_count * corrupt_count
                elif name == "corrupt_corrupt":
                    expected_pairs = math.comb(corrupt_count, 2)
                else:
                    _, left, right = name.split("_")
                    left, right = int(left), int(right)
                    expected_pairs = math.comb(counts[left], 2) if left == right else counts[left] * counts[right]
                checks.check("posthoc_pair_counts", count == expected_pairs, (key, name))
                if count:
                    require(all(math.isfinite(stats[field]) and -1 <= stats[field] <= 1
                                for field in ("min", "mean", "max")),
                            f"Invalid saved cosine aggregate at {key}/{name}.")
                    checks.check("posthoc_cosine_order", stats["min"] <= stats["mean"] <= stats["max"], (key, name))
                    checks.check("posthoc_supported_pair_integer", math.isclose(count * fraction,
                                 round(count * fraction), rel_tol=0, abs_tol=1e-9), (key, name))
                a = pairs[(key[1], name)]
                a["pairs"] += count
                a["supported_pairs"] += count * (fraction or 0.0)
                a["cosine_sum"] += count * (stats["mean"] or 0.0)
                if count:
                    a["min"] = min(a.get("min", stats["min"]), stats["min"])
                    a["max"] = max(a.get("max", stats["max"]), stats["max"])
            normal_pair, cross_pair = posthoc["normal_normal"], posthoc["normal_corrupt"]
            pair_conditions = ((not normal_pair["n_pairs"] or normal_pair["interval_support_fraction"] == 1)
                               and (not cross_pair["n_pairs"] or cross_pair["interval_support_fraction"] == 0))
            pool_conditions[key[1]]["n_scenes"] += 1
            pool_conditions[key[1]]["full_pool_pair_conditions"] += pair_conditions
            scenes[key] = {"labels": labels, "subtypes": subtypes, "availability": availability,
                           "candidate_fingerprint": candidate_hash,
                           "failure_uniform_fingerprint": row["failure_uniform_fingerprint"],
                           "availability_fingerprints": row["availability_fingerprints"],
                           "random_method_seed": row["random_method_seed"]}
    expected = {(seed, q) for seed in seeds for q in qs}
    require(set(scenes) == expected, "Saved candidate pool grid is incomplete.")
    require(len(scenes) == manifest["counts"]["candidate_pools"], "Manifest candidate count mismatch.")
    return scenes, pairs, pool_conditions


def read_runs(path: Path, manifest: dict, scenes: dict, checks: Checks):
    protocol = manifest["protocol"]
    seeds, qs, ps, methods = (protocol[name] for name in ("seeds", "q_values", "failure_values", "methods"))
    sizes, initial, needed = protocol["K_values"], protocol["initial_K"], protocol["M"]
    groups, by_k, paired = defaultdict(Counter), defaultdict(Counter), defaultdict(dict)
    seen = defaultdict(set)
    totals = Counter()
    with path.open(encoding="utf-8-sig", newline="") as stream:
        for line_number, row in enumerate(csv.DictReader(stream), 2):
            method, seed, q, p = row["method"], int(row["seed"]), float(row["q_C_requested"]), float(row["p_fail_requested"])
            key, scene_key = (method, q, p), (seed, q)
            context = f"raw line {line_number}, {method}/{seed}/{q}/{p}"
            require(method in methods and seed in seeds and q in qs and p in ps
                    and seed not in seen[key], f"Unknown or duplicate raw record: {context}")
            seen[key].add(seed)
            require(all(row[name] == manifest[name] for name in
                        ("run_id", "algorithm_version", "code_version", "dataset_fingerprint")),
                    f"Mixed raw identity: {context}")
            scene = scenes[scene_key]
            require(row["candidate_fingerprint"] == scene["candidate_fingerprint"]
                    and row["failure_uniform_fingerprint"] == scene["failure_uniform_fingerprint"]
                    and row["availability_fingerprint"] == scene["availability_fingerprints"][str(p)],
                    f"Raw/scenario fingerprint mismatch: {context}")
            checks.check("random_seed_identity", int(row["random_method_seed"]) == scene["random_method_seed"], context)
            k = int(row["final_K"])
            require(k in sizes, f"Unknown final K: {context}")
            selected, trace = json.loads(row["selected_temporary_ids"]), json.loads(row["gt_trace"])
            require(isinstance(selected, list) and all(isinstance(i, int) and 0 <= i < k for i in selected)
                    and len(selected) == len(set(selected)), f"Invalid selected IDs: {context}")
            available, labels, subtypes = scene["availability"][p], scene["labels"], scene["subtypes"]
            formed, reliable = int(row["gt_success"]), int(row["reliable_success"])
            require(formed in (0, 1) and reliable in (0, 1), f"Invalid binary outcome: {context}")
            corrupt = sum(labels[i] for i in selected)
            all_responsive = bool(selected) and all(available[i] for i in selected)
            valid_formation = len(selected) == needed
            valid_reliable = valid_formation and all_responsive and corrupt == 0
            checks.check("formation_metric", formed == valid_formation and (formed or not selected), context)
            checks.check("reliability_metric", reliable == valid_reliable, context)
            checks.check("selected_evaluation", int(row["gt_size"]) == len(selected)
                         and int(row["gt_corrupt_count"]) == corrupt
                         and int(row["gt_normal_count"]) == len(selected) - corrupt
                         and int(row["gt_all_responsive"]) == int(all_responsive), context)
            checks.check("selected_availability", not selected or all_responsive, context)
            expected_status = "gt_unavailable" if not formed else ("gt_invalid_response" if not all_responsive
                              else "gt_contaminated" if corrupt else "reliable_success")
            checks.check("status_metric", row["status"] == expected_status
                         and bool(row["failure_reason"]) == (expected_status != "reliable_success"), context)
            responsive = sum(available[:k])
            checks.check("actual_response_counts", int(row["responsive_count"]) == responsive
                         and int(row["pool_responsive_count"]) == sum(available), context)
            for field, expected in (("q_C_realized", sum(labels) / len(labels)),
                                    ("initial_K_realized_cr", sum(labels[:initial]) / initial),
                                    ("K75_realized_cr", sum(labels[:75]) / 75),
                                    ("K100_realized_cr", sum(labels[:100]) / 100),
                                    ("p_fail_realized", 1 - sum(available) / len(available)),
                                    ("initial_K_p_fail_realized", 1 - sum(available[:initial]) / initial),
                                    ("final_K_p_fail_realized", 1 - responsive / k)):
                checks.check("actual_composition_and_failure", close(row[field], expected), context + "/" + field)
            for field, expected in (("N", protocol["N"]), ("d", protocol["d"]), ("H", protocol["H"]),
                                    ("global_cr", protocol["global_cr_percent"]),
                                    ("global_cr_realized", protocol["global_cr_percent"]),
                                    ("tau_lower", protocol["tau_lower"]), ("tau_upper", protocol["tau_upper"]),
                                    ("initial_K", initial), ("M", needed),
                                    ("max_candidate_pool", protocol["max_candidate_pool"]),
                                    ("candidate_pool_size", protocol["max_candidate_pool"])):
                checks.check("raw_protocol", close(row[field], expected), context + "/" + field)
            expected_sizes = sizes[:sizes.index(k) + 1]
            checks.check("nested_expansion_trace", [step["candidate_size"] for step in trace] == expected_sizes
                         and int(row["candidate_expansion_count"]) == sizes.index(k), context)
            require(bool(trace), f"Missing construction trace: {context}")
            majority = method in ("RACA-HIM", "Fixed-K Majority")
            for step_index, step in enumerate(trace):
                step_k = int(step["candidate_size"])
                require(step_k in sizes, f"Invalid trace K: {context}")
                count = int(step["passing_count"])
                step_responsive = sum(available[:step_k])
                checks.check("trace_response_counts", step["responsive_count"] == step_responsive
                             and 0 <= count <= step_responsive, context)
                checks.check("majority_threshold", step["threshold"] == (admission_threshold(protocol, step_k) if majority else None), context)
                comparisons = step_responsive * (step_responsive - 1) if majority else 0
                checks.check("trace_logical_comparisons", step["pairwise_comparison_count"] == comparisons, context)
                if not majority:
                    checks.check("baseline_trace_count", count == step_responsive, context)
                if step_index < len(trace) - 1:
                    checks.check("stop_when_M_reached", count < needed, context)
            checks.check("formed_vs_final_trace", formed == (trace[-1]["passing_count"] >= needed), context)
            checks.check("total_logical_comparisons", int(row["pairwise_comparison_count"])
                         == sum(step["pairwise_comparison_count"] for step in trace), context)
            checks.check("method_expansion_boundary", (k == initial if method == "Fixed-K Majority"
                         else formed or k == sizes[-1]), context)
            if method == "Top-M" and formed:
                checks.check("top_M_fixed_order", selected == [i for i in range(k) if available[i]][:needed], context)
            if majority:
                paired[(seed, q, p)][method] = (formed, reliable, k, tuple(selected))
            normal_responsive = sum(available[i] and labels[i] == 0 for i in range(k))
            local = by_k[(method, q, p, k)]
            group = groups[key]
            for target in (local, group):
                target["n"] += 1
                target["formed"] += formed
                target["reliable"] += reliable
                target["contaminated"] += int(formed and corrupt > 0)
                target["unformed"] += 1 - formed
                target["normal_responsive_sum"] += normal_responsive
                target["strict_normal_majority"] += normal_responsive > k / 2
                target["normal_responsive_meets_admission_threshold"] += normal_responsive >= admission_threshold(protocol, k)
                target["normal_responsive_at_least_M"] += normal_responsive >= needed
                target["expanded"] += k > initial
                target["reliable_expansion_rescues"] += int(k > initial and reliable)
                target["normal_selected"] += len(selected) - corrupt
                target["selected_size"] += len(selected)
                target["K_sum"] += k
                target["expansions_sum"] += sizes.index(k)
                target["responsive_sum"] += responsive
                target["comparisons_sum"] += int(row["pairwise_comparison_count"])
                for subtype in range(4):
                    target[f"selected_{SUBTYPES[subtype]}"] += sum(subtypes[i] == subtype for i in selected)
            totals["method_runs"] += 1
            totals["gt_formed"] += formed
            totals["reliable_success"] += reliable
            totals["gt_unavailable"] += 1 - formed
            totals["gt_contaminated"] += int(formed and corrupt > 0)
    expected_groups = {(m, q, p) for m in methods for q in qs for p in ps}
    require(set(groups) == expected_groups and all(values == set(seeds) for values in seen.values()),
            "Raw result grid or per-point seed inventory is incomplete.")
    for name, actual in totals.items():
        checks.check("manifest_record_counts", manifest["counts"][name] == actual, name)
    checks.check("manifest_scenario_count", manifest["counts"]["scenarios"] == len(paired), "scenarios")
    for key, values in paired.items():
        require(set(values) == {"RACA-HIM", "Fixed-K Majority"}, f"Missing majority method pair at {key}.")
        raca, fixed = values["RACA-HIM"], values["Fixed-K Majority"]
        checks.check("paired_fixed_K_ablation", raca == fixed if raca[2] == initial else fixed[0] == 0, key)
    return groups, by_k, totals


def check_saved_statistics(paths: dict[str, Path], manifest: dict, groups: dict, checks: Checks) -> None:
    for name in ("summary", "diagnostics"):
        seen = set()
        with paths[name].open(encoding="utf-8-sig", newline="") as stream:
            for row in csv.DictReader(stream):
                key = (row["method"], float(row["q_C"]), float(row["p_fail"]))
                require(key in groups and key not in seen, f"Duplicate/unknown {name} key: {key}")
                seen.add(key)
                require(all(row[field] == manifest[field] for field in
                            ("run_id", "algorithm_version", "code_version", "dataset_fingerprint")),
                        f"Mixed {name} provenance at {key}.")
                a = groups[key]
                n, formed, rate = a["n"], a["formed"], a["reliable"] / a["n"]
                if name == "summary":
                    lo, hi = wilson(a["reliable"], n)
                    expected = {"n_total": n, "n_gt_formed": formed, "n_reliable_success": a["reliable"],
                                "gt_formation_rate": formed / n, "reliable_gt_rate": rate, "mean": rate,
                                "std": math.sqrt(rate * (1 - rate) * n / (n - 1)) if n > 1 else math.nan,
                                "ci95_lower": lo, "ci95_upper": hi}
                    checks.check("summary_ci_method", row["ci_method"] == "Wilson 95%", key)
                else:
                    expected = {"n_total": n, "n_gt_formed": formed, "n_gt_insufficient": a["unformed"],
                                "n_gt_contaminated": a["contaminated"], "gt_formation_rate": formed / n,
                                "unconditional_gt_contamination_rate": a["contaminated"] / n,
                                "formed_gt_conditional_contamination_rate": a["contaminated"] / formed if formed else math.nan,
                                "formed_gt_conditional_normal_fraction": a["normal_selected"] / a["selected_size"] if a["selected_size"] else math.nan,
                                "mean_final_K": a["K_sum"] / n, "mean_candidate_expansions": a["expansions_sum"] / n,
                                "mean_responsive_candidate_count": a["responsive_sum"] / n,
                                "mean_pairwise_comparison_count": a["comparisons_sum"] / n}
                for field, value in expected.items():
                    checks.check(name + "_matches_raw", close(row[field], value), (key, field))
        require(seen == set(groups), f"Saved {name} is incomplete.")


def audit_rows(manifest: dict, groups: dict, by_k: dict):
    final_rows, subtype_rows = [], []
    for method in manifest["protocol"]["methods"]:
        for q in manifest["protocol"]["q_values"]:
            for p in manifest["protocol"]["failure_values"]:
                group = groups[(method, q, p)]
                sizes = [manifest["protocol"]["initial_K"]] if method == "Fixed-K Majority" else manifest["protocol"]["K_values"]
                for k in sizes:
                    a = by_k.get((method, q, p, k), Counter())
                    common = {field: manifest[field] for field in
                              ("run_id", "algorithm_version", "code_version", "dataset_fingerprint")}
                    common.update(audit_version=AUDIT_VERSION, method=method, q_C=q, p_fail=p, final_K=k,
                                  n_group_total=group["n"], n_final_K=a["n"], n_gt_formed=a["formed"])
                    final_rows.append({**common, "n_reliable": a["reliable"], "n_contaminated": a["contaminated"],
                                       "n_unformed": a["unformed"], "reliable_contribution_to_group_rate": a["reliable"] / group["n"],
                                       "formed_conditional_reliable_rate": a["reliable"] / a["formed"] if a["formed"] else math.nan,
                                       "n_strict_normal_responsive_majority": a["strict_normal_majority"],
                                       "n_normal_responsive_meets_admission_threshold": a["normal_responsive_meets_admission_threshold"],
                                       "n_normal_responsive_at_least_M": a["normal_responsive_at_least_M"],
                                       "mean_normal_responsive_count": a["normal_responsive_sum"] / a["n"] if a["n"] else math.nan,
                                       "n_expanded_trials_in_group": group["expanded"],
                                       "n_reliable_expansion_rescues_in_group": group["reliable_expansion_rescues"]})
                    subtype_rows.append({**common, "n_selected_members": a["selected_size"],
                                         **{f"n_selected_{name}": a[f"selected_{name}"] for name in SUBTYPES}})
    return final_rows, subtype_rows


def markdown_table(headers, rows) -> list[str]:
    return ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |",
            *("| " + " | ".join(str(value) for value in row) + " |" for row in rows)]


def build_report(root, manifest, hashes, checks, groups, by_k, pairs, pool_conditions, totals) -> str:
    protocol = manifest["protocol"]
    initial, needed, pool = protocol["initial_K"], protocol["M"], protocol["max_candidate_pool"]
    timestamp = datetime.now(timezone(timedelta(hours=8))).isoformat(timespec="seconds")
    lines = ["# 第四组已保存实验结果审计", "", f"审计时间：{timestamp}（Asia/Shanghai）。", "",
             f"输入：`{root}`。原始run：`{manifest['run_id']}`。", "",
             "本报告只读取用户已运行并保存的CSV/JSONL/manifest，执行标准库统计核对。未加载Dataset、未计算余弦相似度、未运行GT算法、实验、测试或绘图，未改写results目录。", "",
             "## 结论与证据范围", ""]
    if checks.failures:
        lines += [f"检测到{sum(checks.failures.values())}处保存证据不一致，须先审查下列失败项。不能将本报告解释为算法正确性证明。", ""]
    else:
        lines += [f"保存文件SHA-256及身份核验通过；{totals['method_runs']:,}条方法记录的后置指标、响应状态、门槛、停止条件与摘要一致性核对未发现不一致。", "",
                  "现有证据未发现必须通过强制单调来修理的实现错误。RACA-HIM在污染增加后回升，可以由按M提前结束的候选扩展机制与mild损坏的相似度重叠解释。本审计未修理、平滑或替换趋势，也没有重新验证完整GT内核。", ""]
    thresholds = ", ".join(f"K={k}:θ={admission_threshold(protocol, k)}" for k in protocol["K_values"])
    lines += [f"门槛来自输入manifest保存的政策：{thresholds}。当前项目源码与本审计入口都不能覆盖旧run的政策身份。", ""]
    if protocol.get("quorum_fraction") is None:
        lines += ["本次用户已授权重新设计GT规则并同步修订论文；新政策使用2/3 quorum及history后按支持数、支持pair均值、临时ID排序。本报告审计的是原strict-majority版本，不能视为新政策的执行结果；本次审计没有生成新实验数值。", ""]
    lines += ["## 保存结果来源", "", f"原算法版本：`{manifest['algorithm_version']}`。", "",
              f"原源码身份：`{manifest['code_version']}`。", "", f"原数据身份：`{manifest['dataset_fingerprint']}`。", "",
              "审计使用原manifest保存的源码和数据文件清单核验身份的内部一致性，不要求它们等于当前修改后的源码，也不重新标记旧结果为新代码运行结果。以下保留本次读取文件的内容身份。", ""]
    lines += markdown_table(["证据", "SHA-256"], [(name, f"`{digest}`") for name, digest in hashes.items()])
    lines += ["", "## 核对项", ""]
    lines += markdown_table(["项目", "核对次数", "不一致次数"],
                            [(name, count, checks.failures[name]) for name, count in sorted(checks.counts.items())])
    if checks.examples:
        lines += ["", "失败示例：", "", *(f"- `{example}`" for example in checks.examples)]
    lines += ["", "## p_fail=0 的可靠率与候选扩展分解", "",
              "每行使用该点全部seed为分母。污染是已形成GT但含真实损坏；不足是未形成GT，两者没有从可靠率分母删除。", ""]
    selected_qs = [q for q in protocol["q_values"] if q >= 0.25]
    lines += markdown_table(["q_C", "总数", "可靠", "污染", "不足", "扩展次数≥1的trial", "扩展后可靠贡献"],
                            [(f"{q:.0%}", a["n"], a["reliable"], a["contaminated"], a["unformed"],
                              a["expanded"], f"{a['reliable_expansion_rescues'] / a['n']:.3f}")
                             for q in selected_qs if (a := groups.get(("RACA-HIM", q, 0.0)))])
    lines += ["", "扩展后可靠贡献等于final_K>初始K的可靠trial数除以全部trial数。配对核对要求RACA-HIM扩展时Fixed-K没有形成GT，因此在核对通过时这些是相对于禁止扩展的额外可靠成功。此贡献分解不代表强制所有trial扩展后的反事实结果。", ""]
    detail_qs = [q for q in (0.30, 0.35, 0.50) if q in protocol["q_values"]]
    lines += markdown_table(["q_C", "最终K", "该K结束", "可靠", "污染", "不足", "正常可用严格多数"],
                            [(f"{q:.0%}", k, a["n"], a["reliable"], a["contaminated"], a["unformed"], a["strict_normal_majority"])
                             for q in detail_qs for k in protocol["K_values"]
                             if (a := by_k.get(("RACA-HIM", q, 0.0, k)))])
    for q in detail_qs:
        a = groups[("RACA-HIM", q, 0.0)]
        at_initial = by_k.get(("RACA-HIM", q, 0.0, initial), Counter())
        lines += ["", f"q_C={q:.0%}：{at_initial['n']}/{a['n']}次在K={initial}结束，其中{at_initial['contaminated']}次污染；"
                       f"{a['expanded']}次扩展，扩展后{a['reliable_expansion_rescues']}次形成可靠GT。最终可靠率{a['reliable'] / a['n']:.3f}。"]
    lines += ["", "按论文/Prompt规定，只有当前通过多数的候选不足M才扩展，达到M即停止。污染降低通过数量，触发更大K和新的多数阈值；mild候选可能因此被排除。可靠率同时取决于是否形成足够GT以及形成后是否纯净，这两个条件不保证随q_C或p_fail单调。增加p_fail也可能减少污染或触发扩展，在高失败率时则主要造成GT不足。", "",
              "## 已选GT的subtype及已保存相似度证据", ""]
    lines += markdown_table(["q_C（p_fail=0）", *SUBTYPES],
                            [(f"{q:.0%}", *(groups[("RACA-HIM", q, 0.0)][f"selected_{name}"] for name in SUBTYPES))
                             for q in selected_qs])
    pair_names = ("normal_normal", "normal_corrupt", "subtype_0_3", "subtype_1_3", "subtype_2_2", "subtype_3_3")
    lines += ["", "下表完全来自JSONL已经保存的100候选全池posthoc统计；支持率按pair数加权，未重新计算相似度。support不是同一seed独立样本，不能把pair总数用于实验可靠率的置信区间。", ""]
    lines += markdown_table(["q_C", "pair类别", "pair总数", "阈值内比例", "cosine均值", "保存的min", "保存的max"],
                            [(f"{q:.0%}", name, int(a["pairs"]), f"{a['supported_pairs'] / a['pairs']:.6f}",
                              f"{a['cosine_sum'] / a['pairs']:.6f}", f"{a['min']:.6f}", f"{a['max']:.6f}")
                             for q in detail_qs for name in pair_names
                             if (a := pairs.get((q, name))) and a["pairs"]])
    lines += ["", "subtype_0_3是普通正常–mild损坏，subtype_1_3是高异构正常–mild损坏。即使binary_label为损坏，mild也可能获得正常候选支持；即使binary_label为正常，高异构正常之间也可能不满足阈值。保存记录里的标签只用于本审计后置统计，没有反馈给任何算法。", "",
              "## 定理前提与不能核验的部分", "",
              "正常且可用的候选严格多数是原定理的一项计数前提，不能仅凭该条件证明GT纯净，也不能将该充分条件解释为GT形成的必要条件。原定理同时要求正常候选两两满足合法区间，且可检测异常与正常候选全部不相似。这里真实损坏类别包含可与正常相似的mild，因此原定理不保证排除所有真实损坏。新增quorum版本还需要按其保存门槛核查正常可用数量，CSV另行记录了该计数。", ""]
    lines += markdown_table(["q_C", "候选池数", "全池正常clique且所有损坏与正常分离的池数"],
                            [(f"{q:.0%}", pool_conditions[q]["n_scenes"], pool_conditions[q]["full_pool_pair_conditions"])
                             for q in selected_qs])
    lines += ["", "上述全池pair条件将所有binary_label损坏都当作需分离异常，强于论文只针对可检测异常的保证。该统计只描述全池，不替代每个实际前缀与可用子集上的定理检查。", "",
              f"全池q_C=50%时正常为{pool // 2}个、K={pool}的保存录取门槛为{admission_threshold(protocol, pool)}。即使p_fail=0，严格正常多数也不成立；某些可靠GT可通过正常候选获得mild跨类别支持形成。这属于原定理充分条件之外的实际结果。", "",
              "输入没有保存完整cosine矩阵、各前缀逐候选support_counts和accepted集合。因此本审计能够核对门槛、trace响应数、停止规则与后置污染；不能独立验证passing_count是否由正确逐pair判定得到，不能重验每个GT成员的录取资格或支持排序/固定tie-break，不能证明early-stopping等价性。descriptor字节没有保存到这些结果文件，无法独立重算descriptor/cosine指纹；candidate指纹核对依赖保存的descriptor指纹。", "",
              "## Top-M与Random-M为何快速接近0", "",
              f"主指标要求{needed}个GT全部正常。p_fail=0、随机候选顺序和相同h0下，Top-M以及Random-M选出的{needed}个候选在完整{pool}池上边际分布为不看标签的均匀子集。因此纯净概率为 C({pool}-B,{needed}) / C({pool},{needed})，其中B为全池损坏数量。这是纯净事件的理论参考，没有替换实际统计。", ""]
    lines += markdown_table(["q_C", "B", "全部M纯净理论概率", "Top-M已保存可靠率", "Random-M已保存可靠率"],
                            [(f"{q:.0%}", round(pool * q),
                              f"{(math.comb(pool - round(pool * q), needed) / math.comb(pool, needed)) if pool - round(pool * q) >= needed else 0:.8g}",
                              f"{groups[('Top-M', q, 0.0)]['reliable'] / groups[('Top-M', q, 0.0)]['n']:.3f}",
                              f"{groups[('Random-M', q, 0.0)]['reliable'] / groups[('Random-M', q, 0.0)]['n']:.3f}")
                             for q in protocol["q_values"]])
    lines += ["", "纯净概率随污染迅速减小，两条baseline接近0是指标与随机抽样的结果，不足以判定代码异常；它们没有多数筛选，h0相同使Top-M排序无标签信息。p_fail>0时还可能发生响应不足，实际均值应以保存的全部seed统计为准。", "",
              "## 后续处理", "", "保留原始结果，使用final_K分层和污染subtype解释曲线。不要为了单调性强制用K=100、过滤mild/高异构正常、读取标签改变录取、降低多数门槛或改动失败mask。用户授权的新算法政策需要单独标注实现与实验版本并由用户重新运行，不能把本次审计当作新实验结果。", "",
              "输出文件：`04_ground_truth_audit_final_k.csv`按method/q_C/p_fail/final_K分解可靠、污染、不足与严格多数计数；`04_ground_truth_audit_subtypes.csv`统计同一分层中实际GT成员subtype。零trial的final_K分层也保留，条件统计使用nan。", ""]
    return "\n".join(lines)


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Read-only audit of saved Group 04 evidence; no GT or plot execution")
    parser.add_argument("--result-root", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--report-root", type=Path, default=DEFAULT_REPORTS)
    args = parser.parse_args(argv)
    root, reports = args.result_root.resolve(), args.report_root.resolve()
    require(not reports.is_relative_to(root), "Audit reports must be outside the input result directory.")
    manifest, paths, hashes = load_evidence(root)
    protocol = manifest["protocol"]
    require(protocol["K_values"] == [50, 75, 100] and protocol["initial_K"] == 50
            and protocol["M"] == 25 and protocol["max_candidate_pool"] == 100,
            "This audit supports the saved Group 04 50/75/100, M=25 protocol.")
    require(0.0 in protocol["failure_values"] and set(protocol["methods"])
            == {"RACA-HIM", "Fixed-K Majority", "Top-M", "Random-M"}, "Unsupported saved method grid.")
    checks = Checks()
    scenes, pairs, pool_conditions = read_scenarios(paths["scenarios"], manifest, checks)
    groups, by_k, totals = read_runs(paths["raw"], manifest, scenes, checks)
    check_saved_statistics(paths, manifest, groups, checks)
    # Check again before writing: concurrent changes invalidate the audit.
    require(sha256_file(root / "logs" / "run_manifest.json") == hashes["manifest"],
            "Manifest changed while the audit was reading.")
    for name, path in paths.items():
        require(sha256_file(path) == hashes[name], f"Saved {name} changed while auditing.")
    final_rows, subtype_rows = audit_rows(manifest, groups, by_k)
    report = build_report(root, manifest, hashes, checks, groups, by_k, pairs, pool_conditions, totals)
    reports.mkdir(parents=True, exist_ok=True)
    report_path = reports / "04_ground_truth_result_audit.md"
    write_csv(reports / "04_ground_truth_audit_final_k.csv", final_rows)
    write_csv(reports / "04_ground_truth_audit_subtypes.csv", subtype_rows)
    report_path.write_text(report, encoding="utf-8")
    print(f"Saved-result audit: {totals['method_runs']} records, {sum(checks.failures.values())} inconsistencies.")
    print(f"Report: {report_path}")
    if checks.failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
