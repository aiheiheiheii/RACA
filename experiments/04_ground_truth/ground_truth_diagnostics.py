"""Post-construction evidence; labels never enter the GT method boundary.

This module reads a completed MethodResult and the already computed exact
cosine cache. It neither invokes a method nor computes new descriptor cosines.
All subtype counts and theorem conditions are evaluation-only observations.
Stage averages are conditional on visiting that K; final-K rates retain the
full trial denominator. Missing pair conditions are None, never vacuous True.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np

from experiments.common.similarity import within_interval
from .config_ground_truth import (
    GT_SIZE_M, K_VALUES, MAX_CANDIDATE_POOL_SIZE, METHODS, Q_VALUES,
    FAILURE_VALUES, REPETITIONS, TAU_LOWER, TAU_UPPER,
    GT_QUORUM_NUMERATOR, GT_QUORUM_DENOMINATOR,
)

SUBTYPE_NAMES = (
    "common_normal", "high_heterogeneity_normal", "severe_corrupt", "mild_corrupt",
)
COUNT_NAMES = ("total", "normal", "corrupt", *SUBTYPE_NAMES)
MAJORITY_METHODS = frozenset(("RACA-HIM", "Fixed-K Majority"))
CONDITION_NAMES = (
    "normal_responder_strict_majority", "normal_responder_quorum_met", "normal_normal_legal",
    "normal_corrupt_separated", "theorem_premises_satisfied", "full_purity_premises_satisfied",
)


def _counts(subtypes: np.ndarray, mask: np.ndarray) -> dict[str, int]:
    result = {name: int(np.count_nonzero(mask & (subtypes == index)))
              for index, name in enumerate(SUBTYPE_NAMES)}
    result.update(total=int(np.count_nonzero(mask)),
                  normal=result[SUBTYPE_NAMES[0]] + result[SUBTYPE_NAMES[1]],
                  corrupt=result[SUBTYPE_NAMES[2]] + result[SUBTYPE_NAMES[3]])
    return result


def _support_summary(subtypes, responsive, supports) -> dict:
    """Summaries exclude self and exclude unavailable candidates."""
    result = {}
    for index, name in enumerate(SUBTYPE_NAMES):
        mask = responsive & (subtypes == index)
        values = None if supports is None else supports[mask]
        result[name] = {
            "n_responders": int(np.count_nonzero(mask)),
            "peer_support_min": int(values.min()) if values is not None and len(values) else None,
            "peer_support_mean": float(values.mean()) if values is not None and len(values) else None,
            "peer_support_max": int(values.max()) if values is not None and len(values) else None,
        }
    return result


def diagnose_construction(result, labels, subtypes, available, similarity_cache) -> dict:
    """Audit visited stages after selection, returning only JSON-safe values.

    A corrupted responder is called ``undetectable`` here if it has at least
    one interval-valid comparison to a responsive normal candidate. This is
    precisely a violation of the theorem's detectable-abnormal separation
    premise, not a claim about model semantics or the returned GT's purity.
    Both the majority and quorum denominator are complete K, including failed
    slots. Quorum follows the declared redesigned rule in config. The subset
    theorem excludes detectable abnormal candidates only; its premises do not
    imply that all label-corrupted candidates are detectable. Full label purity
    needs the additional separation condition reported independently here.
    """
    truth, subtype = np.asarray(labels), np.asarray(subtypes)
    response, cache = np.asarray(available), np.asarray(similarity_cache)
    size = MAX_CANDIDATE_POOL_SIZE
    if (truth.shape != (size,) or subtype.shape != (size,)
            or not np.isin(truth, (0, 1)).all() or not np.isin(subtype, (0, 1, 2, 3)).all()
            or not np.array_equal(truth, (subtype >= 2).astype(truth.dtype))):
        raise ValueError("Posthoc labels/subtypes must describe the original 100 candidate slots")
    if response.shape != (size,) or response.dtype != np.bool_:
        raise ValueError("Posthoc responder availability must match the 100 candidate slots")
    if (cache.shape != (size, size) or not np.all(np.isfinite(cache))
            or np.any((cache < -1) | (cache > 1))):
        raise ValueError("Diagnostics require the previously computed exact 100x100 cosine cache")
    if result.method not in METHODS or result.final_K not in K_VALUES:
        raise ValueError("Unknown completed method or final K")
    selected = np.asarray(result.selected_indices)
    if (selected.ndim != 1 or not np.issubdtype(selected.dtype, np.integer)
            or len(np.unique(selected)) != len(selected)
            or np.any((selected < 0) | (selected >= result.final_K))):
        raise ValueError("Completed selection contains invalid or repeated candidate slots")
    if (bool(result.success) != (len(selected) == GT_SIZE_M)
            or (not result.success and len(selected)) or not np.all(response[selected])):
        raise ValueError("Completed selection disagrees with formation or responder availability")
    expected_sizes = K_VALUES[:K_VALUES.index(result.final_K) + 1]
    if tuple(int(step["candidate_size"]) for step in result.trace) != expected_sizes:
        raise ValueError("Diagnostics must describe exactly the stages visited by the method")
    majority = result.method in MAJORITY_METHODS
    selected_mask = np.zeros(size, dtype=bool)
    selected_mask[selected] = True
    selected_counts = _counts(subtype, selected_mask)
    formed = bool(result.success)
    reliable = bool(formed and selected_counts["corrupt"] == 0)
    stages = []
    for step in result.trace:
        k = int(step["candidate_size"])
        stage_subtype, responsive = subtype[:k], response[:k]
        candidate_counts = _counts(stage_subtype, np.ones(k, dtype=bool))
        responder_counts = _counts(stage_subtype, responsive)
        strict_majority = responder_counts["normal"] >= (k + 2) // 2
        quorum = (GT_QUORUM_NUMERATOR * k + GT_QUORUM_DENOMINATOR - 1) // GT_QUORUM_DENOMINATOR
        quorum_met = responder_counts["normal"] >= quorum
        normal_size_sufficient = responder_counts["normal"] >= GT_SIZE_M
        stage = {
            "candidate_size": k, "threshold": int(step["threshold"]) if majority else None,
            "passing_count": int(step["passing_count"]),
            "candidate_counts": candidate_counts, "responder_counts": responder_counts,
            "admitted_counts": None,
            "normal_responder_strict_majority": bool(strict_majority),
            "normal_responder_quorum_met": bool(quorum_met),
            "normal_responder_size_sufficient": bool(normal_size_sufficient),
            "normal_normal_legal": None, "normal_corrupt_separated": None,
            "normal_normal_pair_count": None, "normal_normal_supported_pair_count": None,
            "normal_corrupt_pair_count": None, "normal_corrupt_supported_pair_count": None,
            "undetectable_corrupt_count": None, "detectable_corrupt_count": None,
            "detectable_corrupt_admitted_count": None,
            "theorem_premises_satisfied": None, "full_purity_premises_satisfied": None,
            "support_by_subtype": _support_summary(stage_subtype, responsive, None),
            "pair_condition_notes": {
                "normal_normal_legal": "baseline_does_not_use_similarity",
                "normal_corrupt_separated": "baseline_does_not_use_similarity",
                "undetectable_corrupt_count": "baseline_does_not_use_similarity",
            },
        }
        if int(step["responsive_count"]) != responder_counts["total"]:
            raise ValueError("Completed trace responder count disagrees with the saved availability")
        if majority:
            threshold = quorum
            if stage["threshold"] != threshold:
                raise ValueError("Completed admission threshold differs from the declared current-K quorum")
            support = np.array(within_interval(cache[:k, :k], TAU_LOWER, TAU_UPPER), copy=True)
            support &= responsive[:, None] & responsive[None, :]
            np.fill_diagonal(support, False)
            supports = support.sum(axis=1, dtype=np.int64)
            admitted = responsive & (supports + 1 >= threshold)
            if int(np.count_nonzero(admitted)) != stage["passing_count"]:
                raise ValueError("Posthoc majority admission disagrees with the completed method trace")
            if k == result.final_K and formed and not np.all(admitted[selected]):
                raise ValueError("Completed selection includes a candidate not admitted by the declared quorum")
            stage["admitted_counts"] = _counts(stage_subtype, admitted)
            stage["support_by_subtype"] = _support_summary(stage_subtype, responsive, supports)
            normal = responsive & (stage_subtype < 2)
            corrupt = responsive & (stage_subtype >= 2)
            normal_indices, corrupt_indices = np.flatnonzero(normal), np.flatnonzero(corrupt)
            n_normal, n_corrupt = len(normal_indices), len(corrupt_indices)
            nn_pairs = n_normal * (n_normal - 1) // 2
            nn_support = int(np.count_nonzero(np.triu(support & normal[:, None] & normal[None, :], 1)))
            nc_pairs = n_normal * n_corrupt
            cross = support[np.ix_(corrupt_indices, normal_indices)]
            nc_support = int(np.count_nonzero(cross))
            normal_legal = bool(nn_support == nn_pairs) if nn_pairs else None
            separated = bool(nc_support == 0) if nc_pairs else None
            undetectable = int(np.count_nonzero(np.any(cross, axis=1))) if n_normal else None
            detectable_mask = ~np.any(cross, axis=1) if n_normal else None
            detectable = int(np.count_nonzero(detectable_mask)) if detectable_mask is not None else None
            detectable_admitted = (int(np.count_nonzero(admitted[corrupt_indices] & detectable_mask))
                                   if detectable_mask is not None else None)
            # No corrupt responders means separation is unnecessary, but its
            # pair statistic remains None because no cross-label pair exists.
            premises = (False if not quorum_met or not normal_size_sufficient
                        else bool(normal_legal) if normal_legal is not None else None)
            full_purity = (False if premises is False else
                           bool(separated if n_corrupt else True) if premises is True
                           and (separated is not None or not n_corrupt) else None)
            stage.update(
                normal_normal_legal=normal_legal, normal_corrupt_separated=separated,
                normal_normal_pair_count=int(nn_pairs), normal_normal_supported_pair_count=nn_support,
                normal_corrupt_pair_count=int(nc_pairs), normal_corrupt_supported_pair_count=nc_support,
                undetectable_corrupt_count=undetectable, detectable_corrupt_count=detectable,
                detectable_corrupt_admitted_count=detectable_admitted,
                theorem_premises_satisfied=premises, full_purity_premises_satisfied=full_purity,
                pair_condition_notes={
                    "normal_normal_legal": None if nn_pairs else "fewer_than_two_responsive_normals",
                    "normal_corrupt_separated": None if nc_pairs else "no_responsive_normal_corrupt_pairs",
                    "undetectable_corrupt_count": None if n_normal else "no_responsive_normal_candidates",
                },
            )
        elif stage["passing_count"] != responder_counts["total"]:
            raise ValueError("Baseline trace must count responders without majority filtering")
        if k != result.final_K and stage["passing_count"] >= GT_SIZE_M:
            raise ValueError("Completed method expanded despite having M candidates at an earlier stage")
        if k == result.final_K and formed != (stage["passing_count"] >= GT_SIZE_M):
            raise ValueError("Completed formation disagrees with final-stage candidate count")
        stages.append(stage)
    return {
        "method": result.method, "final_K": int(result.final_K), "stages": stages,
        "selected_subtype_counts": selected_counts,
        "outcome": {
            "formed": int(formed), "reliable_success": int(reliable),
            "status": "reliable_success" if reliable else "gt_contaminated" if formed else "gt_unavailable",
            "gt_normal_count": selected_counts["normal"], "gt_corrupt_count": selected_counts["corrupt"],
        },
    }


def _stage_stats() -> dict:
    return {
        "n_visited": 0,
        "counts": {kind: {name: 0 for name in COUNT_NAMES}
                   for kind in ("candidate", "responder", "admitted")},
        "conditions": {name: {"n_defined": 0, "n_true": 0} for name in CONDITION_NAMES},
        "n_admission_defined": 0, "undetectable_sum": 0, "n_undetectable_defined": 0,
        "detectable_sum": 0, "detectable_admitted_sum": 0, "n_detectable_defined": 0,
    }


class DiagnosticAccumulator:
    """Bounded memory: aggregate counts only, with no descriptors or per-seed arrays.

    ``finish`` returns all method/q/p/K rows, including K values never visited.
    ``n_total`` is the actual point denominator (1000 for a complete formal run).
    Existing SummaryAccumulator remains responsible for unique-seed checks.
    """

    def __init__(self) -> None:
        self.totals = defaultdict(int)
        self.final = defaultdict(lambda: {
            "n_at_final_K": 0, "n_reliable": 0, "n_contaminated": 0, "n_insufficient": 0,
        })
        self.admission = defaultdict(_stage_stats)

    def add(self, row: dict, diagnostic: dict) -> None:
        key = (row["method"], float(row["q_C_requested"]), float(row["p_fail_requested"]))
        if key[0] not in METHODS or key[1] not in Q_VALUES or key[2] not in FAILURE_VALUES:
            raise ValueError("Posthoc diagnostic row is outside the declared method/q/p grid")
        final_k = int(row["final_K"])
        outcome = diagnostic["outcome"]
        if (diagnostic["method"] != key[0] or int(diagnostic["final_K"]) != final_k
                or final_k not in K_VALUES
                or int(outcome["formed"]) != int(row["gt_success"])
                or int(outcome["reliable_success"]) != int(row["reliable_success"])
                or int(outcome["gt_corrupt_count"]) != int(row["gt_corrupt_count"])
                or outcome["status"] != row["status"]):
            raise ValueError("Posthoc construction diagnostics disagree with the raw completed outcome")
        stages = diagnostic["stages"]
        if tuple(int(stage["candidate_size"]) for stage in stages) != K_VALUES[:K_VALUES.index(final_k) + 1]:
            raise ValueError("Posthoc stage sequence differs from the completed construction")
        self.totals[key] += 1
        final = self.final[(*key, final_k)]
        final["n_at_final_K"] += 1
        final["n_reliable"] += int(outcome["reliable_success"])
        final["n_contaminated"] += int(outcome["status"] == "gt_contaminated")
        final["n_insufficient"] += int(outcome["status"] == "gt_unavailable")
        for stage in stages:
            stats = self.admission[(*key, int(stage["candidate_size"]))]
            stats["n_visited"] += 1
            for kind in ("candidate", "responder", "admitted"):
                counts = stage[f"{kind}_counts"]
                if counts is not None:
                    for name in COUNT_NAMES:
                        stats["counts"][kind][name] += int(counts[name])
            stats["n_admission_defined"] += int(stage["admitted_counts"] is not None)
            for name in CONDITION_NAMES:
                value = stage[name]
                if value is not None:
                    stats["conditions"][name]["n_defined"] += 1
                    stats["conditions"][name]["n_true"] += int(bool(value))
            if stage["undetectable_corrupt_count"] is not None:
                stats["n_undetectable_defined"] += 1
                stats["undetectable_sum"] += int(stage["undetectable_corrupt_count"])
            if stage["detectable_corrupt_count"] is not None:
                stats["n_detectable_defined"] += 1
                stats["detectable_sum"] += int(stage["detectable_corrupt_count"])
                stats["detectable_admitted_sum"] += int(stage["detectable_corrupt_admitted_count"])

    def finish(self, *, require_complete: bool = True) -> tuple[list[dict], list[dict]]:
        expected = {(method, q, p) for method in METHODS for q in Q_VALUES for p in FAILURE_VALUES}
        if require_complete and (set(self.totals) != expected
                                 or any(self.totals[key] != REPETITIONS for key in expected)):
            raise ValueError("Posthoc summaries require all 1000 completed trials at every declared point")
        final_rows, admission_rows = [], []
        for method in METHODS:
            for q in Q_VALUES:
                for p in FAILURE_VALUES:
                    key = (method, q, p)
                    total = self.totals.get(key, 0)
                    for k in K_VALUES:
                        common = {"method": method, "q_C": q, "p_fail": p, "K": k, "n_total": total}
                        final = self.final.get((*key, k), {
                            "n_at_final_K": 0, "n_reliable": 0, "n_contaminated": 0, "n_insufficient": 0,
                        })
                        final_rows.append({
                            **common, **final,
                            "final_K_fraction": final["n_at_final_K"] / total if total else None,
                            "unconditional_reliable_rate": final["n_reliable"] / total if total else None,
                            "unconditional_contamination_rate": final["n_contaminated"] / total if total else None,
                            "unconditional_insufficient_rate": final["n_insufficient"] / total if total else None,
                        })
                        stats = self.admission.get((*key, k), _stage_stats())
                        visited = stats["n_visited"]
                        entry = {
                            **common, "n_visited": visited,
                            "visited_fraction": visited / total if total else None,
                            "n_admission_defined": stats["n_admission_defined"],
                            "n_undetectable_defined": stats["n_undetectable_defined"],
                            "n_detectable_defined": stats["n_detectable_defined"],
                            "mean_undetectable_corrupt_count": (
                                stats["undetectable_sum"] / stats["n_undetectable_defined"]
                                if stats["n_undetectable_defined"] else None),
                            "mean_detectable_corrupt_count": (
                                stats["detectable_sum"] / stats["n_detectable_defined"]
                                if stats["n_detectable_defined"] else None),
                            "mean_detectable_corrupt_admitted_count": (
                                stats["detectable_admitted_sum"] / stats["n_detectable_defined"]
                                if stats["n_detectable_defined"] else None),
                        }
                        for kind in ("candidate", "responder", "admitted"):
                            denominator = stats["n_admission_defined"] if kind == "admitted" else visited
                            for name in COUNT_NAMES:
                                entry[f"mean_{kind}_{name}"] = (
                                    stats["counts"][kind][name] / denominator if denominator else None)
                        for name in CONDITION_NAMES:
                            condition = stats["conditions"][name]
                            entry[f"n_defined_{name}"] = condition["n_defined"]
                            entry[f"n_true_{name}"] = condition["n_true"]
                            entry[f"{name}_rate"] = (
                                condition["n_true"] / condition["n_defined"] if condition["n_defined"] else None)
                        admission_rows.append(entry)
        return final_rows, admission_rows
