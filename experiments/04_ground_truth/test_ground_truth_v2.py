"""Focused checks for the revised GT rule; written for user-run pytest only.

These checks use synthetic descriptors, read an existing v1 manifest without
modifying it, and never invoke the formal experiment or produce paper figures.
They assert admission, selection, compatibility, and failure accounting rather
than a monotonic reliability curve, which the method does not guarantee.
"""
from __future__ import annotations

from dataclasses import fields
import importlib
import inspect
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pytest

from experiments.common.config import DetectionConfig
from experiments.common.dataset_adapter import AlgorithmInput
from experiments.common.similarity import cosine_pairwise

kernel = importlib.import_module("experiments.01_detection.proposed_detection")
config = importlib.import_module("experiments.04_ground_truth.config_ground_truth")
methods = importlib.import_module("experiments.04_ground_truth.ground_truth_methods")
diagnostics = importlib.import_module("experiments.04_ground_truth.ground_truth_diagnostics")


def _observations(vectors) -> AlgorithmInput:
    x = np.asarray(vectors, dtype=np.float32)
    return AlgorithmInput(
        X=x, model_ids=np.arange(len(x), dtype=np.int64),
        history=np.full(len(x), config.H0),
        reference_indices=np.arange(len(x), dtype=np.int64),
        evaluation_indices=np.empty(0, dtype=np.int64),
    )


def _one_cluster() -> np.ndarray:
    return np.repeat(np.eye(config.D, dtype=np.float32)[0:1], 100, axis=0)


def _expanding_clusters() -> np.ndarray:
    basis = np.eye(config.D, dtype=np.float32)
    # Neither 25-member cluster reaches theta_50=34. The added 25 members
    # bring the first cluster to exactly theta_75=50.
    return np.concatenate((
        np.repeat(basis[0:1], 25, axis=0),
        np.repeat(basis[1:2], 25, axis=0),
        np.repeat(basis[0:1], 50, axis=0),
    ))


@pytest.mark.parametrize("k,old_threshold,new_threshold", [(50, 26, 34), (75, 38, 50), (100, 51, 67)])
def test_legacy_defaults_and_revised_quorum_are_distinct(k, old_threshold, new_threshold):
    data = _observations(_one_cluster())
    cache = cosine_pairwise(data.X)
    legacy = kernel.build_ground_truth(data, DetectionConfig(K=k), pairwise_cache=cache,
                                       allow_expansion=False)
    revised = kernel.build_ground_truth(
        data, DetectionConfig(K=k), pairwise_cache=cache, allow_expansion=False,
        support_quorum=(2, 3), selection_policy="history_consensus",
    )
    assert legacy.threshold == kernel.majority_threshold(k) == old_threshold
    assert revised.threshold == kernel.majority_threshold(k, (2, 3)) == new_threshold
    assert legacy.success and revised.success
    assert legacy.selection_evidence is None
    assert np.array_equal(legacy.selected_indices, np.arange(25))
    assert np.array_equal(revised.selected_indices, legacy.selected_indices)


@pytest.mark.parametrize("k,threshold", [(50, 34), (75, 50), (100, 67)])
def test_revised_quorum_keeps_failed_slots_and_counts_self_once(k, threshold):
    data = _observations(_one_cluster())
    cache = cosine_pairwise(data.X)
    for responding, succeeds in ((threshold - 1, False), (threshold, True)):
        available = np.arange(100) < responding
        result = kernel.build_ground_truth(
            data, DetectionConfig(K=k), responder_available=available,
            pairwise_cache=cache, allow_expansion=False,
            support_quorum=(2, 3), selection_policy="history_consensus",
        )
        assert result.success is succeeds
        assert result.threshold == threshold and len(result.candidate_indices) == k
        assert len(result.expansion_trace) == 1
        assert np.all(result.support_counts[:responding] == responding - 1)
        assert np.all(result.support_counts[responding:] == 0)
        assert np.all(result.selection_evidence[:responding] == 1.0)
        assert np.all(result.selection_evidence[responding:] == -1.0)
        assert not result.selection_evidence.flags.writeable
        assert result.pairwise_comparison_count == responding * (responding - 1)
        assert len(result.selected_indices) == (25 if succeeds else 0)
        assert np.all(available[result.selected_indices])


def test_consensus_ranking_respects_all_four_keys_and_input_permutation():
    candidates = np.array([5, 3, 1, 4, 0, 2], dtype=np.int64)
    history = np.full(6, 0.1)
    history[0] = 0.05  # History remains primary despite a lower support count.
    model_ids = np.array([20, 13, 12, 11, 14, 15], dtype=np.int64)
    full_supports = np.array([1, 4, 4, 4, 4, 3], dtype=np.int64)
    full_means = np.array([0.997, 0.999, 0.999, 0.998, 0.997, 1.0])
    expected = np.array([0, 2, 1, 3, 4, 5], dtype=np.int64)
    for rows in (candidates, candidates[::-1]):
        ranked = kernel.rank_ground_truth_candidates(
            rows, history, model_ids, full_supports[rows], full_means[rows],
            policy="history_consensus",
        )
        assert np.array_equal(ranked, expected)
    # Legacy tie breaking still ignores both evidence vectors.
    old_ranked = kernel.rank_ground_truth_candidates(
        candidates, history, model_ids, None, None, policy="history",
    )
    assert np.array_equal(old_ranked, [0, 3, 2, 1, 4, 5])


def test_support_mean_uses_only_responsive_legal_peers_and_excludes_self():
    vectors = _one_cluster()
    # The first 49 rows retain identical legal descriptors; all later rows are
    # orthogonal. One otherwise identical peer is unavailable.
    vectors[49:] = np.eye(config.D, dtype=np.float32)[1]
    data = _observations(vectors)
    available = np.ones(100, dtype=bool)
    available[48] = False
    result = kernel.build_ground_truth(
        data, DetectionConfig(), responder_available=available,
        pairwise_cache=cosine_pairwise(data.X), allow_expansion=False,
        support_quorum=(2, 3), selection_policy="history_consensus",
    )
    assert result.success
    assert np.all(result.support_counts[:48] == 47)
    assert np.all(result.selection_evidence[:48] == 1.0)
    assert result.support_counts[48] == result.support_counts[49] == 0
    assert result.selection_evidence[48] == result.selection_evidence[49] == -1.0
    assert np.array_equal(result.selected_indices, np.arange(25))


def test_raca_and_fixed_k_use_the_same_admission_and_selection_rule():
    data = _observations(_expanding_clusters())
    available = np.ones(100, dtype=bool)
    cache = cosine_pairwise(data.X)
    full = methods.run_method("RACA-HIM", data, available, 2026, pairwise_cache=cache)
    fixed = methods.run_method("Fixed-K Majority", data, available, 2026, pairwise_cache=cache)
    assert full.success and full.final_K == 75 and full.expansion_count == 1
    assert not fixed.success and fixed.final_K == 50 and fixed.expansion_count == 0
    assert [(step["candidate_size"], step["threshold"], step["passing_count"])
            for step in full.trace] == [(50, 34, 0), (75, 50, 50)]
    for key in ("candidate_size", "threshold", "passing_count", "responsive_count",
                "pairwise_comparison_count", "selection_policy"):
        assert full.trace[0][key] == fixed.trace[0][key]
    assert np.array_equal(full.selected_indices, np.arange(25))
    assert full.trace[-1]["selected_peer_support_counts"] == [49] * 25
    assert full.trace[-1]["selected_mean_support_similarity"] == [1.0] * 25
    # At a successful initial step, expansion cannot affect either method.
    initial = _observations(_one_cluster())
    same = [methods.run_method(name, initial, available, 2026)
            for name in ("RACA-HIM", "Fixed-K Majority")]
    assert all(result.success and result.final_K == 50 for result in same)
    assert same[0].trace == same[1].trace
    assert np.array_equal(same[0].selected_indices, same[1].selected_indices)


@pytest.fixture(scope="module")
def saved_v1_manifest():
    current = config.RESULT_ROOT / "logs" / "run_manifest.json"
    archived = sorted((PROJECT_ROOT / "results" / "archive").glob(
        "04_ground_truth_*/logs/run_manifest.json"))
    for path in (current, *archived):
        if path.is_file():
            payload = path.read_bytes()
            manifest = json.loads(payload.decode("utf-8-sig"))
            if manifest.get("algorithm_version") == config.LEGACY_IMPLEMENTATION_VERSION:
                return path, payload, manifest
    pytest.skip("No saved v1 result manifest is present in this checkout")


def test_legacy_protocol_identity_matches_saved_manifest_without_rewriting(saved_v1_manifest):
    path, original, manifest = saved_v1_manifest
    legacy = config.protocol_metadata(config.LEGACY_IMPLEMENTATION_VERSION)
    assert legacy == manifest["protocol"]
    assert path.read_bytes() == original
    revised = config.protocol_metadata()
    assert revised != legacy
    assert revised["gt_quorum_numerator"] == 2
    assert revised["gt_quorum_denominator"] == 3
    assert revised["gt_selection_policy"] == "history_consensus"
    assert "gt_quorum_numerator" not in legacy


def test_algorithm_boundary_contains_no_offline_truth_or_prototype():
    assert {field.name for field in fields(AlgorithmInput)} == {
        "X", "model_ids", "history", "reference_indices", "evaluation_indices",
    }
    assert set(inspect.signature(methods.run_method).parameters) == {
        "method", "algorithm_input", "available_mask", "random_seed", "pairwise_cache",
    }
    assert set(inspect.signature(kernel.build_ground_truth).parameters) == {
        "data", "config", "responder_available", "pairwise_cache", "allow_expansion",
        "support_quorum", "selection_policy",
    }
    data = _observations(_one_cluster())
    available = np.ones(100, dtype=bool)
    before = {field.name: getattr(data, field.name).copy() for field in fields(AlgorithmInput)}
    result = methods.run_method("RACA-HIM", data, available, 2026)
    assert result.success
    for field in fields(AlgorithmInput):
        assert np.array_equal(getattr(data, field.name), before[field.name])
        assert not getattr(data, field.name).flags.writeable
    with pytest.raises(TypeError):
        methods.run_method("RACA-HIM", {"observations": data, "labels": np.zeros(100)},
                           available, 2026)


@pytest.fixture(scope="module")
def completed_cases():
    cases = []
    for name in ("reliable", "contaminated", "insufficient"):
        data = _observations(_expanding_clusters() if name == "contaminated" else _one_cluster())
        subtype = np.zeros(100, dtype=np.int8)
        available = np.ones(100, dtype=bool)
        if name == "contaminated":
            subtype[25:50] = 2
            subtype[0] = 3  # An overlapping mild corruption is unobservable to admission.
        elif name == "insufficient":
            available[:40] = False
        truth = (subtype >= 2).astype(np.int8)
        cache = cosine_pairwise(data.X)
        result = methods.run_method("RACA-HIM", data, available, 2026, pairwise_cache=cache)
        posthoc = diagnostics.diagnose_construction(result, truth, subtype, available, cache)
        cases.append((name, data, available, cache, result, posthoc))
    return cases


def test_posthoc_admission_matches_core_evidence_and_is_json_safe(completed_cases):
    for name, data, available, cache, result, posthoc in completed_cases:
        core = kernel.build_ground_truth(
            data, DetectionConfig(), responder_available=available, pairwise_cache=cache,
            support_quorum=(2, 3), selection_policy="history_consensus",
        )
        admitted = available[core.candidate_indices] & (core.support_counts + 1 >= core.threshold)
        final = posthoc["stages"][-1]
        assert result.final_K == len(core.candidate_indices)
        assert final["passing_count"] == int(np.count_nonzero(admitted))
        assert final["admitted_counts"]["total"] == final["passing_count"]
        assert [stage["passing_count"] for stage in posthoc["stages"]] == [
            step.passing_count for step in core.expansion_trace]
        assert np.array_equal(core.selected_indices, result.selected_indices)
        saved = json.loads(json.dumps(posthoc, allow_nan=False))
        assert saved["outcome"]["formed"] == int(result.success)
        if name == "reliable":
            assert result.final_K == 50
            assert final["theorem_premises_satisfied"] is True
            assert final["full_purity_premises_satisfied"] is True
            assert final["normal_corrupt_separated"] is None
            assert saved["outcome"]["reliable_success"] == 1
        elif name == "contaminated":
            assert result.final_K == 75 and result.success
            assert final["admitted_counts"]["corrupt"] == 1
            assert final["undetectable_corrupt_count"] == 1
            assert saved["outcome"]["gt_corrupt_count"] == 1
            assert saved["outcome"]["status"] == "gt_contaminated"
        else:
            assert result.final_K == 100 and not result.success
            assert final["normal_responder_strict_majority"] is True
            assert final["normal_responder_quorum_met"] is False
            assert saved["outcome"]["status"] == "gt_unavailable"


def test_detectable_subset_guarantee_does_not_imply_full_label_purity():
    data = _observations(_one_cluster())
    available = np.ones(100, dtype=bool)
    subtype = np.zeros(100, dtype=np.int8)
    subtype[0] = 3  # Its descriptor is indistinguishable from the normal clique.
    truth = (subtype >= 2).astype(np.int8)
    cache = cosine_pairwise(data.X)
    result = methods.run_method("RACA-HIM", data, available, 2026, pairwise_cache=cache)
    posthoc = diagnostics.diagnose_construction(result, truth, subtype, available, cache)
    stage = posthoc["stages"][-1]
    assert stage["normal_responder_quorum_met"] is True
    assert stage["normal_normal_legal"] is True
    assert stage["theorem_premises_satisfied"] is True
    assert stage["normal_corrupt_separated"] is False
    assert stage["full_purity_premises_satisfied"] is False
    assert stage["detectable_corrupt_count"] == stage["detectable_corrupt_admitted_count"] == 0
    assert stage["undetectable_corrupt_count"] == 1
    assert posthoc["outcome"]["status"] == "gt_contaminated"


def test_final_k_decomposition_retains_all_1000_trials_including_failures(completed_cases):
    accumulator = diagnostics.DiagnosticAccumulator()
    for index, seed in enumerate(config.SEEDS):
        _, _, _, _, result, posthoc = completed_cases[0 if index < 250 else 1 if index < 500 else 2]
        outcome = posthoc["outcome"]
        row = {
            "method": result.method, "q_C_requested": 0.30, "p_fail_requested": 0.10,
            "seed": seed, "final_K": result.final_K, "gt_success": int(result.success),
            "reliable_success": outcome["reliable_success"],
            "gt_corrupt_count": outcome["gt_corrupt_count"], "status": outcome["status"],
        }
        accumulator.add(row, posthoc)
    final_rows, admission_rows = accumulator.finish(require_complete=False)
    assert len(final_rows) == len(admission_rows) == len(config.METHODS) * len(config.Q_VALUES) * len(
        config.FAILURE_VALUES) * len(config.K_VALUES)

    def point(rows):
        return [row for row in rows if row["method"] == "RACA-HIM"
                and row["q_C"] == 0.30 and row["p_fail"] == 0.10]

    final = point(final_rows)
    assert [row["K"] for row in final] == [50, 75, 100]
    assert all(row["n_total"] == 1000 for row in final)
    assert [row["n_at_final_K"] for row in final] == [250, 250, 500]
    assert sum(row["final_K_fraction"] for row in final) == 1.0
    assert sum(row["unconditional_reliable_rate"] for row in final) == 0.25
    assert sum(row["unconditional_contamination_rate"] for row in final) == 0.25
    assert sum(row["unconditional_insufficient_rate"] for row in final) == 0.50
    assert all(row["n_at_final_K"] == row["n_reliable"] + row["n_contaminated"] + row["n_insufficient"]
               for row in final)
    assert [row["n_visited"] for row in point(admission_rows)] == [1000, 750, 500]
    assert [row["visited_fraction"] for row in point(admission_rows)] == [1.0, 0.75, 0.5]
    json.dumps({"final": final_rows, "admission": admission_rows}, allow_nan=False)
    with pytest.raises(ValueError):
        accumulator.finish()  # One complete point cannot stand in for the formal grid.
