"""Group 02 scientific-boundary tests; authored without executing the experiment.

Run explicitly with ``python -m pytest experiments/02_overhead/test_overhead.py``.
Small synthetic workloads test protocol semantics and do not generate figures.
"""
from __future__ import annotations

import ast
from dataclasses import replace
import hashlib
import importlib
import math
from pathlib import Path
import sys

sys.dont_write_bytecode = True
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pytest

from dataset.load_dataset import load_group, resource_costs, resource_profiles
from experiments.common.dataset_adapter import AlgorithmInput
from experiments.common.method_names import RACA_HIM
from experiments.common import similarity

cfg = importlib.import_module("experiments.02_overhead.config_overhead")
baseline = importlib.import_module("experiments.02_overhead.overhead_baselines")
ground_truth = importlib.import_module("experiments.01_detection.proposed_detection")
runner = importlib.import_module("experiments.02_overhead.overhead_experiment")
simulator = importlib.import_module("experiments.02_overhead.overhead_simulator")
counter_module = importlib.import_module("experiments.02_overhead.communication_counter")


def _descriptors(count: int = 12, dim: int = 10) -> np.ndarray:
    """Repeated raw-content anchor plus distinct legitimate-looking targets."""
    values = np.zeros((count, dim), dtype=np.float32)
    values[:, 0] = 1.0
    for index in range(4, count):
        values[index, 1] = np.float32((index - 3) * 0.0001)
    return values


def _raw_sha(row: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(row, dtype="<f4").tobytes()).hexdigest()


def _reference_case():
    x = _descriptors()
    model_ids = np.arange(1_000_000, 1_000_000 + len(x), dtype=np.int64)
    verifiers = np.arange(4, dtype=np.int64)
    targets = np.arange(4, len(x), dtype=np.int64)
    reference = baseline.prepare_reference(x, verifiers, model_ids,
                                           np.full(len(x), 0.10))
    return x, model_ids, verifiers, targets, reference


def _gt_input(x: np.ndarray) -> AlgorithmInput:
    n = len(x)
    return AlgorithmInput(
        X=x, model_ids=np.arange(1_000_000, 1_000_000 + n, dtype=np.int64),
        history=np.full(n, 0.10), reference_indices=np.arange(n, dtype=np.int64),
        evaluation_indices=np.empty(0, dtype=np.int64),
    )


def test_formal_grid_and_budget_use_n_minus_m():
    config = cfg.OverheadConfig()
    assert cfg.N_VALUES == tuple(range(1000, 5001, 500))
    assert cfg.D_PANEL_VALUES == (10, 20, 30, 40, 50)
    assert cfg.SEEDS == (2026,)
    assert cfg.EVALUATION_MODES == ('fixed_workload',)
    assert len(cfg.configurations()) == 9 * 5
    assert len(set(cfg.configurations())) == 9 * 5
    assert config.T_round_max_ms == 800.0
    assert config.budget(1000) == 98
    assert config.budget(5000) == 498
    with pytest.raises(ValueError):
        config.budget(config.M)


@pytest.mark.parametrize("change", [
    {"M": 51}, {"p_fail": 0.01}, {"round_id": 2},
    {"T_round_max_ms": 0.0}, {"beta_B": 0.0},
    {"iedi_tag_modulus": 1}, {"ctcv_collusion_bound": 25},
])
def test_invalid_protocol_configuration_is_rejected(change):
    with pytest.raises(ValueError):
        cfg.OverheadConfig(**change)


def test_resource_sigma_zero_exact_cost_and_float32_payload():
    profiles = resource_profiles(seed=2026, n=12, rounds=1)
    costs = resource_costs(np.array([4, 5]), np.array([0, 1]), d=30,
                           sigma=0.0, round_id=1, profiles=profiles)
    assert np.allclose(costs["compute_ms"], 4.0)
    assert np.allclose(costs["network_ms"], 10.0)
    assert np.allclose(costs["transfer_ms"], 120 * 8 / 50e6 * 1000)
    assert np.allclose(costs["total_ms"], 14.0192)
    assert _descriptors(dim=30)[0].nbytes == 120


def test_resource_lookup_is_order_independent_and_requires_row_indices():
    profiles = resource_profiles(seed=2027, n=12, rounds=1)
    targets, verifiers = np.array([4, 5, 6]), np.array([0, 1, 2])
    first = resource_costs(targets, verifiers, d=10, sigma=0.5, profiles=profiles)
    perm = np.array([2, 0, 1])
    shuffled = resource_costs(targets[perm], verifiers[perm], d=10,
                              sigma=0.5, profiles=profiles)
    for key in first:
        assert np.array_equal(first[key][perm], shuffled[key])
    with pytest.raises(IndexError):
        resource_costs(np.array([1_000_004]), np.array([1_000_000]),
                       d=10, profiles=profiles)


def test_gt_expansion_recomputes_actual_majority_threshold():
    # The initial four contain two members of each cluster; no member has 3
    # support including self. The expanded six contain four A and two B.
    x = np.zeros((6, 10), dtype=np.float32)
    x[[0, 1, 4, 5], 0] = 1.0
    x[[2, 3], 1] = 1.0
    result = ground_truth.build_ground_truth(
        _gt_input(x), cfg.OverheadConfig(K=4, M=3, expansion_step=2, dim=10))
    assert result.success
    assert [(step.candidate_size, step.threshold, step.passing_count)
            for step in result.expansion_trace] == [(4, 3, 0), (6, 4, 4)]
    assert result.selected_indices.tolist() == [0, 1, 4]
    assert result.support_counts.tolist() == [3, 3, 1, 1, 3, 3]


def test_gt_unavailable_keeps_failure_instead_of_inventing_reference():
    x = np.zeros((4, 10), dtype=np.float32)
    x[np.arange(4), np.arange(4)] = 1.0
    result = ground_truth.build_ground_truth(
        _gt_input(x), cfg.OverheadConfig(K=4, M=2, expansion_step=2, dim=10))
    assert not result.success
    assert len(result.selected_indices) == 0
    assert result.expansion_trace[-1].candidate_size == 4
    assert result.reason


def test_reference_uses_exact_raw_hash_frequency_and_observable_ties():
    x = _descriptors(8)
    # Two equally frequent hashes: history then temporary identity break ties.
    x[2:4] = x[4]
    ids = np.array([1004, 1003, 1002, 1001, 1005, 1006, 1007, 1008])
    history = np.full(8, 0.1)
    history[2] = 0.0
    reference = baseline.prepare_reference(x, np.arange(4), ids, history)
    assert reference.anchor_index == 2
    assert reference.anchor_id == 1002
    assert reference.sha256 == _raw_sha(x[2])
    assert reference.payload == np.ascontiguousarray(x[2], dtype="<f4").tobytes()
    with pytest.raises(ValueError):
        baseline.prepare_reference(x.astype(np.float64), np.arange(4), ids, history)


@pytest.mark.parametrize("name", ["choose_dvap", "choose_iedi", "choose_ctcv"])
def test_native_assignment_has_unique_targets_and_never_exceeds_budget(name):
    targets = np.arange(4, 12, dtype=np.int64)
    verifiers = np.arange(4, dtype=np.int64)
    costs = np.arange(1, 33, dtype=np.float64).reshape(8, 4)
    plan = getattr(baseline, name)(targets, verifiers, costs, 3, 80.0)
    assert len(plan.assignments) <= 3
    seen = [int(target) for target, _ in plan.assignments]
    assert len(seen) == len(set(seen))
    assert set(seen).issubset(set(targets))
    assert all(int(verifier) in verifiers and target != verifier
               for target, verifier in plan.assignments)
    load = {int(v): 0.0 for v in verifiers}
    for target, verifier in plan.assignments:
        load[int(verifier)] += costs[int(np.flatnonzero(targets == target)[0]),
                                    int(np.flatnonzero(verifiers == verifier)[0])]
    assert max(load.values()) <= 80.0


@pytest.mark.parametrize("name", ["choose_dvap", "choose_iedi", "choose_ctcv"])
def test_fixed_workload_keeps_every_supplied_target(name):
    targets = np.array([7, 4, 6, 5], dtype=np.int64)
    verifiers = np.arange(4, dtype=np.int64)
    costs = np.array([[1., 6., 9., 8.], [8., 1., 6., 9.],
                      [9., 8., 1., 6.], [6., 9., 8., 1.]])
    plan = getattr(baseline, name)(targets, verifiers, costs, 4, 800.0,
                                  fixed_workload=True)
    assert sorted(int(target) for target, _ in plan.assignments) == sorted(targets)
    assert len(plan.assignments) == len(targets)


def test_baseline_verification_executes_without_any_similarity(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Baselines may not use a similarity threshold")
    for name in ("cosine_pairwise", "cosine_aligned", "within_interval"):
        monkeypatch.setattr(similarity, name, forbidden)
    x, ids, verifiers, targets, reference = _reference_case()
    config = cfg.OverheadConfig(dim=10)
    assignments = tuple((int(target), int(verifiers[0])) for target in targets)
    for name in ("verify_dvap", "verify_iedi", "verify_ctcv"):
        trace = getattr(baseline, name)(
            x, targets, reference, config, assignments=assignments,
            model_ids=ids, verifiers=verifiers, seed=2026)
        assert trace.predictions.shape == (len(targets),)


def test_baseline_module_has_no_similarity_or_oracle_dependencies():
    source = (PROJECT_ROOT / "experiments" / "02_overhead" /
              "overhead_baselines.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    forbidden = {"binary_label", "subtype_label", "prototype", "sample_id",
                 "cosine_pairwise", "cosine_aligned", "within_interval",
                 "build_ground_truth", "detect_proposed"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            name = node.func.id if isinstance(node.func, ast.Name) else (
                node.func.attr if isinstance(node.func, ast.Attribute) else "")
            assert name not in forbidden
        if isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Constant):
            assert node.slice.value not in forbidden


def test_iedi_predictions_are_actual_sha256_modulo_tags():
    x, ids, verifiers, targets, reference = _reference_case()
    config = cfg.OverheadConfig(dim=10)
    trace = baseline.verify_iedi(x, targets, reference, config,
                                 model_ids=ids, verifiers=verifiers)
    anchor_tag = int(reference.sha256, 16) % 7919
    expected = np.array([int(int(_raw_sha(x[target]), 16) % 7919 != anchor_tag)
                         for target in targets], dtype=np.int8)
    assert np.array_equal(trace.predictions, expected)
    assert trace.events and trace.operations


def test_ctcv_vector_collision_verdict_is_not_overridden_by_full_hash():
    x, ids, verifiers, targets, reference = _reference_case()
    # ell=1 maps every actual and reference tag to the same proof position.
    # The vector protocol must retain that collision even when bytes differ.
    config = cfg.OverheadConfig(dim=10, ctcv_proof_vector_length=1)
    assert all(_raw_sha(x[target]) != reference.sha256 for target in targets)
    trace = baseline.verify_ctcv(x, targets, reference, config,
                                 model_ids=ids, verifiers=verifiers, seed=2026)
    assert np.array_equal(trace.predictions, np.zeros(len(targets), dtype=np.int8))
    assert trace.events and trace.operations


def _summary_rows(values, *, failed_seed=None, different_workload_seed=None):
    rows = []
    for offset, value in enumerate(values):
        seed = 2026 + offset
        for method in cfg.METHODS:
            rows.append({
                "N": 1000, "d": 10, "seed": seed, "cr": 3.0, "H": 0.5,
                "method": method, "evaluation_mode": "fixed_workload", "B": 98,
                "selected_count": 98, "attempted_count": 98,
                "completed_count": 97 if seed == failed_seed and method == cfg.METHODS[-1] else 98,
                "deadline_violations": 1 if seed == failed_seed and method == cfg.METHODS[-1] else 0,
                "status": "success", "total_time_ms": float(value),
                "total_bytes": int(value * 1000),
                "fixed_workload_sha256": "other-targets" if (
                    seed == different_workload_seed and method == cfg.METHODS[-1]) else f"targets-{seed}",
            })
    return rows


def _small_summary_grid(monkeypatch, count):
    monkeypatch.setattr(runner, "N_VALUES", (1000,))
    monkeypatch.setattr(runner, "D_PANEL_VALUES", (10,))
    monkeypatch.setattr(runner, "SEEDS", tuple(range(2026, 2026 + count)))
    monkeypatch.setattr(runner, "configurations", lambda: tuple(
        (1000, 10, seed) for seed in range(2026, 2026 + count)))


def test_ci_uses_independent_seed_values_and_sample_standard_deviation(monkeypatch):
    from scipy.stats import t
    _small_summary_grid(monkeypatch, 3)
    summaries, diagnostics = runner.summarize_runs(_summary_rows([10, 20, 30]), "same-run")
    row = next(item for item in summaries["time"]
               if item["method"] == RACA_HIM and item["evaluation_mode"] == "fixed_workload")
    assert row["n_valid"] == 3
    assert row["mean"] == 20.0
    assert row["std"] == 10.0
    half = t.ppf(.975, 2) * 10.0 / math.sqrt(3)
    assert row["ci95_lower"] == pytest.approx(20 - half)
    assert row["ci95_upper"] == pytest.approx(20 + half)
    assert diagnostics["fixed_workload_eligible_paired_seeds"] == 3


def test_ci_excludes_whole_incomplete_pair_and_records_failure_counts(monkeypatch):
    _small_summary_grid(monkeypatch, 3)
    rows = _summary_rows([10, 20, 999], failed_seed=2028)
    summaries, diagnostics = runner.summarize_runs(rows, "same-run")
    fixed = [row for row in summaries["time"] if row["evaluation_mode"] == "fixed_workload"]
    assert all(row["n_valid"] == 2 and row["mean"] == 15.0 for row in fixed)
    assert all(row["n_failed"] == 1 and row["run_count"] == 3 for row in fixed)
    failed = next(row for row in fixed if row["method"] == cfg.METHODS[-1])
    assert failed["n_incomplete_workload"] == 1
    assert failed["deadline_violations_total"] == 1
    assert diagnostics["fixed_workload_excluded_paired_seeds"] == {"incomplete_or_failed_method": 1}


def test_fixed_summary_rejects_different_target_identity_even_equal_counts(monkeypatch):
    _small_summary_grid(monkeypatch, 2)
    summaries, diagnostics = runner.summarize_runs(
        _summary_rows([10, 999], different_workload_seed=2027), "same-run")
    fixed = [row for row in summaries["time"] if row["evaluation_mode"] == "fixed_workload"]
    assert all(row["n_valid"] == 1 and row["mean"] == 10.0 for row in fixed)
    assert all(math.isnan(row["std"]) and math.isnan(row["ci95_lower"]) for row in fixed)
    assert diagnostics["fixed_workload_excluded_paired_seeds"] == {
        "missing_or_different_fixed_target_fingerprint": 1}


def _checkpoint_payload(fingerprint="same-run"):
    return {
        "run_fingerprint": fingerprint, "complete": True,
        "configuration": {"N": 1000, "d": 10, "seed": 2026},
        "results": [{"row": {"method": method, "evaluation_mode": mode,
                             "implementation_version":cfg.OVERHEAD_IMPLEMENTATION_VERSION,
                             "run_fingerprint":fingerprint,"N":1000,"d":10,"seed":2026,
                             "configuration_id":"N1000_d10_seed2026"},
                     "messages":[],"runtime":[],"tasks":[],"diagnostics":{}}
                    for method in cfg.METHODS for mode in cfg.EVALUATION_MODES],
    }


def test_atomic_checkpoint_round_trip_and_exact_resume_identity(tmp_path):
    path = tmp_path / "N1000_d10_seed2026.json.gz"
    payload = _checkpoint_payload()
    runner._atomic_json(path, payload, compressed=True)
    assert runner._checkpoint(path, "same-run", 1000, 10, 2026) == payload
    assert not path.with_name(path.name + ".tmp").exists()
    for fingerprint, n, d, seed in (("other-source-config-data", 1000, 10, 2026),
                                    ("same-run", 1500, 10, 2026),
                                    ("same-run", 1000, 20, 2026),
                                    ("same-run", 1000, 10, 2027)):
        assert runner._checkpoint(path, fingerprint, n, d, seed) is None


@pytest.mark.parametrize("mutation", ["incomplete", "missing", "duplicate", "corrupt"])
def test_checkpoint_never_resumes_partial_or_corrupt_method_results(tmp_path, mutation):
    path = tmp_path / "point.json.gz"
    payload = _checkpoint_payload()
    if mutation == "incomplete":
        payload["complete"] = False
    elif mutation == "missing":
        payload["results"].pop()
    elif mutation == "duplicate":
        payload["results"][-1] = payload["results"][0]
    if mutation == "corrupt":
        path.write_bytes(b"not a gzip checkpoint")
    else:
        runner._atomic_json(path, payload, compressed=True)
    assert runner._checkpoint(path, "same-run", 1000, 10, 2026) is None


def test_run_identity_changes_on_config_data_source_or_runtime_environment():
    config = cfg.OverheadConfig()
    snapshot = {"banks/example.npz": "original-file-sha"}
    environment = {"python": "test-runtime", "packages": {"numpy": "test-version"}}
    original = runner.run_identity(config, snapshot, "original-source", environment)
    assert runner.run_identity(config, snapshot, "original-source", environment) == original
    alternatives = (
        (replace(config, sigma=0.0), snapshot, "original-source", environment),
        (config, {"banks/example.npz": "different-file-sha"}, "original-source", environment),
        (config, snapshot, "different-source", environment),
        (config, snapshot, "original-source", {"python": "other-runtime"}),
    )
    assert all(runner.run_identity(*values) != original for values in alternatives)


def test_prepare_case_never_accesses_labels_prototype_or_source_ids():
    class XOnly(dict):
        def __getitem__(self, key):
            if key != "X":
                raise AssertionError(f"Forbidden algorithm input access: {key}")
            return super().__getitem__(key)
    x = _descriptors(60)
    config = cfg.OverheadConfig(K=4, M=2, expansion_step=2, dim=10)
    original = XOnly(X=x, binary_label=np.zeros(60), prototype=np.ones(10),
                     sample_id=np.arange(60))
    poisoned = XOnly(X=x, binary_label=np.ones(60), subtype_label=np.full(60, 3),
                     prototype=np.full(10, np.nan), sample_id=["do-not-read"] * 60)
    first = simulator.prepare_case(original, 60, 10, 2026, config)
    second = simulator.prepare_case(poisoned, 60, 10, 2026, config)
    for field in ("X", "model_ids", "history", "reference_indices", "evaluation_indices"):
        assert np.array_equal(getattr(first.data, field), getattr(second.data, field))
    for field in ("targets", "verifiers", "baseline_targets", "baseline_verifiers", "fixed_targets"):
        assert np.array_equal(getattr(first, field), getattr(second, field))
    assert first.workload_sha256 == second.workload_sha256
    assert np.array_equal(first.gt.selected_indices, second.gt.selected_indices)
    assert first.gt.expansion_trace == second.gt.expansion_trace
    assert len(first.data.reference_indices) == 60  # Full domain, rather than Group 01's 100 rows.
    assert len(first.fixed_targets) == config.budget(60)
    assert set(first.fixed_targets).isdisjoint(first.verifiers)
    assert set(first.fixed_targets).isdisjoint(first.baseline_verifiers)


def test_baseline_verifier_identities_are_independent_of_cosine_gt_content():
    config = cfg.OverheadConfig(K=4, M=2, expansion_step=2, dim=10)
    first = simulator.prepare_case({"X": _descriptors(60)}, 60, 10, 2026, config)
    changed = _descriptors(60)
    # Change the descriptors of the first neutral identities substantially.
    # Their availability as baseline references must remain an external fixture.
    changed[first.baseline_verifiers, 0] = 0.0
    changed[first.baseline_verifiers, 1] = 1.0
    second = simulator.prepare_case({"X": changed}, 60, 10, 2026, config)
    assert np.array_equal(first.data.model_ids, second.data.model_ids)
    assert np.array_equal(first.baseline_verifiers, second.baseline_verifiers)
    assert len(second.baseline_targets) == 60 - config.M
    assert set(second.baseline_targets).isdisjoint(second.baseline_verifiers)


def test_greedy_does_not_stop_at_negative_first_gain_and_uses_parallel_slack():
    assignments, diagnostics = simulator.joint_schedule(
        np.array([4, 5]), np.array([0, 1]), np.array([0.1, 0.1]),
        np.full((2, 2), 9.0), 2, 10.0, 1.0)
    assert len(assignments) == 2
    assert diagnostics["generated_count"] == diagnostics["chosen_prefix"] == 2
    assert diagnostics["marginal_gains"][0] < 0
    assert diagnostics["marginal_gains"][1] == pytest.approx(0.05)
    assert diagnostics["prefix_makespans_ms"] == [9.0, 9.0]
    assert len({verifier for _, verifier in assignments}) == 2


def test_greedy_generates_all_prefixes_but_returns_best_nonempty_objective():
    assignments, diagnostics = simulator.joint_schedule(
        np.array([4, 5, 6]), np.array([0]), np.array([1.0, 0.0, 0.0]),
        np.ones((3, 1)), 3, 10.0, 0.5)
    assert diagnostics["generated_count"] == 3
    assert diagnostics["chosen_prefix"] == 1
    assert assignments == [(4, 0)]
    objectives = diagnostics["prefix_objectives"]
    assert objectives[0] == max(objectives)
    assert objectives == pytest.approx([1 / 3 - .05, 1 / 3 - .10, 1 / 3 - .15])


def test_greedy_deadline_and_single_assignment_load_invariants():
    targets, verifiers = np.array([4, 5, 6]), np.array([0, 1])
    costs = np.array([[4., 6.], [6., 4.], [9., 9.]])
    assignments, diagnostics = simulator.joint_schedule(
        targets, verifiers, np.ones(3), costs, 3, 8.0, 0.5,
        fixed_workload=True)
    assert len(assignments) == 2
    assert len({target for target, _ in assignments}) == 2
    loads = {int(verifier): 0.0 for verifier in verifiers}
    for target, verifier in assignments:
        loads[verifier] += costs[int(np.flatnonzero(targets == target)[0]),
                                 int(np.flatnonzero(verifiers == verifier)[0])]
    assert max(loads.values()) == 4.0
    assert all(span <= 8.0 for span in diagnostics["prefix_makespans_ms"])
    empty, diagnostics = simulator.joint_schedule(
        targets, verifiers, np.ones(3), np.full((3, 2), 9.0), 3, 8.0)
    assert empty == []
    assert diagnostics["generated_count"] == 0


def test_communication_partitions_count_each_message_once():
    ledger = counter_module.CommunicationCounter()
    ledger.add_message(0, 4, "gt", "gt_request", control_bytes=64)
    ledger.add_message(4, 0, "gt", "gt_descriptor", payload_bytes=120)
    ledger.add_message(5, 1, "verification", "descriptor", payload_bytes=120,
                       control_bytes=8)
    ledger.add_message(1, -1, "verification", "result", control_bytes=32)
    ledger.validate()
    totals = ledger.totals()
    assert totals["gt_bytes"] == totals["verification_bytes"] == 120
    assert totals["control_bytes"] == 104
    assert totals["total_bytes"] == 344
    assert totals["total_bytes"] == sum(event["total_bytes"] for event in ledger.events)
    assert totals["message_count"] == 4
    assert totals["gt_messages"] == totals["verification_messages"] == 2
    assert totals["control_messages"] == 3
    with pytest.raises(ValueError):
        ledger.add_message(0, 1, "verification", "invalid", payload_bytes=-1)


def test_message_duration_scales_actual_bytes_without_double_descriptor_transfer():
    profiles = resource_profiles(seed=2026, n=6, rounds=1)
    config = cfg.OverheadConfig(dim=30, sigma=0.0)
    sim = simulator.EventSimulator(profiles, 30, config)
    transfer, propagation = sim.message_duration(4, 0, 120)
    assert transfer == pytest.approx(0.0192)
    assert propagation == 10.0
    proof_transfer, proof_propagation = sim.message_duration(4, 0, 32)
    assert proof_transfer == pytest.approx(0.00512)
    assert proof_propagation == 10.0
    end = sim.send({"sender": 4, "receiver": 0, "message_type": "descriptor",
                    "payload_bytes": 120}, "verification", 0.0)
    assert end == pytest.approx(10.0192)
    assert sim.counter.totals()["total_bytes"] == 120
    assert sim.counter.totals()["message_count"] == 1


def test_one_pair_extracts_numpy_singleton_values_as_python_scalars():
    profiles=resource_profiles(seed=2026,n=6,rounds=1)
    sim=simulator.EventSimulator(profiles,30,cfg.OverheadConfig(dim=30,sigma=0.0))
    costs=sim.pair(4,0)
    assert all(isinstance(value,float) for value in costs.values())
    assert costs['compute_ms']==4.0
    assert costs['total_ms']==pytest.approx(14.0192)
    assert sim.pair(4,0) is costs


def test_cpu_operations_are_serial_per_node_and_parallel_between_nodes():
    config = cfg.OverheadConfig(dim=30, sigma=0.0)
    sim = simulator.EventSimulator(resource_profiles(n=6, rounds=1), 30, config)
    first = sim.compute({"node": 0, "operation": "cosine", "comparisons": 1},
                        "verification", 0.0)
    same = sim.compute({"node": 0, "operation": "cosine", "comparisons": 1},
                       "verification", 0.0)
    parallel = sim.compute({"node": 1, "operation": "cosine", "comparisons": 1},
                           "verification", 0.0)
    assert (first, same, parallel) == (4.0, 8.0, 4.0)
    assert sim.runtime[1]["start_ms"] == first
    assert sim.runtime[2]["start_ms"] == 0.0


def test_gt_event_checks_and_messages_are_actual_directed_candidate_pairs():
    config = cfg.OverheadConfig(K=4, M=2, expansion_step=2, dim=30, sigma=0.0)
    case = simulator.prepare_case({"X": _descriptors(8, 30)}, 8, 30, 2026, config)
    sim = simulator.EventSimulator(case.profiles, 30, config)
    end = simulator._gt_preparation(case, sim)
    checks = [operation for operation in sim.runtime if operation["operation"] == "gt_cosine"]
    assert len(checks) == 4 * (4 - 1)
    messages = sim.counter.events
    assert sum(event["message_type"] == "gt_request" for event in messages) == 12
    assert sum(event["message_type"] == "gt_descriptor" for event in messages) == 12
    assert sum(event["message_type"] == "gt_support_report" for event in messages) == 4
    assert sum(event["message_type"] == "gt_membership" for event in messages) == 2
    assert sim.counter.totals()["gt_bytes"] == 12 * 120
    assert sim.counter.totals()["message_count"] == 30
    assert end == pytest.approx(3 * (10.01024 + 10.0192 + 4.0) + 10.00512 + 10.01024)
    assert config.T_round_max_ms == 800.0  # Preparation is not charged to this verification deadline.


@pytest.mark.parametrize('n', tuple(range(1000,5001,500)))
@pytest.mark.parametrize('d', (10,20,30,40,50))
def test_all_required_dataset_shapes_and_seed_determinism(n,d):
    args=dict(dataset_root=PROJECT_ROOT/'dataset',n=n,corruption_rate_percent=3.0,
              dim=d,H=.5,seed=2026)
    first,second=load_group(**args),load_group(**args)
    assert first['X'].shape==(n,d)
    assert first['X'].dtype==np.float32
    assert np.array_equal(first['X'],second['X'])


def test_full_configuration_records_eight_methods_modes_with_real_event_totals():
    config=cfg.OverheadConfig(K=4,M=2,dim=10,expansion_step=2,sigma=0.0)
    results=simulator.simulate_configuration({'X':_descriptors(80)},80,10,2026,config)
    assert {(r.row['method'],r.row['evaluation_mode']) for r in results}=={
        (method,mode) for method in cfg.METHODS for mode in cfg.EVALUATION_MODES}
    for result in results:
        assert result.row['status']!='failure',result.row.get('reason')
        assert result.row['selected_count']<=config.budget(80)
        assert result.row['message_count']==len(result.messages)
        assert result.row['total_bytes']==sum(e['total_bytes'] for e in result.messages)
        assert result.row['total_bytes']==sum(result.row[k] for k in ('gt_bytes','verification_bytes','control_bytes'))
        assert result.row['total_time_ms']==pytest.approx(
            result.row['gt_time_ms']+result.row['decision_cpu_ms']+result.row['verification_time_ms'])
        assert len({t['target_row'] for t in result.tasks})==len(result.tasks)
        assert all(t['target_row']!=t['verifier_row'] for t in result.tasks)


def test_digest_is_transmitted_after_hash_and_compared_after_arrival():
    x,ids,verifiers,targets,reference=_reference_case()
    config=cfg.OverheadConfig(dim=10,sigma=0.0)
    pair=[(int(targets[0]),int(verifiers[0]))]
    trace=baseline.verify_dvap(x,targets[:1],reference,config,assignments=pair,model_ids=ids)
    sim=simulator.EventSimulator(resource_profiles(n=len(x),rounds=1),10,config)
    simulator._trace_tasks(sim,trace,pair,0.0)
    digest=next(e for e in sim.counter.events if e['message_type']=='sha256_digest')
    producer=next(o for o in sim.runtime if o['operation']=='hash')
    receiver=next(o for o in sim.runtime if o['operation']=='compare' and o['node']==0)
    assert producer['end_ms']<=digest['start_ms']
    assert digest['end_ms']<=receiver['start_ms']


def test_ctcv_contract_waits_for_received_reports_and_result_follows_verdict():
    x,ids,verifiers,targets,reference=_reference_case()
    config=cfg.OverheadConfig(dim=10,sigma=0.0)
    pairs=[(int(t),int(verifiers[pos%len(verifiers)])) for pos,t in enumerate(targets)]
    trace=baseline.verify_ctcv(x,targets,reference,config,assignments=pairs,model_ids=ids,verifiers=verifiers)
    sim=simulator.EventSimulator(resource_profiles(n=len(x),rounds=1),10,config)
    tasks,finish=simulator._trace_tasks(sim,trace,pairs,0.0)
    reports=[e for e in sim.counter.events if e['message_type'] in ('ctcv_correct_combination','ctcv_proof_combination')]
    contract=[o for o in sim.runtime if o['node']==-2]
    result=next(e for e in sim.counter.events if e['message_type']=='ctcv_co_verification_result')
    assert min(o['start_ms'] for o in contract)>=max(e['end_ms'] for e in reports)
    assert result['start_ms']>=max(o['end_ms'] for o in contract)
    assert all(t['end_ms']==finish for t in tasks)
    for row in targets:
        produced=next(e for e in sim.counter.events if e['message_type']=='ctcv_producer_proof' and e['target_row']==row)
        validation=next(o for o in sim.runtime if o.get('proof_hop')==0 and o.get('target_row')==row)
        forwarded=[e for e in sim.counter.events if e['message_type']=='ctcv_modulo_proof' and e['target_row']==row]
        assert validation['start_ms']>=produced['end_ms']
        assert all(e['start_ms']>=validation['end_ms'] for e in forwarded)


def test_two_ctcv_primaries_that_are_collectors_start_concurrently():
    x=_descriptors(4)
    ids=np.arange(4,dtype=np.int64)
    verifiers=np.array([0,1],dtype=np.int64)
    config=cfg.OverheadConfig(dim=10,sigma=0.0)
    reference=baseline.prepare_reference(x,verifiers,ids)
    finishes=[]
    for pairs in ([(2,0),(3,1)],[(3,1),(2,0)]):
        targets=np.array([t for t,g in pairs],dtype=np.int64)
        trace=baseline.verify_ctcv(x,targets,reference,config,assignments=pairs,
                                   model_ids=ids,verifiers=verifiers,collector_rows=verifiers)
        sim=simulator.EventSimulator(resource_profiles(n=4,rounds=1),10,config)
        tasks,finish=simulator._trace_tasks(sim,trace,pairs,0.0)
        assert tasks[0]['start_ms'] == pytest.approx(tasks[1]['start_ms'])
        assert tasks[0]['start_ms'] >= 0.0
        assert all(t['producer_completion_ms']<41.0 for t in tasks)
        # The former sequential reservation made the second primary start ~40
        # ms late and finish ~80 ms, before even performing AV/SC aggregation.
        for node in {op['node'] for op in sim.runtime}:
            operations=sorted((op for op in sim.runtime if op['node']==node),key=lambda op:op['start_ms'])
            assert all(a['end_ms']<=b['start_ms'] for a,b in zip(operations,operations[1:]))
        finishes.append(finish)
    assert finishes[0]==pytest.approx(finishes[1])


def test_systemic_execution_error_stops_exceptions_not_deadline_results():
    failures=[{'row':{'status':'failure','reason':'TypeError: shared interface error'}} for _ in range(8)]
    assert runner._systemic_execution_error(failures)=='TypeError: shared interface error'
    failures[-1]['row']['status']='deadline_failure'
    assert runner._systemic_execution_error(failures) is None


def test_plot_panel_envelope_and_name_match_detection_geometry(monkeypatch,tmp_path):
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from experiments.common import plotting_style as style
    plot=importlib.import_module('experiments.02_overhead.plot_overhead')
    recorded=[]
    # Inspect an in-memory rendering only; these fixture values are never
    # persisted as research data or figures.
    monkeypatch.setattr(Figure,'savefig',lambda fig,*args,**kwargs:recorded.append(fig))
    monkeypatch.setattr(plot,'atomic_write',lambda path,writer:writer(path))
    records={(n,d,method):{'mean':1.0,'n_valid':1} for n in cfg.N_VALUES for d in cfg.D_PANEL_VALUES for method in cfg.METHODS}
    style.apply_style()
    plot._five_panel(records,tmp_path,'time','test-layout-only')
    fig=recorded[0]
    assert len(fig.axes)==5
    assert fig.get_size_inches()==pytest.approx([style.DETECTION_FIGURE_WIDTH,style.DETECTION_FIGURE_HEIGHT])
    width_pt,height_pt=fig.get_size_inches()*72
    expected_height=style.DETECTION_UPPER_HEIGHT_PT+style.DETECTION_LOWER_HEIGHT_PT+style.DETECTION_BREAK_GAP_PT
    for ax in fig.axes:
        assert ax.get_position().width*width_pt==pytest.approx(style.DETECTION_PANEL_WIDTH_PT)
        assert ax.get_position().height*height_pt==pytest.approx(expected_height)
    assert [label.get_text() for label in fig.legends[0].get_texts()]==['RACA-HIM','DVA-P','I-EDI','CTCV']
    canvas=FigureCanvasAgg(fig)
    canvas.draw()
    renderer=canvas.get_renderer()
    for text in [*fig.texts,*fig.legends[0].get_texts()]:
        bounds=text.get_window_extent(renderer)
        assert bounds.x0>=-0.5 and bounds.y0>=-0.5
        assert bounds.x1<=fig.bbox.x1+0.5 and bounds.y1<=fig.bbox.y1+0.5
