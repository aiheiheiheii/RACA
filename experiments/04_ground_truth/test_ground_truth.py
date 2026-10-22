"""User-run scientific checks; deliberately not executed during implementation.

Run later from the project root with:
    python -B -m pytest experiments/04_ground_truth/test_ground_truth.py

Tests use small synthetic descriptor cases and one real logical dataset group;
they never start the formal experiment or render/write paper figures/results.
"""
from __future__ import annotations

import ast
from dataclasses import fields
import importlib
import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace

sys.dont_write_bytecode = True
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pytest

from dataset.load_dataset import load_group
from experiments.common.config import DetectionConfig
from experiments.common.dataset_adapter import AlgorithmInput
from experiments.common.reproducibility import dataset_snapshot
from experiments.common.similarity import cosine_pairwise

kernel = importlib.import_module("experiments.01_detection.proposed_detection")
config = importlib.import_module("experiments.04_ground_truth.config_ground_truth")
metrics = importlib.import_module("experiments.04_ground_truth.ground_truth_metrics")
sampler = importlib.import_module("experiments.04_ground_truth.candidate_sampler")
methods = importlib.import_module("experiments.04_ground_truth.ground_truth_methods")


def _observations(vectors) -> AlgorithmInput:
    x = np.asarray(vectors, dtype=np.float32)
    return AlgorithmInput(
        X=x, model_ids=np.arange(len(x), dtype=np.int64),
        history=np.full(len(x), 0.1),
        reference_indices=np.arange(len(x), dtype=np.int64),
        evaluation_indices=np.empty(0, dtype=np.int64),
    )


def _clusters(counts) -> np.ndarray:
    """Mutually orthogonal, internally identical descriptor clusters."""
    return np.repeat(np.eye(30, dtype=np.float32)[:len(counts)], counts, axis=0)


@pytest.fixture(scope="module", autouse=True)
def dataset_remains_unchanged():
    before = dataset_snapshot(config.DATASET_ROOT)
    yield
    assert dataset_snapshot(config.DATASET_ROOT) == before


@pytest.fixture(scope="module")
def real_group():
    return load_group(config.DATASET_ROOT, n=3000, corruption_rate_percent=3.0,
                      dim=30, H=0.5, seed=2026)


def test_formal_grid_and_label_free_observation_boundary():
    config.validate_protocol()
    assert (config.EXPECTED_SCENARIOS, config.EXPECTED_RUNS) == (55000, 220000)
    assert config.K_VALUES == (50, 75, 100)
    assert len(config.SEEDS) == config.REPETITIONS == 1000
    assert {field.name for field in fields(AlgorithmInput)} == {
        "X", "model_ids", "history", "reference_indices", "evaluation_indices",
    }


def test_real_dataset_shape_subtypes_corrupt_supply_and_repeatability(real_group):
    assert real_group["X"].shape == (3000, 30)
    assert real_group["X"].dtype == np.float32
    assert np.isfinite(real_group["X"]).all()
    assert np.count_nonzero(real_group["binary_label"]) == 90
    assert set(np.unique(real_group["subtype_label"])) == {0, 1, 2, 3}
    repeated = load_group(config.DATASET_ROOT, n=3000, corruption_rate_percent=3.0,
                          dim=30, H=0.5, seed=2026)
    for name in ("X", "binary_label", "subtype_label"):
        assert np.array_equal(real_group[name], repeated[name])


def test_sampler_exact_composition_shuffled_prefixes_and_no_subtype_filter(real_group, monkeypatch):
    original_draw = sampler.draw_candidate_set
    calls = []

    def record_draw(**kwargs):
        calls.append(kwargs)
        return original_draw(**kwargs)

    monkeypatch.setattr(sampler, "draw_candidate_set", record_draw)
    for q_C in config.Q_VALUES:
        scene = sampler.draw_scenario(real_group, 2026, q_C)
        source, ratio = original_draw(real_group, K=100, q_C=q_C, seed=2026,
                                      use_severe_only=False, use_common_only=False)
        assert np.array_equal(scene.source_indices, source)
        assert len(np.unique(source)) == 100
        assert scene.q_C_realized == ratio == q_C
        assert np.count_nonzero(scene.evaluation_labels) == round(100 * q_C)
        assert np.array_equal(scene.algorithm_input.X, real_group["X"][source])
        assert np.array_equal(scene.algorithm_input.model_ids, np.arange(100))
        assert np.array_equal(scene.algorithm_input.reference_indices, np.arange(100))
        assert len(scene.algorithm_input.evaluation_indices) == 0
        assert scene.prefix_corruption_ratios == {
            size: float(scene.evaluation_labels[:size].mean()) for size in (50, 75, 100)
        }
        for field in fields(AlgorithmInput):
            assert not getattr(scene.algorithm_input, field.name).flags.writeable
    assert all(call["K"] == 100 and call["use_severe_only"] is False
               and call["use_common_only"] is False for call in calls)


def test_sampler_failure_uniforms_reproduce_and_failure_sets_are_nested(real_group):
    first = sampler.draw_scenario(real_group, 2026, 0.30)
    second = sampler.draw_scenario(real_group, 2026, 0.30)
    assert first.candidate_fingerprint == second.candidate_fingerprint
    assert first.random_method_seed == second.random_method_seed
    for name in ("source_indices", "evaluation_labels", "failure_uniform"):
        assert np.array_equal(getattr(first, name), getattr(second, name))
    previous_available = np.ones(100, dtype=bool)
    original_uniform = first.failure_uniform.copy()
    for p_fail in config.FAILURE_VALUES:
        available = first.availability(p_fail)
        assert np.array_equal(available, first.failure_uniform >= p_fail)
        assert not np.any(available & ~previous_available)
        assert not available.flags.writeable
        previous_available = available
    assert np.array_equal(first.failure_uniform, original_uniform)


def test_saved_scenario_audit_is_json_serializable_at_zero_and_half_pollution(real_group):
    experiment = importlib.import_module("experiments.04_ground_truth.ground_truth_experiment")
    for q_C in (0.0, 0.5):
        scene = sampler.draw_scenario(real_group, 2026, q_C)
        audit = experiment._scenario_record(scene, cosine_pairwise(scene.algorithm_input.X), "synthetic")
        saved = json.loads(json.dumps(audit, allow_nan=False))
        assert saved["run_id"] == "synthetic"
        assert saved["prefix_corruption_ratios"]["100"] == q_C
        assert saved["physical_cache_dot_product_count"] == 10000
        assert len(saved["failure_uniform"]) == 100
        if q_C == 0.0:
            assert saved["pairwise_cosine_posthoc"]["corrupt_corrupt"]["min"] is None


def test_output_preparation_uses_fixed_fourth_group_without_moving_existing_files(tmp_path, monkeypatch):
    experiment = importlib.import_module("experiments.04_ground_truth.ground_truth_experiment")
    monkeypatch.setattr(experiment, "PROJECT_ROOT", tmp_path)
    root = tmp_path / "results" / "04_ground_truth"
    first = tmp_path / "results" / "01_detection" / "raw" / "marker.txt"
    original = root / "raw" / "marker.txt"
    for path in (first, original):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("preserve", encoding="utf-8")
    output = experiment._prepare_output(root)
    assert output == root
    assert original.read_text(encoding="utf-8") == "preserve"
    assert first.read_text(encoding="utf-8") == "preserve"
    assert all((output / name).is_dir() for name in ("raw", "summary", "figures", "logs"))
    for rejected in (tmp_path / "dataset" / "04_ground_truth", first.parents[1]):
        with pytest.raises(ValueError):
            experiment._prepare_output(rejected)
    assert first.read_text(encoding="utf-8") == "preserve"


def test_default_kernel_and_explicit_all_available_cache_have_same_decisions():
    data = _observations(np.concatenate((_clusters((25, 25)), _clusters((50,)))))
    implicit = kernel.build_ground_truth(data, DetectionConfig())
    explicit = kernel.build_ground_truth(
        data, DetectionConfig(), responder_available=np.ones(100, dtype=bool),
        pairwise_cache=cosine_pairwise(data.X), allow_expansion=True,
    )
    assert implicit.success and explicit.success
    assert implicit.threshold == explicit.threshold == 38
    assert implicit.expansion_trace == explicit.expansion_trace
    assert [(step.candidate_size, step.threshold, step.passing_count)
            for step in implicit.expansion_trace] == [(50, 26, 0), (75, 38, 50)]
    for name in ("candidate_indices", "selected_indices", "support_counts"):
        assert np.array_equal(getattr(implicit, name), getattr(explicit, name))
    assert np.array_equal(implicit.selected_indices, np.arange(25))


@pytest.mark.parametrize("k,threshold", [(50, 26), (75, 38), (100, 51)])
def test_response_failure_keeps_full_k_threshold_and_counts_self_once(k, threshold):
    data = _observations(_clusters((100,)))
    settings = DetectionConfig(K=k)
    cache = cosine_pairwise(data.X)
    for responding, expected_success in ((threshold - 1, False), (threshold, True)):
        available = np.arange(100) < responding
        result = kernel.build_ground_truth(data, settings, responder_available=available,
                                           pairwise_cache=cache, allow_expansion=False)
        assert result.success is expected_success
        assert result.threshold == threshold
        assert len(result.expansion_trace) == 1
        assert len(result.candidate_indices) == k
        assert np.all(result.support_counts[:responding] == responding - 1)
        assert np.all(result.support_counts[responding:] == 0)
        assert result.pairwise_comparison_count == responding * (responding - 1)
        assert len(result.selected_indices) == (25 if expected_success else 0)
        assert available[result.selected_indices].all()


def test_expansion_rechecks_previous_admission_under_new_majority_threshold():
    # One central model agrees with all 49 nearby models; three directional
    # groups disagree with each other. Two later orthogonal groups provide no
    # support to the center. The center passes K=50/75 but must fail at K=100.
    center = np.eye(30, dtype=np.float32)[0]
    nearby = []
    for axis, count in zip((1, 2, 3), (16, 16, 17)):
        vector = center.copy()
        vector[0], vector[axis] = math.cos(0.07), math.sin(0.07)
        nearby.append(np.repeat(vector[None, :], count, axis=0))
    later = np.repeat(np.eye(30, dtype=np.float32)[[4, 5]], (25, 25), axis=0)
    data = _observations(np.concatenate((center[None, :], *nearby, later)))
    result = kernel.build_ground_truth(data, DetectionConfig(),
                                       pairwise_cache=cosine_pairwise(data.X))
    assert not result.success and len(result.selected_indices) == 0
    assert [(step.candidate_size, step.threshold, step.passing_count)
            for step in result.expansion_trace] == [(50, 26, 1), (75, 38, 1), (100, 51, 0)]
    assert result.support_counts[0] == 49
    assert np.array_equal(result.candidate_indices, np.arange(100))
    assert result.reason


def test_fixed_k_is_identical_at_initial_k_but_cannot_expand():
    data = _observations(np.concatenate((_clusters((25, 25)), _clusters((50,)))))
    full = kernel.build_ground_truth(data, DetectionConfig())
    fixed = kernel.build_ground_truth(data, DetectionConfig(), allow_expansion=False)
    assert full.success and not fixed.success
    assert full.expansion_trace[0] == fixed.expansion_trace[0]
    assert len(fixed.candidate_indices) == 50 and len(fixed.expansion_trace) == 1
    assert len(fixed.selected_indices) == 0 and fixed.reason


def test_four_methods_share_unmodified_observations_and_exogenous_failures():
    data = _observations(_clusters((100,)))
    available = np.arange(100) >= 30
    cache = cosine_pairwise(data.X)
    before = {field.name: getattr(data, field.name).copy() for field in fields(AlgorithmInput)}
    results = {name: methods.run_method(name, data, available, 2026, pairwise_cache=cache)
               for name in config.METHODS}
    assert results["RACA-HIM"].success and results["RACA-HIM"].final_K == 100
    assert not results["Fixed-K Majority"].success and results["Fixed-K Majority"].final_K == 50
    for name in ("Top-M", "Random-M"):
        result = results[name]
        assert result.success and result.final_K == 75 and result.expansion_count == 1
        assert result.pairwise_comparison_count == 0
    assert np.array_equal(results["Top-M"].selected_indices, np.arange(30, 55))
    repeated = methods.run_method("Random-M", data, available, 2026)
    assert np.array_equal(results["Random-M"].selected_indices, repeated.selected_indices)
    for result in results.values():
        assert len(result.selected_indices) == (25 if result.success else 0)
        assert len(np.unique(result.selected_indices)) == len(result.selected_indices)
        assert available[result.selected_indices].all()
        assert (result.selected_indices < result.final_K).all()
    for field in fields(AlgorithmInput):
        assert np.array_equal(getattr(data, field.name), before[field.name])
    assert np.array_equal(available, np.arange(100) >= 30)


@pytest.mark.parametrize("name", ["Top-M", "Random-M"])
def test_baseline_does_not_use_majority_or_cache_and_fails_when_responders_exhausted(name, monkeypatch):
    data = _observations(_clusters((100,)))

    def forbidden_majority(*args, **kwargs):
        raise AssertionError("A no-majority baseline invoked the majority kernel")

    class ForbiddenCosineCache:
        def __array__(self, *args, **kwargs):
            raise AssertionError("A no-similarity baseline read the cosine cache")

    monkeypatch.setattr(methods, "build_ground_truth", forbidden_majority)
    available = np.arange(100) < 24
    result = methods.run_method(name, data, available, 2026, pairwise_cache=ForbiddenCosineCache())
    assert not result.success and result.final_K == 100 and result.expansion_count == 2
    assert result.pairwise_comparison_count == 0 and len(result.selected_indices) == 0
    assert result.reason


@pytest.mark.parametrize("mutation", ["wrong_mask_shape", "integer_mask", "nan_cache", "wrong_cache_shape"])
def test_kernel_rejects_misaligned_or_invalid_optional_inputs(mutation):
    data = _observations(_clusters((100,)))
    mask = np.ones(100, dtype=bool)
    cache = cosine_pairwise(data.X)
    if mutation == "wrong_mask_shape":
        mask = mask[:50]
    elif mutation == "integer_mask":
        mask = mask.astype(np.int8)
    elif mutation == "nan_cache":
        cache[0, 1] = np.nan
    else:
        cache = cache[:50, :50]
    with pytest.raises(ValueError):
        kernel.build_ground_truth(data, DetectionConfig(), responder_available=mask,
                                  pairwise_cache=cache)


@pytest.mark.parametrize("case", ["reliable", "short", "empty", "contaminated", "unavailable", "duplicate"])
def test_reliability_requires_exactly_m_unique_normal_responders(case):
    selected = np.arange(25, dtype=np.int64)
    truth = np.zeros(100, dtype=np.int8)
    available = np.ones(100, dtype=bool)
    if case == "short":
        selected = selected[:24]
    elif case == "empty":
        selected = selected[:0]
    elif case == "contaminated":
        truth[0] = 1
    elif case == "unavailable":
        available[0] = False
    elif case == "duplicate":
        selected[:] = 0
    result = SimpleNamespace(success=case != "empty", selected_indices=selected, reason="")
    evaluated = metrics.evaluate_ground_truth(result, truth, available, M=25)
    assert evaluated["reliable_success"] == int(case == "reliable")
    assert evaluated["gt_success"] == int(case in ("reliable", "contaminated", "unavailable"))
    assert evaluated["gt_size"] == len(selected)
    if case == "contaminated":
        assert evaluated["gt_corrupt_count"] == 1
    if case != "reliable":
        assert evaluated["failure_reason"]


def test_summary_retains_unformed_and_contaminated_trials_in_1000_seed_denominator():
    rows = []
    for index, seed in enumerate(config.SEEDS):
        reliable, formed = int(index < 250), int(index < 750)
        rows.append({
            "method": "RACA-HIM", "q_C_requested": 0.30, "p_fail_requested": 0.10,
            "seed": seed, "reliable_success": reliable, "gt_success": formed,
            "gt_corrupt_count": int(formed and not reliable),
            "gt_normal_count": 25 if reliable else 24 if formed else 0,
            "gt_size": 25 if formed else 0, "final_K": 100,
            "candidate_expansion_count": 2, "responsive_count": 90,
            "pairwise_comparison_count": 100,
        })
    summary, diagnostics = metrics.summarize_runs(rows, require_complete=False)
    assert len(summary) == len(diagnostics) == 1
    point = summary[0]
    assert point["n_total"] == 1000
    assert point["n_reliable_success"] == 250 and point["n_gt_formed"] == 750
    assert point["reliable_gt_rate"] == point["mean"] == 0.25
    assert point["gt_formation_rate"] == 0.75
    assert point["std"] == pytest.approx(np.std([1] * 250 + [0] * 750, ddof=1))
    assert diagnostics[0]["n_gt_insufficient"] == 250
    assert diagnostics[0]["n_gt_contaminated"] == 500
    with pytest.raises(ValueError):
        metrics.summarize_runs(rows)  # A partial method grid cannot become formal output.
    with pytest.raises(ValueError):
        metrics.summarize_runs(rows + [rows[0]], require_complete=False)


def test_wilson_zero_and_all_success_endpoints_are_finite_and_non_degenerate():
    zero = metrics.wilson_interval(0, 1000)
    all_success = metrics.wilson_interval(1000, 1000)
    assert zero[0] == 0.0 and all_success[1] == 1.0
    assert zero[1] == pytest.approx(0.003826758485555, rel=1e-10)
    assert all_success[0] == pytest.approx(1.0 - zero[1])
    assert all(math.isfinite(value) and 0 <= value <= 1 for value in (*zero, *all_success))
    with pytest.raises(ValueError):
        metrics.wilson_interval(0, 0)


def test_plot_source_has_no_algorithm_or_dataset_execution():
    path = Path(__file__).with_name("plot_ground_truth.py")
    tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
    forbidden = {"build_ground_truth", "detect_proposed", "load_group", "draw_candidate_set",
                 "run_methods", "run_seed", "run_trial", "fill_between", "interp", "interp1d"}
    calls = {node.func.id if isinstance(node.func, ast.Name) else node.func.attr
             for node in ast.walk(tree) if isinstance(node, ast.Call)
             and isinstance(node.func, (ast.Name, ast.Attribute))}
    assert calls.isdisjoint(forbidden)
    imported = [node.module or "" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
    imported += [alias.name for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names]
    imported += [node.args[0].value for node in ast.walk(tree)
                 if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                 and node.func.attr == "import_module" and node.args
                 and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str)]
    assert not any(any(fragment in name for fragment in (
        "candidate_sampler", "ground_truth_methods", "ground_truth_experiment", "dataset.load_dataset",
        "01_detection", "03_threshold")) for name in imported)
