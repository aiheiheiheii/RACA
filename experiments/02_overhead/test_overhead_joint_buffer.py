"""Independent equivalence checks for the joint scheduler's reused buffers."""
from __future__ import annotations

import importlib
from itertools import product
from pathlib import Path
import sys

sys.dont_write_bytecode = True
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pytest

simulator = importlib.import_module("experiments.02_overhead.overhead_simulator")


def original_formula_schedule(targets, verifiers, risks, costs, budget, deadline,
                              lambda_, *, fixed_workload, enforce_deadline):
    """Reference the original allocating expressions, without calling the SUT.

    For finite nonnegative finish/span, ``max(0, finish-span)`` is exactly the
    manuscript's ``max(span, finish)-span``. The expression intentionally
    retains its multiplication/division order; arbitrary reassociation would
    not justify a bitwise equivalence check.
    """
    costs, risks = np.asarray(costs, np.float64), np.asarray(risks, np.float64)
    if not len(targets) or not len(verifiers):
        return [], dict(generated_count=0, chosen_prefix=0, prefix_objectives=[],
                        prefix_makespans_ms=[], marginal_gains=[])
    loads, used = np.zeros(len(verifiers)), np.zeros(len(targets), bool)
    assignments, objectives, spans, gains = [], [], [], []
    span = benefit = 0.0
    for _ in range(min(budget, len(targets))):
        finish = costs + loads[None, :]
        feasible = (~used[:, None]) & ((finish <= deadline) if enforce_deadline else True)
        delta = np.maximum(0.0, finish - span)
        gain = risks[:, None] / budget - lambda_ * delta / deadline
        gain = np.where(feasible, gain, -np.inf)
        ti, vi = np.unravel_index(int(np.argmax(gain)), gain.shape)
        if not np.isfinite(gain[ti, vi]):
            break
        used[ti] = True
        loads[vi] = finish[ti, vi]
        span = float(loads.max())
        benefit += float(risks[ti]) / budget
        assignments.append((int(targets[ti]), int(verifiers[vi])))
        objectives.append(benefit - lambda_ * span / deadline)
        spans.append(span)
        gains.append(float(gain[ti, vi]))
    chosen = len(assignments) if fixed_workload else (int(np.argmax(objectives)) + 1 if objectives else 0)
    return assignments[:chosen], dict(
        generated_count=len(assignments), chosen_prefix=chosen,
        prefix_objectives=objectives, prefix_makespans_ms=spans,
        marginal_gains=gains, fixed_prefix_override=fixed_workload,
        enforce_deadline=enforce_deadline,
        tie_rule="supplied temporary target/verifier identity order",
    )


def assert_equivalent(targets, verifiers, risks, costs, budget, deadline, lambda_,
                      *, fixed_workload, enforce_deadline):
    original_costs, original_risks = np.array(costs, copy=True), np.array(risks, copy=True)
    expected_pairs, expected_plan = original_formula_schedule(
        targets, verifiers, risks, costs, budget, deadline, lambda_,
        fixed_workload=fixed_workload, enforce_deadline=enforce_deadline,
    )
    actual_pairs, actual_plan = simulator.joint_schedule(
        targets, verifiers, risks, costs, budget, deadline, lambda_,
        fixed_workload=fixed_workload, enforce_deadline=enforce_deadline,
    )
    assert actual_pairs == expected_pairs
    assert actual_plan.keys() == expected_plan.keys()
    for key in actual_plan:
        if key in ("prefix_objectives", "prefix_makespans_ms", "marginal_gains"):
            # Check IEEE754 bits as well as pair selection: no smoothing,
            # changed tie decision or permitted rounding tolerance is needed.
            actual = np.asarray(actual_plan[key], dtype=np.float64).view(np.uint64)
            expected = np.asarray(expected_plan[key], dtype=np.float64).view(np.uint64)
            np.testing.assert_array_equal(actual, expected)
        else:
            assert actual_plan[key] == expected_plan[key]
    np.testing.assert_array_equal(costs, original_costs)
    np.testing.assert_array_equal(risks, original_risks)
    return actual_pairs, actual_plan


@pytest.mark.parametrize("seed", range(8))
@pytest.mark.parametrize("fixed_workload,enforce_deadline", tuple(product((False, True), repeat=2)))
def test_randomized_buffered_scheduler_matches_original_expressions(seed, fixed_workload,
                                                                  enforce_deadline):
    rng = np.random.default_rng(seed)
    count, width = 5 + 2 * seed, 1 + seed % 5
    targets = rng.permutation(np.arange(100, 100 + count)).astype(np.int64)
    verifiers = rng.permutation(np.arange(width)).astype(np.int64)
    costs = rng.lognormal(mean=1., sigma=1.2, size=(count, width))
    risk_cases = (rng.uniform(0., 1., count), np.full(count, .55), np.zeros(count))
    for risks, budget, deadline, lambda_ in product(
        risk_cases, (1, min(count, 7), count + 3),
        (float(costs.min()) / 2., float(np.median(costs)) * 2., float(costs.max()) * count),
        (.05, .5, 4.),
    ):
        assert_equivalent(targets, verifiers, risks, costs, budget, deadline, lambda_,
                          fixed_workload=fixed_workload, enforce_deadline=enforce_deadline)


@pytest.mark.parametrize("fixed_workload,enforce_deadline", tuple(product((False, True), repeat=2)))
def test_equal_cost_and_zero_marginal_ties_follow_supplied_identity_order(fixed_workload,
                                                                       enforce_deadline):
    targets, verifiers = np.array([30, 10, 20]), np.array([9, 8])
    pairs, plan = assert_equivalent(
        targets, verifiers, np.ones(3), np.ones((3, 2)), 3, 100., .5,
        fixed_workload=fixed_workload, enforce_deadline=enforce_deadline,
    )
    assert pairs == [(30, 9), (10, 8), (20, 9)]
    assert plan["generated_count"] == 3
    assert plan["marginal_gains"][1] == 1. / 3.


@pytest.mark.parametrize("fixed_workload,enforce_deadline", tuple(product((False, True), repeat=2)))
def test_float_neighbors_and_exact_deadline_preserve_pair_decisions(fixed_workload,
                                                                  enforce_deadline):
    below, above = np.nextafter(1., 0.), np.nextafter(1., np.inf)
    costs = np.array([[1., above], [below, 2.], [above, below], [2., above]])
    for deadline, lambda_, risks in product(
        (below, 1., above, 10.), (.5, 1e-12, 1e12),
        (np.full(4, .55), np.array([1., below, 0., .55])),
    ):
        assert_equivalent(np.array([11, 14, 12, 13]), np.array([8, 7]),
                          risks, costs, 4, deadline, lambda_,
                          fixed_workload=fixed_workload, enforce_deadline=enforce_deadline)


@pytest.mark.parametrize("fixed_workload", (False, True))
def test_all_negative_gains_keep_feasible_prefix_generation(fixed_workload):
    pairs, plan = assert_equivalent(
        np.array([10, 11, 12]), np.array([0, 1]), np.zeros(3),
        np.array([[1., 2.], [2., 1.], [3., 3.]]), 3, 100., .5,
        fixed_workload=fixed_workload, enforce_deadline=True,
    )
    assert plan["generated_count"] == 3
    assert plan["chosen_prefix"] == (3 if fixed_workload else 1)
    assert len(pairs) == plan["chosen_prefix"]


def test_readonly_noncontiguous_inputs_are_preserved():
    rng = np.random.default_rng(2033)
    costs = rng.uniform(.01, 8., (14, 8))[::2, 1::2]
    risks = rng.uniform(0., 1., 14)[::2]
    costs.setflags(write=False)
    risks.setflags(write=False)
    assert not costs.flags.c_contiguous and not risks.flags.c_contiguous
    assert_equivalent(np.arange(30, 37), np.array([8, 3, 4, 1]), risks, costs,
                      7, 15., .5, fixed_workload=False, enforce_deadline=True)
    assert not costs.flags.writeable and not risks.flags.writeable


@pytest.mark.parametrize("targets,verifiers,costs,risks", (
    (np.empty(0, np.int64), np.array([0, 1]), np.empty((0, 2)), np.empty(0)),
    (np.array([10]), np.empty(0, np.int64), np.empty((1, 0)), np.array([.55])),
))
def test_empty_domain_keeps_original_empty_prefix_contract(targets, verifiers, costs, risks):
    pairs, plan = assert_equivalent(targets, verifiers, risks, costs, 1, 100., .5,
                                  fixed_workload=True, enforce_deadline=True)
    assert pairs == [] and plan["generated_count"] == 0
