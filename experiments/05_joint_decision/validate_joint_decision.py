"""User-invoked invariant checks; never called by the formal experiment entry.

Run this file manually in PyCharm after reviewing the implementation. It uses
hand-built inputs, a scalar full-pair greedy oracle and a scalar RF-TS reference.
Archive/resume checks use isolated temporary files; no formal result files are
written and no figures are drawn. It does not sample a reduced formal experiment. The GT
adapter requires N >= 1000, so its descriptor fixtures use the smallest valid
domain and are entirely synthetic. Optional --check-dataset only loads the
same published descriptor group twice; it performs no dataset optimization.

These checks are supplied as source. Their presence is not evidence that they
have been executed or passed. Use --check-io-only to exercise file-lock retries
with mock waits, isolated temporary files and in-memory I/O migration identities;
no method or dataset fixture is evaluated in that mode.
"""
from __future__ import annotations

import argparse
import ast
from contextlib import contextmanager, nullcontext
from copy import deepcopy
from dataclasses import fields
import hashlib
import importlib
import inspect
import json
import math
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.dont_write_bytecode = True
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

PACKAGE = "experiments.05_joint_decision"
cfg = importlib.import_module(f"{PACKAGE}.config_joint_decision")
history = importlib.import_module(f"{PACKAGE}.historical_state")
inputs = importlib.import_module(f"{PACKAGE}.joint_inputs")
objective = importlib.import_module(f"{PACKAGE}.joint_objective")
methods = importlib.import_module(f"{PACKAGE}.joint_methods")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _close(actual: float, expected: float, message: str) -> None:
    _require(math.isclose(float(actual), float(expected), rel_tol=1e-11, abs_tol=1e-10),
             f"{message}: actual={actual!r}, expected={expected!r}")


def _problem(targets, verifiers, risks, costs, budget, *, deadline=32.0):
    return objective.JointProblem(
        target_ids=np.asarray(targets, dtype=np.int64),
        verifier_ids=np.asarray(verifiers, dtype=np.int64),
        risk_scores=np.asarray(risks, dtype=np.float64),
        costs=np.asarray(costs, dtype=np.float64).reshape(len(verifiers), len(targets)),
        budget=budget, lambda_value=0.5, deadline_ms=deadline,
    )


def _scalar_measure(problem, plan) -> tuple[float, float, float]:
    """Independent Python arithmetic, with no evaluator or optimizer helpers."""
    risks = {int(t): float(r) for t, r in zip(problem.target_ids, problem.risk_scores)}
    target_rows = {int(t): i for i, t in enumerate(problem.target_ids)}
    verifier_rows = {int(g): j for j, g in enumerate(problem.verifier_ids)}
    loads = {g: 0.0 for g in verifier_rows}
    seen = set()
    reward = 0.0
    _require(0 < len(plan) <= problem.budget, "Scalar check requires a nonempty within-budget plan")
    for target, verifier in plan:
        _require(target in target_rows and verifier in verifier_rows,
                 "A returned ID is outside the common target/GT pools")
        _require(target not in seen and target != verifier, "Duplicate target or self verification")
        cost = float(problem.costs[verifier_rows[verifier], target_rows[target]])
        _require(math.isfinite(cost) and cost >= 0.0, "An unavailable pair was scheduled")
        seen.add(target)
        reward += risks[target]
        loads[verifier] += cost
    makespan = max(loads.values(), default=0.0)
    _require(makespan <= problem.deadline_ms + 1e-9, "A returned plan violates its deadline")
    utility = reward / problem.budget - problem.lambda_value * makespan / problem.deadline_ms
    return reward, makespan, utility


def _literal_greedy_oracle(problem) -> tuple[tuple, int, tuple[float, ...]]:
    """Enumerate every legal extension and score its whole assignment afresh.

    This deliberately avoids the vectorized marginal-gain expression, cached
    loads, method internals, and the production evaluator. It is an oracle for
    the declared greedy path and best prefix, not a global-optimality claim.
    The fixtures use binary fractions and power-of-two denominators so exact
    tie cases are representable without floating-point ambiguity.
    """
    prefix = ()
    prefixes, utilities = [], []
    targets = sorted(int(t) for t in problem.target_ids)
    verifiers = sorted(int(g) for g in problem.verifier_ids)
    while len(prefix) < problem.budget:
        used = {target for target, _ in prefix}
        best_candidate, best_value = None, -math.inf
        for target in targets:
            if target in used:
                continue
            for verifier in verifiers:
                candidate = prefix + ((target, verifier),)
                try:
                    value = _scalar_measure(problem, candidate)[2]
                except AssertionError:
                    continue
                if value > best_value:
                    best_candidate, best_value = candidate, value
        if best_candidate is None:
            break
        prefix = best_candidate
        prefixes.append(prefix)
        utilities.append(best_value)
    if not prefixes:
        return (), 0, ()
    best_position = max(range(len(prefixes)), key=lambda i: utilities[i])
    return prefixes[best_position], len(prefixes), tuple(utilities)


def _literal_rf_ts_reference(problem) -> tuple:
    """Preserve the original two-stage RF-TS rule using only Python loops.

    This reference neither truncates to a best-J prefix nor chooses targets or
    assignments using J. It retains every deadline-feasible scheduled target
    until the fixed budget is reached, including a negative final marginal J.
    """
    if problem.budget == 0:
        return ()
    targets = {int(t): i for i, t in enumerate(problem.target_ids)}
    verifiers = {int(g): j for j, g in enumerate(problem.verifier_ids)}
    risk = {int(t): float(r) for t, r in zip(problem.target_ids, problem.risk_scores)}
    order = sorted(targets, key=lambda target: (-risk[target], target))
    loads = {verifier: 0.0 for verifier in verifiers}
    selected = []
    for target in order:
        legal = []
        for verifier in sorted(verifiers):
            cost = float(problem.costs[verifiers[verifier], targets[target]])
            finish = loads[verifier] + cost
            if target != verifier and math.isfinite(cost) and finish <= problem.deadline_ms:
                legal.append((finish, verifier))
        if not legal:
            continue
        finish, verifier = min(legal)
        selected.append((target, verifier))
        loads[verifier] = finish
        if len(selected) == problem.budget:
            break
    return tuple(selected)


def validate_source_boundaries() -> None:
    """Check accidental forbidden-field reads and the plot-only import boundary."""
    forbidden = {"binary_label", "subtype_label", "prototype", "cosine_to_prototype", "sample_id"}
    for filename in ("historical_state.py", "joint_inputs.py", "joint_objective.py", "joint_methods.py"):
        source = PROJECT_ROOT / "experiments" / "05_joint_decision" / filename
        tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
        for node in ast.walk(tree):
            _require(not (isinstance(node, ast.Attribute) and node.attr in forbidden),
                     f"Forbidden label/provenance attribute read in {filename}")
            if isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Constant):
                _require(node.slice.value not in forbidden,
                         f"Forbidden label/provenance dictionary access in {filename}")
            if isinstance(node, ast.Call):
                literal_arguments = [arg.value for arg in node.args if isinstance(arg, ast.Constant)]
                _require(not any(isinstance(value, str) and value in forbidden for value in literal_arguments),
                         f"Forbidden label/provenance lookup call in {filename}")
    _require(set(inspect.signature(history.build_historical_state).parameters) == {"n", "seed", "config"},
             "History generation must accept only public configuration, N and its own seed")
    _require(not forbidden.intersection(f.name for f in fields(objective.JointProblem)),
             "JointProblem must not expose current labels/provenance")
    plot_path = PROJECT_ROOT / "experiments" / "05_joint_decision" / "plot_joint_decision.py"
    plot_tree = ast.parse(plot_path.read_text(encoding="utf-8"), filename=str(plot_path))
    for node in ast.walk(plot_tree):
        if isinstance(node, ast.ImportFrom):
            _require(not any(word in (node.module or "") for word in
                             ("joint_experiment", "joint_methods", "joint_inputs", "dataset")),
                     "The plot-only entry imports an experiment/optimization input module")
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "import_module":
            _require(len(node.args) == 1 and isinstance(node.args[0], ast.Constant)
                     and node.args[0].value == f"{PACKAGE}.config_joint_decision",
                     "The plot-only entry dynamically imports more than its inert configuration")


def validate_historical_observations() -> None:
    cfg.validate_protocol()
    first = history.build_historical_state(257, seed=87654)
    second = history.build_historical_state(257, seed=87654)
    for name in ("anomaly_state", "age", "risk_scores", "last_verified"):
        a, b = getattr(first, name), getattr(second, name)
        _require(np.array_equal(a, b, equal_nan=True), f"History is not reproducible: {name}")
        _require(not a.flags.writeable, f"Historical {name} must be read-only")
    _require(first.event_counts == second.event_counts, "Historical event counts changed for the same seed")
    _require(float(first.risk_scores.min()) >= 0.0 and float(first.risk_scores.max()) <= 1.0,
             "Risk lies outside [0,1]")
    _require(float(first.risk_scores.std()) > 0.0, "The synthetic history erased all model risk variation")
    cases = (
        ("normal", 1.0, 0.0, 0.0, 0.05, 0.05, True),
        ("anomalous", 1.0, 1.0, 0.0, 0.55, 0.05, True),
        ("invalid", 1.0, 0.0, 1.0, 0.325, 1.0, False),
        ("unobserved", 0.0, 0.0, 0.0, 0.10, 1.0, False),
    )
    for name, observed, anomalous, invalid, expected_h, expected_age, valid in cases:
        settings = history.HistoryConfig(rounds=1, verification_probability=observed,
                                         anomalous_probability=anomalous, invalid_probability=invalid)
        state = history.build_historical_state(3, seed=55, config=settings)
        _require(np.allclose(state.anomaly_state, expected_h), f"Wrong h update for {name}")
        _require(np.allclose(state.age, expected_age), f"Wrong verification age for {name}")
        _require(bool(np.all(np.isfinite(state.last_verified))) == valid,
                 f"Only valid observations may reset last_verified: {name}")
        _require(np.allclose(state.risk_scores, 0.5 * expected_age + 0.5 * expected_h),
                 f"Risk formula is wrong for {name}")


def validate_independent_objective() -> None:
    problem = _problem([10, 11, 12], [100, 101], [0.25, 0.5, 0.75],
                       [[10, 20, 30], [15, 25, 35]], 3, deadline=800.0)
    plan = ((10, 100), (11, 100), (12, 101))
    evaluation = objective.evaluate_joint_objective(problem, plan)
    _require(evaluation.feasible, "A known legal plan was rejected")
    reward, makespan, utility = _scalar_measure(problem, plan)
    _close(reward, 1.5, "Selected risk reward")
    _close(makespan, 35.0, "Parallel makespan must be max(30,35), not sum(30,35)")
    _close(evaluation.joint_utility, utility, "Independent utility reconstruction")
    _close(evaluation.normalized_risk_reward, reward / 3.0, "Risk must be normalized by budget B")
    _close(evaluation.normalized_delay_penalty, 0.5 * 35.0 / 800.0, "Delay penalty uses lambda and ms")
    _require(objective.evaluate_joint_objective(problem, plan[:1]).feasible,
             "A legal plan selecting fewer than B targets must remain feasible")
    illegal = (
        ((10, 100), (10, 101)),  # duplicate selected target
        ((10, 999),),           # outside GT
        ((999, 100),),          # outside target pool
        plan + ((10, 101),),    # budget and uniqueness
        (),                    # manuscript requires a nonempty solution
    )
    for candidate in illegal:
        result = objective.evaluate_joint_objective(problem, candidate)
        _require(not result.feasible and result.joint_utility is None,
                 f"Illegal assignment must not enter mean J: {candidate}")
    deadline_problem = _problem([10], [100], [0.25], [[801.0]], 1, deadline=800.0)
    _require(not objective.evaluate_joint_objective(deadline_problem, ((10, 100),)).feasible,
             "A deadline violation was accepted")
    self_problem = _problem([10], [10], [0.5], [[1.0]], 1)
    _require(not objective.evaluate_joint_objective(self_problem, ((10, 10),)).feasible,
             "Self verification was accepted")
    negative = _problem([10], [100], [0.125], [[700.0]], 1, deadline=800.0)
    result = objective.evaluate_joint_objective(negative, ((10, 100),))
    _require(result.feasible and result.joint_utility < 0.0, "A valid negative J was removed or clamped")


def validate_joint_greedy_oracle() -> None:
    # First marginal is negative, but the second verifier adds reward at zero
    # makespan increase. Both generated prefixes are negative; the empty plan
    # must not compete, and construction must cross the first negative gain.
    negative = _problem([12, 11], [102, 101], [0.125, 0.125], [[4, 4], [4, 4]], 2, deadline=8.0)
    shorter = _problem([11, 12], [101], [0.875, 0.125], [[1, 7]], 2, deadline=8.0)
    joint = _problem([11, 12], [101], [0.875, 0.75], [[7, 1]], 2, deadline=8.0)
    fixtures = [negative, shorter, joint]
    rng = np.random.default_rng(94133)
    # Tiny generated fixtures stress feasible pairs, exact ties and deadlines.
    # They are not dataset trials or substitutes for any formal seed.
    for _ in range(12):
        costs = rng.integers(1, 29, size=(2, 4)).astype(float)
        risks = rng.integers(1, 16, size=4) / 16.0
        fixtures.append(_problem([14, 11, 13, 12], [102, 101], risks, costs, 4, deadline=32.0))
    for problem in fixtures:
        expected, count, prefix_utilities = _literal_greedy_oracle(problem)
        actual = methods.raca_him_greedy(problem)
        _require(actual.assignments == expected, "Preserved greedy stage differs from the literal full-pair greedy oracle")
        _require(actual.diagnostics["generated_prefix_count"] == count,
                 "RACA-HIM stopped before all legal greedy prefixes were generated")
        _require(actual.diagnostics["selected_prefix_length"] == len(expected), "Wrong best nonempty prefix length")
        _close(_scalar_measure(problem, actual.assignments)[2], max(prefix_utilities), "Best generated prefix utility")
    result = methods.raca_him_greedy(negative)
    _require(len(result.assignments) == 2 and result.diagnostics["nonpositive_marginals_traversed"] >= 1,
             "RACA-HIM must traverse nonpositive gains and retain its best nonempty negative prefix")
    _require(_scalar_measure(negative, result.assignments)[2] < 0.0, "Negative-prefix fixture unexpectedly lost its negative J")
    result = methods.raca_him_greedy(shorter)
    _require(len(result.assignments) == 1 and result.diagnostics["generated_prefix_count"] == 2,
             "The best prefix can be shorter than B even after all prefixes were generated")
    _require(methods.raca_him_greedy(joint).assignments[0][0] == 12,
             "Joint selection was simplified to descending-risk target selection")


def validate_rf_ts_semantics() -> None:
    """The comparison method must remain assignment-identical to RF-TS v1."""
    fixtures = [
        _problem([14, 11, 13, 12], [102, 101], [0.125, 0.875, 0.5, 0.25],
                 [[15, 6, 7, 3], [4, 18, 5, 10]], 3),
        # Equal risks/completion times must still use temporary-ID ties.
        _problem([12, 11], [102, 101], [0.5, 0.5], [[2, 2], [2, 2]], 2),
        # Skip an unavailable priority target; do not replace the risk rule.
        _problem([11, 12, 13], [101], [0.875, 0.5, 0.25], [[math.inf, 20, 5]], 2),
        # Self verification and per-verifier cumulative capacity remain illegal.
        _problem([101, 11, 12], [102, 101], [0.75, 0.5, 0.25],
                 [[30, 6, 7], [1, 8, math.inf]], 3),
        # Full RF-TS intentionally retains its last negative marginal step.
        _problem([11, 12], [101], [0.875, 0.125], [[1, 7]], 2, deadline=8.0),
        _problem([11], [101], [0.5], [[33]], 1),
        _problem([11], [101], [0.5], [[1]], 0),
        _problem([], [101], [], [[]], 1),
        _problem([11], [], [0.5], [], 1),
    ]
    rng = np.random.default_rng(90217)
    for _ in range(12):
        fixtures.append(_problem([14, 11, 13, 12], [102, 101],
                                 rng.integers(1, 16, size=4) / 16.0,
                                 rng.integers(1, 40, size=(2, 4)), 3))
    for problem in fixtures:
        original = _snapshot(problem)
        expected = _literal_rf_ts_reference(problem)
        actual = methods.risk_first_time_sensitive(problem)
        _require(actual.assignments == expected, "RF-TS assignments changed from the independent v1 reference")
        _require(_snapshot(problem) == original, "RF-TS mutated common risks/costs/IDs/configuration")


def validate_enhanced_raca_candidates() -> None:
    """Check real-J candidate dominance without requiring impossible strict wins."""
    negative = _problem([12, 11], [102, 101], [0.125, 0.125],
                        [[4, 4], [4, 4]], 2, deadline=8.0)
    fixtures = [
        negative,
        _problem([11, 12], [101], [0.875, 0.125], [[1, 7]], 2, deadline=8.0),
        _problem([11, 12], [101], [0.875, 0.75], [[7, 1]], 2, deadline=8.0),
        _problem([101, 11, 12], [102, 101], [0.75, 0.5, 0.25],
                 [[30, 6, 7], [1, 8, math.inf]], 3),
    ]
    rng = np.random.default_rng(73411)
    for _ in range(12):
        fixtures.append(_problem([14, 11, 13, 12], [102, 101],
                                 rng.integers(1, 16, size=4) / 16.0,
                                 rng.integers(1, 40, size=(2, 4)), 3))
    for problem in fixtures:
        original = _snapshot(problem)
        greedy = methods.raca_him_greedy(problem)
        rf = methods.risk_first_time_sensitive(problem)
        actual = methods.raca_him(problem)
        _require(_snapshot(problem) == original, "Enhanced RACA-HIM changed the shared decision problem")
        evaluation = objective.evaluate_joint_objective(problem, actual.assignments)
        _require(evaluation.feasible, "Enhanced RACA-HIM returned an infeasible candidate")
        _close(evaluation.joint_utility, _scalar_measure(problem, actual.assignments)[2],
               "Enhanced RACA-HIM must use the unchanged independent J")
        candidates = [greedy.assignments]
        candidates.extend(rf.assignments[:length] for length in range(1, len(rf.assignments) + 1))
        for candidate in candidates:
            reference = objective.evaluate_joint_objective(problem, candidate)
            if reference.feasible:
                _require(evaluation.joint_utility >= reference.joint_utility,
                         "Enhanced RACA-HIM lost to its preserved greedy or RF-TS candidate")
        _close(actual.diagnostics["final_joint_utility"], evaluation.joint_utility,
               "Reported final J must equal the external evaluator")
        _require(methods.raca_him(problem).assignments == actual.assignments,
                 "Enhanced RACA-HIM must remain deterministic")
        _require(_snapshot(problem) == original, "Repeated enhanced RACA-HIM mutated its input")
    negative_result = methods.raca_him(negative)
    _require(negative_result.assignments and _scalar_measure(negative, negative_result.assignments)[2] < 0.0,
             "The candidate fallback must preserve a feasible nonempty solution when every J is negative")

    # With one target and one legal verifier, RF-TS already attains the unique
    # feasible plan. A correct implementation must permit equality, without an
    # artificial utility offset or a change to the shared objective.
    optimal_tie = _problem([11], [101], [0.75], [[1]], 1, deadline=8.0)
    rf = methods.risk_first_time_sensitive(optimal_tie)
    raca = methods.raca_him(optimal_tie)
    _require(raca.assignments == rf.assignments == ((11, 101),),
             "A unique feasible optimum cannot legitimately yield a strict RACA-HIM advantage")
    _close(_scalar_measure(optimal_tie, raca.assignments)[2],
           _scalar_measure(optimal_tie, rf.assignments)[2], "Correct equality at the unique optimum")

    # Both construction paths put 11 and 12 on 101: J = .5. Reassigning 11 to
    # 102 balances the true loads at (2,2), preserving both risks: J = .5625.
    # This improvement is unavailable through prefix selection alone.
    reassignment = _problem([11, 12], [101, 102], [0.75, 0.625],
                            [[1, 2], [2, 100]], 2, deadline=8.0)
    baseline = methods.risk_first_time_sensitive(reassignment).assignments
    greedy = methods.raca_him_greedy(reassignment).assignments
    improved = methods.raca_him(reassignment)
    construction_values = [_scalar_measure(reassignment, greedy)[2]]
    construction_values.extend(_scalar_measure(reassignment, baseline[:length])[2]
                               for length in range(1, len(baseline) + 1))
    improved_value = _scalar_measure(reassignment, improved.assignments)[2]
    _close(max(construction_values), 0.5, "Hand-built construction fixture")
    _require(improved_value > max(construction_values),
             "Local reassignment failed to improve beyond every construction candidate")
    _close(improved_value, 0.5625, "Hand-built strictly improving real-J reassignment")


def _snapshot(problem) -> str:
    digest = hashlib.sha256()
    for field in fields(problem):
        value = getattr(problem, field.name)
        digest.update(field.name.encode("utf-8"))
        digest.update(value.tobytes() if isinstance(value, np.ndarray) else repr(value).encode("utf-8"))
    return digest.hexdigest()


def validate_five_methods() -> None:
    problem = _problem([14, 11, 13, 12], [102, 101], [0.125, 0.875, 0.5, 0.25],
                       [[15, 6, 7, 3], [4, 18, 5, 10]], 3, deadline=32.0)
    original = _snapshot(problem)
    for name in cfg.METHODS:
        result = methods.run_method(name, problem, seed=88001)
        _require(_snapshot(problem) == original, f"{name} mutated common risks/costs/IDs/configuration")
        expected_reward, expected_span, expected_utility = _scalar_measure(problem, result.assignments)
        evaluation = objective.evaluate_joint_objective(problem, result.assignments)
        _require(evaluation.feasible, f"{name} produced an illegal plan")
        _close(evaluation.risk_reward, expected_reward, f"{name}: independently reconstructed reward")
        _close(evaluation.makespan_ms, expected_span, f"{name}: independently reconstructed parallel loads")
        _close(evaluation.joint_utility, expected_utility, f"{name}: same independent external J")
        repeated = methods.run_method(name, problem, seed=88001)
        _require(repeated.assignments == result.assignments, f"{name} changed on a repeated identical input/seed")
        _require(_snapshot(problem) == original, f"Repeated {name} mutated the shared problem")
    reversed_risks = _problem(problem.target_ids, problem.verifier_ids, problem.risk_scores[::-1],
                              problem.costs, problem.budget, deadline=problem.deadline_ms)
    for name in ("Cost-First", "Random"):
        a = methods.run_method(name, problem, seed=77).assignments
        b = methods.run_method(name, reversed_risks, seed=77).assignments
        _require(a == b, f"{name} must not use risk to rank/sample targets")
    infeasible = _problem([11], [101], [0.5], [[33.0]], 1, deadline=32.0)
    for name in cfg.METHODS:
        result = methods.run_method(name, infeasible, seed=1)
        evaluation = objective.evaluate_joint_objective(infeasible, result.assignments)
        _require(not result.assignments and not evaluation.feasible and evaluation.joint_utility is None,
                 f"{name} must preserve an infeasible scenario without manufacturing J=0")
    for problem, expected_status in (
        (_problem([], [101], [], [[]], 1), "no_legal_target"),
        (_problem([11], [], [0.5], [], 1), "no_legal_verifier"),
        (_problem([11], [101], [0.5], [[1]], 0), "zero_budget"),
    ):
        evaluation = objective.evaluate_joint_objective(problem, ())
        _require(evaluation.status == expected_status, "Distinct unavailable-input cases were merged")


def validate_checkpoint_arithmetic() -> None:
    """Check compact saved assignment evidence in memory; write no checkpoint."""
    storage = importlib.import_module(f"{PACKAGE}.joint_storage")
    problem = _problem([0, 1, 2], [3], [0.25, 0.5, 0.75], [[4, 6, 8]], 2)
    evaluation = objective.evaluate_joint_objective(problem, ((0, 3), (2, 3)))
    record = {**evaluation.to_dict(), "N": 4, "B": 2, "deadline_ms": 32.0}
    evidence = {
        "columns": [[task[field] for task in evaluation.assignment_diagnostics]
                    for field in storage.DIAGNOSTIC_COLUMNS],
        "verifier_loads": {str(key): value for key, value in evaluation.verifier_loads.items()},
    }
    shared = {"gt_verifier_ids": [3]}
    storage.validate_saved_decision(record, evidence, shared)
    corruptions = []
    changed_record = dict(record, joint_utility=record["joint_utility"] + 0.25)
    corruptions.append((changed_record, evidence))
    changed_evidence = deepcopy(evidence)
    changed_evidence["columns"][0][1] = changed_evidence["columns"][0][0]
    corruptions.append((record, changed_evidence))
    changed_evidence = deepcopy(evidence)
    changed_evidence["columns"][3][0] += 1.0
    corruptions.append((record, changed_evidence))
    for candidate_record, candidate_evidence in corruptions:
        try:
            storage.validate_saved_decision(candidate_record, candidate_evidence, shared)
        except ValueError:
            continue
        raise AssertionError("A corrupted saved objective, duplicate ID or pair-cost history was accepted")


def validate_io_retry_policy() -> None:
    """Check bounded lock retries without waiting or using formal result files."""
    storage = importlib.import_module(f"{PACKAGE}.joint_storage")

    # Plain PermissionError also covers Python/platforms without winerror.
    # Other OSError subclasses only qualify for the specific Windows codes.
    retryable = [PermissionError(13, "synthetic permission failure")]
    for code in (5, 32, 33):
        error = OSError(5, f"synthetic Windows lock {code}")
        error.winerror = code
        retryable.append(error)
    for error in retryable:
        calls = []
        sentinel = object()

        def eventually_available():
            calls.append(1)
            if len(calls) <= 2:
                raise error
            return sentinel

        with patch.object(storage.time, "sleep") as sleeper:
            result = storage.retry_io(eventually_available, description="synthetic lock",
                                      delays=(0.01, 0.02))
        _require(result is sentinel and len(calls) == 3,
                 "Two transient locks must allow exactly a third successful attempt")
        _require([call.args for call in sleeper.call_args_list] == [(0.01,), (0.02,)],
                 "Retry waits must use the requested delays without sleeping after success")

    # Missing files, invalid operations and non-I/O exceptions cannot be fixed
    # by waiting for another process to release a Windows file handle.
    nonretryable = [FileNotFoundError(2, "synthetic missing file"),
                    OSError(22, "synthetic invalid operation"),
                    ValueError("synthetic programming error")]
    unrecognized_windows = OSError(5, "synthetic unsupported Windows failure")
    unrecognized_windows.winerror = 123
    nonretryable.append(unrecognized_windows)
    for error in nonretryable:
        calls = []

        def rejected_operation():
            calls.append(1)
            raise error

        with patch.object(storage.time, "sleep") as sleeper:
            try:
                storage.retry_io(rejected_operation, delays=(0.01, 0.02))
            except Exception as caught:
                _require(caught is error, "A nonretryable failure was replaced by another exception")
            else:
                raise AssertionError("A nonretryable failure was suppressed")
        _require(len(calls) == 1 and not sleeper.called,
                 "Nonretryable failures must propagate immediately without a retry or wait")

    original = PermissionError(13, "synthetic permanent lock")
    calls = []

    def permanently_unavailable():
        calls.append(1)
        raise original

    description = "synthetic permanent manifest replacement"
    with patch.object(storage.time, "sleep") as sleeper:
        try:
            storage.retry_io(permanently_unavailable, description=description,
                             delays=(0.01, 0.02))
        except PermissionError as caught:
            _require(caught is original, "Retry exhaustion must preserve the original PermissionError")
            _require(any(description in note for note in getattr(caught, "__notes__", ())),
                     "Retry exhaustion must add the failed operation to an exception note")
        else:
            raise AssertionError("A permanent permission failure was suppressed")
    _require(len(calls) == 3 and sleeper.call_count == 2,
             "Retry exhaustion must stop after one initial attempt and the finite configured retries")


def validate_atomic_json_retry() -> None:
    """Verify real temporary JSON writes retain the old target until commit."""
    storage = importlib.import_module(f"{PACKAGE}.joint_storage")
    real_replace, real_unlink = Path.replace, Path.unlink
    with TemporaryDirectory(prefix="tnse05-io-retry-") as temporary:
        root = Path(temporary)
        target = root / "run_manifest.json"
        old_bytes = b'{"status": "previous-valid-manifest"}\n'
        value = {"status": "replacement", "completed_trials": 318}
        target.write_bytes(old_bytes)
        attempts = []

        def replace_after_two_locks(source, destination):
            destination = Path(destination)
            _require(destination == target and source.parent == target.parent
                     and source.name.startswith(f".{target.name}.") and source.suffix == ".tmp",
                     "Atomic JSON must replace the target from a sibling temporary file")
            _require(target.read_bytes() == old_bytes,
                     "The previous manifest must remain intact until replacement succeeds")
            _require(json.loads(source.read_text(encoding="utf-8")) == value,
                     "The staged temporary JSON is not the complete replacement manifest")
            attempts.append(source)
            if len(attempts) <= 2:
                raise PermissionError(13, "synthetic transient replace lock")
            return real_replace(source, destination)

        with patch.object(Path, "replace", new=replace_after_two_locks), \
                patch.object(storage.time, "sleep") as sleeper:
            storage.atomic_json(target, value)
        _require(len(attempts) == 3 and sleeper.call_count == 2,
                 "atomic_json must retry transient replacement locks and commit once available")
        _require(len(set(attempts)) == 1 and json.loads(target.read_text(encoding="utf-8")) == value,
                 "atomic_json must retry the same complete staged file before committing")
        _require(not list(root.glob(f".{target.name}.*.tmp")),
                 "A successfully committed manifest left its temporary file behind")

        # A completed temporary manifest remains available for manual recovery
        # when replacement retries are exhausted. It must not be discarded.
        target.write_bytes(old_bytes)
        original = PermissionError(13, "synthetic permanent replace lock")
        cleanup_error = PermissionError(13, "synthetic cleanup lock")
        failures, cleanups = [], []

        def locked_replace(source, destination):
            _require(Path(destination) == target and target.read_bytes() == old_bytes,
                     "A failed replacement changed the previous manifest")
            failures.append(source)
            raise original

        def locked_cleanup(source, *args, **kwargs):
            if source.parent == root and source.suffix == ".tmp":
                cleanups.append(source)
                raise cleanup_error
            return real_unlink(source, *args, **kwargs)

        with patch.object(Path, "replace", new=locked_replace), \
                patch.object(Path, "unlink", new=locked_cleanup), \
                patch.object(storage.time, "sleep") as sleeper:
            try:
                storage.atomic_json(target, value)
            except PermissionError as caught:
                _require(caught is original,
                         "Replacement retry exhaustion must preserve the original PermissionError")
                _require(any("recovery" in note.lower() and str(failures[-1]) in note
                             for note in getattr(caught, "__notes__", ())),
                         "A permanently locked JSON replacement must note its recovery file")
            else:
                raise AssertionError("A permanently locked manifest replacement was suppressed")
        _require(1 < len(failures) < 100 and sleeper.call_count == len(failures) - 1,
                 "Permanent atomic JSON replacement must stop after bounded retries")
        _require(not cleanups and target.read_bytes() == old_bytes,
                 "Failed replacement must preserve both the old manifest and the complete staged file")
        _require(len(set(failures)) == 1 and all(
            staged.is_file() and json.loads(staged.read_text(encoding="utf-8")) == value
            for staged in set(failures)), "Completed recovery JSON was removed or damaged")
        for staged in set(failures):
            real_unlink(staged, missing_ok=True)

        # Before a complete JSON file exists, cleanup is best effort. A second
        # permission failure during cleanup must not mask serialization/write
        # failure or touch the previous valid destination.
        real_write_text = Path.write_text
        for failure_stage in ("serialization", "write"):
            original_write = (ValueError("synthetic serialization failure")
                              if failure_stage == "serialization"
                              else OSError(28, "synthetic interrupted JSON write"))
            incomplete, cleanups = [], []

            def interrupted_write(source, data, *args, **kwargs):
                if source.parent == root and source.suffix == ".tmp":
                    real_write_text(source, '{"incomplete":', encoding="utf-8")
                    incomplete.append(source)
                    raise original_write
                return real_write_text(source, data, *args, **kwargs)

            with patch.object(Path, "write_text", new=interrupted_write), \
                    patch.object(Path, "unlink", new=locked_cleanup), \
                    patch.object(Path, "replace") as replacement, \
                    patch.object(storage.time, "sleep") as sleeper:
                serialization = (patch.object(storage.json, "dumps", side_effect=original_write)
                                 if failure_stage == "serialization" else nullcontext())
                with serialization:
                    try:
                        storage.atomic_json(target, value)
                    except Exception as caught:
                        _require(caught is original_write and caught is not cleanup_error,
                                 "Incomplete-file cleanup masked the original JSON generation error")
                    else:
                        raise AssertionError("A JSON serialization/write failure was suppressed")
                _require(not replacement.called and not sleeper.called,
                         "An incomplete JSON write must not be committed or retried for unrelated errors")
            _require(bool(cleanups) and target.read_bytes() == old_bytes,
                     "Incomplete JSON cleanup must be attempted without changing the saved manifest")
            for staged in incomplete:
                real_unlink(staged, missing_ok=True)


def validate_io_patch_resume() -> None:
    """Check the one-time I/O migration with synthetic in-memory identities."""
    driver = importlib.import_module(f"{PACKAGE}.joint_experiment")
    storage = importlib.import_module(f"{PACKAGE}.joint_storage")
    io_names = tuple(driver.IO_PATCH_PREDECESSOR_HASHES)
    old_hashes = {name: f"synthetic-old-io-{index}" for index, name in enumerate(io_names)}
    algorithm_name = "experiments/05_joint_decision/joint_methods.py"
    old_hashes[algorithm_name] = "synthetic-unchanged-algorithm"
    audited_hashes = {name: old_hashes[name] for name in io_names}
    old_code = storage.digest_json(old_hashes)
    current_hashes = dict(old_hashes)
    current_hashes.update({name: f"synthetic-current-io-{index}" for index, name in enumerate(io_names)})
    source = {"code_version": storage.digest_json(current_hashes), "source_sha256": current_hashes}
    identity = {
        "code_version": source["code_version"], "dataset_fingerprint": "synthetic-dataset",
        "protocol": {"synthetic": "unchanged-protocol"}, "algorithm_version": "synthetic-algorithm",
        "numerical_environment": {"python": "synthetic", "numpy": "synthetic"},
    }
    saved = {
        **deepcopy(identity), "run_id": "synthetic-original-run", "code_version": old_code,
        "source_sha256": old_hashes, "run_fingerprint": "synthetic-original-scientific-fingerprint",
        "completed_trials": 318, "checkpoints": {"fixture": {"sha256": "preserved-checkpoint-hash"}},
        "files": {"raw": {"sha256": "preserved-output-hash"}},
    }
    with patch.multiple(driver, IO_PATCH_PREDECESSOR_CODE_VERSION=old_code,
                        IO_PATCH_PREDECESSOR_HASHES=audited_hashes):
        candidate = deepcopy(saved)
        migrated = driver._resume_io_patch(candidate, deepcopy(identity), deepcopy(source))
        _require(migrated is candidate, "An exact audited I/O predecessor should resume the same manifest")
        _require({key: value for key, value in migrated.items() if key != "runtime_patch_history"} == saved,
                 "I/O migration changed original run identity, completed count, checkpoint or export identity")
        patch_history = migrated["runtime_patch_history"]
        _require(len(patch_history) == 1, "One-time I/O migration must append exactly one runtime record")
        record = patch_history[0]
        _require(record["runtime_source_sha256"] == current_hashes
                 and record["runtime_code_version"] == source["code_version"]
                 and record["runtime_run_fingerprint"] == storage.digest_json(identity)
                 and record["changed_source_files"] == sorted(io_names),
                 "I/O migration failed to record its complete runtime source and identity")
        before = deepcopy(migrated)
        repeated = driver._resume_io_patch(migrated, deepcopy(identity), deepcopy(source))
        _require(repeated is migrated and migrated == before,
                 "Resuming the exact recorded I/O runtime must be idempotent")

        for field in ("dataset_fingerprint", "protocol", "algorithm_version", "numerical_environment"):
            changed_identity = deepcopy(identity)
            changed_identity[field] = {"synthetic-change": field}
            _require(driver._resume_io_patch(deepcopy(saved), changed_identity, deepcopy(source)) is None,
                     f"A scientific identity change was incorrectly treated as an I/O repair: {field}")
            _require(driver._resume_io_patch(deepcopy(before), changed_identity, deepcopy(source)) is None,
                     f"A scientific identity change bypassed recorded runtime checks: {field}")

        corrupt_saved = deepcopy(saved)
        corrupt_saved["code_version"] = "unaudited-predecessor"
        _require(driver._resume_io_patch(corrupt_saved, deepcopy(identity), deepcopy(source)) is None,
                 "An unaudited predecessor code version was allowed to resume")
        corrupt_saved = deepcopy(saved)
        corrupt_saved["source_sha256"][io_names[0]] = "tampered-predecessor-hash"
        _require(driver._resume_io_patch(corrupt_saved, deepcopy(identity), deepcopy(source)) is None,
                 "A predecessor source table inconsistent with its digest was accepted")

        unchanged_source = {"code_version": old_code, "source_sha256": deepcopy(old_hashes)}
        _require(driver._resume_io_patch(deepcopy(saved), deepcopy(identity), unchanged_source) is None,
                 "An unchanged source must not create an I/O patch history")
        for changed_name in (algorithm_name, "synthetic-added-file.py"):
            changed_source = deepcopy(source)
            changed_source["source_sha256"][changed_name] = "synthetic-unrelated-change"
            changed_source["code_version"] = storage.digest_json(changed_source["source_sha256"])
            _require(driver._resume_io_patch(deepcopy(saved), deepcopy(identity), changed_source) is None,
                     "A changed algorithm or source-file set was accepted by the I/O migration")

        # The migration permits this exact runtime once. Even another edit to
        # one of the I/O files must return to the normal new-run policy.
        changed_source = deepcopy(source)
        changed_source["source_sha256"][io_names[0]] = "synthetic-later-io-edit"
        changed_source["code_version"] = storage.digest_json(changed_source["source_sha256"])
        _require(driver._resume_io_patch(deepcopy(before), deepcopy(identity), changed_source) is None,
                 "Recorded I/O history became a permanent whitelist for later source edits")
        for field in ("patch_version", "runtime_code_version", "runtime_source_sha256", "runtime_run_fingerprint"):
            corrupt_history = deepcopy(before)
            corrupt_history["runtime_patch_history"][-1][field] = "synthetic-tampering"
            _require(driver._resume_io_patch(corrupt_history, deepcopy(identity), deepcopy(source)) is None,
                     f"A recorded runtime identity mismatch was accepted: {field}")


def validate_restart_archiving() -> None:
    """Preserve old outputs and checkpoint bytes using temporary fixtures only."""
    storage = importlib.import_module(f"{PACKAGE}.joint_storage")
    with TemporaryDirectory(prefix="tnse05-archive-") as temporary:
        root = Path(temporary) / "results"
        manifest = {"run_id": "previous-validation-run", "run_fingerprint": "old-fingerprint",
                    "status": "complete", "checkpoints": {}, "files": {}}
        contents = {
            "raw/joint_runs.csv": b"method,joint_utility\nRF-TS,0.25\n",
            "raw/nested/assignment.csv": b"target,verifier\n1,2\n",
            "summary/joint_summary.csv": b"method,mean\nRF-TS,0.25\n",
            "figures/fixture.png": b"synthetic figure bytes; not an actual plot",
            "logs/execution.log": b"previous synthetic execution log\n",
            "logs/run_manifest.json": (json.dumps(manifest) + "\n").encode("utf-8"),
        }
        checkpoint = root / "logs" / "checkpoints" / manifest["run_id"] / "fixture.json.gz"
        checkpoint.parent.mkdir(parents=True)
        checkpoint.write_bytes(b"previous immutable checkpoint fixture")
        for relative, data in contents.items():
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)

        previous_archives = []
        for _ in range(2):
            result = storage.archive_previous_run(root, manifest, "fingerprint_changed")
            _require(result["previous_run_id"] == manifest["run_id"]
                     and result["previous_run_fingerprint"] == manifest["run_fingerprint"]
                     and result["reason"] == "fingerprint_changed", "Archive linkage lost the old run identity")
            archive = root / result["archive_path"]
            _require(archive.resolve().is_relative_to(root.resolve()) and archive.is_dir(),
                     "Archive must be created inside the selected temporary result root")
            _require(archive not in previous_archives, "Repeated archives overwrite an earlier backup")
            for relative, data in contents.items():
                _require((root / relative).read_bytes() == data, "Archiving modified an active old output")
                _require((archive / relative).read_bytes() == data, "An archived output differs from its source")
            _require(checkpoint.read_bytes() == b"previous immutable checkpoint fixture",
                     "Archiving modified or removed an old checkpoint")
            _require(not (archive / "logs" / "checkpoints").exists(),
                     "Old checkpoints should remain at their original run-scoped path")
            archive_manifest = json.loads((archive / "archive_manifest.json").read_text(encoding="utf-8"))
            _require(all(archive_manifest.get(key) == value for key, value in result.items())
                     and archive_manifest["status"] == "complete"
                     and archive_manifest["checkpoint_copy_policy"] == "retained_in_original_location"
                     and (root / result["checkpoint_dir"]) == checkpoint.parent,
                     "Archive manifest lost its completed identity or retained checkpoint location")
            expected_identities = {
                relative: {"sha256": hashlib.sha256(data).hexdigest(), "size_bytes": len(data)}
                for relative, data in contents.items()
            }
            _require(archive_manifest["source_files"] == expected_identities,
                     "Archive manifest hashes/sizes differ from independently measured source bytes")
            previous_archives.append(archive)
        for archive in previous_archives:
            for relative, data in contents.items():
                _require(hashlib.sha256((archive / relative).read_bytes()).hexdigest()
                         == hashlib.sha256(data).hexdigest(), "A previous archive was changed by a later backup")

        try:
            storage.archive_previous_run(root, dict(manifest, run_id="../outside"), "invalid_run_id")
        except ValueError:
            pass
        else:
            raise AssertionError("A checkpoint path escaping the result root was accepted")

        outside = Path(temporary) / "outside.csv"
        outside.write_bytes(b"external file must remain untouched")
        link = root / "raw" / "external.csv"
        try:
            link.symlink_to(outside)
        except (OSError, NotImplementedError) as error:
            print(f"Skipped unavailable symlink fixture: {type(error).__name__}", flush=True)
        else:
            try:
                storage.archive_previous_run(root, manifest, "unsafe_symlink")
            except ValueError:
                pass
            else:
                raise AssertionError("An output symlink escaping the result root was archived")
            finally:
                link.unlink(missing_ok=True)
            _require(outside.read_bytes() == b"external file must remain untouched",
                     "Rejected archive modified a file outside its result root")


def validate_manifest_restart_policy() -> None:
    """Exercise resume/version-change/force decisions without loading data or scheduling."""
    driver = importlib.import_module(f"{PACKAGE}.joint_experiment")
    storage = importlib.import_module(f"{PACKAGE}.joint_storage")
    source = {"code_version": "synthetic-code-v1", "source_sha256": {"fixture.py": "v1"}}
    environment = {"python": "synthetic-python", "platform": "synthetic-platform",
                   "packages": {"numpy": "synthetic-numpy", "scipy": "synthetic-scipy"}}
    with TemporaryDirectory(prefix="tnse05-restart-") as temporary, patch.multiple(
            driver, source_identity=lambda: deepcopy(source),
            dataset_identity=lambda: {"dataset_fingerprint": "synthetic-dataset"},
            environment_metadata=lambda: deepcopy(environment), threadpool_info=lambda: []):
        root = Path(temporary) / "results"
        manifest_path = root / "logs" / "run_manifest.json"
        first = driver._prepare_manifest(root, False)
        key = "synthetic-checkpoint"
        checkpoint = storage.checkpoint_path(root, first["run_id"], key)
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        checkpoint.write_bytes(b"old run checkpoint; never a new run input")
        first["checkpoints"][key] = storage.file_identity(checkpoint, root)
        first["completed_trials"] = 1
        storage.atomic_json(manifest_path, first)
        before = manifest_path.read_bytes()
        saved = json.loads(before)
        resumed = driver._prepare_manifest(root, False)
        _require(resumed == saved and manifest_path.read_bytes() == before,
                 "Identical fingerprints must resume their existing run and checkpoints")

        source.update(code_version="synthetic-code-v2", source_sha256={"fixture.py": "v2"})
        changed = driver._prepare_manifest(root, False)
        _require(changed["run_fingerprint"] != first["run_fingerprint"],
                 "A code change did not produce a new run fingerprint")
        _require(changed["run_id"] != first["run_id"] and not changed["checkpoints"]
                 and changed["completed_trials"] == 0 and not changed["files"],
                 "A changed run reused old checkpoint/results bookkeeping")
        _require(changed["previous_run"]["previous_run_id"] == first["run_id"]
                 and changed["previous_run"]["reason"] == "identity_changed"
                 and changed["previous_run"]["changed_identity_fields"] == ["code_version"],
                 "The new run did not link to its archived predecessor and changed identity")
        archive = root / changed["previous_run"]["archive_path"]
        _require((archive / "logs" / "run_manifest.json").read_bytes() == before,
                 "The archived manifest differs from the previous saved manifest")
        storage.atomic_json(manifest_path, changed)
        forced = driver._prepare_manifest(root, True)
        _require(forced["run_fingerprint"] == changed["run_fingerprint"]
                 and forced["run_id"] not in {first["run_id"], changed["run_id"]}
                 and not forced["checkpoints"] and forced["completed_trials"] == 0,
                 "Force must start a fresh run even when its fingerprint is unchanged")
        _require(forced["previous_run"]["previous_run_id"] == changed["run_id"]
                 and forced["previous_run"]["reason"] == "forced_restart"
                 and not forced["previous_run"]["changed_identity_fields"],
                 "Forced restart lost the saved predecessor identity")
        _require(checkpoint.read_bytes() == b"old run checkpoint; never a new run input"
                 and not storage.checkpoint_path(root, changed["run_id"], key).exists()
                 and not storage.checkpoint_path(root, forced["run_id"], key).exists(),
                 "New manifests reused or changed an old run-scoped checkpoint")


@contextmanager
def _replace_loader(loader):
    original = inputs.load_group
    inputs.load_group = loader
    try:
        yield
    finally:
        inputs.load_group = original


class _DescriptorsOnly:
    """A loader boundary that fails on every field except the allowed X."""
    def __init__(self, values):
        self.values = values

    def __getitem__(self, key):
        if key != "X":
            raise AssertionError(f"Scheduling adapter read a forbidden/unneeded dataset field: {key}")
        return self.values

    def get(self, key, default=None):
        return self[key]


def validate_shared_inputs_and_resources() -> None:
    n = min(cfg.N_VALUES)
    synthetic_x = np.zeros((n, cfg.D), dtype=np.float32)
    synthetic_x[:, 0] = 1.0
    calls = []

    def synthetic_loader(**kwargs):
        calls.append(kwargs)
        return _DescriptorsOnly(synthetic_x)

    # Exercise the real adapter with a poison loader; no dataset file is read.
    # GT is actually constructed from identical nonzero synthetic descriptors.
    with _replace_loader(synthetic_loader):
        first = inputs.prepare_trial_inputs(n, cfg.SEEDS[0])
        second = inputs.prepare_trial_inputs(n, cfg.SEEDS[0])
    _require(len(calls) == 2 and all(call["n"] == n and call["dim"] == 30
                                   and call["corruption_rate_percent"] == 3.0
                                   and call["H"] == 0.5 for call in calls),
             "The loader did not receive the published descriptor configuration")
    _require(first.ground_truth.success and len(first.verifier_rows) == cfg.M,
             "Shared GT must actually construct M trusted models from descriptors")
    _require(len(first.target_rows) == n - cfg.M
             and not np.intersect1d(first.target_rows, first.verifier_rows).size,
             "GT and target pools do not partition the population")
    _require(np.array_equal(first.model_ids, second.model_ids)
             and np.array_equal(first.history.risk_scores, second.history.risk_scores)
             and np.array_equal(first.verifier_rows, second.verifier_rows),
             "Common history/temporary IDs/GT changed for identical label-free inputs")
    components = inputs.resource_cost_components_for_sigma(first, 0.0)
    repeated = inputs.resource_cost_components_for_sigma(second, 0.0)
    expected_transfer = (4.0 * 30.0 * 8.0 / 50_000_000.0) * 1000.0
    _require(np.allclose(components.compute_ms, 4.0, rtol=1e-6), "sigma=0 compute center is not 4 ms")
    _require(np.allclose(components.transfer_ms, expected_transfer, rtol=1e-6), "sigma=0 float32 descriptor transfer formula differs")
    _require(np.allclose(components.network_ms, 10.0, rtol=1e-6), "sigma=0 network center is not 10 ms")
    _require(np.allclose(components.total_ms, 14.0 + expected_transfer, rtol=1e-6),
             "sigma=0 total must include each resource component exactly once")
    _require(np.array_equal(components.total_ms, repeated.total_ms), "Common pair costs changed across identical inputs")
    for name in ("compute_ms", "transfer_ms", "network_ms", "total_ms"):
        values = getattr(components, name)
        _require(np.all(np.isfinite(values) & (values >= 0.0)) and not values.flags.writeable,
                 f"Resource {name} must be finite, nonnegative and immutable")
    costs = inputs.resource_cost_components_for_sigma(first, 0.5)
    again = inputs.resource_cost_components_for_sigma(second, 0.5)
    _require(np.array_equal(costs.total_ms, again.total_ms), "Heterogeneous costs changed for the shared resource seed")
    _require(np.allclose(costs.total_ms, costs.compute_ms + costs.transfer_ms + costs.network_ms),
             "Heterogeneous costs count the network component more than once")
    # Exercise the actual resource-to-decision boundary, including float32
    # component rounding. No scheduling method runs on this descriptor fixture.
    problem = objective.JointProblem(
        target_ids=first.target_ids[:4], verifier_ids=first.verifier_ids[:2],
        risk_scores=first.history.risk_scores[first.target_rows[:4]],
        costs=costs.total_ms[:2, :4], budget=2,
        compute_costs=costs.compute_ms[:2, :4],
        transfer_costs=costs.transfer_ms[:2, :4],
        network_costs=costs.network_ms[:2, :4],
    )
    _require(np.array_equal(problem.costs, costs.total_ms[:2, :4]),
             "Decision input changed the Dataset's true pair costs")


def validate_ground_truth_domain() -> None:
    """Synthetic admission/failure checks beyond the old 100-candidate pool."""
    from experiments.common.dataset_adapter import AlgorithmInput

    n = min(cfg.N_VALUES)
    basis = np.eye(cfg.D, dtype=np.float32)

    def observations(x):
        return AlgorithmInput(X=x, model_ids=np.arange(n, dtype=np.int64) + 123_000,
                              history=np.arange(n, dtype=np.float64) / n,
                              reference_indices=np.arange(n, dtype=np.int64),
                              evaluation_indices=np.empty(0, dtype=np.int64))

    # The first 100 rows have no quorum. A coherent tail reaches the required
    # 2/3 support at K=300, exercising the large-candidate block implementation.
    x = np.repeat(basis[0:1], n, axis=0)
    x[:100] = basis[1 + np.arange(100) % (cfg.D - 1)]
    admitted = inputs.build_shared_ground_truth(observations(x))
    sizes = [step.candidate_size for step in admitted.expansion_trace]
    _require(admitted.success and sizes == list(range(50, 301, 25)),
             "GT must expand through the full model domain until the first genuine quorum succeeds")
    _require(len(admitted.selected_indices) == 25 and np.all(admitted.selected_indices >= 100),
             "GT selected unsupported early rows instead of the coherent tail")
    # Thirty mutually orthogonal groups are each far below quorum at every
    # candidate size. The public adapter must exhaust N and preserve failure.
    failed_x = basis[np.arange(n) % cfg.D]
    failed = inputs.build_shared_ground_truth(observations(failed_x))
    failed_sizes = [step.candidate_size for step in failed.expansion_trace]
    _require(not failed.success and failed.selected_indices.size == 0,
             "A synthetic no-quorum domain was incorrectly admitted")
    _require(failed_sizes == list(range(50, n + 1, 25))
             and len(failed.candidate_indices) == n,
             "GT failure must exhaust all N candidates without a hidden 100-candidate cap")
    # Inspect the real runner's common-failure branch without entering a method
    # or exporting anything. All five outcomes must retain the same GT failure.
    driver = importlib.import_module(f"{PACKAGE}.joint_experiment")
    unavailable = SimpleNamespace(
        ground_truth=failed, target_rows=np.empty(0, dtype=np.int64),
        verifier_ids=np.empty(0, dtype=np.int64), descriptor_fingerprint="synthetic-no-quorum",
        to_manifest=lambda: {"gt": {"success": False}},
    )
    identity = {"run_id": "in-memory-validation", "code_version": "synthetic",
                "algorithm_version": cfg.ALGORITHM_VERSION,
                "dataset_fingerprint": "synthetic", "run_fingerprint": "synthetic"}
    payload = driver._execute_trial(identity, unavailable, n, cfg.SEEDS[0], 0.10, 0.50, None, None)
    _require([row["method"] for row in payload["records"]] == list(cfg.METHODS)
             and all(not row["feasible"] and row["status"] == "gt_unavailable"
                     and row["joint_utility"] is None for row in payload["records"]),
             "A shared GT failure did not yield five explicit unavailable outcomes")


def validate_dataset_reproducibility() -> None:
    """Optional read-only descriptor check; no history, GT or optimizer is run."""
    from dataset.load_dataset import load_group

    kwargs = dict(dataset_root=cfg.DATASET_ROOT, n=1000,
                  corruption_rate_percent=3.0, dim=30, H=0.5, seed=2026)
    left = load_group(**kwargs)
    right = load_group(**kwargs)
    a, b = left["X"], right["X"]
    _require(a.shape == (1000, 30) and a.dtype == np.float32,
             "Dataset descriptors have the wrong published shape or precision")
    _require(np.array_equal(a, b), "The same published dataset seed does not reproduce identical descriptors")
    _require(np.all(np.isfinite(a)), "Dataset descriptors contain nonfinite values")


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-dataset", action="store_true",
                        help="Also read the same N=1000 dataset descriptor group twice, without running GT/optimization")
    parser.add_argument("--check-io-only", action="store_true",
                        help="Only check file-lock retries and I/O migration using isolated temporary/in-memory fixtures")
    args = parser.parse_args(argv)
    if args.check_io_only and args.check_dataset:
        parser.error("--check-io-only cannot be combined with --check-dataset")
    checks = (
        validate_source_boundaries, validate_historical_observations,
        validate_independent_objective, validate_joint_greedy_oracle,
        validate_rf_ts_semantics, validate_enhanced_raca_candidates,
        validate_five_methods, validate_checkpoint_arithmetic,
        validate_io_retry_policy, validate_atomic_json_retry, validate_io_patch_resume,
        validate_restart_archiving, validate_manifest_restart_policy, validate_shared_inputs_and_resources,
        validate_ground_truth_domain,
    )
    if args.check_io_only:
        checks = (validate_io_retry_policy, validate_atomic_json_retry, validate_io_patch_resume)
    elif args.check_dataset:
        checks += (validate_dataset_reproducibility,)
    for check in checks:
        check()
        print(f"Passed: {check.__name__}", flush=True)
    print("Requested invariant checks completed. No formal experiment results or figures were generated.", flush=True)


if __name__ == "__main__":
    main()
