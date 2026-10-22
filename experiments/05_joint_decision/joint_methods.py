"""Five distinct schedulers for Group 05, with no access to current labels.

The DVA-P implementation is an explicitly declared single-round extension of
the source's binary subset algorithm, not a claim of native peer-assignment
reproduction. See reports/05_joint_decision_baseline_adaptation.md.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .config_joint_decision import (
    RACA_ALGORITHM_VERSION,
    RACA_IMPROVEMENT_EPS as RACA_IMPROVEMENT_TOLERANCE,
    RACA_LOCAL_SEARCH_MAX_STEPS,
    RACA_REPLACEMENT_POOL_LIMIT,
)
from .joint_objective import Assignment, JointProblem, evaluate_joint_objective


@dataclass(frozen=True)
class MethodResult:
    assignments: tuple[Assignment, ...]
    status: str
    diagnostics: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class _WorkingProblem:
    targets: np.ndarray
    verifiers: np.ndarray
    risk: np.ndarray
    cost: np.ndarray
    allowed: np.ndarray
    budget: int
    deadline: float
    lambda_value: float


def _prepare(problem: JointProblem) -> _WorkingProblem:
    """Sort only by temporary IDs so all exact ties are label independent."""
    target_order = np.argsort(problem.target_ids, kind="stable")
    verifier_order = np.argsort(problem.verifier_ids, kind="stable")
    targets = problem.target_ids[target_order]
    verifiers = problem.verifier_ids[verifier_order]
    cost = problem.costs[np.ix_(verifier_order, target_order)]
    allowed = np.isfinite(cost) & (verifiers[:, None] != targets[None, :])
    return _WorkingProblem(
        targets, verifiers, problem.risk_scores[target_order], cost, allowed,
        int(problem.budget), float(problem.deadline_ms), float(problem.lambda_value),
    )


def _empty_reason(w: _WorkingProblem) -> str | None:
    if not w.targets.size:
        return "no_legal_target"
    if not w.verifiers.size:
        return "no_legal_verifier"
    if w.budget == 0:
        return "zero_budget"
    if not np.any(w.allowed):
        return "no_legal_verifier"
    if not np.any(w.allowed & (w.cost <= w.deadline)):
        return "deadline_infeasible"
    return None


def _pairs(w: _WorkingProblem, selected: list[tuple[int, int]]) -> tuple[Assignment, ...]:
    return tuple((int(w.targets[i]), int(w.verifiers[g])) for i, g in selected)


def raca_him_greedy(problem: JointProblem) -> MethodResult:
    """Manuscript IV-C: joint marginal J, then best nonempty feasible prefix.

    Each matrix cell represents the same pair examined by the literal greedy
    loop. Vectorization changes neither its feasible set nor marginal objective.
    Exact marginal ties use ascending target ID then ascending verifier ID.
    Negative marginal gains do not stop construction, and the empty prefix is
    never compared with the generated nonempty prefixes.
    """
    w = _prepare(problem)
    reason = _empty_reason(w)
    if reason is not None:
        return MethodResult((), reason, {"termination": reason})
    loads = np.zeros(w.verifiers.size, dtype=np.float64)
    selected_mask = np.zeros(w.targets.size, dtype=bool)
    selected: list[tuple[int, int]] = []
    makespan, reward = 0.0, 0.0
    best_utility, best_length = -np.inf, 0
    nonpositive_gains = 0
    final_utility = None
    termination = "budget_reached"
    for _ in range(min(w.budget, w.targets.size)):
        completion = loads[:, None] + w.cost
        new_makespan = np.maximum(makespan, completion)
        feasible = w.allowed & (~selected_mask[None, :]) & (new_makespan <= w.deadline)
        if not np.any(feasible):
            termination = "no_remaining_deadline_feasible_pair"
            break
        gains = w.risk[None, :] / w.budget - w.lambda_value * (
            new_makespan - makespan
        ) / w.deadline
        gains[~feasible] = -np.inf
        # Target-major flattening supplies the declared target/verifier ID tie.
        flat = int(np.argmax(gains.T))
        i, g = divmod(flat, w.verifiers.size)
        marginal_gain = float(gains[g, i])
        nonpositive_gains += int(marginal_gain <= 0.0)
        selected.append((i, g))
        selected_mask[i] = True
        loads[g] = completion[g, i]
        makespan = float(np.max(loads))
        reward += float(w.risk[i])
        final_utility = reward / w.budget - w.lambda_value * makespan / w.deadline
        if final_utility > best_utility:
            best_utility, best_length = final_utility, len(selected)
    if best_length == 0:
        return MethodResult((), "deadline_infeasible", {"termination": termination})
    return MethodResult(_pairs(w, selected[:best_length]), "ok", {
        "decision_rule": "maximum exact marginal J over every feasible target-verifier pair",
        "prefix_policy": "maximum J over all generated nonempty feasible prefixes; earliest exact tie",
        "generated_prefix_count": len(selected),
        "selected_prefix_length": best_length,
        "nonpositive_marginals_traversed": nonpositive_gains,
        "internal_best_prefix_utility": float(best_utility),
        "internal_final_prefix_utility": final_utility,
        "termination": termination,
        "tie_breaking": "ascending temporary target ID, then verifier ID",
    })


def _risk_priority_assignment(w: _WorkingProblem) -> tuple[list[tuple[int, int]], int]:
    """The unchanged RF-TS construction, reusable without a baseline result.

    RACA constructs its own candidate from common decision inputs. It neither
    reads nor repairs the RF-TS result recorded by the experiment runner.
    """
    loads = np.zeros(w.verifiers.size, dtype=np.float64)
    selected: list[tuple[int, int]] = []
    order = np.lexsort((w.targets, -w.risk))
    skipped = 0
    for position in order:
        i = int(position)
        completion = loads + w.cost[:, i]
        legal = w.allowed[:, i] & (completion <= w.deadline)
        if not np.any(legal):
            skipped += 1
            continue
        g = int(np.argmin(np.where(legal, completion, np.inf)))
        selected.append((i, g))
        loads[g] = completion[g]
        if len(selected) == w.budget:
            break
    return selected, skipped


def risk_first_time_sensitive(problem: JointProblem) -> MethodResult:
    """Two stages: descending risk priority, then minimum completion time.

    This baseline keeps its original stopping rule and assignment decisions;
    objective-based prefix selection or refinement belongs only to RACA.
    """
    w = _prepare(problem)
    reason = _empty_reason(w)
    if reason is not None:
        return MethodResult((), reason, {"termination": reason})
    selected, skipped = _risk_priority_assignment(w)
    return MethodResult(_pairs(w, selected), "ok" if selected else "deadline_infeasible", {
        "target_rule": "descending risk then temporary target ID",
        "assignment_rule": "minimum current verifier load + pair cost; verifier ID tie",
        "targets_skipped_for_deadline": skipped,
        "termination": "budget_reached" if len(selected) == w.budget else "priority_list_exhausted",
    })


def _best_risk_priority_prefix(
    w: _WorkingProblem, selected: list[tuple[int, int]],
) -> list[tuple[int, int]]:
    """Find the best nonempty RF construction prefix under the common J.

    Prefix scanning uses exactly the same reward/load sums as construction.
    Its returned candidate is independently evaluated before RACA compares it.
    """
    loads = np.zeros(w.verifiers.size, dtype=np.float64)
    reward, best_utility, best_length = 0.0, -np.inf, 0
    for length, (i, g) in enumerate(selected, start=1):
        reward += float(w.risk[i])
        loads[g] += float(w.cost[g, i])
        utility = reward / w.budget - w.lambda_value * float(loads.max()) / w.deadline
        if utility > best_utility:
            best_utility, best_length = utility, length
    return selected[:best_length]


def _refine_raca_candidate(
    problem: JointProblem, w: _WorkingProblem, assignments: tuple[Assignment, ...],
) -> tuple[tuple[Assignment, ...], dict[str, Any]]:
    """Bounded, deterministic, strictly improving feasible one-move search.

    Reassignment, addition and nonempty removal examine every corresponding
    single move. Replacement examines declared finite incoming/outgoing pools.
    Array formulas screen moves; the public evaluator must accept each actual
    move with a strict J increase before the incumbent changes. The initial
    feasible candidate is retained even if screening/evaluation ever disagree.
    """
    target_position = {int(value): i for i, value in enumerate(w.targets)}
    verifier_position = {int(value): g for g, value in enumerate(w.verifiers)}
    selected = [(target_position[i], verifier_position[g]) for i, g in assignments]
    current = assignments
    evaluation = evaluate_joint_objective(problem, current)
    if not evaluation.feasible or evaluation.joint_utility is None:
        raise ValueError("RACA local refinement requires a nonempty feasible candidate")
    initial_utility = float(evaluation.joint_utility)
    move_counts = {kind: 0 for kind in ("reassignment", "addition", "removal", "replacement")}
    screened_counts = {kind: 0 for kind in move_counts}
    accepted_utilities: list[float] = []
    accepted_moves: list[dict[str, Any]] = []
    independent_checks = 0
    termination = "iteration_limit_reached"
    for _ in range(RACA_LOCAL_SEARCH_MAX_STEPS):
        current_utility = float(evaluation.joint_utility)
        reward = float(evaluation.risk_reward)
        loads = np.asarray([evaluation.verifier_loads[int(g)] for g in w.verifiers])
        makespan = float(evaluation.makespan_ms)
        old_i = np.asarray([i for i, _ in selected], dtype=np.int64)
        old_g = np.asarray([g for _, g in selected], dtype=np.int64)
        old_cost = w.cost[old_g, old_i]
        used = np.zeros(w.targets.size, dtype=bool)
        used[old_i] = True
        # Nonnegative costs ensure removing a task cannot violate a deadline.
        after_removal = np.broadcast_to(loads, (len(selected), loads.size)).copy()
        after_removal[np.arange(len(selected)), old_g] -= old_cost
        after_removal = np.maximum(after_removal, 0.0)
        remaining_makespan = np.max(after_removal, axis=1)
        removal_utility = ((reward - w.risk[old_i]) / w.budget
                           - w.lambda_value * remaining_makespan / w.deadline)
        best_score = current_utility
        best_move: tuple[str, int, int, int] | None = None

        def consider(score: float, kind: str, row: int, i: int, g: int) -> None:
            nonlocal best_score, best_move
            if score > best_score + RACA_IMPROVEMENT_TOLERANCE:
                best_score, best_move = score, (kind, row, i, g)

        # Reassign one existing target to any legal verifier. Target risk and
        # cardinality stay fixed; only the true maximum verifier load changes.
        completion = after_removal + w.cost[:, old_i].T
        reassignment_makespan = np.maximum(remaining_makespan[:, None], completion)
        legal = w.allowed[:, old_i].T & (reassignment_makespan <= w.deadline)
        legal[np.arange(len(selected)), old_g] = False
        utilities = reward / w.budget - w.lambda_value * reassignment_makespan / w.deadline
        utilities[~legal] = -np.inf
        screened_counts["reassignment"] += int(legal.sum())
        if np.any(legal):
            row, g = np.unravel_index(int(np.argmax(utilities)), utilities.shape)
            consider(float(utilities[row, g]), "reassignment", int(row), int(old_i[row]), int(g))

        # An addition is considered only below the common cardinality budget.
        if len(selected) < w.budget and np.any(~used):
            completion = loads[:, None] + w.cost
            next_makespan = np.maximum(makespan, completion)
            legal = w.allowed & ~used[None, :] & (next_makespan <= w.deadline)
            utilities = ((reward + w.risk[None, :]) / w.budget
                         - w.lambda_value * next_makespan / w.deadline)
            utilities[~legal] = -np.inf
            screened_counts["addition"] += int(legal.sum())
            if np.any(legal):
                i, g = divmod(int(np.argmax(utilities.T)), w.verifiers.size)
                consider(float(utilities[g, i]), "addition", -1, i, g)

        # Keep the manuscript's nonempty constraint, even for negative J.
        if len(selected) > 1:
            row = int(np.argmax(removal_utility))
            screened_counts["removal"] += len(selected)
            consider(float(removal_utility[row]), "removal", row, int(old_i[row]), int(old_g[row]))

        available = np.flatnonzero(~used & np.any(w.allowed & (w.cost <= w.deadline), axis=0))
        if available.size:
            # Fixed pools prevent a hidden O(B*N*G) replacement scan per step.
            # Incoming: equal shares of highest risk and highest singleton J,
            # followed by descending risk to fill any duplicate slots. Outgoing:
            # best J after deletion. IDs resolve every exact score tie.
            risk_order = available[np.lexsort((w.targets[available], -w.risk[available]))]
            minimum_cost = np.min(np.where(w.allowed[:, available], w.cost[:, available], np.inf), axis=0)
            singleton_utility = (w.risk[available] / w.budget
                                 - w.lambda_value * minimum_cost / w.deadline)
            singleton_order = available[np.lexsort((w.targets[available], -singleton_utility))]
            half = RACA_REPLACEMENT_POOL_LIMIT // 2
            pooled = np.concatenate((risk_order[:half], singleton_order[:half], risk_order))
            _, first_occurrence = np.unique(pooled, return_index=True)
            incoming = pooled[np.sort(first_occurrence)[:RACA_REPLACEMENT_POOL_LIMIT]]
            # Sort IDs after pool selection for stable target/verifier tie order.
            incoming.sort()
            outgoing = np.lexsort((w.targets[old_i], -removal_utility))[:RACA_REPLACEMENT_POOL_LIMIT]
            replacement_costs = w.cost[:, incoming]
            for row in outgoing:
                completion = after_removal[row, :, None] + replacement_costs
                next_makespan = np.maximum(remaining_makespan[row], completion)
                legal = w.allowed[:, incoming] & (next_makespan <= w.deadline)
                utilities = ((reward - w.risk[old_i[row]] + w.risk[incoming][None, :]) / w.budget
                             - w.lambda_value * next_makespan / w.deadline)
                utilities[~legal] = -np.inf
                screened_counts["replacement"] += int(legal.sum())
                if np.any(legal):
                    column, g = divmod(int(np.argmax(utilities.T)), w.verifiers.size)
                    consider(float(utilities[g, column]), "replacement", int(row), int(incoming[column]), g)

        if best_move is None:
            termination = "no_screened_strictly_improving_move"
            break
        kind, row, i, g = best_move
        trial = list(selected)
        if kind == "addition":
            trial.append((i, g))
        elif kind == "removal":
            trial.pop(row)
        else:
            trial[row] = (i, g)
        trial_assignments = _pairs(w, trial)
        trial_evaluation = evaluate_joint_objective(problem, trial_assignments)
        independent_checks += 1
        if (not trial_evaluation.feasible or trial_evaluation.joint_utility is None
                or trial_evaluation.joint_utility <= current_utility + RACA_IMPROVEMENT_TOLERANCE):
            termination = "independent_evaluator_rejected_screened_move"
            break
        accepted_moves.append({
            "type": kind,
            "removed_target_id": int(w.targets[selected[row][0]]) if kind in ("removal", "replacement") else None,
            "target_id": int(w.targets[i]),
            "verifier_id": int(w.verifiers[g]),
            "joint_utility_before": current_utility,
            "joint_utility_after": float(trial_evaluation.joint_utility),
        })
        selected, current, evaluation = trial, trial_assignments, trial_evaluation
        move_counts[kind] += 1
        accepted_utilities.append(float(evaluation.joint_utility))
    return current, {
        "initial_utility": initial_utility,
        "final_utility": float(evaluation.joint_utility),
        "max_steps": RACA_LOCAL_SEARCH_MAX_STEPS,
        "replacement_pool_limit_per_side": RACA_REPLACEMENT_POOL_LIMIT,
        "improvement_tolerance": RACA_IMPROVEMENT_TOLERANCE,
        "accepted_move_count": sum(move_counts.values()),
        "accepted_move_counts": move_counts,
        "screened_feasible_move_counts": screened_counts,
        "independent_move_evaluations": independent_checks,
        "accepted_utilities": accepted_utilities,
        "accepted_moves": accepted_moves,
        "replacement_pool_policy": "incoming high risk / high singleton J plus risk fill; outgoing highest deletion J; fixed per-side limit above",
        "neighborhood": "all single reassignments, additions, nonempty removals; bounded single target replacements",
        "screening_bounds_per_step": {
            "reassignment_pair_slots": int(min(w.budget, w.targets.size) * w.verifiers.size),
            "addition_pair_slots": int(w.targets.size * w.verifiers.size),
            "removal_slots": int(min(w.budget, w.targets.size)),
            "replacement_pair_slots": int(
                RACA_REPLACEMENT_POOL_LIMIT ** 2 * w.verifiers.size
            ),
        },
        "tie_breaking": "move order reassignment/addition/removal/replacement; deterministic row order and temporary IDs",
        "termination": termination,
    }


def raca_him(problem: JointProblem) -> MethodResult:
    """RACA v2: feasible candidate construction followed by bounded improvement.

    GT selection, common risk/resource inputs, budget, deadline, and objective
    remain unchanged. The original greedy/best-prefix construction remains an
    explicit candidate. An independently reconstructed risk-priority schedule
    and its best nonempty prefix provide two additional feasible candidates.
    The best candidate is refined only through independently verified strict
    J improvements. Final selection re-evaluates every retained candidate.

    Consequently J is at least the original greedy J and unchanged RF-TS J
    on identical inputs whenever those schedules are feasible. Strict superiority
    is possible when improvement exists; equality is allowed and never perturbed.
    This is an algorithm-definition extension, not a change to reported scores.
    """
    w = _prepare(problem)
    reason = _empty_reason(w)
    if reason is not None:
        return MethodResult((), reason, {"algorithm_version": RACA_ALGORITHM_VERSION, "termination": reason})
    greedy = raca_him_greedy(problem)
    risk_pairs, skipped = _risk_priority_assignment(w)
    candidates = [
        ("original_all_pair_greedy", greedy.assignments),
        ("independent_risk_priority_full", _pairs(w, risk_pairs)),
        ("independent_risk_priority_best_prefix", _pairs(w, _best_risk_priority_prefix(w, risk_pairs))),
    ]
    initial_utilities: dict[str, float | None] = {}
    best_assignments: tuple[Assignment, ...] = ()
    best_source, best_utility = "none", -np.inf
    for source, candidate in candidates:
        evaluation = evaluate_joint_objective(problem, candidate)
        utility = evaluation.joint_utility if evaluation.feasible else None
        initial_utilities[source] = utility
        if utility is not None and utility > best_utility:
            best_assignments, best_source, best_utility = candidate, source, float(utility)
    if not best_assignments:
        return MethodResult((), "deadline_infeasible", {
            "algorithm_version": RACA_ALGORITHM_VERSION,
            "termination": "no_independently_verified_nonempty_candidate",
            "candidate_utilities": initial_utilities,
        })
    refinement_start = best_source
    refined, refinement_diagnostics = _refine_raca_candidate(problem, w, best_assignments)
    candidates.append(("local_refinement_of_" + refinement_start, refined))
    # Compare all unmodified initial candidates again as well as the refined
    # candidate. No method-reported score is trusted for the final decision.
    candidate_utilities: dict[str, float | None] = {}
    best_utility = -np.inf
    for source, candidate in candidates:
        evaluation = evaluate_joint_objective(problem, candidate)
        utility = evaluation.joint_utility if evaluation.feasible else None
        candidate_utilities[source] = utility
        if utility is not None and utility > best_utility:
            best_assignments, best_source, best_utility = candidate, source, float(utility)
    return MethodResult(best_assignments, "ok", {
        "algorithm_version": RACA_ALGORITHM_VERSION,
        "decision_rule": "maximum independently evaluated common J among original greedy, independent risk-priority full/best-prefix and strict local refinement",
        "selected_candidate_source": best_source,
        "candidate_utilities": candidate_utilities,
        "original_greedy_utility": initial_utilities["original_all_pair_greedy"],
        "rf_full_utility": initial_utilities["independent_risk_priority_full"],
        "rf_best_prefix_utility": initial_utilities["independent_risk_priority_best_prefix"],
        "final_joint_utility": best_utility,
        "refinement_start_source": refinement_start,
        "local_search": refinement_diagnostics,
        "legacy_greedy_diagnostics": greedy.diagnostics,
        "risk_priority_targets_skipped_for_deadline": skipped,
        "uses_recorded_baseline_output": False,
        "guarantee": "J >= original greedy and RF-TS when evaluated on identical inputs; strict > is not guaranteed",
        "nonempty_required": True,
        "termination": "independent_candidate_comparison_completed",
    })


def cost_first(problem: JointProblem) -> MethodResult:
    """Choose minimum marginal makespan; never rank a target by risk."""
    w = _prepare(problem)
    reason = _empty_reason(w)
    if reason is not None:
        return MethodResult((), reason, {"termination": reason})
    loads = np.zeros(w.verifiers.size, dtype=np.float64)
    used = np.zeros(w.targets.size, dtype=bool)
    selected: list[tuple[int, int]] = []
    makespan = 0.0
    for _ in range(min(w.budget, w.targets.size)):
        completion = loads[:, None] + w.cost
        next_makespan = np.maximum(makespan, completion)
        feasible = w.allowed & ~used[None, :] & (next_makespan <= w.deadline)
        if not np.any(feasible):
            break
        increase = next_makespan - makespan
        increase[~feasible] = np.inf
        minimum = float(np.min(increase))
        # Cost is a resource-only tie, followed by temporary IDs. This does
        # not introduce the risk-first rule when many pairs have zero delta T.
        cost_tie = np.where(feasible & (increase == minimum), w.cost, np.inf)
        flat = int(np.argmin(cost_tie.T))
        i, g = divmod(flat, w.verifiers.size)
        selected.append((i, g))
        used[i] = True
        loads[g] = completion[g, i]
        makespan = float(np.max(loads))
    return MethodResult(_pairs(w, selected), "ok" if selected else "deadline_infeasible", {
        "decision_rule": "minimum marginal makespan over every feasible pair",
        "tie_breaking": "pair cost, temporary target ID, temporary verifier ID",
        "risk_used_for_selection": False,
        "termination": "budget_reached" if len(selected) == w.budget else "no_remaining_feasible_pair",
    })


def random_feasible(problem: JointProblem, seed: int) -> MethodResult:
    """One uniformly shuffled permutation of initially legal target/GT pairs.

    Accept the next pair if its target is unused and its current verifier load
    remains feasible. Rejected capacity pairs never become feasible later because
    verifier loads only increase. Risk is never read to rank or sample a pair.
    """
    w = _prepare(problem)
    reason = _empty_reason(w)
    if reason is not None:
        return MethodResult((), reason, {"termination": reason, "seed": int(seed)})
    rng = np.random.default_rng(int(seed))
    eligible = (w.allowed & (w.cost <= w.deadline)).T
    candidates = rng.permutation(np.flatnonzero(eligible))
    loads = np.zeros(w.verifiers.size, dtype=np.float64)
    used = np.zeros(w.targets.size, dtype=bool)
    selected: list[tuple[int, int]] = []
    rejected_deadline = 0
    for candidate in candidates:
        i, g = divmod(int(candidate), w.verifiers.size)
        if used[i]:
            continue
        finish = loads[g] + w.cost[g, i]
        if finish > w.deadline:
            rejected_deadline += 1
            continue
        selected.append((i, g))
        used[i] = True
        loads[g] = finish
        if len(selected) == w.budget:
            break
    return MethodResult(_pairs(w, selected), "ok" if selected else "deadline_infeasible", {
        "sampling_rule": "uniform permutation of initially feasible distinct target-verifier pairs",
        "seed": int(seed), "risk_used_for_selection": False,
        "pairs_rejected_for_current_deadline": rejected_deadline,
        "termination": "budget_reached" if len(selected) == w.budget else "pair_permutation_exhausted",
    })


def dva_slot_objective(
    delay_sum_seconds: float | np.ndarray,
    queue_dot_selected_delay: float | np.ndarray,
    queue_sum: float,
    deadline_seconds: float,
    control_lambda: float = 1.0,
) -> float | np.ndarray:
    """Source Eq.21 without the constant Omega, in seconds.

    exp(-s) avoids exp(s) overflow. The queue contribution includes unselected
    targets' -Q_i*C term. Native Q_i(0)=0 is retained in this single round.
    """
    exponent = np.exp(-np.asarray(delay_sum_seconds, dtype=np.float64))
    return control_lambda * 2.0 * exponent / (1.0 + exponent) + (
        queue_dot_selected_delay - deadline_seconds * queue_sum
    )


def dva_p_adapted(problem: JointProblem) -> MethodResult:
    """Source Algorithm 1 priority/add/single-swap with a declared pair extension.

    The source selects binary inspection endpoints and has no separate trusted
    peer choice. For each source add/swap, first map the endpoint to its minimum
    intrinsic-cost capacity-feasible GT. Only then evaluate source Eq.21 for the
    mapped endpoint cost. Loads restrict feasibility but are not added to cost
    when ranking GTs, unlike RF-TS's least completion-time assignment. Hard B and
    accumulated GT-load constraints are common experimental adapters.

    Zero native queues make Eq.21 strictly decreasing in summed selected delay.
    Its endpoint-subset ordering is compared analytically via total seconds so
    even a saturated sigmoid cannot silently turn unequal candidates into ties.
    This source incentive is not extended to the added GT-choice variable: that
    variable has the separate neutral minimum intrinsic-cost mapping above.
    """
    w = _prepare(problem)
    reason = _empty_reason(w)
    if reason is not None:
        return MethodResult((), reason, {"termination": reason})
    control_lambda = 1.0
    queue = np.zeros(w.targets.size, dtype=np.float64)
    loads = np.zeros(w.verifiers.size, dtype=np.float64)
    selected: list[tuple[int, int, float]] = []
    total_ms = 0.0
    additions, replacements, candidate_checks = 0, 0, 0
    order = np.lexsort((w.targets, -w.risk))
    # In source Algorithm 1, previous-selected is empty and its complement is
    # all targets. In our formal grid B<|V|, so that complement is infeasible
    # under the newly shared budget and the source empty fallback is required.
    priority_initialized = False
    if w.targets.size <= w.budget:
        full: list[tuple[int, int, float]] = []
        trial_loads = loads.copy()
        for position in order:
            i = int(position)
            feasible = w.allowed[:, i] & (trial_loads + w.cost[:, i] <= w.deadline)
            if not np.any(feasible):
                break
            # GT assignment is an explicit neutral adapter, not a source
            # variable. Rank intrinsic pair cost only, with verifier ID ties.
            g = int(np.argmin(np.where(feasible, w.cost[:, i], np.inf)))
            cost = float(w.cost[g, i])
            trial_loads[g] += cost
            full.append((i, g, cost))
        if len(full) == w.targets.size:
            candidate_sum = sum(job[2] for job in full)
            if candidate_sum >= 0.0:  # Eq.21(full) <= Eq.21(empty)
                selected, loads, total_ms = full, trial_loads, candidate_sum
                priority_initialized = True
    if not priority_initialized:
        for position in order:
            i = int(position)
            completion = loads + w.cost[:, i]
            feasible = w.allowed[:, i] & (completion <= w.deadline)
            if len(selected) < w.budget and np.any(feasible):
                g = int(np.argmin(np.where(feasible, w.cost[:, i], np.inf)))
                cost = float(w.cost[g, i])
                # Source ΔDVA<=0 is exact at Q=0 for nonnegative costs.
                candidate_checks += 1
                selected.append((i, g, cost))
                loads[g] += cost
                total_ms += cost
                additions += 1
                continue
            if not selected:
                continue
            # Test all single endpoint replacements (source lines 15--23).
            # After removing each old endpoint, independently map the new
            # endpoint to minimum intrinsic cost among capacity-feasible GTs.
            old_i = np.asarray([job[0] for job in selected], dtype=np.int64)
            old_g = np.asarray([job[1] for job in selected], dtype=np.int64)
            old_cost = np.asarray([job[2] for job in selected], dtype=np.float64)
            after_removal = np.broadcast_to(loads, (len(selected), loads.size)).copy()
            after_removal[np.arange(len(selected)), old_g] -= old_cost
            proposed = after_removal + w.cost[:, i][None, :]
            legal = w.allowed[:, i][None, :] & (proposed <= w.deadline)
            # Other loads were already feasible and removing a task cannot
            # increase them, so the chosen GT row is the only required check.
            mapped_costs = np.where(legal, w.cost[:, i][None, :], np.inf)
            mapped_verifiers = np.argmin(mapped_costs, axis=1)
            mapped_values = mapped_costs[np.arange(len(selected)), mapped_verifiers]
            mapped_feasible = np.any(legal, axis=1)
            sums = total_ms - old_cost + mapped_values
            # Only the mapped endpoint subset uses source Eq.21 ordering.
            # The adapter has already picked the cheapest feasible GT.
            candidates = np.where(mapped_feasible, sums, -np.inf)
            candidate_checks += int(mapped_feasible.sum())
            maximum_sum = float(np.max(candidates))
            if maximum_sum < total_ms or not np.isfinite(maximum_sum):
                continue
            swap_rows = np.flatnonzero(candidates == maximum_sum)
            tie_order = np.lexsort((w.verifiers[mapped_verifiers[swap_rows]], w.targets[old_i[swap_rows]]))
            row = int(swap_rows[tie_order[0]])
            g = int(mapped_verifiers[row])
            old = selected[row]
            cost = float(w.cost[g, i])
            loads[old[1]] -= old[2]
            loads[g] += cost
            selected[row] = (i, g, cost)
            total_ms = maximum_sum
            replacements += 1
    # Source Eq.13, one real slot only. Queues are indexed by target endpoints,
    # as in the source, not manufactured from synthetic anomaly histories.
    for i, _, cost in selected:
        queue[i] = max(queue[i] + (cost - w.deadline) / 1000.0, 0.0)
    final_objective = float(dva_slot_objective(total_ms / 1000.0, 0.0, 0.0, w.deadline / 1000.0, control_lambda))
    return MethodResult(_pairs(w, [(i, g) for i, g, _ in selected]),
                        "ok" if selected else "deadline_infeasible", {
        "adaptation_version": "dva-p-single-round-pair-subset-v1",
        "source_steps": "Eq.13 and Eq.21; Algorithm 1 priority, add and single replacement",
        "native_scope": "binary inspection subset; no separate GT verifier in source",
        "pair_extension": "minimum intrinsic pair cost among capacity-feasible GTs before source Eq.21 add/swap",
        "assignment_adapter": "GT loads screen feasibility only; pair cost c_gi ranks options, verifier ID breaks ties",
        "target_iteration_order": "descending common risk, then temporary target ID",
        "initial_previous_selected": [], "initial_queue": 0.0,
        "priority_complement_initialized": priority_initialized,
        "priority_fallback": "empty if complement exceeds common B or cannot fit GT load deadline",
        "control_lambda": control_lambda, "objective_delay_unit": "seconds",
        "internal_dva_objective_without_Omega": final_objective,
        "sigmoid_comparison": "analytic monotone ordering of total delay under native zero queues",
        "additions_accepted": additions, "replacements_accepted": replacements,
        "candidate_objective_checks": candidate_checks,
        "final_virtual_queue_sum_seconds": float(queue.sum()),
        "long_term_guarantees_evaluated": False,
        "uses_joint_objective_for_decision": False,
        "uses_least_completion_assignment": False,
    })


def run_method(method: str, problem: JointProblem, seed: int = 0) -> MethodResult:
    """Dispatch only; independent evaluation must follow in the experiment."""
    methods = {
        "RACA-HIM": raca_him,
        "DVA-P": dva_p_adapted,
        "DVA-P-Adapted": dva_p_adapted,
        "RF-TS": risk_first_time_sensitive,
        "Cost-First": cost_first,
    }
    if method == "Random":
        return random_feasible(problem, seed)
    if method not in methods:
        raise ValueError(f"Unknown joint decision method: {method}")
    return methods[method](problem)
