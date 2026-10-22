"""Shared decision inputs and an evaluator independent of all five optimizers."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np


Assignment = tuple[int, int]  # (target temporary ID, verifier temporary ID)


@dataclass(frozen=True)
class JointProblem:
    target_ids: np.ndarray
    verifier_ids: np.ndarray
    risk_scores: np.ndarray
    costs: np.ndarray  # [verifier row, target column], in milliseconds
    budget: int
    lambda_value: float = 0.5
    deadline_ms: float = 800.0
    compute_costs: np.ndarray | None = None
    transfer_costs: np.ndarray | None = None
    network_costs: np.ndarray | None = None

    def __post_init__(self) -> None:
        for name in ("target_ids", "verifier_ids"):
            raw = np.asarray(getattr(self, name))
            if raw.ndim != 1 or not np.issubdtype(raw.dtype, np.integer):
                raise ValueError(f"{name} must be a one-dimensional integer ID array")
            if np.unique(raw).size != raw.size:
                raise ValueError(f"{name} contains duplicate temporary IDs")
            array = np.frombuffer(np.asarray(raw, dtype=np.int64).tobytes(), dtype=np.int64)
            object.__setattr__(self, name, array)
        risk = np.array(self.risk_scores, dtype=np.float64, copy=True)
        cost = np.array(self.costs, dtype=np.float64, copy=True)
        if risk.shape != self.target_ids.shape:
            raise ValueError("risk_scores must align with target_ids")
        if not np.all(np.isfinite(risk) & (risk >= 0.0) & (risk <= 1.0)):
            raise ValueError("risk_scores must be finite and lie in [0,1]")
        if cost.shape != (self.verifier_ids.size, self.target_ids.size):
            raise ValueError("costs must have shape (number of GT verifiers, number of targets)")
        if np.any(np.isnan(cost) | (cost < 0.0)):
            raise ValueError("Costs must be nonnegative; +inf may denote an unavailable pair")
        if isinstance(self.budget, bool) or not isinstance(self.budget, (int, np.integer)) or self.budget < 0:
            raise ValueError("budget must be a nonnegative integer")
        if not np.isfinite(self.lambda_value) or self.lambda_value <= 0.0:
            raise ValueError("lambda_value must be positive and finite")
        if not np.isfinite(self.deadline_ms) or self.deadline_ms <= 0.0:
            raise ValueError("deadline_ms must be positive and finite")
        risk = np.frombuffer(risk.tobytes(), dtype=np.float64).reshape(risk.shape)
        cost = np.frombuffer(cost.tobytes(), dtype=np.float64).reshape(cost.shape)
        object.__setattr__(self, "risk_scores", risk)
        object.__setattr__(self, "costs", cost)
        components = (self.compute_costs, self.transfer_costs, self.network_costs)
        if any(component is not None for component in components):
            if not all(component is not None for component in components):
                raise ValueError("Provide all three resource cost components or none")
            for name in ("compute_costs", "transfer_costs", "network_costs"):
                array = np.array(getattr(self, name), dtype=np.float64, copy=True)
                if array.shape != cost.shape or np.any(np.isnan(array) | (array < 0.0)):
                    raise ValueError(f"Invalid {name} resource component")
                array = np.frombuffer(array.tobytes(), dtype=np.float64).reshape(array.shape)
                object.__setattr__(self, name, array)
            total = self.compute_costs + self.transfer_costs + self.network_costs
            # Dataset components can be float32; preserve its supplied total
            # rather than silently replacing it with a float64 re-sum.
            if not np.allclose(total, cost, rtol=1e-6, atol=1e-9):
                raise ValueError("total costs do not equal compute + transfer + network exactly once")


@dataclass(frozen=True)
class ObjectiveEvaluation:
    selected_count: int
    completed_count: int | None
    risk_reward: float | None
    normalized_risk_reward: float | None
    makespan_ms: float | None
    normalized_delay_penalty: float | None
    joint_utility: float | None
    deadline_satisfied: bool | None
    feasible: bool
    status: str
    failure_reason: str
    budget_usage_ratio: float | None
    unused_budget: int
    verifier_loads: dict[int, float]
    assignment_diagnostics: tuple[dict[str, Any], ...]

    @property
    def verification_makespan_ms(self) -> float | None:
        return self.makespan_ms

    def to_dict(self) -> dict[str, Any]:
        """Scalar result columns; detailed assignments/loads stay separate."""
        return {
            "selected_count": self.selected_count,
            "completed_count": self.completed_count,
            "risk_reward": self.risk_reward,
            "normalized_risk_reward": self.normalized_risk_reward,
            "makespan_ms": self.makespan_ms,
            "verification_makespan_ms": self.makespan_ms,
            "normalized_delay_penalty": self.normalized_delay_penalty,
            "joint_utility": self.joint_utility,
            "deadline_satisfied": self.deadline_satisfied,
            "feasible": self.feasible,
            "status": self.status,
            "failure_reason": self.failure_reason,
            "budget_usage_ratio": self.budget_usage_ratio,
            "unused_budget": self.unused_budget,
        }


def evaluate_joint_objective(
    problem: JointProblem, assignments: Iterable[Assignment],
) -> ObjectiveEvaluation:
    """Reconstruct unique targets, pair costs, parallel loads and J from IDs.

    No method-reported utility or load is trusted. An invalid solution is never
    truncated or repaired and receives joint_utility=None. The manuscript's
    nonempty constraint is enforced even when all nonempty utilities are negative.
    With the common p_fail=0 model a feasible scheduled task is completed.
    """
    given = tuple(assignments)
    target_position = {int(value): i for i, value in enumerate(problem.target_ids)}
    verifier_position = {int(value): g for g, value in enumerate(problem.verifier_ids)}
    loads = np.zeros(problem.verifier_ids.size, dtype=np.float64)
    diagnostics: list[dict[str, Any]] = []
    seen: set[int] = set()
    errors: list[str] = []
    risk_reward = 0.0
    reconstructable = True
    for sequence, assignment in enumerate(given, start=1):
        if not isinstance(assignment, (tuple, list, np.ndarray)) or len(assignment) != 2:
            errors.append(f"malformed_assignment_at_{sequence}")
            reconstructable = False
            continue
        target, verifier = assignment
        if (isinstance(target, (bool, np.bool_)) or isinstance(verifier, (bool, np.bool_))
                or not isinstance(target, (int, np.integer)) or not isinstance(verifier, (int, np.integer))):
            errors.append(f"noninteger_id_at_{sequence}")
            reconstructable = False
            continue
        target, verifier = int(target), int(verifier)
        if target in seen:
            errors.append(f"duplicate_target_{target}")
        else:
            seen.add(target)
            if target in target_position:
                risk_reward += float(problem.risk_scores[target_position[target]])
        if target not in target_position or verifier not in verifier_position:
            errors.append(f"unknown_target_or_non_gt_verifier_at_{sequence}")
            reconstructable = False
            continue
        if target == verifier:
            errors.append(f"self_verification_at_{sequence}")
        i, g = target_position[target], verifier_position[verifier]
        pair_cost = float(problem.costs[g, i])
        if not np.isfinite(pair_cost):
            errors.append(f"unavailable_pair_at_{sequence}")
            reconstructable = False
            continue
        loads[g] += pair_cost
        diagnostics.append({
            "sequence": sequence,
            "target_id": target,
            "verifier_id": verifier,
            "target_risk": float(problem.risk_scores[i]),
            "pair_cost_ms": pair_cost,
            "verifier_load_after_ms": float(loads[g]),
            "makespan_after_ms": float(np.max(loads)),
        })
    selected = len(seen)
    if len(given) > problem.budget:
        errors.append("budget_exceeded")
    makespan = float(np.max(loads)) if loads.size else 0.0
    deadline_ok = bool(makespan <= problem.deadline_ms + 1e-9) if reconstructable else None
    if deadline_ok is False:
        errors.append("deadline_exceeded")
    if not given:
        if problem.target_ids.size == 0:
            empty_status = "no_legal_target"
        elif problem.verifier_ids.size == 0:
            empty_status = "no_legal_verifier"
        elif problem.budget == 0:
            empty_status = "zero_budget"
        else:
            allowed = np.isfinite(problem.costs)
            allowed &= problem.verifier_ids[:, None] != problem.target_ids[None, :]
            if not np.any(allowed):
                empty_status = "no_legal_verifier"
            else:
                legal = allowed & (problem.costs <= problem.deadline_ms)
                empty_status = "deadline_infeasible" if not np.any(legal) else "empty_solution"
        errors.append(empty_status)
    feasible = not errors
    status = "ok" if feasible else (empty_status if not given else "invalid_assignment")
    normalized_reward = risk_reward / problem.budget if problem.budget > 0 and reconstructable else None
    delay_penalty = problem.lambda_value * makespan / problem.deadline_ms if reconstructable else None
    utility = normalized_reward - delay_penalty if feasible else None
    return ObjectiveEvaluation(
        selected_count=selected,
        completed_count=selected if feasible else None,
        risk_reward=risk_reward if reconstructable else None,
        normalized_risk_reward=normalized_reward,
        makespan_ms=makespan if reconstructable else None,
        normalized_delay_penalty=delay_penalty,
        joint_utility=utility,
        deadline_satisfied=deadline_ok,
        feasible=feasible,
        status=status,
        failure_reason=";".join(errors),
        budget_usage_ratio=selected / problem.budget if problem.budget > 0 else None,
        unused_budget=max(problem.budget - selected, 0),
        verifier_loads={int(value): float(loads[g]) for g, value in enumerate(problem.verifier_ids)},
        assignment_diagnostics=tuple(diagnostics),
    )


def validate_assignment(problem: JointProblem, assignments: Iterable[Assignment]) -> ObjectiveEvaluation:
    """Explicit user-run validation helper; raises on any illegal returned plan."""
    evaluation = evaluate_joint_objective(problem, assignments)
    if not evaluation.feasible:
        raise ValueError(evaluation.failure_reason)
    return evaluation
