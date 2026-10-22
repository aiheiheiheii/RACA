"""Finite-round, raw-content adaptations of DVA-P, I-EDI and CTCV.

Selection has no access to descriptors, similarity scores or current labels.
Verification hashes actual stored float32 bytes.  First-group v4's O2RS and
CTCV four-Flag/member-bit kernels are imported, rather than reinterpreted.
The message trace describes a logical transport, not a deployed blockchain.
See reports/02_overhead_baseline_adaptation.md for retained and omitted steps.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
import importlib
import math
import struct
from typing import Any

import numpy as np

_raw = importlib.import_module("experiments.01_detection.adaptation_utils")
_iedi = importlib.import_module("experiments.01_detection.iedi_adapted")
_ctcv = importlib.import_module("experiments.01_detection.ctcv_adapted")
canonical_descriptor_bytes = _raw.canonical_descriptor_bytes

AV = -1
SC = -2
# A lossless deterministic request adapter replaces repeated full VS tables.
# uint64 seed + uint32 requested blocks/payload bytes/cyclic start.
CTCV_COMPACT_REQUEST_BYTES = struct.calcsize("<QIII")


@dataclass(frozen=True)
class AssignmentPlan:
    assignments: tuple[tuple[int, int], ...]
    diagnostics: dict[str, Any]


@dataclass(frozen=True)
class ExactReference:
    anchor_index: int
    anchor_id: int
    payload: bytes
    sha256: str
    diagnostics: dict[str, Any]
    events: tuple[dict[str, Any], ...]
    operations: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class VerificationTrace:
    predictions: np.ndarray
    events: tuple[dict[str, Any], ...]
    operations: tuple[dict[str, Any], ...]
    diagnostics: dict[str, Any]


def _inputs(targets, verifiers, cost_ms, budget, deadline_ms):
    targets = np.asarray(targets, dtype=np.int64)
    verifiers = np.asarray(verifiers, dtype=np.int64)
    cost_ms = np.asarray(cost_ms, dtype=np.float64)
    if targets.ndim != 1 or verifiers.ndim != 1 or not len(verifiers):
        raise ValueError("Targets and nonempty verifiers must be one-dimensional.")
    if len(np.unique(targets)) != len(targets) or len(np.unique(verifiers)) != len(verifiers):
        raise ValueError("Targets and verifiers must contain unique row indices.")
    if np.intersect1d(targets, verifiers).size:
        raise ValueError("The shared target and verifier sets must be disjoint.")
    if np.any(targets < 0) or np.any(verifiers < 0):
        raise ValueError("Row indices must be nonnegative.")
    if cost_ms.shape != (len(targets), len(verifiers)) or not np.isfinite(cost_ms).all() or np.any(cost_ms <= 0):
        raise ValueError("cost_ms must be a positive finite target-by-verifier matrix.")
    if isinstance(budget, bool) or not isinstance(budget, (int, np.integer)) or budget < 0:
        raise ValueError("The common budget must be a nonnegative integer.")
    if not math.isfinite(deadline_ms) or deadline_ms <= 0:
        raise ValueError("The verification deadline must be positive and finite.")
    return targets, verifiers, cost_ms, min(int(budget), len(targets)), float(deadline_ms)


def _assign(positions, targets, verifiers, cost_ms, deadline_ms, *, require_feasible):
    """Public neutral adapter: earliest projected completion, identity ties.

    Fixed workload always attempts the supplied targets; an over-deadline job
    stays in the plan so that the simulator can expose unfinished work.
    """
    loads = np.zeros(len(verifiers), dtype=np.float64)
    pairs, rejected, finishes = [], [], []
    for position in positions:
        finish = loads + cost_ms[position]
        selected = int(np.lexsort((verifiers, finish))[0])
        if require_feasible and finish[selected] > deadline_ms:
            rejected.append(int(targets[position]))
            continue
        pairs.append((int(targets[position]), int(verifiers[selected])))
        loads[selected] = finish[selected]
        finishes.append(float(finish[selected]))
    return tuple(pairs), loads, rejected, finishes


def _fixed_plan(targets, verifiers, costs, budget, deadline, method):
    if budget != len(targets):
        raise ValueError("Fixed workload must contain exactly the common budget's targets.")
    pairs, loads, rejected, finishes = _assign(
        range(len(targets)), targets, verifiers, costs, deadline, require_feasible=False)
    return AssignmentPlan(pairs, {
        "method": method, "evaluation_mode": "fixed_workload",
        "source_subset_selection_bypassed": True,
        "source_subset_limitation": "Fixed targets assess execution and the neutral assignment adapter; native selection ability is not evaluated.",
        "assignment_adapter": "earliest projected completion; row identity breaks ties",
        "planned_verifier_loads_ms": loads.tolist(),
        "planned_finish_ms": finishes, "unassigned_target_rows": rejected,
        "planned_deadline_violations": int(np.count_nonzero(np.asarray(finishes) > deadline)),
        "selected_count": len(pairs), "budget": budget,
    })


def _dva_objective(delay_sum_seconds, queue_dot_seconds, queue_sum_seconds,
                   deadline_seconds, control_lambda):
    """Eq. (21), omitting its solution-independent Omega.

    All delay/queue terms use seconds. exp(-s) is numerically stable; the
    queue contribution is sum_i Q_i*(d_i*C_i-C).  Source zero initial queues
    are retained, rather than manufacturing a beneficial multi-round history.
    """
    z = math.exp(-float(delay_sum_seconds))
    return float(control_lambda * 2.0 * z / (1.0 + z)
                 + queue_dot_seconds - deadline_seconds * queue_sum_seconds)


def choose_dvap(targets, verifiers, cost_ms, budget, deadline_ms, *,
                fixed_workload=False, control_lambda=1.0) -> AssignmentPlan:
    """Algorithm 1's priority/add/replace policy with declared finite capacity.

    DVA-P selects *replicas*, not a peer verifier. The common verifier pool
    requires the explicit earliest-completion adapter. Native mode evaluates
    actual marginal Eq. (21) values, including replacement, subject to B and
    accumulated verifier loads. Fixed mode bypasses source subset selection.
    """
    targets, verifiers, costs, budget, deadline = _inputs(
        targets, verifiers, cost_ms, budget, deadline_ms)
    if not math.isfinite(control_lambda) or control_lambda <= 0:
        raise ValueError("DVA-P Lambda must be finite and positive.")
    if fixed_workload:
        return _fixed_plan(targets, verifiers, costs, budget, deadline, "DVA-P-Adapted")
    queue = np.zeros(len(targets), dtype=np.float64)
    selected: list[tuple[int, int, float]] = []  # target position, verifier position, job ms
    loads = np.zeros(len(verifiers), dtype=np.float64)
    delay_sum = 0.0
    objective = _dva_objective(0, 0, 0, deadline / 1000, control_lambda)
    checks, replacements = 0, 0
    priority_initialized = False
    # Source M(0)=empty: previous-unselected complement is the whole set.
    # It may initialize the selection only if the added B/load constraints
    # allow it; otherwise Algorithm 1's empty-selection fallback is used.
    order = np.argsort(targets, kind="stable")
    if len(targets) <= budget:
        seed_pairs, seed_loads, rejected, _ = _assign(
            order, targets, verifiers, costs, deadline, require_feasible=True)
        if not rejected:
            row_to_pos = {int(row): p for p, row in enumerate(targets)}
            verifier_to_pos = {int(row): p for p, row in enumerate(verifiers)}
            for target, verifier in seed_pairs:
                p, v = row_to_pos[target], verifier_to_pos[verifier]
                selected.append((p, v, float(costs[p, v])))
            loads = seed_loads
            delay_sum = sum(job[2] for job in selected) / 1000
            objective = _dva_objective(delay_sum, 0, 0, deadline / 1000, control_lambda)
            priority_initialized = True
    if not priority_initialized:
        for position in order:
            projected = loads + costs[position]
            v = int(np.lexsort((verifiers, projected))[0])
            new_delay = delay_sum + float(costs[position, v]) / 1000
            trial = _dva_objective(new_delay, 0, 0, deadline / 1000, control_lambda)
            checks += 1
            if len(selected) < budget and projected[v] <= deadline and trial <= objective:
                selected.append((int(position), v, float(costs[position, v])))
                loads[v] = projected[v]
                delay_sum, objective = new_delay, trial
                continue
            if not selected:
                continue
            # Source lines 15--23: test replacing each current selected replica
            # and accept the largest nonnegative objective decrease.  Vectorize
            # feasibility checks without changing the add/replace decisions.
            old_positions = np.asarray([job[0] for job in selected], dtype=np.int64)
            old_verifiers = np.asarray([job[1] for job in selected], dtype=np.int64)
            old_delays = np.asarray([job[2] for job in selected], dtype=np.float64)
            replacement_loads = np.broadcast_to(loads, (len(selected), len(verifiers))).copy()
            replacement_loads[np.arange(len(selected)), old_verifiers] -= old_delays
            proposed = replacement_loads + costs[position]
            candidate_verifiers = np.argmin(proposed[:, np.argsort(verifiers)], axis=1)
            candidate_verifiers = np.argsort(verifiers)[candidate_verifiers]
            candidate_costs = costs[position, candidate_verifiers]
            feasible = proposed[np.arange(len(selected)), candidate_verifiers] <= deadline
            sums = delay_sum + (candidate_costs - old_delays) / 1000
            candidate_objectives = np.asarray([
                _dva_objective(total, 0, 0, deadline / 1000, control_lambda) for total in sums])
            decrease = objective - candidate_objectives
            decrease[~feasible] = -np.inf
            checks += len(selected)
            best = int(np.lexsort((targets[old_positions], -decrease))[0])
            if decrease[best] >= 0:
                old = selected[best]
                new_v = int(candidate_verifiers[best])
                loads[old[1]] -= old[2]
                loads[new_v] += float(candidate_costs[best])
                selected[best] = (int(position), new_v, float(candidate_costs[best]))
                delay_sum, objective = float(sums[best]), float(candidate_objectives[best])
                replacements += 1
    # Eq. (13): only one actual time-slot update.  The load-constrained adapter
    # does not fabricate previous slots; all feasible initial queues remain 0.
    for position, _, delay in selected:
        queue[position] = max(queue[position] + (delay - deadline) / 1000, 0.0)
    pairs = tuple((int(targets[p]), int(verifiers[v])) for p, v, _ in selected)
    return AssignmentPlan(pairs, {
        "method": "DVA-P-Adapted", "evaluation_mode": "native_budget",
        "source_subset_selection_bypassed": False,
        "source_steps": "Algorithm 1 priority judgement/add/replace; Equations 13 and 21",
        "initial_previous_selected": [], "initial_queue": 0.0,
        "priority_complement_initialized": priority_initialized,
        "priority_capacity_adaptation": "The previous-unselected complement may initialize selection only if common B and accumulated verifier deadline allow it.",
        "control_lambda": control_lambda, "objective_without_constant_Omega": objective,
        "objective_delay_unit": "seconds", "objective_checks": checks,
        "replacements_accepted": replacements, "final_queue": queue.tolist(),
        "assignment_adapter": "earliest projected completion; capacity/load-feasible Eq21 add/replace",
        "planned_verifier_loads_ms": loads.tolist(), "selected_count": len(pairs), "budget": budget,
        "unassigned_target_rows": [int(row) for row in targets if int(row) not in {p[0] for p in pairs}],
        "long_term_guarantees_evaluated": False,
    })


def choose_iedi(targets, verifiers, cost_ms, budget, deadline_ms, *,
                fixed_workload=False, control_V=1.0,
                minimum_fraction=0.10) -> AssignmentPlan:
    """First-group literal O2RS once, followed by the public B/load wrapper."""
    targets, verifiers, costs, budget, deadline = _inputs(
        targets, verifiers, cost_ms, budget, deadline_ms)
    if not math.isfinite(control_V) or control_V <= 0:
        raise ValueError("I-EDI control V must be finite and positive.")
    if not math.isfinite(minimum_fraction) or not 0 < minimum_fraction <= 1:
        raise ValueError("I-EDI minimum fraction must lie in (0,1].")
    if fixed_workload:
        plan = _fixed_plan(targets, verifiers, costs, budget, deadline, "I-EDI-Adapted")
        plan.diagnostics.update({"reliability_prior": 0.75, "initial_queue": 0.0,
                                 "source_selection_passes_executed": 0})
        return plan
    order = np.argsort(targets, kind="stable")
    gamma = min(budget, max(1, math.ceil(len(targets) * minimum_fraction)))
    if not len(targets) or budget == 0:
        return AssignmentPlan((), {"method": "I-EDI-Adapted", "selected_count": 0,
                                    "budget": budget, "source_selection_passes_executed": 0})
    reliability = np.full(len(targets), 0.75, dtype=np.float64)
    selection = _iedi._o2rs_exclusion_greedy(reliability[order], gamma, 0.0, control_V)
    source_positions = order[selection.selected_mask]
    bounded_positions = source_positions[:budget]
    pairs, loads, rejected, finishes = _assign(
        bounded_positions, targets, verifiers, costs, deadline, require_feasible=True)
    return AssignmentPlan(pairs, {
        "method": "I-EDI-Adapted", "evaluation_mode": "native_budget",
        "source_subset_selection_bypassed": False,
        "source_steps": "first-group v4 O2RS: Eqs4,8,12,20, Algorithm1",
        "qos_conformance_prior": 1.0, "success_rate_prior": 0.5,
        "retrieval_error_prior": 0.5, "qos_weight_beta": 0.5,
        "reliability_prior": 0.75, "unavailable_history_is_neutral_prior": True,
        "initial_queue": 0.0, "final_queue": selection.queue_after,
        "gamma": gamma, "control_V": control_V,
        "queue_update_granularity": "literal per candidate Algorithm1 line10",
        "source_selection_passes_executed": 1,
        "source_selected_count": len(source_positions),
        "source_exclusions_accepted": selection.exclusions_accepted,
        "source_initial_objective": selection.initial_objective,
        "source_final_objective": selection.final_objective,
        "source_stored_minimum": selection.stored_minimum,
        "budget_adapter": "deterministic row order caps the source subset at common maximum B; gamma is the source minimum-count objective, not B",
        "assignment_adapter": "earliest projected completion; row identity ties; no joint objective",
        "planned_verifier_loads_ms": loads.tolist(), "planned_finish_ms": finishes,
        "unassigned_target_rows": rejected, "selected_count": len(pairs), "budget": budget,
    })


def choose_ctcv(targets, verifiers, cost_ms, budget, deadline_ms, *,
                fixed_workload=False, collusion_bound=1, seed=2026) -> AssignmentPlan:
    """Time-constrained responses and seeded k+1 report-collector selection.

    Source CTCV requests all replicas; common B is an external cap, not an
    invented CTCV risk selector. Time eligibility uses shared resource costs.
    Full fanout/report delays are checked by the actual event simulator.
    """
    targets, verifiers, costs, budget, deadline = _inputs(
        targets, verifiers, cost_ms, budget, deadline_ms)
    if isinstance(collusion_bound, bool) or not isinstance(collusion_bound, (int, np.integer)) or collusion_bound < 0:
        raise ValueError("CTCV k must be a nonnegative integer.")
    if len(verifiers) < collusion_bound + 1:
        return AssignmentPlan((), {"method": "CTCV-Adapted", "selected_count": 0,
                                    "status": "collector_unavailable", "budget": budget,
                                    "collector_required_count": collusion_bound + 1})
    rng = np.random.default_rng(int(seed))
    collectors = rng.permutation(np.sort(verifiers))[:collusion_bound + 1]
    if fixed_workload:
        plan = _fixed_plan(targets, verifiers, costs, budget, deadline, "CTCV-Adapted")
    else:
        pairs, loads, rejected, finishes = _assign(
            range(budget), targets, verifiers, costs, deadline, require_feasible=True)
        plan = AssignmentPlan(pairs, {
            "method": "CTCV-Adapted", "evaluation_mode": "native_budget",
            "source_subset_selection_bypassed": False,
            "source_steps": "time-constrained response; Algorithm3 seeded k+1 collector choice",
            "source_target_policy": "request every replica subject to the external common B cap; supplied target order is label-independent",
            "assignment_adapter": "earliest projected completion for primary queues; actual k+1 proof fanout and reports are simulated separately",
            "planned_verifier_loads_ms": loads.tolist(), "planned_finish_ms": finishes,
            "unassigned_target_rows": rejected, "selected_count": len(pairs), "budget": budget,
        })
    plan.diagnostics.update({"collector_rows": collectors.tolist(),
                             "collusion_bound": int(collusion_bound),
                             "collector_required_count": int(collusion_bound) + 1,
                             "collector_selection_seed": int(seed),
                             "deadline_filter_cost_is_estimate": True})
    return plan


def _event(sender, receiver, message_type, payload=b"", *, control_bytes=0,
           stage="task", target_row=None):
    result = {"sender": int(sender), "receiver": int(receiver),
              "message_type": message_type, "payload_bytes": len(payload),
              "control_bytes": int(control_bytes), "stage": stage}
    if payload:
        result["payload_sha256"] = hashlib.sha256(payload).hexdigest()
    if target_row is not None:
        result["target_row"] = int(target_row)
    return result


def _operation(node, operation, input_bytes=0, comparisons=0, *,
               stage="task", target_row=None, **metadata):
    result = {"node": int(node), "operation": operation,
              "input_bytes": int(input_bytes), "comparisons": int(comparisons),
              "stage": stage, **metadata}
    if target_row is not None:
        result["target_row"] = int(target_row)
    return result


def prepare_reference(X, verifiers, model_ids=None, history=None, *, config=None) -> ExactReference:
    """Independently estimate an anchor using full raw hash frequency only.

    This does not authenticate a vendor original. Every method must call it
    independently; sharing this algorithm-specific transcript is prohibited.
    Physical wire events use row indices, while model IDs only break ties.
    """
    X = np.asarray(X)
    pool = np.asarray(verifiers, dtype=np.int64)
    if X.ndim != 2 or pool.ndim != 1 or not len(pool) or len(np.unique(pool)) != len(pool):
        raise ValueError("The raw-reference pool must be nonempty and unique.")
    if np.any(pool < 0) or np.any(pool >= len(X)):
        raise ValueError("Reference rows must be inside X.")
    ids = np.arange(len(X), dtype=np.int64) if model_ids is None else np.asarray(model_ids, dtype=np.int64)
    hist = np.zeros(len(X), dtype=np.float64) if history is None else np.asarray(history, dtype=np.float64)
    if ids.shape != (len(X),) or hist.shape != (len(X),) or not np.isfinite(hist).all():
        raise ValueError("Reference model IDs/history must align with X.")
    payloads = tuple(canonical_descriptor_bytes(X[row]) for row in pool)
    hashes = tuple(hashlib.sha256(payload).hexdigest() for payload in payloads)
    counts = Counter(hashes)
    chosen = min(range(len(pool)), key=lambda p: (-counts[hashes[p]], float(hist[pool[p]]), int(ids[pool[p]])))
    anchor = int(pool[chosen])
    events, operations = [], []
    request_bytes = int(getattr(config, 'request_bytes', 64))
    for row, payload in zip(pool, payloads):
        events.extend((_event(AV, row, "reference_request", control_bytes=request_bytes, stage="setup"),
                       _event(row, AV, "reference_descriptor", payload, stage="setup")))
        operations.append(_operation(AV, "hash", len(payload), stage="setup",
                                     provenance="actual SHA256 for complete raw hash frequency"))
    operations.append(_operation(AV, "compare", comparisons=len(pool), stage="setup",
                                 provenance="hash-frequency/history/identity anchor selection"))
    return ExactReference(anchor, int(ids[anchor]), payloads[chosen], hashes[chosen], {
        "reference_constructor": "most frequent complete raw float32 SHA256; history and temporary ID break ties",
        "anchor_index": anchor, "anchor_id": int(ids[anchor]),
        "reference_sha256": hashes[chosen], "reference_hash_support": counts[hashes[chosen]],
        "reference_pool_size": len(pool), "reference_pool_rows": pool.tolist(),
        "anchor_authenticated": False, "content_comparison": "exact_hash",
        "shared_GT_scores_or_similarity_read": False,
    }, tuple(events), tuple(operations))


def _verification_inputs(X, targets, reference, assignments, model_ids):
    X = np.asarray(X)
    targets = np.asarray(targets, dtype=np.int64)
    if X.ndim != 2 or targets.ndim != 1 or len(np.unique(targets)) != len(targets):
        raise ValueError("Verification requires a unique one-dimensional target set.")
    if np.any(targets < 0) or np.any(targets >= len(X)):
        raise ValueError("Verification target rows must be inside X.")
    ids = np.arange(len(X), dtype=np.int64) if model_ids is None else np.asarray(model_ids, dtype=np.int64)
    if ids.shape != (len(X),):
        raise ValueError("Verification model IDs must align with X.")
    if assignments is None:
        routing = {int(row): reference.anchor_index for row in targets}
    else:
        pairs = assignments.assignments if isinstance(assignments, AssignmentPlan) else tuple(assignments)
        routing = dict((int(t), int(v)) for t, v in pairs)
        if len(routing) != len(pairs) or set(routing) != set(targets.tolist()):
            raise ValueError("Assignments must cover each requested target exactly once.")
    if any(v < 0 or v >= len(X) or t == v for t, v in routing.items()):
        raise ValueError("Every verification route requires a legal distinct verifier row.")
    if canonical_descriptor_bytes(X[reference.anchor_index]) != reference.payload:
        raise ValueError("The independent anchor bytes changed after preparation.")
    return X, targets, ids, routing


def _control_lengths(config):
    request = int(getattr(config, "request_bytes", 64))
    result = int(getattr(config, "result_bytes", 32))
    if request < 0 or result < 0:
        raise ValueError("Declared control-message byte sizes must be nonnegative.")
    return request, result


def _verify_hash_tag(X, targets, reference, config, *, modulus=None,
                     assignments=None, model_ids=None):
    X, targets, ids, routing = _verification_inputs(X, targets, reference, assignments, model_ids)
    request_bytes, result_bytes = _control_lengths(config)
    assignment_bytes = int(getattr(config, 'assignment_bytes', request_bytes))
    full_anchor = int(reference.sha256, 16)
    correct = full_anchor if modulus is None else full_anchor % modulus
    width = 32 if modulus is None else max(1, ((modulus - 1).bit_length() + 7) // 8)
    expected_payload = correct.to_bytes(width, "big")
    events, operations, tags, actual_hashes = [], [], [], []
    if modulus is not None:
        operations.append(_operation(AV, 'compare', comparisons=1, stage='setup',
                                     provenance='actual correct anchor modulo eta', arithmetic='modulo'))
    for verifier in sorted(set(routing.values())):
        events.append(_event(AV, verifier, "correct_digest" if modulus is None else "correct_modulo_tag",
                             expected_payload, stage="setup"))
    predictions = np.empty(len(targets), dtype=np.int8)
    for position, row in enumerate(targets):
        row, verifier = int(row), routing[int(row)]
        payload = canonical_descriptor_bytes(X[row])
        actual_hash = hashlib.sha256(payload).hexdigest()
        tag = int(actual_hash, 16) if modulus is None else int(actual_hash, 16) % modulus
        predictions[position] = int(tag != correct)
        tags.append(tag)
        actual_hashes.append(actual_hash)
        events.extend((
            _event(AV, verifier, "assignment_control", control_bytes=assignment_bytes, target_row=row),
            _event(AV, row, "verification_challenge", control_bytes=request_bytes, target_row=row),
            _event(row, verifier, "sha256_digest" if modulus is None else "iedi_modulo_tag",
                   tag.to_bytes(width, "big"), target_row=row),
            _event(verifier, AV, "verification_result", control_bytes=result_bytes, target_row=row),
        ))
        operations.extend((
            _operation(row, "hash", len(payload), target_row=row, provenance="actual SHA256 of stored raw descriptor"),
            _operation(verifier, "compare", comparisons=1, target_row=row, provenance="actual exact proof equality"),
        ))
        if modulus is not None:
            operations.append(_operation(row, "compare", comparisons=1, target_row=row,
                                         provenance="actual integer modulo eta", arithmetic="modulo"))
    full_hash_equal = np.asarray([value == reference.sha256 for value in actual_hashes], dtype=bool)
    diagnostics = {
        "content_comparison": "exact_hash", "verification_rule": "actual proof != independent anchor correct proof",
        "target_rows": targets.tolist(), "target_ids": ids[targets].tolist(),
        "correct_tag": correct, "actual_tags": tags, "proof_byte_width": width,
        "full_hash_equal_count": int(full_hash_equal.sum()),
        "tag_equal_count": int(np.count_nonzero(predictions == 0)),
        "additional_modulus_collision_count": int(np.count_nonzero((predictions == 0) & ~full_hash_equal)),
        "anchor_authenticated": False,
        "authentication_and_security_wrappers_simulated": False,
    }
    predictions.setflags(write=False)
    return VerificationTrace(predictions, tuple(events), tuple(operations), diagnostics)


def verify_dvap(X, targets, reference, config, *, assignments=None, model_ids=None,
                verifiers=None, seed=2026, collector_rows=None) -> VerificationTrace:
    """DVA-P is an assignment plug-in; use declared actual SHA256 execution."""
    trace = _verify_hash_tag(X, targets, reference, config,
                             assignments=assignments, model_ids=model_ids)
    trace.diagnostics.update({"method": "DVA-P-Adapted",
                              "integrity_protocol_adapter": "actual full SHA256; DVA-P paper does not uniquely prescribe this plug-in protocol"})
    return trace


def verify_iedi(X, targets, reference, config, *, assignments=None, model_ids=None,
                verifiers=None, seed=2026, collector_rows=None) -> VerificationTrace:
    """Literal Eq. (39) experimental tag, default eta=7919, retaining collisions."""
    modulus = getattr(config, "iedi_tag_modulus", 7919)
    if isinstance(modulus, bool) or not isinstance(modulus, (int, np.integer)) or modulus < 2:
        raise ValueError("I-EDI eta must be an integer >=2.")
    trace = _verify_hash_tag(X, targets, reference, config, modulus=int(modulus),
                             assignments=assignments, model_ids=model_ids)
    trace.diagnostics.update({"method": "I-EDI-Adapted", "tag_modulus": int(modulus),
                              "tag_formula": "int(SHA256(raw little-endian float32 C-order bytes),16) mod eta",
                              "tag_modulus_source": "Section V-A eta=7919" if modulus == 7919 else "explicit override",
                              "randomized_challenge_security_guarantees_inherited": False})
    return trace


def _ctcv_partition(request_seed: int, requested_blocks: int, payload_size: int):
    """Reconstruct the nonempty-byte partition from a public request seed.

    This is the same declared byte-cap adaptation used in Group 01. Sending
    the seed, block count and byte size transmits the complete rule without
    copying every boundary/permutation to every target. It is a transport
    adaptation; source blockchain broadcast/consensus remains unmodeled.
    """
    if requested_blocks < 1 or payload_size < 1:
        raise ValueError("CTCV partition needs positive blocks and payload size.")
    block_count = min(int(requested_blocks), int(payload_size))
    rng = np.random.default_rng(int(request_seed))
    cuts = np.sort(rng.choice(np.arange(1, payload_size), block_count - 1, replace=False))
    boundaries = np.concatenate(([0], cuts, [payload_size])).astype(np.int64)
    return boundaries, rng.permutation(block_count)


def verify_ctcv(X, targets, reference, config, *, assignments=None, model_ids=None,
                verifiers=None, seed=2026, collector_rows=None) -> VerificationTrace:
    """Real reordered-byte tags, packed vectors and v4 CR-member-bit verdicts.

    The first-group byte cap and deterministic cyclic starts are retained.
    VS is losslessly described by a public seed, block count, byte size and
    target cyclic start, instead of sending its complete table to each target.
    Selected collectors receive compact tags through a logical fanout adapter.
    Hashes of raw responses are diagnostics only; they cannot override Flag
    cases or CR[actual_tag] membership collisions.
    """
    X, targets, ids, routing = _verification_inputs(X, targets, reference, assignments, model_ids)
    if not len(targets):
        return VerificationTrace(np.empty(0, dtype=np.int8), (), (), {
            "method": "CTCV-Adapted", "status": "no_selected_targets"})
    request_bytes, result_bytes = _control_lengths(config)
    assignment_bytes = int(getattr(config, 'assignment_bytes', request_bytes))
    k = getattr(config, "ctcv_collusion_bound", 1)
    configured_ell = getattr(config, "ctcv_proof_vector_length", 0)
    block_override = getattr(config, "ctcv_block_count", 0)
    for value, name in ((k, "k"), (configured_ell, "ell"), (block_override, "block_count")):
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 0:
            raise ValueError(f"CTCV {name} must be a nonnegative integer.")
    ell = int(configured_ell) or 1000 * len(targets)
    pool = np.asarray(sorted(set(routing.values())) if verifiers is None else verifiers, dtype=np.int64)
    if len(np.unique(pool)) != len(pool) or np.intersect1d(pool, targets).size:
        raise ValueError("CTCV collector rows must be distinct and disjoint from targets.")
    if len(pool) < k + 1:
        raise ValueError("CTCV requires at least k+1 eligible collectors.")
    if collector_rows is None:
        collectors = np.random.default_rng(int(seed)).permutation(np.sort(pool))[:int(k) + 1]
    else:
        collectors = np.asarray(collector_rows, dtype=np.int64)
        if len(collectors) != k + 1 or len(np.unique(collectors)) != len(collectors) or not np.isin(collectors, pool).all():
            raise ValueError("CTCV must use exactly k+1 distinct eligible collector rows.")
    requested_blocks = int(block_override) or len(targets)
    block_count = min(requested_blocks, len(reference.payload))
    request_seed = int.from_bytes(hashlib.sha256(
        b"CTCV-raw-request-v3\0" + ids.astype("<i8").tobytes()).digest()[:8], "little")
    boundaries, shuffled = _ctcv_partition(request_seed, requested_blocks, len(reference.payload))
    # ell/deadline/identity remain in the common control header. This explicit
    # 20-byte payload lets each receiver reconstruct all VS bytes locally.
    expanded_partition_bytes = 4 * (len(boundaries) + len(shuffled))
    tag_width = max(1, ((ell - 1).bit_length() + 7) // 8)
    events, operations, actual_tags, correct_tags, hash_equal = [], [], [], [], []
    operations.append(_operation(AV, "vector", expanded_partition_bytes, stage="setup",
                                 provenance="actual seeded AV.Request partition and shuffle"))
    correct_cache: dict[int, tuple[str, int]] = {}
    for row in targets:
        row = int(row)
        verifier = routing[row]
        raw = canonical_descriptor_bytes(X[row])
        if len(raw) != len(reference.payload):
            raise ValueError("CTCV raw targets and reference must have equal byte sizes.")
        cyclic_start = int(ids[row]) % block_count
        request_payload = struct.pack("<QIII", request_seed, requested_blocks,
                                      len(reference.payload), cyclic_start)
        # Decode the real compact wire representation rather than treating
        # its claimed length as a substitute for an executable request.
        wire_seed, wire_blocks, wire_size, wire_start = struct.unpack("<QIII", request_payload)
        target_boundaries, target_shuffled = _ctcv_partition(wire_seed, wire_blocks, wire_size)
        order = np.roll(target_shuffled, -wire_start)
        transformed = b"".join(raw[target_boundaries[b]:target_boundaries[b + 1]] for b in order)
        response_hash = hashlib.sha256(transformed).hexdigest()
        actual = int(response_hash, 16) % ell
        # The byte-cap adapter repeats cyclic starts when targets outnumber
        # bytes. Identical correct reference challenges are computed once;
        # each actual target response is still hashed independently above.
        if cyclic_start not in correct_cache:
            correct_payload = b"".join(reference.payload[boundaries[b]:boundaries[b + 1]] for b in order)
            correct_hash = hashlib.sha256(correct_payload).hexdigest()
            correct_cache[cyclic_start] = (correct_hash, int(correct_hash, 16) % ell)
            operations.extend((
                _operation(AV, "vector", len(reference.payload), stage="aggregate", provenance="actual unique reference-byte block reordering"),
                _operation(AV, "hash", len(correct_payload), stage="aggregate", provenance="actual AV.Inquiry correct response SHA256; request-local cyclic-start cache"),
                _operation(AV, "compare", comparisons=1, stage="aggregate", provenance="actual unique correct integer modulo ell", arithmetic="modulo"),
            ))
        correct_hash, expected = correct_cache[cyclic_start]
        actual_tags.append(actual)
        correct_tags.append(expected)
        hash_equal.append(response_hash == correct_hash)
        events.extend((
            _event(AV, verifier, "assignment_control", control_bytes=assignment_bytes, target_row=row),
            _event(AV, row, "ctcv_request", request_payload, control_bytes=request_bytes, target_row=row),
        ))
        operations.extend((
            _operation(row, "vector", expanded_partition_bytes, target_row=row,
                       provenance="actual compact-request partition reconstruction"),
            _operation(row, "vector", len(raw), target_row=row, provenance="actual raw-byte block reordering"),
            _operation(row, "hash", len(transformed), target_row=row, provenance="actual Eq4b response SHA256"),
            _operation(row, "compare", comparisons=1, target_row=row, provenance="actual integer modulo ell", arithmetic="modulo"),
        ))
        tag_payload = actual.to_bytes(tag_width, "big")
        # A primary is an actual logical producer/relay, rather than an idle
        # identity used solely to obtain fictitious parallel queues. It receives
        # and range-checks the real response before forwarding compact tags.
        if not 0 <= actual < ell:
            raise ValueError('CTCV producer received an out-of-range modulo tag')
        producer_event = _event(row, verifier, 'ctcv_producer_proof', tag_payload, target_row=row)
        producer_event['proof_hop'] = 0
        events.append(producer_event)
        operations.append(_operation(verifier, 'compare', comparisons=1, target_row=row,
                                     proof_hop=0, provenance='actual producer modulo-tag range validation'))
        for collector in collectors:
            if int(collector) != verifier:
                forwarded = _event(verifier, collector, "ctcv_modulo_proof", tag_payload, target_row=row)
                forwarded['proof_hop'] = 1
                events.append(forwarded)
            operations.append(_operation(collector, "compare", comparisons=1, target_row=row,
                                         provenance="set actual modulo-proof occupied bit"))
    actual_tags = np.asarray(actual_tags, dtype=np.int64)
    correct_tags = np.asarray(correct_tags, dtype=np.int64)
    correct = _ctcv._bitvector(correct_tags, ell)
    # Every report is independently built from the actual received tags. No
    # byzantine state or voted report is fabricated from ground-truth labels.
    reports = tuple(_ctcv._bitvector(actual_tags, ell) for _ in collectors)
    correct_wire = np.packbits(correct, bitorder="big").tobytes()
    events.append(_event(AV, SC, "ctcv_correct_combination", correct_wire, stage="aggregate"))
    operations.append(_operation(AV, "vector", ell, stage="aggregate",
                                 provenance="actual correct-vector construction and packing"))
    for collector, report in zip(collectors, reports):
        report_wire = np.packbits(report, bitorder="big").tobytes()
        events.extend((
            _event(SC, collector, "ctcv_collector_request", control_bytes=request_bytes, stage="aggregate"),
            _event(collector, SC, "ctcv_proof_combination", report_wire, stage="aggregate"),
        ))
        operations.extend((
            _operation(collector, "vector", ell, stage="aggregate", provenance="actual independently built ell-bit report and packing"),
            _operation(SC, "vector", ell, stage="aggregate", provenance="actual report/correct comparison and report consistency scan"),
        ))
    verdict = _ctcv.co_verify_reports(reports, correct, collectors, ids[targets], int(k),
                                     response_tags=actual_tags)
    if not verdict["resolved"]:
        raise RuntimeError(verdict["reason"])
    predictions = np.asarray(verdict["predictions"], dtype=np.int8)
    operations.append(_operation(SC, "compare", comparisons=len(targets), stage="aggregate",
                                 provenance="actual four-Flag and CR[actual_tag] localization"))
    bad_ids = ids[targets][predictions != 0].astype("<i8").tobytes()
    events.append(_event(SC, AV, "ctcv_co_verification_result", bad_ids,
                         control_bytes=result_bytes, stage="aggregate"))
    membership = correct[actual_tags] == 1
    individual_match = actual_tags == correct_tags
    full_match = np.asarray(hash_equal, dtype=bool)
    predictions.setflags(write=False)
    return VerificationTrace(predictions, tuple(events), tuple(operations), {
        "method": "CTCV-Adapted", "content_comparison": "exact_hash",
        "source_steps": "Algorithms1-3; Eq4a/4b; CR member bits; original four Flag cases",
        "target_rows": targets.tolist(), "target_ids": ids[targets].tolist(),
        "collector_rows": collectors.tolist(), "collector_count": len(collectors),
        "primary_role": "logical response producer/relay; receives and validates a real tag before collector fanout",
        "collusion_bound": int(k), "collector_selection_seed": int(seed),
        "proof_vector_length": ell, "proof_byte_width": tag_width,
        "packed_report_bytes": len(correct_wire), "request_partition_bytes": CTCV_COMPACT_REQUEST_BYTES,
        "expanded_partition_bytes": expanded_partition_bytes,
        "request_partition_encoding": "little-endian uint64 request seed, uint32 requested blocks, payload byte size and cyclic start; receivers reconstruct the complete partition",
        "correct_challenges_computed": len(correct_cache),
        "correct_challenge_cache_hits": len(targets) - len(correct_cache),
        "correct_challenge_cache_scope": "one request, one exact anchor payload, one ell; no target response or verdict is cached",
        "request_seed": request_seed, "requested_block_count": requested_blocks,
        "block_count": block_count, "physical_byte_cap": len(reference.payload),
        "block_boundaries_bytes": boundaries.tolist(), "shuffled_block_order": shuffled.tolist(),
        "original_node_block_mapping_preserved": block_count == len(targets),
        "block_count_adaptation": "first-group v4 nonempty-byte block cap and repeated temporary-ID cyclic starts",
        "actual_proof_tags": actual_tags.tolist(), "correct_proof_tags": correct_tags.tolist(),
        "correct_combination_sha256": _ctcv._combination_digest(correct),
        "report_combination_sha256": [_ctcv._combination_digest(report) for report in reports],
        "correct_combination_set_bits": np.flatnonzero(correct).tolist(),
        "flag": verdict["flag"], "VR": verdict["VR"],
        "content_rule": "original vector Flag then CR[actual_modulo_tag] membership; individual correct tags and full hashes are auxiliary only",
        "full_hash_equal_count": int(full_match.sum()),
        "mod_tag_equal_count": int(individual_match.sum()),
        "membership_match_count": int(membership.sum()),
        "cross_tag_membership_collision_count": int(np.count_nonzero(membership & ~individual_match)),
        "modulus_collision_count": int(np.count_nonzero(individual_match & ~full_match)),
        "combination_or_modulus_masked_hash_mismatch_count": int(np.count_nonzero((predictions == 0) & ~full_match)),
        "anchor_authenticated": False,
        "consensus_transport_adapter": "target to assigned logical producer, then compact-tag fanout to k+1 collectors; local collector copies are not network sends",
        "unmodeled_protocol_components": ["blockchain consensus", "block producer signatures", "full-chain replica broadcast", "security wrappers"],
        "original_collusion_security_guarantees_inherited": False,
        "inquiry_depends_on_all_responses": True,
    })
