"""I-EDI O²RS sequencing with actual exact-hash vendor-tag inspection.

Source: Zhao et al. (2025), Intelligent Edge Data Integrity Verification With
Dynamic Unreliable Data Replica Selection. Equations (4), (8), (12), (20),
Algorithm 1 and Algorithm 2/Equation (39) are the retained reference points.

The static experiment has no cache QoS traces, learned verification history or
trusted vendor original. Uniform priors feed the original reliability formula;
O²RS determines audit order. A most-frequent complete reference-pool hash supplies
an explicitly estimated vendor anchor. Equation (39) is evaluated literally as
int(SHA256(raw little-endian float32 bytes)) mod eta, and exact tag inequality
determines corruption. No descriptor tolerance or similarity is introduced.
The default eta=7919 comes from the original Section V-A experimental setup.
Algorithm 2 describes random challenges, while Theorem 6 uses a different
nonce-prefixed hash definition; this implementation follows the published
Eq. (39) experimental path and does not inherit that theorem's guarantees.
All common targets must receive a prediction before accuracy is reported.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math

import numpy as np

from experiments.common.config import DetectionConfig
from experiments.common.dataset_adapter import AlgorithmInput
if __package__:
    from .adaptation_utils import descriptor_hash, exact_hash_anchor, readonly
    from .proposed_detection import DetectionResult
else:
    from adaptation_utils import descriptor_hash, exact_hash_anchor, readonly
    from proposed_detection import DetectionResult


# These are declared neutral initial conditions, not measurements or learned
# prehistory. QoS=1 follows Eq. (1b)'s equal-attribute normalization convention.
# With unavailable success/retrieval histories, symmetric SR=ER=0.5 gives VR=0.5.
QOS_CONFORMANCE_PRIOR = 1.0
SUCCESS_RATE_PRIOR = 0.5
RETRIEVAL_ERROR_PRIOR = 0.5
QOS_WEIGHT_BETA = 0.5


@dataclass(frozen=True)
class _SelectionPass:
    selected_mask: np.ndarray
    queue_after: float
    initial_objective: float
    final_objective: float
    stored_minimum: float
    exclusions_accepted: int
    maximum_queue: float


def _o2rs_objective(
    reliability_sum: float,
    selected_count: int,
    minimum_count: int,
    queue: float,
    control_V: float,
) -> float:
    """Exact deterministic objective in Eq. (20), including constant Ω=γ²/2.

    Current observations make the conditional expectation deterministic:
    Ω + Υ Σ_i d_i R_i + Q (γ - Σ_i d_i).
    Υ is named ``iedi_control_V`` in the experiment configuration.
    """

    omega = 0.5 * minimum_count * minimum_count
    return float(omega + control_V * reliability_sum + queue * (minimum_count - selected_count))


def _o2rs_exclusion_greedy(
    reliability: np.ndarray,
    minimum_count: int,
    queue: float,
    control_V: float,
) -> _SelectionPass:
    """Literal Algorithm 1 all-selected initialization and exclusion sweep.

    Traversal order is supplied by the caller and is fixed independently of
    labels. A candidate is excluded only if its Eq. (20) value is strictly below
    the stored minimum. As in Algorithm 1 line 10, the Eq. (12) queue update is
    inside the candidate loop; the stored minimum is not silently recomputed
    after that update. Running reliability sums make each candidate check O(1).
    """

    scores = np.asarray(reliability, dtype=np.float64)
    if scores.ndim != 1 or not np.isfinite(scores).all() or np.any((scores < 0) | (scores > 1)):
        raise ValueError("I-EDI reliability scores must be finite and in [0,1].")
    n = len(scores)
    if not 1 <= minimum_count <= n:
        raise ValueError("I-EDI requires 1 <= gamma <= the pending target count.")
    if not math.isfinite(queue) or queue < 0 or not math.isfinite(control_V) or control_V <= 0:
        raise ValueError("I-EDI requires a nonnegative finite queue and positive finite control V.")

    selected = np.ones(n, dtype=bool)
    selected_count = n
    reliability_sum = float(scores.sum())
    stored_minimum = _o2rs_objective(reliability_sum, n, minimum_count, queue, control_V)
    initial_objective = stored_minimum
    maximum_queue = float(queue)
    exclusions = 0
    for position, score in enumerate(scores):
        trial_sum = reliability_sum - float(score)
        trial_count = selected_count - 1
        trial_objective = _o2rs_objective(trial_sum, trial_count, minimum_count, queue, control_V)
        if trial_objective < stored_minimum:
            selected[position] = False
            reliability_sum = trial_sum
            selected_count = trial_count
            stored_minimum = trial_objective
            exclusions += 1
        queue = max(queue + minimum_count - selected_count, 0.0)
        maximum_queue = max(maximum_queue, queue)
    return _SelectionPass(
        selected_mask=readonly(selected, bool),
        queue_after=float(queue),
        initial_objective=initial_objective,
        final_objective=_o2rs_objective(reliability_sum, selected_count, minimum_count, queue, control_V),
        stored_minimum=stored_minimum,
        exclusions_accepted=exclusions,
        maximum_queue=maximum_queue,
    )


def detect_iedi(data: AlgorithmInput, config: DetectionConfig) -> DetectionResult:
    """Run O²RS-selected audit batches until all common evaluation targets pass.

    Each chosen target returns its actual SHA256-mod-eta tag, which is compared
    exactly with the independent surrogate vendor reference's tag. The remaining
    targets enter subsequent selection passes; prediction order is restored to
    the shared evaluation order. Queue changes affect sequencing only and
    cannot gain an accuracy advantage by dropping hard targets.
    """

    fraction = float(config.iedi_budget_fraction)
    control_V = float(config.iedi_control_V)
    modulus = getattr(config, "iedi_tag_modulus", 7919)
    if not math.isfinite(fraction) or not 0 < fraction <= 1:
        raise ValueError("iedi_budget_fraction must be finite and in (0,1].")
    if not math.isfinite(control_V) or control_V <= 0:
        raise ValueError("iedi_control_V must be positive and finite.")
    if isinstance(modulus, bool) or not isinstance(modulus, int) or modulus < 2:
        raise ValueError("iedi_tag_modulus must be an integer >= 2.")

    targets = np.asarray(data.evaluation_indices, dtype=np.int64)
    target_ids = readonly(data.model_ids[targets])
    reference_index, pool_hashes = exact_hash_anchor(data)
    reference_index = int(reference_index)
    anchor_hash = descriptor_hash(data.X[reference_index])
    correct_tag = int(anchor_hash, 16) % modulus
    verification_reliability = (1 + SUCCESS_RATE_PRIOR - RETRIEVAL_ERROR_PRIOR) / 2
    reliability_value = (
        QOS_WEIGHT_BETA * QOS_CONFORMANCE_PRIOR
        + (1 - QOS_WEIGHT_BETA) * verification_reliability
    )
    reliability = np.full(len(targets), reliability_value, dtype=np.float64)
    minimum_per_pass = max(1, math.ceil(len(targets) * fraction))
    queue = 0.0  # Eq. (12): original Q(0)=0, with no fabricated warm-up history.
    predictions = np.empty(len(targets), dtype=np.int8)
    coverage = np.zeros(len(targets), dtype=bool)
    # Source Algorithm 1 does not specify its candidate traversal order.
    # Temporary identity order supplies a fixed deterministic tie convention.
    pending = np.argsort(target_ids, kind="stable")
    audit_order: list[int] = []
    passes: list[dict] = []
    idle_passes = 0
    response_tags: list[int | None] = [None] * len(targets)
    response_hash_equal = np.zeros(len(targets), dtype=bool)
    # Literal source queue updates can repeatedly return an empty selection for
    # extreme control settings. Record failure instead of forcing a favorable
    # target or manufacturing predictions. This is a computational guard only.
    max_passes = max(8, 2 * len(targets))

    while len(pending) and len(passes) < max_passes:
        gamma = min(minimum_per_pass, len(pending))
        queue_before = queue
        selection = _o2rs_exclusion_greedy(reliability[pending], gamma, queue, control_V)
        queue = selection.queue_after
        chosen_positions = pending[selection.selected_mask]
        passes.append({
            "audit_pass": len(passes) + 1,
            "pending_count": len(pending), "gamma": gamma,
            "selected_count": len(chosen_positions),
            "queue_before": queue_before, "queue_after": queue,
            "maximum_queue": selection.maximum_queue,
            "initial_objective": selection.initial_objective,
            "final_objective": selection.final_objective,
            "stored_minimum": selection.stored_minimum,
            "exclusions_accepted": selection.exclusions_accepted,
        })
        if not len(chosen_positions):
            idle_passes += 1
            continue

        chosen_rows = targets[chosen_positions]
        # Algorithm 2 / Eq. (39): actual tags from actual cached content.
        # The paper's experiment uses eta=7919, retaining finite-modulus
        # collisions. Tags are calculated, never randomized or perturbed.
        batch_predictions = np.empty(len(chosen_rows), dtype=np.int8)
        for offset, (position, row_index) in enumerate(zip(chosen_positions, chosen_rows)):
            response_hash = descriptor_hash(data.X[row_index])
            actual_tag = int(response_hash, 16) % modulus
            response_tags[int(position)] = actual_tag
            response_hash_equal[position] = response_hash == anchor_hash
            batch_predictions[offset] = int(actual_tag != correct_tag)
        if np.any(coverage[chosen_positions]):
            raise AssertionError("I-EDI attempted to evaluate a target more than once.")
        predictions[chosen_positions] = batch_predictions
        coverage[chosen_positions] = True
        audit_order.extend(data.model_ids[chosen_rows].tolist())
        pending = pending[~selection.selected_mask]

    reference_pool = np.asarray(data.reference_indices, dtype=np.int64)
    reference_hash_support = sum(str(value) == anchor_hash for value in pool_hashes)
    covered_tag_equal = sum(tag == correct_tag for tag in response_tags if tag is not None)
    full_hash_equal_count = int(response_hash_equal[coverage].sum())
    tag_byte_width = max(1, ((modulus - 1).bit_length() + 7) // 8)
    tag_stream = hashlib.sha256(b"I-EDI-actual-tags-v4\0")
    equal_target_ids = []
    for position, tag in enumerate(response_tags):
        if tag is not None:
            tag_stream.update(int(target_ids[position]).to_bytes(8, "little", signed=True))
            tag_stream.update(tag.to_bytes(tag_byte_width, "big"))
            if tag == correct_tag:
                equal_target_ids.append(int(target_ids[position]))
    diagnostics = {
        "adaptation": "I-EDI O2RS static full-coverage sequencing with actual exact-hash tags",
        "content_comparison": "exact_hash",
        "source_steps": "Equations 4,8,12,20; Algorithm 1; Algorithm 2 and Equation 39; Section V-A eta=7919",
        "reference_constructor": "most frequent complete raw-descriptor SHA256 in the shared reference pool; history/temporary ID ties",
        "reference_indices": [reference_index],
        "reference_ids": [int(data.model_ids[reference_index])],
        "reference_sha256": anchor_hash,
        "reference_hash_support": reference_hash_support,
        "reference_pool_size": len(reference_pool),
        "qos_conformance_prior": QOS_CONFORMANCE_PRIOR,
        "success_rate_prior": SUCCESS_RATE_PRIOR,
        "retrieval_error_prior": RETRIEVAL_ERROR_PRIOR,
        "qos_weight_beta": QOS_WEIGHT_BETA,
        "verification_reliability_prior": verification_reliability,
        "reliability_score": reliability_value,
        "control_V": control_V, "minimum_count_fraction": fraction,
        "initial_queue": 0.0, "final_queue": queue,
        "queue_update_granularity": "per candidate, literal Algorithm 1 line 10",
        "objective": "gamma^2/2 + V*sum(selected_R_i) + Q*(gamma-selected_count)",
        "batch_sizes": [row["selected_count"] for row in passes],
        "selection_passes": passes, "idle_passes": idle_passes,
        "coverage_audit_order": audit_order,
        "coverage_count": int(coverage.sum()), "target_count": len(targets),
        "full_coverage": bool(coverage.all()),
        "tag_formula": "int(SHA256(canonical raw little-endian float32 C-order bytes),16) mod eta",
        "tag_modulus": modulus,
        "tag_modulus_source": "original Section V-A experimental setting" if modulus == 7919 else "explicit fixed configuration override",
        "challenge_parameter_adaptation": "fixed-modulus Eq39 experiment path, default eta=7919 from Section V-A; Theorem6's independently sampled nonce-prefixed hash is not used",
        "fixed_eta_is_full_sha256_domain": modulus == 2**256,
        "correct_tag": correct_tag,
        "target_tag_stream_sha256": tag_stream.hexdigest(),
        "tag_stream_order": "evaluation order; each covered temporary ID is followed by its actual tag bytes",
        "tag_byte_width": tag_byte_width,
        "tag_equal_target_ids": equal_target_ids,
        "full_hash_equal_count": full_hash_equal_count,
        "tag_equal_count": covered_tag_equal,
        "additional_modulus_collision_count": covered_tag_equal - full_hash_equal_count,
        "verification_rule": "corrupted iff actual hash-mod-eta tag != independently constructed anchor correct tag",
        "anchor_limitation": "hash-frequency-selected anchor is not an authenticated vendor original; legitimate unequal model bytes are rejected",
        "guarantees_not_inherited": "original randomized-challenge freshness/security; trusted vendor correctness; long-term L-URS guarantees; observed QoS/history discrimination",
    }
    if len(pending):
        return DetectionResult(
            method="I-EDI-Adapted", status="failure", predictions=None,
            target_ids=target_ids, verifier_ids=None, gt_success=True, ground_truth=None,
            reason=(f"Literal O2RS sequencing did not cover {len(pending)} remaining targets "
                    f"within {max_passes} audit passes; no partial-coverage accuracy is reported."),
            diagnostics=diagnostics,
        )
    if len(audit_order) != len(targets) or len(set(audit_order)) != len(targets):
        raise AssertionError("I-EDI full-coverage audit is not a permutation of the common targets.")
    return DetectionResult(
        method="I-EDI-Adapted", status="success", predictions=readonly(predictions, np.int8),
        target_ids=target_ids,
        verifier_ids=readonly(np.full(len(targets), data.model_ids[reference_index]), data.model_ids.dtype),
        gt_success=True, ground_truth=None, diagnostics=diagnostics,
    )
