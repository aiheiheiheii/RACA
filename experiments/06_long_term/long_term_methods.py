"""Causal, independently stateful methods for the 200-round experiment.

RACA-HIM reuses the existing manuscript GT kernel and the exact Section IV-C
all-pair marginal-gain/best-prefix scheduler. The Group 05 v2 local-search
extension is deliberately not dispatched here: it is absent from the supplied
latest manuscript. Published replica baselines retain their own subset
objectives and actual hash/tag verification, with explicit resource adapters.
No method accepts a corruption mask, event table, generator prototype or future
response trace. See reports/06_long_term_baseline_adaptation.md for scope.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import importlib
import math
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np

from experiments.common.config import DetectionConfig
from experiments.common.dataset_adapter import AlgorithmInput
from experiments.common.similarity import cosine_pairwise, within_interval

_gt = importlib.import_module("experiments.01_detection.proposed_detection")
_iedi = importlib.import_module("experiments.01_detection.iedi_adapted")
_joint = importlib.import_module("experiments.05_joint_decision.joint_methods")
_objective = importlib.import_module("experiments.05_joint_decision.joint_objective")
_native_iedi = importlib.import_module("experiments.06_long_term.native_iedi")

ALGORITHM_VERSION = "long-term-native-iedi-dvap-ediv-v1"
GROUND_TRUTH_SOURCE = "experiments.01_detection.proposed_detection.build_ground_truth"
JOINT_DECISION_SOURCE = "experiments.05_joint_decision.joint_methods.raca_him_greedy"
IEDI_SELECTION_SOURCE = "experiments.01_detection.iedi_adapted._o2rs_exclusion_greedy"
GT_FULL_MATRIX_LIMIT = 256
GT_BLOCK_ROWS = 128
METHODS = ("RACA-HIM", "I-EDI-Adapted", "DVA-P-Adapted", "Age-Only", "Random")
NATIVE_METHODS = ("RACA-HIM", "I-EDI", "DVA-P+EDI-V")


def _frozen(values: Any, dtype=None) -> np.ndarray:
    array = np.ascontiguousarray(values, dtype=dtype)
    return np.frombuffer(array.tobytes(), dtype=array.dtype).reshape(array.shape)


@dataclass
class MethodState:
    """Private-to-one-method observations; initialized without prehistory."""

    anomaly: np.ndarray
    last_valid_round: np.ndarray
    normal_count: np.ndarray
    corrupted_count: np.ndarray
    failed_count: np.ndarray
    attempted_count: np.ndarray
    qos_history: list[list[tuple[int, float]]]
    service_history: list[list[tuple[int, float, float]]]
    virtual_number_queue: float = 0.0
    virtual_delay_queues: np.ndarray = field(default_factory=lambda: np.empty(0))
    previous_selected: set[int] = field(default_factory=set)
    last_round: int = 0

    def snapshot(self) -> dict[str, Any]:
        """Return copies for auditing; no caller can mutate the live history."""
        return {
            "anomaly": _frozen(self.anomaly),
            "last_valid_round": _frozen(self.last_valid_round),
            "normal_count": _frozen(self.normal_count),
            "corrupted_count": _frozen(self.corrupted_count),
            "failed_count": _frozen(self.failed_count),
            "attempted_count": _frozen(self.attempted_count),
            "virtual_number_queue": float(self.virtual_number_queue),
            "virtual_delay_queues": _frozen(self.virtual_delay_queues),
            "previous_selected": tuple(sorted(self.previous_selected)),
            "last_round": int(self.last_round),
        }


@dataclass(frozen=True)
class MethodRoundResult:
    assignments: tuple[tuple[int, int], ...]
    outcomes: tuple[Any, ...]
    gt_success: bool | None
    gt_rows: tuple[int, ...]
    budget: int
    status: str
    failure_reason: str
    age_before: np.ndarray
    h_before: np.ndarray
    risk_before: np.ndarray
    diagnostics: dict[str, Any]
    verification_diagnostics: tuple[dict[str, Any], ...]

    def round_stats(self) -> dict[str, Any]:
        """Scalar method-round fields, separate from offline event scoring."""
        loads: dict[int, float] = {}
        for outcome in self.outcomes:
            if outcome.attempted:
                loads[outcome.verifier_id] = loads.get(outcome.verifier_id, 0.0) + outcome.total_ms
        attempted = sum(bool(item.attempted) for item in self.outcomes)
        completed = sum(bool(item.completed) for item in self.outcomes)
        makespan = max(loads.values(), default=0.0)
        return {
            "budget": self.budget,
            "selected_count": len(self.assignments),
            "attempted_count": attempted,
            "completed_count": completed,
            "missed_count": len(self.assignments) - completed,
            "gt_success": self.gt_success,
            "gt_size": len(self.gt_rows),
            "mean_verification_age": float(self.age_before.mean()),
            "mean_historical_anomaly": float(self.h_before.mean()),
            "mean_risk": float(self.risk_before.mean()),
            "verification_makespan_ms": makespan,
            "deadline_satisfied": all(bool(item.deadline_satisfied) for item in self.outcomes),
            "status": self.status,
            "failure_reason": self.failure_reason,
            "gt_observation_count": self.diagnostics.get("gt_observation_count", 0),
            "gt_failed_round": self.gt_success is False,
            "method_algorithm_version": ALGORITHM_VERSION,
            "virtual_number_queue": self.diagnostics.get("virtual_number_queue_after"),
            "virtual_delay_queue_sum_seconds": self.diagnostics.get("virtual_delay_queue_sum_seconds"),
        }


class LongTermMethod:
    """One persistent method operating only through a current-round view.

    ``config`` is the uppercase ``config_long_term`` module. ``temporary_ids``
    is a seeded label-independent permutation aligned with physical model rows.
    Only the hash baselines store ``enrollment_tags``; those must originate from
    the simulator's explicit t=0 enrollment, before any dynamic event.
    """

    def __init__(self, method: str, config: Any, seed: int,
                 temporary_ids: np.ndarray,
                 enrollment_tags: Mapping[int, str] | None = None) -> None:
        if method not in set(METHODS) | set(NATIVE_METHODS):
            raise ValueError(f"Unknown long-term method: {method}")
        self.method, self.config = method, config
        self.family = {"I-EDI-Adapted": "I-EDI", "DVA-P-Adapted": "DVA-P",
                       "DVA-P+EDI-V": "DVA-P"}.get(method, method)
        self.native = method in ("I-EDI", "DVA-P+EDI-V")
        self.n, self.d = int(config.N), int(config.D)
        ids = np.asarray(temporary_ids)
        if (ids.shape != (self.n,) or not np.issubdtype(ids.dtype, np.integer)
                or len(np.unique(ids)) != self.n):
            raise ValueError("Temporary IDs must uniquely identify all N model rows")
        if not 1 <= int(config.M) <= int(config.K) <= self.n:
            raise ValueError("Require 1 <= M <= K <= N")
        self.temporary_ids = _frozen(ids, np.int64)
        self._rows_by_id = np.argsort(self.temporary_ids, kind="stable")
        self._id_to_row = {int(value): row for row, value in enumerate(ids)}
        self._rng = np.random.default_rng(int(seed))
        self._enrollment: Mapping[int, str] = MappingProxyType({})
        if self.family in ("I-EDI", "DVA-P"):
            if enrollment_tags is None or set(enrollment_tags) != set(range(self.n)):
                raise ValueError("Hash baselines require trusted t=0 enrollment of every model")
            tags = {int(row): str(tag) for row, tag in enrollment_tags.items()}
            if any(len(tag) != 64 or any(c not in "0123456789abcdef" for c in tag)
                   for tag in tags.values()):
                raise ValueError("Enrollment must contain actual SHA256 hexadecimal digests")
            self._enrollment = MappingProxyType(tags)
        self.state = MethodState(
            anomaly=np.full(self.n, float(config.H0), dtype=np.float64),
            last_valid_round=np.full(self.n, -1, dtype=np.int64),
            normal_count=np.zeros(self.n, dtype=np.int64),
            corrupted_count=np.zeros(self.n, dtype=np.int64),
            failed_count=np.zeros(self.n, dtype=np.int64),
            attempted_count=np.zeros(self.n, dtype=np.int64),
            qos_history=[[] for _ in range(self.n)],
            service_history=[[] for _ in range(self.n)],
            virtual_delay_queues=np.zeros(self.n, dtype=np.float64),
        )

    def visible_history(self, round_id: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Current age/risk use only valid outcomes in preceding rounds."""
        if int(round_id) <= self.state.last_round:
            raise ValueError("History may only be requested for a later round")
        last = self.state.last_valid_round
        age = np.ones(self.n, dtype=np.float64)
        valid = last >= 0
        age[valid] = np.minimum((int(round_id) - last[valid]) / float(self.config.A_MAX), 1.0)
        h = self.state.anomaly.copy()
        risk = float(self.config.ALPHA) * age + (1.0 - float(self.config.ALPHA)) * h
        if not (np.all((age >= 0) & (age <= 1)) and np.all((h >= 0) & (h <= 1))
                and np.all((risk >= 0) & (risk <= 1))):
            raise ValueError("Observable age, anomaly state and risk must remain in [0,1]")
        return _frozen(age), _frozen(h), _frozen(risk)

    def _large_gt_step(self, data: AlgorithmInput, candidates: np.ndarray,
                       available: np.ndarray) -> Any:
        """Shared GT mathematics in bounded row blocks for a necessary large K."""
        size = len(candidates)
        counts = np.empty(size, dtype=np.int64)
        means = np.empty(size, dtype=np.float64)
        candidate_x, responsive = data.X[candidates], available[candidates]
        for start in range(0, size, GT_BLOCK_ROWS):
            stop = min(size, start + GT_BLOCK_ROWS)
            similarity = cosine_pairwise(candidate_x[start:stop], candidate_x)
            support = within_interval(similarity, self.config.TAU_LOWER, self.config.TAU_UPPER)
            support &= responsive[start:stop, None] & responsive[None, :]
            support[np.arange(stop - start), np.arange(start, stop)] = False
            count = support.sum(axis=1, dtype=np.int64)
            counts[start:stop] = count
            totals = np.where(support, similarity, 0.0).sum(axis=1, dtype=np.float64)
            means[start:stop] = np.divide(totals, count, out=np.full(stop - start, -1.0), where=count > 0)
        threshold = _gt.majority_threshold(size, (2, 3))
        admitted = responsive & (counts + 1 >= threshold)
        selected = _gt.rank_ground_truth_candidates(
            candidates[admitted], data.history, data.model_ids,
            counts[admitted], means[admitted], policy="history_consensus",
        )[:int(self.config.M)]
        success = len(selected) == int(self.config.M)
        return _gt.GroundTruthResult(
            success=success,
            selected_indices=_frozen(selected if success else [], np.int64),
            candidate_indices=_frozen(candidates, np.int64),
            support_counts=_frozen(counts, np.int64), threshold=threshold,
            expansion_trace=(_gt.GroundTruthStep(size, threshold, int(admitted.sum())),),
            reason="" if success else "Insufficient responsive two-thirds-supported candidates",
            pairwise_comparison_count=int(responsive.sum()) * (int(responsive.sum()) - 1),
            selection_evidence=_frozen(means),
        )

    def _ground_truth(self, view: Any, h: np.ndarray) -> tuple[Any, dict[int, np.ndarray], dict]:
        """Interact only with the history-ranked candidate prefix actually needed."""
        ordered = np.lexsort((self.temporary_ids, h))
        # Unobserved rows are finite placeholders, outside the kernel reference
        # partition. They never contribute peer support or reveal responses.
        x = np.zeros((self.n, self.d), dtype=np.float32)
        x[:, 0] = 1.0
        available = np.zeros(self.n, dtype=bool)
        observed = 0
        size = int(self.config.K)
        trace: list[dict[str, int]] = []
        comparisons = 0
        while True:
            added = ordered[observed:size]
            response = view.observe_candidates(added)
            if not np.array_equal(np.asarray(response.ids), added):
                raise ValueError("Candidate response IDs are not aligned with the requested model rows")
            present = np.asarray(response.available, dtype=bool)
            descriptors = np.asarray(response.descriptors)
            if present.shape != (len(added),) or descriptors.shape != (len(added), self.d):
                raise ValueError("Candidate responses have invalid dimensions")
            if np.any(present):
                x[added[present]] = descriptors[present]
            available[added] = present
            observed = size
            candidates = ordered[:size]
            data = AlgorithmInput(
                X=x, model_ids=self.temporary_ids, history=h,
                reference_indices=candidates,
                evaluation_indices=ordered[size:],
            )
            if size <= GT_FULL_MATRIX_LIMIT:
                settings = DetectionConfig(
                    K=size, M=int(self.config.M), reference_pool_size=self.n,
                    expansion_step=int(self.config.DELTA_K), h0=float(self.config.H0),
                    tau_lower=float(self.config.TAU_LOWER), tau_upper=float(self.config.TAU_UPPER),
                    dim=self.d, H=float(self.config.H), p_fail=0.0,
                )
                result = _gt.build_ground_truth(
                    data, settings, responder_available=available, allow_expansion=False,
                    support_quorum=(2, 3), selection_policy="history_consensus",
                )
            else:
                result = self._large_gt_step(data, candidates, available)
            trace.extend(asdict(step) for step in result.expansion_trace)
            comparisons += result.pairwise_comparison_count
            if result.success or size == self.n:
                references = {int(row): _frozen(x[row], np.float32) for row in result.selected_indices}
                return result, references, {
                    "gt_kernel_source": GROUND_TRUTH_SOURCE,
                    "gt_support_quorum": [2, 3], "gt_selection_policy": "history_consensus",
                    "gt_refresh": "every round from this method's preceding history",
                    "gt_observation_count": observed,
                    "gt_failed_responses": int((~available[candidates]).sum()),
                    "gt_expansion_trace": trace,
                    "gt_pairwise_comparison_count": int(comparisons),
                    "gt_failure_reason": result.reason,
                    "gt_failed_candidate_rows": candidates[~available[candidates]].tolist(),
                }
            size = min(size + int(self.config.DELTA_K), self.n)

    def _baseline_roles(self, round_id: int) -> tuple[np.ndarray, np.ndarray]:
        """Label-independent rotating processors; enrolled digests are references."""
        start = ((int(round_id) - 1) * int(self.config.M)) % self.n
        positions = (start + np.arange(int(self.config.M))) % self.n
        verifiers = self._rows_by_id[positions]
        targets = np.setdiff1d(np.arange(self.n), verifiers, assume_unique=True)
        return targets, verifiers

    def _priority_plan(self, order: np.ndarray, targets: np.ndarray,
                       verifiers: np.ndarray, cost: np.ndarray,
                       budget: int) -> tuple[tuple[tuple[int, int], ...], dict]:
        """Neutral minimum projected completion mapping; no marginal J."""
        loads = np.zeros(len(verifiers), dtype=np.float64)
        pairs: list[tuple[int, int]] = []
        skipped = 0
        for position in order:
            i = int(position)
            finish = loads + cost[:, i]
            legal = np.isfinite(cost[:, i]) & (verifiers != targets[i]) & (finish <= self.config.T_ROUND_MAX_MS)
            if not np.any(legal):
                skipped += 1
                continue
            options = np.flatnonzero(legal)
            g = int(options[np.lexsort((self.temporary_ids[verifiers[options]], finish[options]))[0]])
            pairs.append((int(targets[i]), int(verifiers[g])))
            loads[g] = finish[g]
            if len(pairs) >= budget:
                break
        return tuple(pairs), {
            "assignment_adapter": "minimum projected completion; temporary verifier ID ties",
            "targets_skipped_for_deadline": skipped,
            "planned_makespan_ms": float(loads.max(initial=0.0)),
        }

    def _reliability(self, round_id: int) -> tuple[np.ndarray, dict]:
        """I-EDI Eqs.2--4/8 from this method's completed observations only."""
        if self.native:
            return self._native_reliability(round_id)
        normal, anomalous = self.state.normal_count, self.state.corrupted_count
        sr = (normal + 0.5) / (normal + anomalous + 1.0)
        er = (self.state.failed_count + 0.5) / (self.state.attempted_count + 1.0)
        vr = (1.0 + sr - er) / 2.0
        floor = float(getattr(self.config, "IEDI_QOS_DECAY_FLOOR", 0.5))
        scale = float(getattr(self.config, "IEDI_QOS_DECAY_SCALE", 20.0))
        qos = np.ones(self.n, dtype=np.float64)
        for row, records in enumerate(self.state.qos_history):
            if records:
                rounds, values = np.asarray(records, dtype=np.float64).T
                weights = (1.0 - floor) * np.exp((rounds - round_id) / scale) + floor
                qos[row] = float(np.dot(weights, values) / weights.sum())
        beta = float(getattr(self.config, "IEDI_QOS_WEIGHT", 0.5))
        reliability = beta * qos + (1.0 - beta) * vr
        return reliability, {
            "iedi_qos_proxy": "own past valid task timeliness max(1-finish_ms/deadline,0); failed interactions=0",
            "iedi_qos_no_history_prior": 1.0,
            "iedi_qos_normalization_bounds_ms": [0.0, float(self.config.T_ROUND_MAX_MS)],
            "iedi_qos_decay_floor": floor, "iedi_qos_decay_scale_rounds": scale,
            "iedi_sr_proxy": "(own normal verdicts+0.5)/(own valid verdicts+1)",
            "iedi_er_proxy": "(own failed interactions+0.5)/(own attempted interactions+1)",
            "iedi_verification_reliability": "VR=(1+SR-ER)/2, source Eq4",
            "iedi_qos_weight": beta,
            "mean_iedi_reliability": float(reliability.mean()),
            "min_iedi_reliability": float(reliability.min()),
            "max_iedi_reliability": float(reliability.max()),
        }

    def _native_reliability(self, round_id: int) -> tuple[np.ndarray, dict]:
        """Source equations with explicitly disclosed causal service inputs.

        The supplied workload has no QWS service log. Past actual response
        latency/throughput supply two QoS attributes; corruption verdicts never
        determine SR/ER. No-history values are explicit initialization choices.
        """
        attempts = self.state.attempted_count
        responses = self.state.normal_count + self.state.corrupted_count
        prior_rate = float(getattr(self.config, "IEDI_NO_HISTORY_RATE_PRIOR", .5))
        sr = np.divide(responses, attempts, out=np.full(self.n, prior_rate), where=attempts > 0)
        er = np.divide(self.state.failed_count, attempts, out=np.full(self.n, prior_rate), where=attempts > 0)
        vr = _native_iedi.verification_representation(sr, er)
        qos = np.full(self.n, float(getattr(self.config, "IEDI_NO_HISTORY_QOS_PRIOR", .5)))
        population = [entry[1:] for history in self.state.service_history for entry in history]
        if population:
            observed = np.asarray(population, dtype=np.float64)
            lower, upper = observed.min(axis=0), observed.max(axis=0)
            for row, history in enumerate(self.state.service_history):
                if not history:
                    continue
                values = np.asarray(history, dtype=np.float64)
                normalized = _native_iedi.normalize_qos_attributes(
                    values[:, 1:], (False, True), lower, upper)
                conformance = _native_iedi.qos_conformance(normalized)
                qos[row] = float(_native_iedi.long_term_qos(
                    values[:, 0], conformance, current_time=round_id,
                    alpha=float(self.config.IEDI_QOS_DECAY_FLOOR),
                    decay_scale=float(self.config.IEDI_QOS_DECAY_SCALE)))
        reliability = _native_iedi.reliability_representation(qos, vr, beta=float(self.config.IEDI_QOS_WEIGHT))
        return reliability, {
            "iedi_reliability_equations": "source Eqs1-4 and8",
            "iedi_qos_input_adapter": "past own response finish_ms (negative) and descriptor bytes/total_ms (positive); timeout deadline and zero throughput",
            "iedi_qos_bounds": "extrema of all own service observations strictly before the scheduling round",
            "iedi_sr_input_adapter": "received responses/attempted requests; tag mismatch still counts as a response",
            "iedi_er_input_adapter": "failed response interactions/attempted messages; no integrity-label input",
            "iedi_no_history_rate_prior": prior_rate,
            "iedi_no_history_qos_prior": float(getattr(self.config, "IEDI_NO_HISTORY_QOS_PRIOR", .5)),
            "iedi_original_qws_log_available": False,
            "mean_iedi_reliability": float(reliability.mean()),
        }

    def _iedi_plan(self, targets: np.ndarray, verifiers: np.ndarray,
                   cost: np.ndarray, budget: int, age: np.ndarray,
                   round_id: int) -> tuple[tuple[tuple[int, int], ...], dict]:
        reliability, diagnostics = self._reliability(round_id)
        fraction = float(getattr(self.config, "IEDI_MINIMUM_FRACTION", 0.10))
        gamma = min(budget, max(1, math.ceil(fraction * len(targets))))
        # The source does not define traversal order. Inspect exclusion options
        # from most reliable to least reliable, so a source exclusion removes
        # reliable replicas first. Youngest age and temporary ID resolve ties.
        traversal = (np.argsort(self.temporary_ids[targets], kind="stable") if self.native
                     else np.lexsort((self.temporary_ids[targets], age[targets], -reliability[targets])))
        queue_before = float(self.state.virtual_number_queue)
        selection = _iedi._o2rs_exclusion_greedy(
            reliability[targets[traversal]], gamma, queue_before,
            float(getattr(self.config, "IEDI_CONTROL_V", 1.0)),
        )
        source_positions = traversal[selection.selected_mask]
        # A hard maximum budget is additional to the paper's minimum-count
        # objective. Only source-selected targets remain eligible; never fill
        # an empty source choice or replenish a failed verification.
        source_priority = source_positions[np.lexsort((
            self.temporary_ids[targets[source_positions]], reliability[targets[source_positions]],
        ))] if self.native else source_positions[np.lexsort((
            self.temporary_ids[targets[source_positions]],
            -age[targets[source_positions]], reliability[targets[source_positions]],
        ))]
        bounded = source_priority[:budget]
        pairs, mapping = self._priority_plan(bounded, targets, verifiers, cost, budget)
        # The literal candidate-loop queue is retained across rounds. Explicit
        # cap/capacity correction charges source-selected work never dispatched.
        adapter_shortfall = len(source_positions) - len(pairs)
        self.state.virtual_number_queue = (selection.queue_after if self.native
                                          else max(selection.queue_after + adapter_shortfall, 0.0))
        diagnostics.update(mapping)
        diagnostics.update({
            "iedi_selection_source": IEDI_SELECTION_SOURCE,
            "source_steps": "Eqs2,3,4,8,12,20 and Algorithm1; actual Eq39 modulo tags",
            "source_selected_count": len(source_positions),
            "source_exclusions_accepted": selection.exclusions_accepted,
            "source_initial_objective": selection.initial_objective,
            "source_final_objective": selection.final_objective,
            "source_stored_minimum": selection.stored_minimum,
            "gamma": gamma,
            "virtual_number_queue_before": queue_before,
            "virtual_number_queue_after_literal_sweep": selection.queue_after,
            "virtual_number_queue_adapter_shortfall": adapter_shortfall,
            "virtual_number_queue_after": float(self.state.virtual_number_queue),
            "queue_update_granularity": "literal source per-candidate Eq12; one post-sweep common-capacity shortfall correction",
            "target_traversal_adapter": "descending own reliability for exclusion, ascending age and temporary ID ties",
            "budget_adapter": "among source-selected targets retain at most B by ascending reliability, descending age, temporary ID",
            "tag_modulus": int(getattr(self.config, "IEDI_TAG_MODULUS", 7919)),
        })
        if self.native:
            diagnostics.update({
                "queue_update_granularity": "literal source per-candidate Eq12; capacity shortfall logged only",
                "target_traversal_adapter": "temporary ID; no Age/h ordering",
                "budget_adapter": "among source-selected targets retain at most B by ascending reliability then temporary ID",
                "tag_modulus": None,
                "tag_protocol": "Algorithm2 independent eta per replica per round; Eq39 SHA256(payload)%eta",
                "challenge_mode": self.config.IEDI_CHALLENGE_MODE,
                "challenge_domain_reproduction_choice": [self.config.IEDI_ETA_MIN, self.config.IEDI_ETA_MAX],
            })
        return pairs, diagnostics

    def _dvap_plan(self, targets: np.ndarray, verifiers: np.ndarray,
                   cost: np.ndarray, budget: int, age: np.ndarray) -> tuple[tuple[tuple[int, int], ...], dict]:
        """DVA-P Algorithm1 priority/add/swap with persistent native queues.

        Source variables are endpoint inspection bits, not peer assignments.
        Minimum intrinsic-cost feasible mapping introduces that missing degree
        of freedom. Native Eq21 still decides endpoint add/replacement, while
        shared B and maximum accumulated verifier load screen feasibility.
        """
        deadline = float(self.config.T_ROUND_MAX_MS)
        control = float(getattr(self.config, "DVAP_CONTROL_LAMBDA", 1.0))
        queue = self.state.virtual_delay_queues[targets]
        order = (np.argsort(self.temporary_ids[targets], kind="stable") if self.native
                 else np.lexsort((self.temporary_ids[targets], -age[targets])))
        verifier_order = np.argsort(self.temporary_ids[verifiers], kind="stable")
        loads = np.zeros(len(verifiers), dtype=np.float64)
        selected: list[tuple[int, int, float]] = []
        total_ms, queue_dot = 0.0, 0.0
        additions, replacements, checks = 0, 0, 0
        zero_queues = bool(np.all(queue == 0.0))
        if zero_queues:
            # This ordering depends only on the current resource matrix and
            # legal identities, not on any selected-set load or outcome.
            intrinsic_legal = (np.isfinite(cost)
                               & (verifiers[:, None] != targets[None, :]))
            intrinsic_choices = np.where(intrinsic_legal[verifier_order],
                                         cost[verifier_order], np.inf)
            intrinsic_best = verifier_order[np.argmin(intrinsic_choices, axis=0)]

        def value(total: float, weighted: float) -> float:
            z = math.exp(-total / 1000.0)
            return float(control * 2.0 * z / (1.0 + z) + weighted
                         - deadline / 1000.0 * float(queue.sum()))

        def map_set(positions: list[int]) -> tuple[list[tuple[int, int, float]], np.ndarray] | None:
            if len(positions) > budget:
                return None
            mapped, trial_loads = [], np.zeros(len(verifiers), dtype=np.float64)
            for i in positions:
                legal = (np.isfinite(cost[:, i]) & (verifiers != targets[i])
                         & (trial_loads + cost[:, i] <= deadline))
                if not np.any(legal):
                    return None
                options = np.flatnonzero(legal)
                g = int(options[np.lexsort((self.temporary_ids[verifiers[options]], cost[options, i]))[0]])
                amount = float(cost[g, i])
                mapped.append((i, g, amount))
                trial_loads[g] += amount
            return mapped, trial_loads

        previous = [int(i) for i in order if int(targets[i]) in self.state.previous_selected]
        complement = [int(i) for i in order if int(targets[i]) not in self.state.previous_selected]
        previous_plan, complement_plan = map_set(previous), map_set(complement)
        priority_initialized = False
        if complement_plan is not None:
            comp, comp_loads = complement_plan
            comp_total = sum(job[2] for job in comp)
            comp_dot = sum(queue[i] * amount / 1000.0 for i, _, amount in comp)
            if previous_plan is None:
                prefer = True
            else:
                prev, _ = previous_plan
                prev_total = sum(job[2] for job in prev)
                prev_dot = sum(queue[i] * amount / 1000.0 for i, _, amount in prev)
                prefer = (comp_total >= prev_total if zero_queues
                          else value(comp_total, comp_dot) <= value(prev_total, prev_dot))
            if prefer:
                selected, loads = comp, comp_loads
                total_ms, queue_dot = comp_total, comp_dot
                priority_initialized = True
        selected_ids = {i for i, _, _ in selected}
        for position in order:
            i = int(position)
            if i in selected_ids:
                continue
            legal = (np.isfinite(cost[:, i]) & (verifiers != targets[i])
                     & (loads + cost[:, i] <= deadline))
            if len(selected) < budget and np.any(legal):
                options = np.flatnonzero(legal)
                g = int(options[np.lexsort((self.temporary_ids[verifiers[options]], cost[options, i]))[0]])
                amount = float(cost[g, i])
                new_dot = queue_dot + float(queue[i]) * amount / 1000.0
                checks += 1
                improves = (amount >= 0.0 if zero_queues
                            else value(total_ms + amount, new_dot) <= value(total_ms, queue_dot))
                if improves:
                    selected.append((i, g, amount))
                    selected_ids.add(i)
                    loads[g] += amount
                    total_ms += amount
                    queue_dot = new_dot
                    additions += 1
                    continue
            if not selected:
                continue
            # Enumerate exactly the native single-endpoint replacement options;
            # independently map the new endpoint after each proposed removal.
            old_i = np.asarray([job[0] for job in selected], dtype=np.int64)
            old_amount = np.asarray([job[2] for job in selected], dtype=np.float64)
            if zero_queues and legal[intrinsic_best[i]]:
                # The global intrinsic-cost minimum already fits. Removing
                # any old task cannot make it infeasible or reveal a cheaper
                # processor, including the same temporary-ID tie break.
                # Retain the original vector arithmetic/comparisons below.
                best_g = int(intrinsic_best[i])
                new_g = np.full(len(selected), best_g, dtype=np.int64)
                amounts = cost[new_g, i]
                possible = np.ones(len(selected), dtype=bool)
            else:
                old_g = np.asarray([job[1] for job in selected], dtype=np.int64)
                removed = np.broadcast_to(loads, (len(selected), len(verifiers))).copy()
                removed[np.arange(len(selected)), old_g] -= old_amount
                feasible = (np.isfinite(cost[:, i])[None, :]
                            & (verifiers != targets[i])[None, :]
                            & (removed + cost[:, i][None, :] <= deadline))
                # Sorting by temporary IDs supplies exact ties after the
                # intrinsic cost minimization, without a load/risk preference.
                choices = np.where(feasible[:, verifier_order], cost[verifier_order, i][None, :], np.inf)
                new_g = verifier_order[np.argmin(choices, axis=1)]
                amounts = cost[new_g, i]
                possible = feasible[np.arange(len(selected)), new_g]
            sums = total_ms - old_amount + amounts
            dots = queue_dot - queue[old_i] * old_amount / 1000.0 + queue[i] * amounts / 1000.0
            checks += int(possible.sum())
            if zero_queues:
                scores = np.where(possible, sums, -np.inf)
                best = float(scores.max(initial=-np.inf))
                if not np.isfinite(best) or best < total_ms:
                    continue
                ties = np.flatnonzero(scores == best)
            else:
                scores = np.full(len(selected), np.inf)
                for j in np.flatnonzero(possible):
                    scores[j] = value(float(sums[j]), float(dots[j]))
                best = float(scores.min(initial=np.inf))
                if not np.isfinite(best) or best > value(total_ms, queue_dot):
                    continue
                ties = np.flatnonzero(scores == best)
            tie_order = np.lexsort((self.temporary_ids[verifiers[new_g[ties]]],
                                    self.temporary_ids[targets[old_i[ties]]]))
            j = int(ties[tie_order[0]])
            old = selected[j]
            g, amount = int(new_g[j]), float(amounts[j])
            loads[old[1]] -= old[2]
            loads[g] += amount
            selected_ids.remove(old[0])
            selected_ids.add(i)
            selected[j] = (i, g, amount)
            total_ms, queue_dot = float(sums[j]), float(dots[j])
            replacements += 1
        pairs = tuple((int(targets[i]), int(verifiers[g])) for i, g, _ in selected)
        return pairs, {
            "source_steps": "DVA-P Algorithm1 previous-selected/complement priority, native add/swap, Eq13 queues, Eq21 objective",
            "target_iteration_adapter": ("temporary ID; no Age/h ordering" if self.native
                                         else "descending own verification age, then temporary ID"),
            "assignment_adapter": "minimum intrinsic pair cost among currently capacity-feasible processors; temporary ID ties",
            "source_scope": "binary replica subset; source has no independent peer-verifier decision",
            "priority_previous_count": len(previous),
            "priority_unselected_complement_count": len(complement),
            "priority_complement_initialized": priority_initialized,
            "priority_capacity_adapter": "the full native complement may initialize only if common B and accumulated verifier deadline permit it; otherwise source empty fallback",
            "control_lambda": control, "source_objective_delay_unit": "seconds",
            "source_objective_without_Omega": value(total_ms, queue_dot),
            "source_additions_accepted": additions,
            "source_replacements_accepted": replacements,
            "source_objective_checks": checks,
            "source_queue_before_sum_seconds": float(self.state.virtual_delay_queues.sum()),
            "zero_queue_comparison": "analytic monotone ordering of summed endpoint delays avoids saturated-sigmoid ties",
            "uses_joint_objective_for_decision": False,
            "planned_makespan_ms": float(loads.max(initial=0.0)),
        }

    def _update(self, row: int, outcome: str, round_id: int) -> None:
        rho, eta = float(self.config.RHO), float(self.config.ETA)
        h = float(self.state.anomaly[row])
        if outcome == "normal":
            self.state.anomaly[row] = (1.0 - rho) * h
            self.state.last_valid_round[row] = round_id
        elif outcome == "corrupted":
            self.state.anomaly[row] = h + rho * (1.0 - h)
            self.state.last_valid_round[row] = round_id
        elif outcome in ("timeout", "invalid"):
            self.state.anomaly[row] = h + rho * eta * (1.0 - h)
        else:
            raise ValueError(f"Unrecognized observable verification outcome: {outcome}")

    def step(self, view: Any) -> MethodRoundResult:
        """Plan, interact, then update once; never inspect offline event truth."""
        t = int(view.round_id)
        if t != self.state.last_round + 1 or int(view.n) != self.n or int(view.d) != self.d:
            raise ValueError("Methods must consume consecutive aligned rounds starting at one")
        age, h, risk = self.visible_history(t)
        self._current_step_round = t
        self._round_last_before = self.state.last_valid_round.copy()
        diagnostics: dict[str, Any] = {"method": self.method, "algorithm_version": ALGORITHM_VERSION}
        gt_success: bool | None = None
        references: dict[int, np.ndarray] = {}
        if self.method in ("RACA-HIM", "Age-Only", "Random"):
            ground_truth, references, details = self._ground_truth(view, h)
            diagnostics.update(details)
            gt_success = bool(ground_truth.success)
            verifiers = np.asarray(ground_truth.selected_indices, dtype=np.int64)
            targets = np.setdiff1d(np.arange(self.n), verifiers, assume_unique=True)
            if not gt_success:
                self.state.last_round = t
                # A failed GT candidate interaction is an actual invalid
                # observation, not a state verdict. It raises h only next round.
                records = []
                for row in diagnostics["gt_failed_candidate_rows"]:
                    self._update(int(row), "invalid", t)
                    records.append(self._history_record(int(row), None, "gt_interaction", "invalid", age, h, risk))
                return MethodRoundResult(
                    (), (), False, (), int(math.ceil(self.config.BETA_B * (self.n - self.config.M))),
                    "gt_unavailable", str(ground_truth.reason), age, h, risk,
                    diagnostics, tuple(records),
                )
        else:
            targets, verifiers = self._baseline_roles(t)
            diagnostics.update({
                "reference_protocol": "per-target trusted t=0 enrollment; no RACA-HIM GT",
                "processor_panel_adapter": "M processors rotate by temporary-ID order each round; no response queried before scheduling",
                "processor_rows": verifiers.tolist(), "gt_observation_count": 0,
            })
        budget = int(math.ceil(float(self.config.BETA_B) * len(targets)))
        components = view.resource_matrix(targets, verifiers)
        cost = np.asarray(components["total_ms"], dtype=np.float64)
        if cost.shape != (len(verifiers), len(targets)):
            raise ValueError("Resource matrix must have axes (verifier, target)")
        if self.method == "RACA-HIM":
            problem = _objective.JointProblem(
                target_ids=self.temporary_ids[targets], verifier_ids=self.temporary_ids[verifiers],
                risk_scores=risk[targets], costs=cost, budget=budget,
                lambda_value=float(self.config.LAMBDA), deadline_ms=float(self.config.T_ROUND_MAX_MS),
                compute_costs=components["compute_ms"], transfer_costs=components["transfer_ms"],
                network_costs=components["network_ms"],
            )
            decision = _joint.raca_him_greedy(problem)
            pairs = tuple((self._id_to_row[i], self._id_to_row[g]) for i, g in decision.assignments)
            diagnostics.update(decision.diagnostics)
            diagnostics["joint_decision_source"] = JOINT_DECISION_SOURCE
        elif self.family == "I-EDI":
            pairs, details = self._iedi_plan(targets, verifiers, cost, budget, age, t)
            diagnostics.update(details)
        elif self.family == "DVA-P":
            pairs, details = self._dvap_plan(targets, verifiers, cost, budget, age)
            diagnostics.update(details)
        else:
            if self.method == "Age-Only":
                order = np.lexsort((self.temporary_ids[targets], -age[targets]))
                diagnostics["target_rule"] = "descending verification age then temporary ID; h excluded"
            else:
                # First sort the neutral domain to make the independent RNG
                # semantics stable even if a caller changes row enumeration.
                order = self._rng.permutation(np.argsort(self.temporary_ids[targets], kind="stable"))
                diagnostics["target_rule"] = "uniform independent target permutation"
            pairs, details = self._priority_plan(order, targets, verifiers, cost, budget)
            diagnostics.update(details)
        if (len(pairs) > budget or len({i for i, _ in pairs}) != len(pairs)
                or any(i == g or i not in targets or g not in verifiers for i, g in pairs)):
            raise ValueError("Method generated an illegal verification assignment")
        protocol = {"I-EDI-Adapted": "iedi_tag", "DVA-P-Adapted": "sha256",
                    "I-EDI": "iedi_native", "DVA-P+EDI-V": "edi_v"}.get(self.method, "cosine")
        diagnostics["verification_protocol"] = protocol
        expected: Mapping[int, str | int] | None = None
        if protocol == "sha256":
            expected = self._enrollment
        elif protocol == "iedi_tag":
            modulus = int(getattr(self.config, "IEDI_TAG_MODULUS", 7919))
            expected = {row: int(digest, 16) % modulus for row, digest in self._enrollment.items()}
        outcomes = tuple(view.execute(
            pairs, protocol=protocol, references=references, expected_tags=expected,
            budget=budget, allowed_verifier_ids=verifiers,
            tau_lower=float(self.config.TAU_LOWER), tau_upper=float(self.config.TAU_UPPER),
        ))
        if tuple((item.target_id, item.verifier_id) for item in outcomes) != pairs:
            raise ValueError("Execution outcomes are not aligned with this method's assignments")
        if self.method == "I-EDI":
            diagnostics["source_malicious_flag_count"] = sum(
                bool(item.protocol_evidence.get("source_flagged_malicious")) for item in outcomes)
            diagnostics["source_timeout_flag_count"] = sum(
                item.protocol_evidence.get("source_status") == "timeout" for item in outcomes)
            diagnostics["cdr_scope"] = "completed corrupted-content verdicts only; source timeout flags reported separately"
        elif self.method == "DVA-P+EDI-V":
            diagnostics["vendor_tree_hash_operations_performed"] = sum(
                item.protocol_evidence["vendor_tree_hash_operations"] for item in outcomes)
            diagnostics["response_hash_operations_performed"] = sum(
                item.protocol_evidence["response_hash_operations_performed"] for item in outcomes)
            diagnostics["resource_adapter"] = "common dataset pair cost; native hash operations recorded separately, not wall-clock timing"
        records: list[dict[str, Any]] = []
        # GT feedback is applied only after all choices and outcomes. Failed
        # candidates later attempted as targets get their actual target event
        # once; do not multiply one external failure into two h updates.
        target_rows = {item.target_id for item in outcomes if item.attempted}
        for row in diagnostics.get("gt_failed_candidate_rows", []):
            if row not in target_rows:
                self._update(int(row), "invalid", t)
                records.append(self._history_record(int(row), None, "gt_interaction", "invalid", age, h, risk))
        if gt_success:
            for row in verifiers:
                self._update(int(row), "normal", t)
                records.append(self._history_record(int(row), None, "gt_admission", "normal", age, h, risk))
        consumed = np.zeros(self.n, dtype=np.float64)
        for outcome in outcomes:
            row = int(outcome.target_id)
            if outcome.attempted:
                self._update(row, outcome.outcome, t)
                self.state.attempted_count[row] += 1
                consumed[row] = float(outcome.total_ms) / 1000.0
                if outcome.completed and outcome.outcome == "normal":
                    self.state.normal_count[row] += 1
                elif outcome.completed and outcome.outcome == "corrupted":
                    self.state.corrupted_count[row] += 1
                else:
                    self.state.failed_count[row] += 1
                timeliness = (max(0.0, 1.0 - float(outcome.finish_ms) / float(self.config.T_ROUND_MAX_MS))
                              if outcome.completed else 0.0)
                self.state.qos_history[row].append((t, timeliness))
                if self.method == "I-EDI":
                    latency = float(outcome.finish_ms) if outcome.completed else float(self.config.T_ROUND_MAX_MS)
                    throughput = self.d * 4.0 / max(float(outcome.total_ms), 1e-12) if outcome.completed else 0.0
                    self.state.service_history[row].append((t, latency, throughput))
            record = self._history_record(row, int(outcome.verifier_id), "target", outcome.outcome, age, h, risk)
            record.update(asdict(outcome))
            record["phase"] = "target"
            records.append(record)
        if self.family == "DVA-P":
            self.state.virtual_delay_queues = np.maximum(
                self.state.virtual_delay_queues + consumed - float(self.config.T_ROUND_MAX_MS) / 1000.0, 0.0,
            )
            self.state.previous_selected = {int(i) for i, _ in pairs}
            diagnostics["virtual_delay_queue_sum_seconds"] = float(self.state.virtual_delay_queues.sum())
            diagnostics["virtual_delay_queue_max_seconds"] = float(self.state.virtual_delay_queues.max(initial=0.0))
            diagnostics["queue_update"] = "Eq13 once per real round; failed attempts consume their full pair cost; no artificial backlog"
        self.state.last_round = t
        status = "ok" if pairs else "no_tasks_selected"
        reason = "" if pairs else "No target selected by the method under its source rule and common capacity constraints"
        return MethodRoundResult(
            pairs, outcomes, gt_success,
            tuple(int(row) for row in verifiers) if gt_success else (), budget,
            status, reason, age, h, risk, diagnostics, tuple(records),
        )

    def _history_record(self, row: int, verifier: int | None, phase: str,
                        outcome: str, age: np.ndarray, h: np.ndarray,
                        risk: np.ndarray) -> dict[str, Any]:
        return {
            "phase": phase, "target_id": row, "verifier_id": verifier,
            "temporary_target_id": int(self.temporary_ids[row]),
            "temporary_verifier_id": None if verifier is None else int(self.temporary_ids[verifier]),
            "outcome": outcome, "age_before": float(age[row]),
            "h_before": float(h[row]), "risk_before": float(risk[row]),
            "h_after": float(self.state.anomaly[row]),
            "age_after": (1.0 if self.state.last_valid_round[row] < 0 else min(
                (self._current_step_round - int(self.state.last_valid_round[row])) / float(self.config.A_MAX), 1.0)),
            "last_valid_round_before": int(self._round_last_before[row]),
            "last_valid_round_after": int(self.state.last_valid_round[row]),
            "attempted": True,
            "completed": phase == "gt_admission",
        }
