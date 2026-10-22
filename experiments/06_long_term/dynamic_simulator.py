"""Causal, method-local interaction ports for the Group 06 dynamic workload.

Only the orchestrator owns ``DynamicScenario``. Algorithms receive a narrow
``MethodEnvironment`` containing callables, never the case, resource profiles,
truth table, baseline descriptors, or failure matrix. This is an API boundary
for research code, not a security sandbox against Python introspection.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
from typing import Callable, Mapping, Sequence

import numpy as np

from dataset.load_dataset import resource_costs
from experiments.common.similarity import cosine_pairwise, within_interval
from . import native_edi_v, native_iedi


def _immutable(values: np.ndarray, dtype=None) -> np.ndarray:
    array = np.ascontiguousarray(values, dtype=dtype)
    return np.frombuffer(array.tobytes(), dtype=array.dtype).reshape(array.shape)


def descriptor_digest(descriptor: np.ndarray) -> str:
    """Canonical little-endian float32 bytes; no generator labels enter tags."""
    return hashlib.sha256(np.asarray(descriptor, dtype="<f4").tobytes(order="C")).hexdigest()


def validate_dynamic_case(case: Mapping, *, n: int, d: int, rounds: int,
                          cr_percent: float | None = None) -> None:
    """Runtime guards for the user's future run; never invoked at import."""
    required = {"normal_X", "corrupted_event_X", "event_model_ids", "corrupted_mask",
                "fail_uniform", "onset_round", "recovery_round"}
    missing = required.difference(case)
    if missing:
        raise ValueError(f"Dynamic case is missing fields: {sorted(missing)}")
    normal = np.asarray(case["normal_X"])
    event_x = np.asarray(case["corrupted_event_X"])
    ids = np.asarray(case["event_model_ids"])
    onsets = np.asarray(case["onset_round"])
    recoveries = np.asarray(case["recovery_round"])
    mask = np.asarray(case["corrupted_mask"])
    failures = np.asarray(case["fail_uniform"])
    count = len(ids)
    if normal.shape != (n, d) or event_x.shape != (count, d):
        raise ValueError("Normal/event descriptor dimensions do not match the configuration")
    if normal.dtype != np.float32 or event_x.dtype != np.float32:
        raise ValueError("Dynamic descriptors must preserve the dataset float32 precision")
    if (ids.shape != (count,) or ids.dtype.kind not in "iu"
            or len(np.unique(ids)) != count or np.any(ids < 0) or np.any(ids >= n)):
        raise ValueError("Each event must map to one unique real model ID")
    if onsets.shape != (count,) or recoveries.shape != (count,):
        raise ValueError("Event times must align row-for-row with event_model_ids")
    if (onsets.dtype.kind not in "iu" or recoveries.dtype.kind not in "iu"
            or np.any(onsets < 1) or np.any(onsets > rounds)
            or np.any(recoveries <= onsets) or np.any(recoveries > rounds + 1)):
        raise ValueError("Events must use valid onset-inclusive/recovery-exclusive times")
    if mask.shape != (rounds, n) or mask.dtype.kind != "b":
        raise ValueError("corrupted_mask must be a boolean (rounds,N) matrix")
    if (failures.shape != (rounds, n) or not np.all(np.isfinite(failures))
            or np.any(failures < 0.0) or np.any(failures >= 1.0)):
        raise ValueError("fail_uniform must contain (rounds,N) finite draws in [0,1)")
    if not np.all(np.isfinite(normal)) or not np.all(np.isfinite(event_x)):
        raise ValueError("Descriptors must be finite")
    if np.any(np.linalg.norm(normal, axis=1) == 0) or np.any(np.linalg.norm(event_x, axis=1) == 0):
        raise ValueError("Zero descriptors cannot support cosine verification")
    t = np.arange(1, rounds + 1)[:, None]
    expected = np.zeros((rounds, n), dtype=bool)
    expected[:, ids] = (t >= onsets[None, :]) & (t < recoveries[None, :])
    if not np.array_equal(mask, expected):
        raise ValueError("Event table and corrupted_mask disagree")
    if cr_percent is not None:
        # Dataset's documented half-up event-count convention.
        from decimal import Decimal, ROUND_HALF_UP
        expected_count = int((Decimal(n) * Decimal(str(cr_percent)) / 100)
                             .quantize(Decimal("1"), rounding=ROUND_HALF_UP))
        if count != expected_count:
            raise ValueError("Event identity count does not match cr_dynamic")


@dataclass(frozen=True)
class CandidateResponses:
    ids: np.ndarray
    descriptors: np.ndarray
    available: np.ndarray

    def __post_init__(self) -> None:
        for name in ("ids", "descriptors", "available"):
            object.__setattr__(self, name, _immutable(getattr(self, name)))


@dataclass(frozen=True)
class VerificationOutcome:
    target_id: int
    verifier_id: int
    outcome: str
    completed: bool
    attempted: bool
    similarity: float | None
    digest: str | None
    tag: int | None
    compute_ms: float
    transfer_ms: float
    network_ms: float
    total_ms: float
    start_ms: float
    finish_ms: float
    deadline_satisfied: bool
    failure_reason: str = ""
    protocol: str = ""
    protocol_evidence: dict = field(default_factory=dict)


class MethodEnvironment:
    """A method-local current-round port. No offline truth fields are exposed."""

    __slots__ = ("n", "d", "rounds", "deadline_ms", "_begin", "_observe",
                 "_resource", "_execute", "_enroll", "_state")

    def __init__(self, *, n: int, d: int, rounds: int, deadline_ms: float,
                 begin: Callable, observe: Callable, resource: Callable,
                 execute: Callable, enroll: Callable, state: Callable):
        self.n, self.d, self.rounds = n, d, rounds
        self.deadline_ms = float(deadline_ms)
        self._begin, self._observe, self._resource = begin, observe, resource
        self._execute, self._enroll, self._state = execute, enroll, state

    @property
    def round_id(self) -> int:
        return int(self._state()["round_id"])

    @property
    def gt_observation_count(self) -> int:
        return int(self._state()["gt_observation_count"])

    def begin_round(self, round_id: int) -> None:
        self._begin(round_id)

    def observe_candidates(self, ids: Sequence[int] | np.ndarray) -> CandidateResponses:
        """Attempt GT candidate interaction, exposing current responses only.

        This port is for the declared GT construction. It is not a selected
        target verification, does not reset target Age, and cannot detect an
        event for the offline evaluator. Repeated candidate requests are cached.
        """
        return self._observe(ids)

    def resource_matrix(self, target_ids: Sequence[int], verifier_ids: Sequence[int]) -> dict:
        """Public current-round pair costs, shape (verifier,target)."""
        return self._resource(target_ids, verifier_ids)

    def trusted_enrollment(self) -> dict[int, str]:
        """t=0 trusted registration; returns digests, never normal descriptors."""
        return self._enroll()

    def execute(self, assignments: Sequence[tuple[int, int]], *, protocol: str,
                references: Mapping[int, np.ndarray] | None = None,
                expected_tags: Mapping[int, str | int] | None = None,
                allowed_verifier_ids: Sequence[int] | None = None,
                budget: int | None = None, tau_lower: float = 0.997,
                tau_upper: float = 1.0) -> list[VerificationOutcome]:
        return self._execute(assignments, protocol=protocol, references=references,
                             expected_tags=expected_tags,
                             allowed_verifier_ids=allowed_verifier_ids, budget=budget,
                             tau_lower=tau_lower, tau_upper=tau_upper)


class DynamicScenario:
    """Private exogenous workload shared by all independent method ports."""

    def __init__(self, case: Mapping, profiles: Mapping, *, p_fail: float = 0.05,
                 sigma: float = 0.5, deadline_ms: float = 800.0,
                 beta_budget: float = 0.10, validate: bool = True,
                 protocol_settings: Mapping | None = None):
        normal = np.asarray(case["normal_X"])
        self.n, self.d = normal.shape
        self.rounds = len(case["corrupted_mask"])
        if validate:
            validate_dynamic_case(case, n=self.n, d=self.d, rounds=self.rounds)
        if not (0 <= p_fail <= 1 and sigma >= 0 and deadline_ms > 0 and 0 < beta_budget <= 1):
            raise ValueError("Invalid dynamic resource/failure/budget parameters")
        self._case = case
        self._profiles = profiles
        self._p_fail, self._sigma = float(p_fail), float(sigma)
        self._deadline, self._beta = float(deadline_ms), float(beta_budget)
        self._event_position = np.full(self.n, -1, dtype=np.int64)
        self._event_position[np.asarray(case["event_model_ids"], dtype=int)] = np.arange(
            len(case["event_model_ids"]))
        self._method_names: set[str] = set()
        self._protocol_settings = dict(protocol_settings or {})

    def _ids(self, values: Sequence[int], *, unique: bool = True) -> np.ndarray:
        ids = np.asarray(values)
        if ids.ndim != 1 or (ids.size and ids.dtype.kind not in "iu"):
            raise ValueError("IDs must be a one-dimensional integer sequence")
        ids = ids.astype(np.int64, copy=False)
        if np.any(ids < 0) or np.any(ids >= self.n):
            raise ValueError("Model ID outside the configured domain")
        if unique and len(np.unique(ids)) != len(ids):
            raise ValueError("Repeated IDs are not allowed")
        return ids

    def _descriptors(self, round_id: int, ids: np.ndarray) -> np.ndarray:
        # Only requested rows are copied; no (T,N,d) cube is materialized.
        values = np.array(self._case["normal_X"][ids], copy=True)
        active = self._case["corrupted_mask"][round_id - 1, ids]
        if np.any(active):
            positions = self._event_position[ids[active]]
            if np.any(positions < 0):
                raise RuntimeError("Active model has no matching event descriptor")
            values[active] = self._case["corrupted_event_X"][positions]
        return values

    def for_method(self, method: str) -> MethodEnvironment:
        if method in self._method_names:
            raise ValueError(f"A method port already exists: {method}")
        self._method_names.add(str(method))
        state = {"round_id": 0, "executed": False, "enrolled": False,
                 "gt_observation_count": 0}
        observed: dict[int, tuple[bool, np.ndarray | None]] = {}
        vendor: native_edi_v.NativeEDIV | None = None
        settings = self._protocol_settings
        challenge_rng = np.random.default_rng(int(settings.get("iedi_seed", 0)))
        enrolled_digests: dict[int, str] = {}

        def current_round() -> int:
            t = int(state["round_id"])
            if not 1 <= t <= self.rounds:
                raise RuntimeError("Current-round interaction requires begin_round(t)")
            return t

        def begin(round_id: int) -> None:
            if isinstance(round_id, bool) or int(round_id) != round_id:
                raise ValueError("round_id must be an integer")
            if int(round_id) != int(state["round_id"]) + 1 or round_id > self.rounds:
                raise ValueError("Method ports advance once through consecutive rounds 1..T")
            state.update(round_id=int(round_id), executed=False, gt_observation_count=0)
            observed.clear()

        def observe(ids: Sequence[int]) -> CandidateResponses:
            t = current_round()
            if state["executed"]:
                raise RuntimeError("GT observations must precede the round's target execution")
            rows = self._ids(ids)
            unseen = np.asarray([int(i) for i in rows if int(i) not in observed], dtype=np.int64)
            if len(unseen):
                responses = self._descriptors(t, unseen)
                for i, response in zip(unseen, responses):
                    available = bool(self._case["fail_uniform"][t - 1, i] >= self._p_fail)
                    observed[int(i)] = (available, _immutable(response) if available else None)
                state["gt_observation_count"] += len(unseen)
            # Missing responses contain NaN, never an inaccessible descriptor.
            matrix = np.full((len(rows), self.d), np.nan, dtype=np.float32)
            available = np.asarray([observed[int(i)][0] for i in rows], dtype=bool)
            for j, i in enumerate(rows):
                if available[j]:
                    matrix[j] = observed[int(i)][1]
            return CandidateResponses(rows, matrix, available)

        def resources(target_ids: Sequence[int], verifier_ids: Sequence[int]) -> dict:
            t = current_round()
            targets, verifiers = self._ids(target_ids), self._ids(verifier_ids)
            target_grid = np.broadcast_to(targets[None, :], (len(verifiers), len(targets)))
            verifier_grid = np.broadcast_to(verifiers[:, None], target_grid.shape)
            return {key: _immutable(value, np.float64) for key, value in resource_costs(
                target_grid, verifier_grid, d=self.d, sigma=self._sigma,
                round_id=t, profiles=self._profiles).items()}

        def enroll() -> dict[int, str]:
            nonlocal vendor
            if state["round_id"] != 0 or state["enrolled"]:
                raise RuntimeError("Trusted enrollment is available exactly once at t=0")
            # Legitimate registration assumption requires no active event at t=0.
            if np.any(np.asarray(self._case["onset_round"]) < 1):
                raise ValueError("Trusted enrollment requires an event-free initial state")
            state["enrolled"] = True
            enrolled_digests.update({i: descriptor_digest(row)
                                     for i, row in enumerate(self._case["normal_X"])})
            if method == "DVA-P+EDI-V":
                vendor = native_edi_v.NativeEDIV(
                    {i: np.asarray(row, dtype="<f4").tobytes(order="C")
                     for i, row in enumerate(self._case["normal_X"])},
                    seed=int(settings["dvap_ediv_seed"]),
                    block_bytes=int(settings["dvap_ediv_block_bytes"]),
                    detection_probability=float(settings["dvap_ediv_detection_probability"]),
                    assumed_corrupted_blocks=int(settings["dvap_ediv_min_corrupted_blocks"]),
                )
            return dict(enrolled_digests)

        def execute(assignments, *, protocol, references, expected_tags,
                    allowed_verifier_ids, budget, tau_lower, tau_upper):
            t = current_round()
            if state["executed"]:
                raise RuntimeError("Targets may only be executed once per method-round")
            raw_tasks = list(assignments)
            targets = self._ids([i for i, _ in raw_tasks])
            verifiers = self._ids([g for _, g in raw_tasks], unique=False)
            tasks = list(zip(map(int, targets), map(int, verifiers)))
            if protocol not in {"cosine", "sha256", "iedi_tag", "iedi_native", "edi_v"}:
                raise ValueError(f"Unknown verification protocol: {protocol}")
            required_protocol = {"I-EDI": "iedi_native", "DVA-P+EDI-V": "edi_v"}.get(method)
            if required_protocol is not None and protocol != required_protocol:
                raise ValueError("Formal native baseline cannot substitute another protocol")
            if np.any(targets == verifiers):
                raise ValueError("A target cannot verify itself")
            if allowed_verifier_ids is None:
                allowed = set(references or {}) if protocol == "cosine" else set(range(self.n))
            else:
                allowed = set(map(int, self._ids(allowed_verifier_ids)))
            if not set(map(int, verifiers)).issubset(allowed):
                raise ValueError("Assignment uses a verifier outside the declared legal pool")
            # The pool is disjoint from targets for the common N-M budget.
            if set(map(int, targets)).intersection(allowed):
                raise ValueError("Verifier identities must be excluded from the target pool")
            upper = int(np.ceil(self._beta * (self.n - len(allowed))))
            if budget is not None and (isinstance(budget, bool) or int(budget) != budget):
                raise ValueError("Budget must be an integer target count")
            selected_budget = upper if budget is None else int(budget)
            if selected_budget < 0 or selected_budget > upper or len(tasks) > selected_budget:
                raise ValueError("Assignment exceeds the common per-round target budget")
            if protocol == "cosine":
                if references is None or not 0 <= tau_lower <= tau_upper <= 1:
                    raise ValueError("Cosine verification requires legal current GT references")
                for g in allowed:
                    available, response = observed.get(g, (False, None))
                    if (not available or g not in references
                            or not np.array_equal(response, references[g])):
                        raise ValueError("GT reference must come from a successful current-round observation")
            elif protocol in {"sha256", "iedi_tag"}:
                if not state["enrolled"] or expected_tags is None:
                    raise ValueError("Native tag verification requires method-local trusted enrollment")
                if not set(map(int, targets)).issubset(expected_tags):
                    raise ValueError("Missing a target's registered integrity tag")
            elif not state["enrolled"] or (protocol == "edi_v" and vendor is None):
                raise ValueError("Native protocols require their own explicit t=0 enrollment")
            costs = resource_costs(targets, verifiers, d=self.d, sigma=self._sigma,
                                   round_id=t, profiles=self._profiles)
            loads: dict[int, float] = {}
            scheduled: list[tuple[float, float]] = []
            for j, g in enumerate(verifiers):
                start = loads.get(int(g), 0.0)
                finish = start + float(costs["total_ms"][j])
                scheduled.append((start, finish))
                loads[int(g)] = finish
            # Native policies and ablations all obey the same hard deadline.
            if any(finish > self._deadline + 1e-9 for _, finish in scheduled):
                raise ValueError(f"Selected assignment exceeds the {self._deadline:g} ms verifier-load deadline")
            state["executed"] = True
            target_x = self._descriptors(t, targets)
            # Challenges are drawn only after the scheduling decision. Their
            # RNG streams are independent of selection, failures and events.
            edi_requests = vendor.prepare_round(t, targets.tolist()) if protocol == "edi_v" else {}
            iedi_requests = {}
            if protocol == "iedi_native":
                mode = str(settings["iedi_challenge_mode"])
                domain = ({"eta_min": int(settings["iedi_eta_min"]),
                           "eta_max": int(settings["iedi_eta_max"])}
                          if mode == native_iedi.ALGORITHM2_RANDOM_MODULUS else {})
                iedi_requests = {int(i): native_iedi.make_challenge(
                    t, int(i), rng=challenge_rng, mode=mode,
                    fixed_eta=int(settings["iedi_fixed_eta"]), **domain,
                ) for i in targets}
            result: list[VerificationOutcome] = []
            for j, (i, g) in enumerate(tasks):
                start, finish = scheduled[j]
                failed_target = bool(self._case["fail_uniform"][t - 1, i] < self._p_fail)
                failed_verifier = bool(self._case["fail_uniform"][t - 1, g] < self._p_fail)
                score, digest, tag = None, None, None
                reason = ""
                evidence: dict = {}
                payload = np.asarray(target_x[j], dtype="<f4").tobytes(order="C")
                if protocol == "edi_v":
                    evidence = vendor.diagnostics(edi_requests[i])
                    evidence["response_hash_operations_performed"] = 0
                elif protocol == "iedi_native":
                    request = iedi_requests[i]
                    evidence = {"source": "I-EDI Algorithm2/Eq39", "round_id": t,
                                "replica_id": i, "eta": request.eta, "mode": request.mode}
                if failed_target or failed_verifier:
                    outcome, completed = "timeout", False
                    reason = ("target_response_failure" if failed_target else "verifier_response_failure")
                    if protocol == "iedi_native":
                        checked = native_iedi.verify_digest_response(
                            iedi_requests[i], enrolled_digests[i], None,
                            arrival_ms=None, deadline_ms=self._deadline)
                        evidence.update(source_flagged_malicious=checked.flagged_malicious,
                                        source_status=checked.status, response_tag=None)
                    elif protocol == "edi_v":
                        evidence["proof"] = None
                elif protocol == "cosine":
                    score = float(cosine_pairwise(target_x[j:j + 1], np.asarray(references[g])[None, :])[0, 0])
                    outcome = "normal" if bool(within_interval(np.asarray(score), tau_lower, tau_upper)) else "corrupted"
                    completed = True
                elif protocol == "iedi_native":
                    request = iedi_requests[i]
                    tag = native_iedi.compute_tag(payload, request.eta)
                    checked = native_iedi.verify_digest_response(
                        request, enrolled_digests[i], tag,
                        arrival_ms=finish, deadline_ms=self._deadline)
                    completed = checked.status != "timeout"
                    outcome = ("normal" if checked.integrity_match else "corrupted") if completed else "timeout"
                    evidence.update(response_tag=tag, expected_tag=checked.expected_tag,
                                    source_flagged_malicious=checked.flagged_malicious,
                                    source_status=checked.status)
                elif protocol == "edi_v":
                    request = edi_requests[i]
                    proof = vendor.respond(i, payload, request)
                    matches = vendor.verify(i, request, proof)
                    evidence["proof"] = proof.as_record()
                    evidence["integrity_match"] = matches
                    evidence["response_hash_operations_performed"] = evidence["response_subtree_hash_operations"]
                    outcome, completed = ("normal" if matches else "corrupted"), True
                else:
                    digest = descriptor_digest(target_x[j])
                    if protocol == "iedi_tag":
                        tag = int(digest, 16) % 7919
                        expected = expected_tags[i]
                        expected = int(expected, 16) % 7919 if isinstance(expected, str) else int(expected)
                        matches = tag == expected
                    else:
                        matches = digest == str(expected_tags[i])
                    outcome, completed = ("normal" if matches else "corrupted"), True
                result.append(VerificationOutcome(
                    target_id=i, verifier_id=g, outcome=outcome, completed=completed,
                    attempted=True, similarity=score, digest=digest, tag=tag,
                    compute_ms=float(costs["compute_ms"][j]),
                    transfer_ms=float(costs["transfer_ms"][j]),
                    network_ms=float(costs["network_ms"][j]),
                    total_ms=float(costs["total_ms"][j]), start_ms=start, finish_ms=finish,
                    deadline_satisfied=finish <= self._deadline + 1e-9, failure_reason=reason,
                    protocol=protocol, protocol_evidence=evidence,
                ))
            return result

        return MethodEnvironment(
            n=self.n, d=self.d, rounds=self.rounds, deadline_ms=self._deadline,
            begin=begin, observe=observe, resource=resources, execute=execute,
            enroll=enroll, state=lambda: dict(state),
        )
