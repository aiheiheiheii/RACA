"""I-EDI source-protocol primitives; no workload, scheduler or QoS fabrication.

Source: Zhao et al., TIFS 20 (2025), pp. 13189--13204,
DOI 10.1109/TIFS.2025.3638138. References below use the supplied PDF pages.

* PDF p.4: Eqs. (1a--b), (2--4), (8), QoS and reliability representations.
* PDF p.5: Eq. (12), virtual number queue.
* PDF p.8: Algorithm 2, independently generated per-replica, per-round eta;
  Eq. (39), hash(replica bytes) modulo eta, followed by exact tag comparison.
* PDF p.9: late/missing response is flagged malicious; Section V-A uses
  SHA-256 and fixed eta=7919, whereas Theorem 6 defines H(eta || message).

The three tag descriptions in that source are distinct. This module implements
Algorithm 2/Eq. (39), with an explicitly configured random-modulus domain, and
an explicitly named fixed-modulus experimental option. It never substitutes
the theorem's nonce-prefixed hash or claims to inherit its security bound.
The random-modulus sampling domain is unspecified by the paper and therefore
must be supplied by the caller and disclosed as a reproduction setting.

QoS functions accept externally supplied observations/rates. They never invent
QWS traces, infer throughput from integrity verdicts, or read corruption labels.
Trusted vendor bytes, observed bytes, QoS logs and any missing-history priors
remain explicit responsibilities of the workload/integration layer.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from typing import Literal

import numpy as np

ALGORITHM2_RANDOM_MODULUS = "algorithm2-random-modulus"
EXPERIMENT_FIXED_MODULUS = "section-va-fixed-modulus"
SOURCE_EXPERIMENT_ETA = 7919
SOURCE_DOI = "10.1109/TIFS.2025.3638138"


def _integer(value, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    result = int(value)
    if result < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return result


def _finite(value, name: str, *, minimum: float | None = None) -> float:
    result = float(value)
    if not math.isfinite(result) or (minimum is not None and result < minimum):
        raise ValueError(f"{name} must be finite" + (f" and >= {minimum}" if minimum is not None else ""))
    return result


def _unit_values(values, name: str) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64)
    if not np.isfinite(result).all() or np.any((result < 0.0) | (result > 1.0)):
        raise ValueError(f"{name} must contain finite fractions in [0, 1]")
    return result


@dataclass(frozen=True)
class IediChallenge:
    """Association metadata and the actual Eq. (39) modulus sent to a server."""

    round_id: int
    replica_id: int
    eta: int
    mode: str

    def __post_init__(self) -> None:
        _integer(self.round_id, "round_id", minimum=1)
        _integer(self.replica_id, "replica_id")
        _integer(self.eta, "eta", minimum=2)
        if self.mode not in {ALGORITHM2_RANDOM_MODULUS, EXPERIMENT_FIXED_MODULUS}:
            raise ValueError("Unknown I-EDI challenge mode")


@dataclass(frozen=True)
class IediVerification:
    """Keep a missing response separate from an observed tag mismatch.

    Both mismatch and timeout are malicious flags under the source framework.
    A timeout supplies no observed tag and does not prove content corruption.
    The offline evaluator must state how such flags enter its chosen metric.
    """

    status: Literal["match", "mismatch", "timeout"]
    expected_tag: int
    response_tag: int | None
    flagged_malicious: bool
    integrity_match: bool | None


def make_challenge(
    round_id: int,
    replica_id: int,
    *,
    rng: np.random.Generator | None = None,
    mode: str = ALGORITHM2_RANDOM_MODULUS,
    eta_min: int | None = None,
    eta_max: int | None = None,
    fixed_eta: int = SOURCE_EXPERIMENT_ETA,
) -> IediChallenge:
    """Algorithm 2 line 4, or the explicitly named Section V-A option.

    Random draws are uniform independent draws on the inclusive integer domain
    [eta_min, eta_max]. Repeated draws are allowed; forcing uniqueness would
    change that distribution. The source gives no such domain/distribution.
    """
    _integer(round_id, "round_id", minimum=1)
    _integer(replica_id, "replica_id")
    if mode == ALGORITHM2_RANDOM_MODULUS:
        if rng is None or eta_min is None or eta_max is None:
            raise ValueError("Algorithm 2 requires an RNG and an explicitly disclosed eta_min/eta_max")
        low = _integer(eta_min, "eta_min", minimum=2)
        high = _integer(eta_max, "eta_max", minimum=2)
        if low >= high or high >= np.iinfo(np.int64).max:
            raise ValueError("Random eta requires 2 <= eta_min < eta_max < int64 maximum")
        eta = int(rng.integers(low, high + 1))
    elif mode == EXPERIMENT_FIXED_MODULUS:
        if eta_min is not None or eta_max is not None:
            raise ValueError("Fixed-modulus mode must not declare an unused random domain")
        eta = _integer(fixed_eta, "fixed_eta", minimum=2)
    else:
        raise ValueError("Unknown I-EDI challenge mode")
    return IediChallenge(int(round_id), int(replica_id), eta, mode)


def compute_tag(payload: bytes, eta: int) -> int:
    """Eq. (39), using the paper's Section V-A SHA-256 choice.

    Pass actual serialized replica bytes at each endpoint. This function never
    accesses a descriptor prototype, label, tolerance, cosine, or server state.
    """
    if not isinstance(payload, bytes):
        raise TypeError("I-EDI payload must be explicitly serialized bytes")
    modulus = _integer(eta, "eta", minimum=2)
    return int.from_bytes(hashlib.sha256(payload).digest(), "big") % modulus


def verify_response(
    challenge: IediChallenge,
    trusted_payload: bytes,
    response_tag: int | None,
    *,
    arrival_ms: float | None,
    deadline_ms: float,
) -> IediVerification:
    """Algorithm 2 lines 13--17 and the p.9 response-time threshold rule.

    The vendor computes its own expected tag from its trusted original bytes.
    The caller must obtain response_tag via compute_tag(current_server_bytes,
    challenge.eta); providing a digest of a different reference is not valid.
    This is called once the response arrives or the declared deadline expires.
    """
    expected = compute_tag(trusted_payload, challenge.eta)
    return _verify_expected_tag(challenge, expected, response_tag,
                                arrival_ms=arrival_ms, deadline_ms=deadline_ms)


def verify_digest_response(
    challenge: IediChallenge,
    trusted_sha256_hex: str,
    response_tag: int | None,
    *,
    arrival_ms: float | None,
    deadline_ms: float,
) -> IediVerification:
    """Algorithm 2 vendor check from the SHA-256 of its trusted original.

    Since eta is the modulus outside SHA-256 in Eq. (39), retaining the full
    256-bit original digest gives exactly the same expected tag as retaining
    every original payload byte. This does not implement H(eta || message),
    for which such digest-only trusted registration would not be equivalent.
    The digest must have been obtained from trusted registration bytes, never
    inferred from current server responses, reference voting or truth labels.
    """
    if (not isinstance(trusted_sha256_hex, str) or len(trusted_sha256_hex) != 64
            or any(character not in "0123456789abcdefABCDEF" for character in trusted_sha256_hex)):
        raise ValueError("Trusted SHA-256 must be exactly 64 hexadecimal characters")
    expected = int(trusted_sha256_hex, 16) % challenge.eta
    return _verify_expected_tag(challenge, expected, response_tag,
                                arrival_ms=arrival_ms, deadline_ms=deadline_ms)


def _verify_expected_tag(
    challenge: IediChallenge,
    expected: int,
    response_tag: int | None,
    *,
    arrival_ms: float | None,
    deadline_ms: float,
) -> IediVerification:
    """One shared timeout/tag comparison path for both trusted representations."""
    deadline = _finite(deadline_ms, "deadline_ms", minimum=0.0)
    arrival = None if arrival_ms is None else _finite(arrival_ms, "arrival_ms", minimum=0.0)
    if response_tag is None:
        return IediVerification("timeout", expected, None, True, None)
    received = _integer(response_tag, "response_tag")
    if received >= challenge.eta:
        raise ValueError("An Eq. (39) response tag must be less than eta")
    if arrival is None:
        raise ValueError("A returned tag requires an observed arrival_ms")
    if arrival > deadline:
        return IediVerification("timeout", expected, received, True, None)
    matches = received == expected
    return IediVerification("match" if matches else "mismatch", expected, received, not matches, matches)


def normalize_qos_attributes(values, positive_attributes, lower_bounds, upper_bounds) -> np.ndarray:
    """Eq. (1b) for (..., g) external measurements and declared per-metric bounds.

    The caller supplies the observation population and its extrema; this module
    does not fit normalization bounds using future rounds. No clipping is used.
    Equal upper/lower bounds map to one exactly as in the source equation.
    """
    measured = np.asarray(values, dtype=np.float64)
    positive = np.asarray(positive_attributes)
    lower = np.asarray(lower_bounds, dtype=np.float64)
    upper = np.asarray(upper_bounds, dtype=np.float64)
    if measured.ndim == 0 or measured.shape[-1] == 0:
        raise ValueError("QoS measurements require at least one attribute on the last axis")
    g = measured.shape[-1]
    if positive.shape != (g,) or lower.shape != (g,) or upper.shape != (g,):
        raise ValueError("Classes and bounds must have one entry per QoS attribute")
    if not np.isin(positive, (False, True)).all():
        raise ValueError("Attribute class must be 1/True for positive or 0/False for negative")
    if (not np.isfinite(measured).all() or not np.isfinite(lower).all()
            or not np.isfinite(upper).all() or np.any(upper < lower)
            or np.any(measured < lower) or np.any(measured > upper)):
        raise ValueError("QoS values must be finite and within their valid normalization bounds")
    span = upper - lower
    numerator = np.where(positive.astype(bool), measured - lower, upper - measured)
    return np.divide(numerator, span, out=np.ones_like(measured), where=span != 0.0)


def qos_conformance(normalized_attributes) -> np.ndarray:
    """Eq. (1a): mean of genuine normalized attributes on the final axis."""
    values = _unit_values(normalized_attributes, "QoS conformance attributes")
    if values.ndim == 0 or values.shape[-1] == 0:
        raise ValueError("At least one normalized QoS attribute is required")
    return np.mean(values, axis=-1)


def long_term_qos(
    observation_times,
    conformances,
    *,
    current_time: float,
    alpha: float = 0.5,
    decay_scale: float = 1.0,
) -> np.ndarray:
    """Eqs. (2--3), accepting snapshots shaped (observations, ...replicas).

    Each snapshot already averages attributes using Eq. (1a). Time and C must
    use the same declared unit. Missing histories are rejected rather than
    silently replaced by fabricated measurements or completeness priors.
    """
    times = np.asarray(observation_times, dtype=np.float64)
    values = _unit_values(conformances, "QoS history")
    now = _finite(current_time, "current_time")
    floor = _finite(alpha, "alpha", minimum=0.0)
    scale = _finite(decay_scale, "decay_scale", minimum=0.0)
    if floor > 1.0 or scale <= 0.0:
        raise ValueError("Eq. (3) requires alpha in [0, 1] and C > 0")
    if (times.ndim != 1 or not times.size or values.ndim == 0
            or values.shape[0] != times.size or not np.isfinite(times).all()
            or np.any(times > now)):
        raise ValueError("QoS history requires aligned nonempty finite observations from the present or past")
    exponents = (times - now) / scale
    if floor == 0.0:
        # Common rescaling cancels in Eq. (2), preventing underflow when alpha=0.
        weights = np.exp(exponents - np.max(exponents))
    else:
        weights = (1.0 - floor) * np.exp(exponents) + floor
    return np.asarray(np.tensordot(weights, values, axes=(0, 0)) / weights.sum())


def verification_representation(success_rate, retrieval_error_rate) -> np.ndarray:
    """Eq. (4): VR=(1+SR-ER)/2 for externally supplied fractional rates.

    Section V-A obtains SR from NumberResponses/NumberRequests and ER from
    NumberErrors/TotalMessages in QWS v2, not from known corruption labels or
    the count of successful integrity matches. Any substitute data source must
    be explicitly documented by the caller.
    """
    sr = _unit_values(success_rate, "SR")
    er = _unit_values(retrieval_error_rate, "ER")
    return np.asarray((1.0 + sr - er) / 2.0)


def reliability_representation(qos, verification_reliability, *, beta: float = 0.5) -> np.ndarray:
    """Eq. (8) per-replica summands beta*QoS + (1-beta)*VR.

    The O2RS objective sums only those terms whose decision d_i equals one.
    This helper does not select targets or impose a workload budget.
    """
    weight = _finite(beta, "beta", minimum=0.0)
    if weight > 1.0:
        raise ValueError("beta must be in [0, 1]")
    qos_values = _unit_values(qos, "QoS")
    vr = _unit_values(verification_reliability, "VR")
    return np.asarray(weight * qos_values + (1.0 - weight) * vr)


def update_number_queue(queue: float, minimum_count: int, selected_count: int) -> float:
    """Eq. (12), Q_next=max(Q+gamma-sum(d_i),0).

    Algorithm 1 line 10 places this update inside its exclusion sweep. The
    calling selector owns that update schedule; this function changes no state.
    """
    backlog = _finite(queue, "queue", minimum=0.0)
    gamma = _integer(minimum_count, "minimum_count")
    selected = _integer(selected_count, "selected_count")
    return max(backlog + gamma - selected, 0.0)
