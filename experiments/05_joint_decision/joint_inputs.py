"""Shared label-blind inputs for the single-round Group 05 comparison.

The Group 04 wrapper intentionally accepts only its 100-candidate stress-test
pool, so this module reuses its Group 01 GT kernel directly. Small candidate
steps call that kernel unchanged; larger steps use row blocks and the same
cosine, quorum, and final ranking functions, avoiding an N-by-N allocation.
No dataset label, source identity, or generation prototype leaves load_group.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import hashlib
import importlib
import json
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

import numpy as np

from dataset.load_dataset import load_group, resource_costs
from experiments.common.config import DetectionConfig
from experiments.common.dataset_adapter import AlgorithmInput
from experiments.common.similarity import cosine_pairwise, within_interval

from .config_joint_decision import (
    ALPHA, A_MAX, CORRUPTION_RATE_PERCENT, D, DATASET_NAME, DATASET_ROOT,
    DELTA_K, ETA, H, H0, HISTORY_ANOMALY_PROB, HISTORY_INVALID_PROB,
    HISTORY_ROUNDS, HISTORY_VERIFICATION_PROB, K, M, N_VALUES, RHO,
    ROUND_ID, SEEDS, TAU_LOWER, TAU_UPPER,
)
from .historical_state import HistoricalState, HistoryConfig, build_historical_state

_gt = importlib.import_module("experiments.01_detection.proposed_detection")
_resources = importlib.import_module("experiments.02_overhead.overhead_resources")
GroundTruthResult = _gt.GroundTruthResult
GroundTruthStep = _gt.GroundTruthStep
GROUND_TRUTH_KERNEL_SOURCE = "experiments.01_detection.proposed_detection.build_ground_truth"
RESOURCE_FIXTURE_SOURCE = "experiments.02_overhead.overhead_resources.shared_resource_profiles"
RESOURCE_COST_SOURCE = "dataset.load_dataset.resource_costs"
INPUT_SCHEMA = "joint-label-blind-shared-inputs-v1"
GT_FULL_STEP_LIMIT = 256
GT_BLOCK_ROWS = 128
RESOURCE_DOMAIN_MAX = 5000


def _immutable(values: np.ndarray, dtype=None) -> np.ndarray:
    """Bytes-backed arrays cannot have writeability re-enabled by a method."""
    array = np.ascontiguousarray(values, dtype=dtype)
    return np.frombuffer(array.tobytes(), dtype=array.dtype).reshape(array.shape)


def _array_digest(values: np.ndarray) -> str:
    array = np.ascontiguousarray(values)
    header = json.dumps({"dtype": array.dtype.str, "shape": list(array.shape)},
                        sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(header + b"\n" + array.tobytes()).hexdigest()


def derive_seed_streams(trial_seed: int) -> Mapping[str, int]:
    """Keep dataset seeds published; derive stable independent exogenous streams.

    Temporary IDs supply the manuscript's only GT tie breaker, so the
    gt_tiebreak stream is used to generate their permutation. No extra random
    decision is made while constructing GT.
    """
    if (isinstance(trial_seed, (bool, np.bool_))
            or not isinstance(trial_seed, (int, np.integer)) or trial_seed < 0):
        raise ValueError("trial_seed must be a nonnegative integer")
    trial_seed = int(trial_seed)
    names = {"history_seed": 5101, "resource_seed": 5102,
             "gt_tiebreak_seed": 5103, "random_baseline_seed": 5104}
    return MappingProxyType({
        "dataset_seed": trial_seed,
        **{name: int(np.random.SeedSequence([trial_seed, code]).generate_state(1)[0])
           for name, code in names.items()},
    })


def _frozen_gt(result: GroundTruthResult, trace: list,
               comparison_count: int) -> GroundTruthResult:
    return GroundTruthResult(
        success=bool(result.success),
        selected_indices=_immutable(result.selected_indices, np.int64),
        candidate_indices=_immutable(result.candidate_indices, np.int64),
        support_counts=_immutable(result.support_counts, np.int64),
        threshold=int(result.threshold), expansion_trace=tuple(trace),
        reason=str(result.reason), pairwise_comparison_count=int(comparison_count),
        selection_evidence=(None if result.selection_evidence is None
                            else _immutable(result.selection_evidence, np.float64)),
    )


def _blocked_gt_step(data: AlgorithmInput, ordered: np.ndarray,
                     size: int) -> GroundTruthResult:
    """Exactly the GT kernel's peer evidence, accumulated one row block at a time.

    A row's complete peer vector is reduced in the original candidate order,
    using the shared float64 cosine implementation and inclusive interval.
    Self support is excluded from peer counts/means and added only at admission.
    All candidates are responsive in this experiment (p_fail=0).
    """
    candidates = ordered[:size]
    candidate_x = data.X[candidates]
    counts = np.empty(size, dtype=np.int64)
    means = np.empty(size, dtype=np.float64)
    for start in range(0, size, GT_BLOCK_ROWS):
        stop = min(start + GT_BLOCK_ROWS, size)
        similarities = cosine_pairwise(candidate_x[start:stop], candidate_x)
        mask = within_interval(similarities, TAU_LOWER, TAU_UPPER)
        mask[np.arange(stop - start), np.arange(start, stop)] = False
        block_counts = mask.sum(axis=1, dtype=np.int64)
        totals = np.where(mask, similarities, 0.0).sum(axis=1, dtype=np.float64)
        counts[start:stop] = block_counts
        means[start:stop] = np.divide(
            totals, block_counts, out=np.full(stop - start, -1.0),
            where=block_counts > 0,
        )
    threshold = _gt.majority_threshold(size, (2, 3))
    admitted = counts + 1 >= threshold
    passed = candidates[admitted]
    success = len(passed) >= M
    selected = (_gt.rank_ground_truth_candidates(
        passed, data.history, data.model_ids, counts[admitted], means[admitted],
        policy="history_consensus",
    )[:M] if success else np.empty(0, dtype=np.int64))
    return GroundTruthResult(
        success=success, selected_indices=selected, candidate_indices=candidates,
        support_counts=counts, threshold=threshold,
        expansion_trace=(GroundTruthStep(size, threshold, len(passed)),),
        reason=("" if success else f"Only {len(passed)} quorum-supported references at K={size}."),
        pairwise_comparison_count=size * (size - 1), selection_evidence=means,
    )


def build_shared_ground_truth(data: AlgorithmInput) -> GroundTruthResult:
    """Expand over the full N-domain, retaining every actual construction step.

    The legacy Group 04 wrapper cannot be reused because it enforces N=100.
    The shared kernel is called with each actual K and expansion disabled so
    that the outer adapter can switch to bounded-memory evidence accumulation.
    Only storage changes at large K; admission, ordering, and stopping agree
    with Section IV-A, including exhausting the entire model domain on failure.
    """
    if not isinstance(data, AlgorithmInput):
        raise TypeError("GT requires the common label-free AlgorithmInput")
    n = len(data.X)
    if (n not in N_VALUES or data.X.shape != (n, D)
            or not np.array_equal(data.reference_indices, np.arange(n))
            or len(data.evaluation_indices)):
        raise ValueError("Joint GT candidate domain must contain every current N model")
    ordered = data.reference_indices[np.lexsort((data.model_ids, data.history))]
    trace: list = []
    comparisons = 0
    size = K
    while True:
        if size <= GT_FULL_STEP_LIMIT:
            config = DetectionConfig(
                K=size, M=M, reference_pool_size=n, expansion_step=DELTA_K,
                h0=H0, tau_lower=TAU_LOWER, tau_upper=TAU_UPPER,
                dim=D, H=H, p_fail=0.0,
            )
            step = _gt.build_ground_truth(
                data, config, allow_expansion=False, support_quorum=(2, 3),
                selection_policy="history_consensus",
            )
        else:
            step = _blocked_gt_step(data, ordered, size)
        if (len(step.expansion_trace) != 1
                or step.expansion_trace[0].candidate_size != size
                or not np.array_equal(step.candidate_indices, ordered[:size])):
            raise RuntimeError("GT kernel no longer preserves the declared candidate order")
        trace.extend(step.expansion_trace)
        comparisons += int(step.pairwise_comparison_count)
        if step.success:
            if len(step.selected_indices) != M:
                raise RuntimeError("Shared GT kernel returned a size different from M")
            return _frozen_gt(step, trace, comparisons)
        if size == n:
            final = GroundTruthResult(
                success=False, selected_indices=np.empty(0, dtype=np.int64),
                candidate_indices=step.candidate_indices, support_counts=step.support_counts,
                threshold=step.threshold, expansion_trace=(),
                reason=(f"Only {step.expansion_trace[0].passing_count} quorum-supported "
                        f"references remained after exhausting the complete {n}-model domain; "
                        f"{M} required."),
                selection_evidence=step.selection_evidence,
            )
            return _frozen_gt(final, trace, comparisons)
        size = min(size + DELTA_K, n)


@dataclass(frozen=True)
class SharedTrialInputs:
    n: int
    trial_seed: int
    X: np.ndarray
    model_ids: np.ndarray
    history: HistoricalState
    ground_truth: GroundTruthResult
    target_rows: np.ndarray
    verifier_rows: np.ndarray
    streams: Mapping[str, int]
    resource_profiles: Mapping
    descriptor_fingerprint: str

    def __post_init__(self) -> None:
        for name, dtype in (("X", np.float32), ("model_ids", np.int64),
                            ("target_rows", np.int64), ("verifier_rows", np.int64)):
            object.__setattr__(self, name, _immutable(getattr(self, name), dtype))
        object.__setattr__(self, "streams", MappingProxyType(dict(self.streams)))
        if self.X.shape != (self.n, D) or self.model_ids.shape != (self.n,):
            raise ValueError("Shared descriptors and temporary IDs must align with N")
        if self.history.risk_scores.shape != (self.n,):
            raise ValueError("Shared history must be aligned with every model")
        if self.ground_truth.success:
            if (len(self.verifier_rows) != M or len(self.target_rows) != self.n - M
                    or len(np.intersect1d(self.verifier_rows, self.target_rows))
                    or not np.array_equal(np.sort(np.concatenate(
                        (self.verifier_rows, self.target_rows))), np.arange(self.n))):
                raise ValueError("GT and target pools must partition all N model rows")
        elif len(self.verifier_rows) or len(self.target_rows):
            raise ValueError("Unavailable GT has no schedulable verifier or target pool")

    @property
    def target_ids(self) -> np.ndarray:
        return _immutable(self.model_ids[self.target_rows], np.int64)

    @property
    def verifier_ids(self) -> np.ndarray:
        return _immutable(self.model_ids[self.verifier_rows], np.int64)

    def to_manifest(self) -> dict:
        gt = self.ground_truth
        return {
            "input_schema": INPUT_SCHEMA, "N": self.n, "trial_seed": self.trial_seed,
            "dataset_name": DATASET_NAME, "descriptor_fingerprint": self.descriptor_fingerprint,
            "descriptor_shape": list(self.X.shape), "descriptor_precision": "float32",
            "seed_streams": dict(self.streams), "temporary_id_source": "gt_tiebreak_seed permutation",
            "temporary_ids_sha256": _array_digest(self.model_ids),
            "history": self.history.to_manifest(),
            "history_anomaly_sha256": _array_digest(self.history.anomaly_state),
            "history_risk_sha256": _array_digest(self.history.risk_scores),
            "resource": dict(_resources.resource_fixture_metadata(self.resource_profiles)),
            "gt": {
                "kernel_source": GROUND_TRUTH_KERNEL_SOURCE,
                "success": bool(gt.success), "reason": str(gt.reason),
                "candidate_domain_size": self.n, "selected_rows": self.verifier_rows.tolist(),
                "selected_model_ids": self.verifier_ids.tolist(),
                "target_pool_size": len(self.target_rows) if gt.success else None,
                "support_quorum": [2, 3], "selection_policy": "history_consensus",
                "full_matrix_step_limit": GT_FULL_STEP_LIMIT, "block_rows": GT_BLOCK_ROWS,
                "pairwise_comparison_count": int(gt.pairwise_comparison_count),
                "expansion_trace": [{"candidate_size": int(step.candidate_size),
                                     "threshold": int(step.threshold),
                                     "passing_count": int(step.passing_count)}
                                    for step in gt.expansion_trace],
            },
        }


def prepare_trial_inputs(n: int, trial_seed: int,
                         dataset_root: str | Path = DATASET_ROOT) -> SharedTrialInputs:
    """Load one descriptor group and construct shared history/GT/resources once.

    load_group necessarily returns its documented full dictionary. This adapter
    retrieves only X and immediately releases that dictionary; labels and source
    metadata are never inspected or placed in a scheduling input container.
    """
    if (isinstance(n, (bool, np.bool_)) or not isinstance(n, (int, np.integer))
            or n not in N_VALUES):
        raise ValueError("n must be one of the nine formal N values")
    if (isinstance(trial_seed, (bool, np.bool_))
            or not isinstance(trial_seed, (int, np.integer)) or trial_seed not in SEEDS):
        raise ValueError("trial_seed must be one of the thirty formal seeds, 2026..2055")
    n, trial_seed = int(n), int(trial_seed)
    streams = derive_seed_streams(trial_seed)
    group = load_group(dataset_root=Path(dataset_root).resolve(), n=n,
                       corruption_rate_percent=CORRUPTION_RATE_PERCENT,
                       dim=D, H=H, seed=streams["dataset_seed"])
    raw_x = np.asarray(group["X"])
    if raw_x.shape != (n, D) or raw_x.dtype != np.float32:
        raise ValueError("Dataset must return the requested float32 (N,30) descriptors")
    x = _immutable(raw_x)
    del raw_x, group
    model_ids = _immutable(
        np.random.default_rng(streams["gt_tiebreak_seed"]).permutation(n), np.int64,
    )
    history_config = HistoryConfig(
        rounds=HISTORY_ROUNDS, verification_probability=HISTORY_VERIFICATION_PROB,
        anomalous_probability=HISTORY_ANOMALY_PROB, invalid_probability=HISTORY_INVALID_PROB,
        initial_anomaly=H0, max_age=A_MAX, rho=RHO, eta=ETA, alpha=ALPHA,
        decision_round=ROUND_ID,
    )
    history = build_historical_state(n, seed=streams["history_seed"], config=history_config)
    observations = AlgorithmInput(
        X=x, model_ids=model_ids, history=history.anomaly_state,
        reference_indices=np.arange(n, dtype=np.int64),
        evaluation_indices=np.empty(0, dtype=np.int64),
    )
    ground_truth = build_shared_ground_truth(observations)
    verifiers = ground_truth.selected_indices
    targets = (np.setdiff1d(np.arange(n, dtype=np.int64), verifiers, assume_unique=True)
               if ground_truth.success else np.empty(0, dtype=np.int64))
    profiles = _resources.shared_resource_profiles(
        seed=streams["resource_seed"], n=n, rounds=1, n_max=RESOURCE_DOMAIN_MAX,
    )
    return SharedTrialInputs(
        n=n, trial_seed=trial_seed, X=x, model_ids=model_ids, history=history,
        ground_truth=ground_truth, target_rows=targets, verifier_rows=verifiers,
        streams=streams, resource_profiles=profiles, descriptor_fingerprint=_array_digest(x),
    )


@dataclass(frozen=True)
class ResourceCostMatrix:
    """Dataset pair costs in ms; rows=GT verifiers, columns=target pool."""
    compute_ms: np.ndarray
    transfer_ms: np.ndarray
    network_ms: np.ndarray
    total_ms: np.ndarray
    sigma: float

    def __post_init__(self) -> None:
        if not np.isfinite(self.sigma) or not 0.0 <= self.sigma <= 1.0:
            raise ValueError("sigma must be finite and in [0,1]")
        shape = np.asarray(self.total_ms).shape
        if len(shape) != 2 or not all(shape):
            raise ValueError("Resource costs require nonempty (G,T) matrices")
        for name in ("compute_ms", "transfer_ms", "network_ms", "total_ms"):
            values = np.asarray(getattr(self, name), dtype=np.float64)
            if values.shape != shape or not np.all(np.isfinite(values)) or np.any(values < 0):
                raise ValueError("All resource costs must be aligned, finite and nonnegative")
            object.__setattr__(self, name, _immutable(values))
        # total_ms already contains network_ms once. Do not add it again when
        # producing solver input or computing a verifier's serial task load.
        component_total = self.compute_ms + self.transfer_ms + self.network_ms
        if not np.allclose(self.total_ms, component_total, rtol=1e-6, atol=1e-9):
            raise ValueError("Dataset total_ms must equal computation + transfer + network")

    def to_manifest(self) -> dict:
        return {
            "cost_source": RESOURCE_COST_SOURCE, "fixture_source": RESOURCE_FIXTURE_SOURCE,
            "shape": list(self.total_ms.shape), "axis_order": ["verifier", "target"],
            "sigma": float(self.sigma), "round_id": ROUND_ID, "dimension": D,
            "descriptor_payload_bytes": 4 * D, "network_count_per_task": 1,
            "total_cost_sha256": _array_digest(self.total_ms),
            "component_sha256": {name: _array_digest(getattr(self, name))
                                 for name in ("compute_ms", "transfer_ms", "network_ms")},
            "min_total_ms": float(self.total_ms.min()), "max_total_ms": float(self.total_ms.max()),
        }


@lru_cache(maxsize=8)
def _cached_cost_components(resource_seed: int, n: int, target_rows: tuple[int, ...],
                            verifier_rows: tuple[int, ...], sigma: float) -> ResourceCostMatrix:
    profiles = _resources.shared_resource_profiles(
        seed=resource_seed, n=n, rounds=1, n_max=RESOURCE_DOMAIN_MAX,
    )
    shape = (len(verifier_rows), len(target_rows))
    targets = np.broadcast_to(np.asarray(target_rows, dtype=np.int32)[None, :], shape)
    verifiers = np.broadcast_to(np.asarray(verifier_rows, dtype=np.int32)[:, None], shape)
    # Resource IDs are internal row positions, never temporary model ID values.
    costs = resource_costs(target_ids=targets, verifier_ids=verifiers, d=D,
                           sigma=sigma, round_id=ROUND_ID, profiles=profiles)
    return ResourceCostMatrix(
        compute_ms=costs["compute_ms"], transfer_ms=costs["transfer_ms"],
        network_ms=costs["network_ms"], total_ms=costs["total_ms"], sigma=sigma,
    )


def resource_cost_components_for_sigma(shared: SharedTrialInputs,
                                        sigma: float) -> ResourceCostMatrix:
    """One immutable common environment for every method at the requested sigma."""
    sigma = float(sigma)
    if not np.isfinite(sigma) or not 0.0 <= sigma <= 1.0:
        raise ValueError("sigma must be finite and in [0,1]")
    if not shared.ground_truth.success or not len(shared.verifier_rows) or not len(shared.target_rows):
        raise ValueError("Cannot construct a scheduling cost environment without shared GT and targets")
    return _cached_cost_components(
        shared.streams["resource_seed"], shared.n,
        tuple(int(row) for row in shared.target_rows),
        tuple(int(row) for row in shared.verifier_rows), sigma,
    )


def cost_matrix_for_sigma(shared: SharedTrialInputs, sigma: float) -> np.ndarray:
    return resource_cost_components_for_sigma(shared, sigma).total_ms
