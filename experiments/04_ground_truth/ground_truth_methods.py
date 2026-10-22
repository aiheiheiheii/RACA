"""Label-blind GT strategies; quorum methods call the shared Group 01 core.

No method accepts an offline scene, truth labels, source IDs, or a prototype.
The response mask is exogenous and stays fixed through nested expansion.
Comparison counts represent logical, directed peer-similarity accesses. Any
physical full-pool cache construction is accounted for by the experiment.
"""

from __future__ import annotations

from dataclasses import dataclass
import importlib
from typing import Any

import numpy as np

from experiments.common.config import DetectionConfig
from experiments.common.dataset_adapter import AlgorithmInput

from .config_ground_truth import (
    D, DELTA_K, GT_SIZE_M, H, H0, INITIAL_K, MAX_CANDIDATE_POOL_SIZE,
    METHODS, TAU_LOWER, TAU_UPPER,
    GT_QUORUM_NUMERATOR, GT_QUORUM_DENOMINATOR, GT_SELECTION_POLICY,
)

_proposed = importlib.import_module("experiments.01_detection.proposed_detection")
build_ground_truth = _proposed.build_ground_truth
majority_threshold = _proposed.majority_threshold
rank_ground_truth_candidates = _proposed.rank_ground_truth_candidates
GROUND_TRUTH_KERNEL_SOURCE = "experiments.01_detection.proposed_detection.build_ground_truth"


def _readonly(values: np.ndarray, dtype: Any = None) -> np.ndarray:
    result = np.array(values, dtype=dtype, copy=True)
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class MethodResult:
    """Construction evidence; ``success`` means formation, not label purity.

    A failed construction returns an empty selected set and keeps its final K
    and expansion evidence. Evaluation determines reliability independently.
    """

    method: str
    success: bool
    selected_indices: np.ndarray
    final_K: int
    expansion_count: int
    pairwise_comparison_count: int
    status: str
    reason: str
    trace: tuple[dict[str, Any], ...]

    def __post_init__(self) -> None:
        selected = np.asarray(self.selected_indices)
        if (selected.ndim != 1 or not np.issubdtype(selected.dtype, np.integer)
                or len(np.unique(selected)) != len(selected)
                or np.any((selected < 0) | (selected >= self.final_K))):
            raise ValueError("Returned GT must contain distinct valid current-pool indices")
        if self.method not in METHODS:
            raise ValueError("Unknown GT method in result")
        sizes = tuple(range(INITIAL_K, MAX_CANDIDATE_POOL_SIZE + 1, DELTA_K))
        if self.final_K not in sizes or self.expansion_count != sizes.index(self.final_K):
            raise ValueError("GT result has an invalid candidate size or expansion count")
        if self.method == "Fixed-K Majority" and self.final_K != INITIAL_K:
            raise ValueError("Fixed-K Majority must never expand")
        if self.success:
            if len(selected) != GT_SIZE_M or self.status != "success" or self.reason:
                raise ValueError("A formed GT must contain exactly M candidates and success status")
        elif len(selected) != 0 or self.status != "gt_unavailable" or not self.reason:
            raise ValueError("An unavailable GT must preserve its reason without fabricated members")
        if (isinstance(self.pairwise_comparison_count, bool)
                or not isinstance(self.pairwise_comparison_count, (int, np.integer))
                or self.pairwise_comparison_count < 0):
            raise ValueError("Pairwise comparison count must be a nonnegative integer")
        if len(self.trace) != self.expansion_count + 1:
            raise ValueError("GT result trace must include each actual candidate size")
        object.__setattr__(self, "selected_indices", _readonly(selected, np.int64))


def _validate_observations(data: AlgorithmInput, available_mask: np.ndarray) -> np.ndarray:
    if not isinstance(data, AlgorithmInput):
        raise TypeError("Methods require label-free AlgorithmInput, not an offline scene")
    order = np.arange(MAX_CANDIDATE_POOL_SIZE, dtype=np.int64)
    if (data.X.shape != (MAX_CANDIDATE_POOL_SIZE, D)
            or data.X.dtype != np.float32
            or not np.array_equal(data.model_ids, order)
            or not np.array_equal(data.reference_indices, order)
            or len(data.evaluation_indices) != 0 or not np.all(data.history == H0)):
        raise ValueError("Group 04 methods require the ordered 100-candidate input and common h0")
    available = np.asarray(available_mask)
    if available.shape != (MAX_CANDIDATE_POOL_SIZE,) or available.dtype != np.bool_:
        raise ValueError("Responder availability must be a Boolean vector aligned with all 100 rows")
    if any(array.flags.writeable for array in (
        data.X, data.model_ids, data.history, data.reference_indices, data.evaluation_indices,
    )):
        raise ValueError("Shared algorithm observations must remain read-only")
    return available


def _majority_method(
    method: str, data: AlgorithmInput, available: np.ndarray,
    pairwise_cache: np.ndarray | None,
) -> MethodResult:
    config = DetectionConfig(
        K=INITIAL_K, M=GT_SIZE_M, reference_pool_size=MAX_CANDIDATE_POOL_SIZE,
        expansion_step=DELTA_K, h0=H0, tau_lower=TAU_LOWER,
        tau_upper=TAU_UPPER, dim=D, H=H, p_fail=0.0,
    )
    result = build_ground_truth(
        data, config, responder_available=available, pairwise_cache=pairwise_cache,
        allow_expansion=(method == "RACA-HIM"),
        support_quorum=(GT_QUORUM_NUMERATOR, GT_QUORUM_DENOMINATOR),
        selection_policy=GT_SELECTION_POLICY,
    )
    if not result.expansion_trace:
        raise ValueError("The full candidate input requires GT construction evidence")
    selected = np.asarray(result.selected_indices)
    final_k = int(result.expansion_trace[-1].candidate_size)
    candidates = np.asarray(result.candidate_indices)
    supports = np.asarray(result.support_counts)
    means = np.asarray(result.selection_evidence)
    if (not np.array_equal(candidates, np.arange(final_k))
            or supports.shape != candidates.shape or not np.issubdtype(supports.dtype, np.integer)
            or np.any((supports < 0) | (supports >= final_k))
            or np.any(supports[~available[candidates]] != 0)
            or means.shape != candidates.shape or not np.issubdtype(means.dtype, np.number)
            or not np.all(np.isfinite(means)) or np.any((means < -1.0) | (means > 1.0))
            or np.any(means[supports == 0] != -1.0)):
        raise ValueError("Shared GT core changed the nested candidate order or support evidence")
    accepted_mask = available[candidates] & (supports + 1 >= result.threshold)
    accepted = candidates[accepted_mask]
    if bool(result.success):
        expected = rank_ground_truth_candidates(
            accepted, data.history, data.model_ids, supports[accepted_mask], means[accepted_mask],
            policy=GT_SELECTION_POLICY,
        )[:GT_SIZE_M]
        if not np.array_equal(selected, expected) or len(selected) != GT_SIZE_M:
            raise ValueError("Shared GT core must select the first M candidates in the declared consensus ranking")
    elif len(selected) or len(accepted) >= GT_SIZE_M:
        raise ValueError("GT failure conflicts with returned majority admission evidence")
    if not result.success and method == "RACA-HIM" and final_k != MAX_CANDIDATE_POOL_SIZE:
        raise ValueError("RACA-HIM may fail only after exhausting the complete candidate pool")
    trace: list[dict[str, Any]] = []
    for index, step in enumerate(result.expansion_trace):
        size = INITIAL_K + index * DELTA_K
        responsive = int(available[:size].sum())
        if (step.candidate_size != size
                or step.threshold != majority_threshold(size, (GT_QUORUM_NUMERATOR, GT_QUORUM_DENOMINATOR))
                or not 0 <= step.passing_count <= responsive
                or (index < len(result.expansion_trace) - 1 and step.passing_count >= GT_SIZE_M)):
            raise ValueError("Shared GT core returned an invalid majority or expansion trace")
        trace.append({
            "candidate_size": size, "threshold": int(step.threshold),
            "passing_count": int(step.passing_count), "responsive_count": responsive,
            "pairwise_comparison_count": responsive * (responsive - 1),
            "selection_policy": GT_SELECTION_POLICY,
        })
    # The full candidate-aligned means remain in the shared core result. Save
    # the selected scores in the trial trace for review without duplicating a
    # complete similarity cache or reading any offline truth labels.
    trace[-1]["selected_peer_support_counts"] = supports[selected].tolist()
    trace[-1]["selected_mean_support_similarity"] = means[selected].tolist()
    logical_count = int(result.pairwise_comparison_count)
    if logical_count != sum(step["pairwise_comparison_count"] for step in trace):
        raise ValueError("Shared GT core comparison count disagrees with full effective peer checks")
    if result.threshold != trace[-1]["threshold"] or trace[-1]["passing_count"] != len(accepted):
        raise ValueError("Final GT threshold or admission count disagrees with core evidence")
    return MethodResult(
        method=method, success=bool(result.success), selected_indices=selected,
        final_K=final_k, expansion_count=len(trace) - 1,
        pairwise_comparison_count=logical_count,
        status="success" if result.success else "gt_unavailable",
        reason="" if result.success else result.reason, trace=tuple(trace),
    )


def _baseline_method(
    method: str, data: AlgorithmInput, available: np.ndarray, random_seed: int,
) -> MethodResult:
    """Expand only for insufficient responders; perform no similarity checks."""
    trace: list[dict[str, Any]] = []
    for size in range(INITIAL_K, MAX_CANDIDATE_POOL_SIZE + 1, DELTA_K):
        responsive = np.flatnonzero(available[:size])
        trace.append({
            "candidate_size": size, "threshold": None,
            "passing_count": len(responsive), "responsive_count": len(responsive),
            "pairwise_comparison_count": 0,
        })
        if len(responsive) >= GT_SIZE_M:
            if method == "Top-M":
                # IDs encode the pre-shuffled order and break equal history ties.
                rank = np.lexsort((data.model_ids[responsive], data.history[responsive]))
                selected = responsive[rank[:GT_SIZE_M]]
            else:
                # No label or history is read by the Random-M selection rule.
                selected = np.random.default_rng(random_seed).choice(
                    responsive, size=GT_SIZE_M, replace=False,
                )
            return MethodResult(
                method=method, success=True, selected_indices=selected,
                final_K=size, expansion_count=len(trace) - 1,
                pairwise_comparison_count=0, status="success", reason="", trace=tuple(trace),
            )
    return MethodResult(
        method=method, success=False, selected_indices=np.empty(0, dtype=np.int64),
        final_K=MAX_CANDIDATE_POOL_SIZE, expansion_count=len(trace) - 1,
        pairwise_comparison_count=0, status="gt_unavailable",
        reason=(f"Only {len(responsive)} responsive candidates remained after exhausting "
                f"the {MAX_CANDIDATE_POOL_SIZE}-member pool; {GT_SIZE_M} required."),
        trace=tuple(trace),
    )


def run_method(
    method: str,
    algorithm_input: AlgorithmInput,
    available_mask: np.ndarray,
    random_seed: int,
    pairwise_cache: np.ndarray | None = None,
) -> MethodResult:
    """Build GT using observations only; no label-dependent retry is possible."""
    if method not in METHODS:
        raise ValueError(f"Unknown GT method {method!r}; expected one of {METHODS}")
    available = _validate_observations(algorithm_input, available_mask)
    if (isinstance(random_seed, (bool, np.bool_))
            or not isinstance(random_seed, (int, np.integer)) or random_seed < 0):
        raise ValueError("Random-M seed must be a fixed nonnegative integer")
    if method in ("RACA-HIM", "Fixed-K Majority"):
        result = _majority_method(method, algorithm_input, available, pairwise_cache)
    else:
        result = _baseline_method(method, algorithm_input, available, int(random_seed))
    if len(result.selected_indices) and not np.all(available[result.selected_indices]):
        raise ValueError("Unavailable responders cannot enter the returned GT")
    return result
