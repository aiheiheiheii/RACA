"""TNSE Section IV-A ground truth and Section IV-D corruption detection.

Only observable descriptors, temporary identities, history, and the fixed
reference/evaluation split enter these functions. Ground truth is constructed
by peer support; each target is then checked by exactly one selected verifier.
The fixed reference pool and balanced cyclic assignments are the controlled
Group 01 protocol, rather than the manuscript's resource scheduling algorithm.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from experiments.common.config import DetectionConfig
from experiments.common.dataset_adapter import AlgorithmInput
from experiments.common.method_names import RACA_HIM
from experiments.common.similarity import cosine_aligned, cosine_pairwise, within_interval


def _readonly(values: np.ndarray, dtype=None) -> np.ndarray:
    result = np.array(values, dtype=dtype, copy=True)
    result.setflags(write=False)
    return result


def majority_threshold(K: int, quorum: tuple[int, int] | None = None) -> int:
    """Return the full-slot admission quorum using exact integer arithmetic.

    ``None`` retains the Group 01/03 strict-majority rule. A fraction ``(a, b)``
    requests ceil(a*K/b); response failures never reduce this denominator.
    """
    if isinstance(K, (bool, np.bool_)) or not isinstance(K, (int, np.integer)) or K < 1:
        raise ValueError("Ground-truth candidate size K must be a positive integer.")
    if quorum is None:
        return (int(K) + 2) // 2
    if not isinstance(quorum, tuple) or len(quorum) != 2:
        raise ValueError("support_quorum must be None or a (numerator, denominator) tuple.")
    numerator, denominator = quorum
    if any(isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
           for value in quorum) or not 1 <= numerator <= denominator:
        raise ValueError("support_quorum requires integers 1 <= numerator <= denominator.")
    return (int(numerator) * int(K) + int(denominator) - 1) // int(denominator)


def rank_ground_truth_candidates(
    candidates: np.ndarray,
    history: np.ndarray,
    model_ids: np.ndarray,
    support_counts: np.ndarray | None,
    mean_support_similarity: np.ndarray | None,
    policy: str = "history",
) -> np.ndarray:
    """Order admitted global rows using only observable construction evidence.

    History and IDs are full-row vectors; peer evidence is aligned with the
    supplied ``candidates``. The legacy policy uses (history, temporary ID).
    ``history_consensus`` uses ascending history, descending peer support,
    descending mean cosine over responsive interval-valid peers, then ascending
    temporary ID. Neither peer score includes self or an unavailable response.
    """
    if policy not in ("history", "history_consensus"):
        raise ValueError("selection_policy must be 'history' or 'history_consensus'.")
    rows = np.asarray(candidates, dtype=np.int64)
    if rows.ndim != 1:
        raise ValueError("Ground-truth candidates must be a one-dimensional row vector.")
    history_values, ids = np.asarray(history), np.asarray(model_ids)
    if (history_values.ndim != 1 or ids.shape != history_values.shape
            or np.any((rows < 0) | (rows >= len(history_values)))):
        raise ValueError("History and temporary IDs must align with the candidate rows.")
    if policy == "history":
        return rows[np.lexsort((ids[rows], history_values[rows]))]
    supports, means = np.asarray(support_counts), np.asarray(mean_support_similarity)
    if (supports.shape != rows.shape or not np.issubdtype(supports.dtype, np.integer)
            or np.any((supports < 0) | (supports >= len(history_values))) or means.shape != rows.shape
            or not np.issubdtype(means.dtype, np.number) or not np.all(np.isfinite(means))
            or np.any((means < -1.0) | (means > 1.0))):
        raise ValueError("Consensus ranking needs aligned peer counts and finite mean cosines in [-1, 1].")
    # Counts are bounded by the full observable pool. Signed conversion also
    # keeps descending ordering correct when a caller supplies unsigned counts.
    supports = supports.astype(np.int64, copy=False)
    return rows[np.lexsort((ids[rows], -means, -supports, history_values[rows]))]


@dataclass(frozen=True)
class GroundTruthStep:
    """One full candidate check, including its recomputed admission quorum."""

    candidate_size: int
    threshold: int
    passing_count: int


@dataclass(frozen=True)
class GroundTruthResult:
    """Global row indices and peer supports at the final construction step.

    ``support_counts`` excludes self; a responsive candidate earns one self
    support in the acceptance test.
    Its order matches ``candidate_indices``. An unsuccessful construction has
    an empty ``selected_indices`` and retains diagnostics for its final step.
    """

    success: bool
    selected_indices: np.ndarray
    candidate_indices: np.ndarray
    support_counts: np.ndarray
    threshold: int
    expansion_trace: tuple[GroundTruthStep, ...]
    reason: str = ""
    # Directed effective peer relations checked across construction steps.
    # This is a decision-access count, not the physical cost of computing a
    # full cosine matrix; a caller providing a shared cache records that cost.
    pairwise_comparison_count: int = 0
    # Optional final-step mean cosine to effective supporting peers, excluding
    # self. Aligned with candidate_indices; -1 means no supporting peers.
    # The default history-only protocol keeps this field None.
    selection_evidence: np.ndarray | None = None


@dataclass(frozen=True)
class DetectionResult:
    """Prediction records aligned with the complete shared target ID vector.

    Unavailable runs have no predictions: unverified targets must never be
    silently treated as normal. ``verifier_ids`` identifies an actual single
    reference when applicable. Multi-witness protocols use ``diagnostics``;
    a representative witness must not be misreported as the sole verifier.
    """

    method: str
    status: str
    predictions: np.ndarray | None
    target_ids: np.ndarray
    verifier_ids: np.ndarray | None
    gt_success: bool | None
    ground_truth: GroundTruthResult | None
    reason: str = ""
    diagnostics: dict = field(default_factory=dict)


def build_ground_truth(
    data: AlgorithmInput,
    config: DetectionConfig,
    *,
    responder_available: np.ndarray | None = None,
    pairwise_cache: np.ndarray | None = None,
    allow_expansion: bool = True,
    support_quorum: tuple[int, int] | None = None,
    selection_policy: str = "history",
) -> GroundTruthResult:
    """Construct history-ranked, quorum-supported references without labels.

    The default policy breaks history ties by ascending temporary model IDs;
    the adapter assigns those IDs through a predeclared seeded permutation
    independent of the split.
    Expansion adds ``expansion_step`` new pool members without resampling.
    Peer support and the full actual-K quorum are recomputed at every step; a
    supplied cosine cache changes only where exact similarity values are read.
    The number of required GT references is the predeclared ``config.M``.

    Optional response availability, exact cosine caches, support quorum, and
    consensus tie breaking extend the same kernel for Group 04. Availability
    and caches are aligned with all rows of ``data.X``. A failed
    candidate keeps its position in K, but supplies neither peer nor self
    support and cannot enter GT. Caches contain raw exact pairwise cosines,
    never previous majority decisions. ``allow_expansion=False`` provides the
    Fixed-K ablation. ``support_quorum=(2, 3)`` uses ceil(2*K/3), and
    ``selection_policy='history_consensus'`` resolves equal histories by peer
    count, then supporting-peer mean cosine, then temporary ID. Expansion still
    stops at the first step admitting M candidates. Omitted options retain the
    Group 01/03 strict majority and history/temporary-ID decisions.
    """

    needed = int(config.M)
    initial_size = int(config.K)
    increment = int(config.expansion_step)
    if needed < 1 or initial_size < 1 or needed > initial_size:
        raise ValueError("Ground-truth construction requires 1 <= M <= K.")
    if increment < 1:
        raise ValueError("Candidate expansion_step must be positive.")
    if float(config.p_fail) != 0.0:
        raise ValueError("Group 01 detection requires p_fail=0; failure simulation is not implemented.")
    if not isinstance(allow_expansion, (bool, np.bool_)):
        raise ValueError("allow_expansion must be a boolean.")
    initial_threshold = majority_threshold(initial_size, support_quorum)
    if selection_policy not in ("history", "history_consensus"):
        raise ValueError("selection_policy must be 'history' or 'history_consensus'.")

    row_count = len(data.X)
    if responder_available is None:
        available = np.ones(row_count, dtype=bool)
    else:
        available = np.asarray(responder_available)
        if available.shape != (row_count,) or available.dtype != np.bool_:
            raise ValueError("responder_available must be a boolean vector aligned with data.X rows.")

    cached = None
    if pairwise_cache is not None:
        cached = np.asarray(pairwise_cache, dtype=np.float64)
        if cached.shape != (row_count, row_count):
            raise ValueError("pairwise_cache must be a square matrix aligned with all data.X rows.")
        if not np.all(np.isfinite(cached)) or np.any((cached < -1.0) | (cached > 1.0)):
            raise ValueError("pairwise_cache must contain finite cosine values in [-1, 1].")
        # These checks establish structural cosine invariants without computing
        # a second matrix. The caller supplies the exact common cosine routine's
        # output for these descriptors and records its source/provenance.
        tolerance = 16.0 * np.finfo(np.float64).eps
        if (not np.allclose(cached, cached.T, rtol=0.0, atol=tolerance)
                or not np.allclose(np.diag(cached), 1.0, rtol=0.0, atol=tolerance)):
            raise ValueError("pairwise_cache must be symmetric with unit self-cosines.")

    reference = np.asarray(data.reference_indices, dtype=np.int64)
    if reference.ndim != 1 or len(np.unique(reference)) != len(reference):
        raise ValueError("Reference indices must be a one-dimensional set of unique rows.")
    if np.any(reference < 0) or np.any(reference >= len(data.X)):
        raise ValueError("Reference indices are outside the descriptor matrix.")
    if len(reference) < initial_size:
        return GroundTruthResult(
            success=False,
            selected_indices=_readonly(np.empty(0), np.int64),
            candidate_indices=_readonly(np.empty(0), np.int64),
            support_counts=_readonly(np.empty(0), np.int64),
            threshold=initial_threshold,
            expansion_trace=(),
            reason=f"Reference pool has {len(reference)} members but initial K={initial_size} is required.",
        )

    rank = np.lexsort((data.model_ids[reference], data.history[reference]))
    ordered = reference[rank]
    candidate_size = initial_size
    trace: list[GroundTruthStep] = []
    comparison_count = 0

    while True:
        candidates = ordered[:candidate_size]
        similarities = (cosine_pairwise(data.X[candidates]) if cached is None
                        else cached[np.ix_(candidates, candidates)])
        support_mask = np.array(
            within_interval(similarities, config.tau_lower, config.tau_upper),
            dtype=bool,
            copy=True,
        )
        candidate_available = available[candidates]
        support_mask &= candidate_available[:, None] & candidate_available[None, :]
        np.fill_diagonal(support_mask, False)
        support_counts = support_mask.sum(axis=1, dtype=np.int64)
        threshold = majority_threshold(candidate_size, support_quorum)
        mean_support_similarity = None
        if selection_policy == "history_consensus":
            # Reuse already accessed exact cosines. The mean includes only
            # interval-valid, responsive peers; self remains excluded.
            totals = np.where(support_mask, similarities, 0.0).sum(axis=1, dtype=np.float64)
            mean_support_similarity = np.divide(
                totals, support_counts,
                out=np.full(candidate_size, -1.0, dtype=np.float64),
                where=support_counts > 0,
            )
        # Only a responsive candidate earns its one self-support. The quorum
        # denominator remains the complete current K, including failed slots.
        accepted_mask = (
            candidate_available & (support_counts + candidate_available.astype(np.int64) >= threshold)
        )
        accepted = candidates[accepted_mask]
        responsive_count = int(np.count_nonzero(candidate_available))
        comparison_count += responsive_count * (responsive_count - 1)
        trace.append(GroundTruthStep(candidate_size, threshold, len(accepted)))

        if len(accepted) >= needed:
            selected = rank_ground_truth_candidates(
                accepted, data.history, data.model_ids, support_counts[accepted_mask],
                None if mean_support_similarity is None else mean_support_similarity[accepted_mask],
                policy=selection_policy,
            )[:needed]
            return GroundTruthResult(
                success=True,
                selected_indices=_readonly(selected, np.int64),
                candidate_indices=_readonly(candidates, np.int64),
                support_counts=_readonly(support_counts, np.int64),
                threshold=threshold,
                expansion_trace=tuple(trace),
                pairwise_comparison_count=comparison_count,
                selection_evidence=(None if mean_support_similarity is None
                                    else _readonly(mean_support_similarity, np.float64)),
            )

        if not allow_expansion or candidate_size == len(ordered):
            return GroundTruthResult(
                success=False,
                selected_indices=_readonly(np.empty(0), np.int64),
                candidate_indices=_readonly(candidates, np.int64),
                support_counts=_readonly(support_counts, np.int64),
                threshold=threshold,
                expansion_trace=tuple(trace),
                reason=(
                    f"Only {len(accepted)} majority-supported references were available; "
                    + (f"{needed} required after exhausting the {len(ordered)}-member reference pool."
                       if allow_expansion else f"{needed} required at fixed K={candidate_size}; expansion disabled.")
                ),
                pairwise_comparison_count=comparison_count,
                selection_evidence=(None if mean_support_similarity is None
                                    else _readonly(mean_support_similarity, np.float64)),
            )
        candidate_size = min(candidate_size + increment, len(ordered))


def detect_proposed(data: AlgorithmInput, config: DetectionConfig) -> DetectionResult:
    """RACA-HIM: assign each shared target one GT reference in a balanced cycle."""

    ground_truth = build_ground_truth(data, config)
    targets = np.asarray(data.evaluation_indices, dtype=np.int64)
    target_ids = _readonly(data.model_ids[targets])
    if not ground_truth.success:
        return DetectionResult(
            method=RACA_HIM,
            status="gt_unavailable",
            predictions=None,
            target_ids=target_ids,
            verifier_ids=None,
            gt_success=False,
            ground_truth=ground_truth,
            reason=ground_truth.reason,
        )

    selected = ground_truth.selected_indices
    verifiers = selected[np.arange(len(targets)) % len(selected)]
    similarities = cosine_aligned(data.X[verifiers], data.X[targets])
    predictions = (~within_interval(similarities, config.tau_lower, config.tau_upper)).astype(np.int8)
    return DetectionResult(
        method=RACA_HIM,
        status="success",
        predictions=_readonly(predictions, np.int8),
        target_ids=target_ids,
        verifier_ids=_readonly(data.model_ids[verifiers]),
        gt_success=True,
        ground_truth=ground_truth,
    )
