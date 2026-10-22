"""Group 02 RACA-HIM core adaptation with incremental majority support.

The manuscript's history ordering, inclusive cosine interval, majority rule,
necessary candidate expansion and single-verifier detection remain intact.
This overhead adapter uses a trusted coordinator: each candidate uploads its
unchanged descriptor once, and the coordinator computes candidate support.
It therefore does not represent the previous all-to-all wire protocol.

The symmetric cosine relation is reused in both directions. Previously tested
pairs and normalized rows are retained while the candidate set expands; the
majority threshold and accepted set are still recomputed at every expansion.
No label, prototype or source identity enters these functions.
"""
from __future__ import annotations

from dataclasses import dataclass
import importlib
import math
import time

import numpy as np

from experiments.common.dataset_adapter import AlgorithmInput
from experiments.common.config import DetectionConfig
from experiments.common.similarity import within_interval

_gt = importlib.import_module("experiments.01_detection.proposed_detection")

CORE_PROTOCOL = "trusted-coordinator descriptor collection and incremental majority support"


def _readonly(values: np.ndarray) -> np.ndarray:
    result = np.array(values, dtype=np.int64, copy=True)
    result.setflags(write=False)
    return result


def _unit_rows(values: np.ndarray) -> np.ndarray:
    """Same float64 normalization as the shared cosine kernel, cached once."""
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] == 0 or not np.isfinite(values).all():
        raise ValueError("Descriptors must be a finite matrix with nonzero feature dimension")
    scale = np.max(np.abs(values), axis=1, keepdims=True)
    if np.any(scale == 0.0):
        raise ValueError("Cosine similarity is undefined for a zero vector")
    scaled = values / scale
    return scaled / np.linalg.norm(scaled, axis=1, keepdims=True)


@dataclass(frozen=True)
class GroundTruthCoreStep:
    candidate_size: int
    new_candidate_indices: np.ndarray
    new_pair_count: int
    kernel_dot_count: int
    kernel_cpu_ms: float


@dataclass(frozen=True)
class GroundTruthCoreResult(_gt.GroundTruthResult):
    """Compatible result plus the actual incremental execution plan.

    ``new_pair_count`` counts distinct unordered non-self comparisons. The
    vectorized diagonal block additionally computes its symmetric entries and
    diagonal: ``kernel_dot_count`` records those actual matrix dot products.
    Neither count is used as an artificial runtime reduction multiplier.
    """

    core_steps: tuple[GroundTruthCoreStep, ...] = ()
    core_protocol: str = CORE_PROTOCOL
    preflight_cpu_ms: float = 0.0

    @property
    def unique_pair_count(self) -> int:
        return sum(step.new_pair_count for step in self.core_steps)

    @property
    def normalized_row_count(self) -> int:
        return sum(len(step.new_candidate_indices) for step in self.core_steps)


def build_ground_truth_core(data: AlgorithmInput, config: DetectionConfig) -> GroundTruthCoreResult:
    """Build the same majority GT while reusing unchanged candidate pairs."""
    preflight_start = time.perf_counter()
    needed, initial_size, increment = int(config.M), int(config.K), int(config.expansion_step)
    if needed < 1 or initial_size < 1 or needed > initial_size:
        raise ValueError("Ground-truth construction requires 1 <= M <= K")
    if increment < 1 or float(config.p_fail) != 0.0:
        raise ValueError("Positive expansion_step and p_fail=0 are required")
    reference = np.asarray(data.reference_indices, dtype=np.int64)
    if (reference.ndim != 1 or len(np.unique(reference)) != len(reference)
            or np.any(reference < 0) or np.any(reference >= len(data.X))):
        raise ValueError("Reference indices must be a one-dimensional set of valid unique rows")
    if len(reference) < initial_size:
        return GroundTruthCoreResult(
            success=False, selected_indices=_readonly(np.empty(0)),
            candidate_indices=_readonly(np.empty(0)), support_counts=_readonly(np.empty(0)),
            threshold=(initial_size + 2) // 2, expansion_trace=(),
            reason=f"Reference pool has {len(reference)} members but initial K={initial_size} is required.",
            preflight_cpu_ms=(time.perf_counter() - preflight_start) * 1000.0,
        )

    ordered = reference[np.lexsort((data.model_ids[reference], data.history[reference]))]
    candidate_size, previous_size = initial_size, 0
    units = np.empty((0, data.X.shape[1]), dtype=np.float64)
    supports = np.empty(0, dtype=np.int64)
    trace, core_steps = [], []
    preflight_cpu_ms = (time.perf_counter() - preflight_start) * 1000.0
    while True:
        kernel_start = time.perf_counter()
        candidates = ordered[:candidate_size]
        new_candidates = candidates[previous_size:]
        new_units = _unit_rows(data.X[new_candidates])
        added = len(new_candidates)
        new_supports = np.zeros(added, dtype=np.int64)
        if previous_size:
            cross = np.clip(units @ new_units.T, -1.0, 1.0)
            cross_support = within_interval(cross, config.tau_lower, config.tau_upper)
            supports += cross_support.sum(axis=1, dtype=np.int64)
            new_supports += cross_support.sum(axis=0, dtype=np.int64)
        diagonal_block = np.clip(new_units @ new_units.T, -1.0, 1.0)
        block_support = np.array(
            within_interval(diagonal_block, config.tau_lower, config.tau_upper), copy=True,
        )
        np.fill_diagonal(block_support, False)
        new_supports += block_support.sum(axis=1, dtype=np.int64)
        supports = np.concatenate((supports, new_supports))
        units = np.concatenate((units, new_units), axis=0)
        threshold = (candidate_size + 2) // 2
        accepted = candidates[supports + 1 >= threshold]
        trace.append(_gt.GroundTruthStep(candidate_size, threshold, len(accepted)))
        core_steps.append(GroundTruthCoreStep(
            candidate_size=candidate_size, new_candidate_indices=_readonly(new_candidates),
            new_pair_count=previous_size * added + added * (added - 1) // 2,
            kernel_dot_count=previous_size * added + added * added,
            kernel_cpu_ms=(time.perf_counter() - kernel_start) * 1000.0,
        ))
        success = len(accepted) >= needed
        if success or candidate_size == len(ordered):
            return GroundTruthCoreResult(
                success=success,
                selected_indices=_readonly(accepted[:needed] if success else np.empty(0)),
                candidate_indices=_readonly(candidates), support_counts=_readonly(supports),
                threshold=threshold, expansion_trace=tuple(trace), core_steps=tuple(core_steps),
                preflight_cpu_ms=preflight_cpu_ms,
                reason="" if success else (
                    f"Only {len(accepted)} majority-supported references were available; "
                    f"{needed} required after exhausting the {len(ordered)}-member reference pool."
                ),
            )
        previous_size = candidate_size
        candidate_size = min(candidate_size + increment, len(ordered))


def execute_ground_truth_core(case, sim, *, cpu_ms: float | None = None,
                              notify_membership: bool = True) -> float:
    """Replay uploads and actual measured GT kernels with expansion barriers.

    ``case`` and ``sim`` follow the existing overhead simulator interfaces.
    CPU uses the same host execution clock as scheduling, explicitly recorded
    as measured CPU; network transfers retain the shared resource simulator.
    The caller must apply this measured-kernel convention consistently when
    comparing baseline preparation. This is a hybrid core-overhead estimate.

    The fixed-workload adapter can merge membership notification into its
    later batch assignment header. In that case no standalone membership is
    sent; the batch still waits for the entire GT decision to finish.
    """
    gt = case.gt
    record_operation = getattr(sim, 'record_operation', sim.runtime.append)
    if not isinstance(gt, GroundTruthCoreResult):
        raise TypeError("execute_ground_truth_core requires an incremental core GT result")
    total_cpu = float(case.gt_cpu_ms if cpu_ms is None else cpu_ms)
    if not math.isfinite(total_cpu) or total_cpu < 0:
        raise ValueError("Measured GT CPU duration must be finite and nonnegative")
    measured = gt.preflight_cpu_ms + sum(step.kernel_cpu_ms for step in gt.core_steps)
    prefix_cpu = total_cpu * gt.preflight_cpu_ms / measured if measured else total_cpu
    clock = max(0.0, sim.node_ready.get(-1, 0.0)) + prefix_cpu
    sim.node_ready[-1] = clock
    record_operation(dict(
        node=-1, operation="gt_candidate_selection", phase="gt",
        start_ms=clock - prefix_cpu, end_ms=clock, cpu_ms=prefix_cpu,
        measurement_kind="measured_cpu", included_in="gt_preparation",
        core_protocol=CORE_PROTOCOL,
    ))
    for step_id, step in enumerate(gt.core_steps):
        arrivals = []
        for row in step.new_candidate_indices:
            cursor = sim.send(dict(
                sender=-1, receiver=int(row), message_type="gt_candidate_request",
                control_bytes=sim.config.request_bytes, expansion_step=step_id,
                candidate_row=int(row), core_protocol=CORE_PROTOCOL,
            ), "gt", clock)
            cursor = sim.send(dict(
                sender=int(row), receiver=-1, message_type="gt_candidate_upload",
                payload_bytes=4 * sim.d, expansion_step=step_id,
                candidate_row=int(row), core_protocol=CORE_PROTOCOL,
            ), "gt", cursor)
            arrivals.append(cursor)
        begin = max(max(arrivals, default=clock), sim.node_ready.get(-1, 0.0))
        duration = total_cpu * step.kernel_cpu_ms / measured if measured else 0.0
        clock = begin + duration
        sim.node_ready[-1] = clock
        record_operation(dict(
            node=-1, operation="gt_incremental_vectorized_majority", phase="gt",
            start_ms=begin, end_ms=clock, cpu_ms=duration,
            measurement_kind="measured_cpu", included_in="gt_preparation",
            distinct_unordered_pairs=step.new_pair_count,
            kernel_dot_count=step.kernel_dot_count, candidate_size=step.candidate_size,
            expansion_step=step_id, core_protocol=CORE_PROTOCOL,
        ))
    if gt.success and notify_membership:
        clock = max((sim.send(dict(
            sender=-1, receiver=int(row), message_type="gt_membership",
            control_bytes=sim.config.assignment_bytes, core_protocol=CORE_PROTOCOL,
        ), "gt", clock) for row in gt.selected_indices), default=clock)
    return clock
