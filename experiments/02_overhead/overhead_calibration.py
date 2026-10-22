"""One shared host-kernel cost calibration for the controlled overhead sweep.

Every operation class uses the same unit cost, regardless of the method. A
single, predeclared timed batch per kernel provides enough work for a useful
clock resolution; these batches are calibration work, not extra verification
rounds or independent experimental observations. The caller calibrates once
on the full population for a dimension and reuses this immutable record across
all N values. No implicit cache, label, prototype, prediction, or method name
is used here.

Cosine includes both raw-row normalizations and the aligned dot product. Hash
includes canonical raw float32 byte serialization, SHA256, and hexdigest. The
generic vector unit performs a copy, comparison, and bit packing. Unit costs
are host elapsed kernel times; the shared resource factor scales those costs
at each simulated node. Network costs are accounted for separately.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from types import MappingProxyType
from typing import Mapping
import time

import numpy as np

from experiments.common.similarity import cosine_aligned


CALIBRATION_SCHEMA = "shared-host-kernels-v1"
COSINE_BATCH_PAIRS = 4096
HASH_BATCH_DESCRIPTORS = 4096
SCALAR_BATCH_ITEMS = 131072
VECTOR_BATCH_BYTES = 1024 * 1024
CALIBRATION_SAMPLE_LIMIT = 256
SCALAR_MODULUS = 7919
_REQUIRED_COSTS = ("cosine_per_pair_ms", "sha256_per_descriptor_ms",
                   "scalar_compare_ms", "scalar_modulo_ms", "vector_per_byte_ms")


@dataclass(frozen=True)
class KernelCalibration:
    """Immutable unit costs plus serializable calibration provenance."""

    costs_ms: Mapping[str, float]
    seed: int
    dimension: int
    input_shape: tuple[int, int]
    input_sha256: str
    sample_count: int
    benchmark_cpu_ms: float
    kernel_batch_cpu_ms: Mapping[str, float]
    schema: str = CALIBRATION_SCHEMA

    def __post_init__(self) -> None:
        costs = {key: float(self.costs_ms[key]) for key in _REQUIRED_COSTS}
        if any(not math.isfinite(value) or value <= 0 for value in costs.values()):
            raise ValueError("Calibrated kernel unit costs must be positive and finite")
        timings = {key: float(value) for key, value in self.kernel_batch_cpu_ms.items()}
        if any(not math.isfinite(value) or value <= 0 for value in timings.values()):
            raise ValueError("Calibration batch times must be positive and finite")
        if not math.isfinite(self.benchmark_cpu_ms) or self.benchmark_cpu_ms <= 0:
            raise ValueError("Calibration total time must be positive and finite")
        if self.dimension < 1 or self.input_shape[1] != self.dimension:
            raise ValueError("Calibration input shape and dimension disagree")
        if not 1 <= self.sample_count <= self.input_shape[0]:
            raise ValueError("Calibration sample count is outside its input population")
        object.__setattr__(self, "costs_ms", MappingProxyType(costs))
        object.__setattr__(self, "kernel_batch_cpu_ms", MappingProxyType(timings))
        object.__setattr__(self, "input_shape", tuple(self.input_shape))

    @property
    def descriptor_bytes(self) -> int:
        return 4 * self.dimension

    def to_dict(self) -> dict:
        """Return a detached JSON-compatible record; mutating it changes nothing."""
        return dict(schema=self.schema, costs_ms=dict(self.costs_ms), seed=self.seed,
                    dimension=self.dimension, input_shape=list(self.input_shape),
                    input_sha256=self.input_sha256, sample_count=self.sample_count,
                    benchmark_cpu_ms=self.benchmark_cpu_ms,
                    kernel_batch_cpu_ms=dict(self.kernel_batch_cpu_ms),
                    clock="perf_counter_ns elapsed host kernel time",
                    calibration_scope="one timed batch per kernel; shared across methods and N",
                    cosine_batch_pairs=COSINE_BATCH_PAIRS,
                    hash_batch_descriptors=HASH_BATCH_DESCRIPTORS,
                    scalar_batch_items=SCALAR_BATCH_ITEMS,
                    vector_batch_input_bytes=VECTOR_BATCH_BYTES,
                    scalar_modulus=SCALAR_MODULUS,
                    cosine_scope="complete raw-row normalization and aligned dot product",
                    hash_scope="canonical float32 bytes, SHA256 and hexdigest",
                    vector_scope="copy, compare and packbits per input byte")

    @classmethod
    def from_dict(cls, payload: Mapping) -> "KernelCalibration":
        """Rehydrate only an intact record of the declared calibration policy."""
        if not isinstance(payload, Mapping) or payload.get("schema") != CALIBRATION_SCHEMA:
            raise ValueError("Unsupported or absent kernel calibration schema")
        try:
            record = cls(
                costs_ms=payload["costs_ms"], seed=payload["seed"],
                dimension=payload["dimension"], input_shape=tuple(payload["input_shape"]),
                input_sha256=payload["input_sha256"], sample_count=payload["sample_count"],
                benchmark_cpu_ms=payload["benchmark_cpu_ms"],
                kernel_batch_cpu_ms=payload["kernel_batch_cpu_ms"], schema=payload["schema"],
            )
            if (isinstance(record.seed, bool) or not isinstance(record.seed, int) or record.seed < 0
                    or isinstance(record.dimension, bool) or not isinstance(record.dimension, int)
                    or len(record.input_shape) != 2
                    or any(isinstance(value, bool) or not isinstance(value, int) or value < 1
                           for value in record.input_shape)
                    or isinstance(record.sample_count, bool) or not isinstance(record.sample_count, int)
                    or record.sample_count != min(record.input_shape[0], CALIBRATION_SAMPLE_LIMIT)
                    or not isinstance(record.input_sha256, str) or len(record.input_sha256) != 64
                    or any(character not in "0123456789abcdef" for character in record.input_sha256)):
                raise ValueError("Invalid kernel calibration input provenance")
            expected_timings = dict(
                cosine=record.costs_ms["cosine_per_pair_ms"] * COSINE_BATCH_PAIRS,
                sha256=record.costs_ms["sha256_per_descriptor_ms"] * HASH_BATCH_DESCRIPTORS,
                scalar_compare=record.costs_ms["scalar_compare_ms"] * SCALAR_BATCH_ITEMS,
                scalar_modulo=record.costs_ms["scalar_modulo_ms"] * SCALAR_BATCH_ITEMS,
                vector=record.costs_ms["vector_per_byte_ms"] * VECTOR_BATCH_BYTES,
            )
            if (set(record.kernel_batch_cpu_ms) != set(expected_timings)
                    or any(not math.isclose(record.kernel_batch_cpu_ms[key], value,
                                            rel_tol=1e-12, abs_tol=0.)
                           for key, value in expected_timings.items())
                    or not math.isclose(record.benchmark_cpu_ms, sum(record.kernel_batch_cpu_ms.values()),
                                        rel_tol=1e-12, abs_tol=0.)
                    or record.to_dict() != dict(payload)):
                raise ValueError("Kernel calibration values or policy metadata disagree")
            return record
        except (KeyError, TypeError, IndexError) as exc:
            raise ValueError("Incomplete kernel calibration record") from exc


def _timed_ms(operation) -> tuple[float, object]:
    start = time.perf_counter_ns()
    result = operation()
    elapsed = (time.perf_counter_ns() - start) / 1_000_000
    if not math.isfinite(elapsed) or elapsed <= 0:
        raise RuntimeError("Host clock did not resolve the declared calibration batch")
    return elapsed, result


def calibrate_kernel_costs(X: np.ndarray, seed: int, d: int) -> KernelCalibration:
    """Calibrate once from full raw descriptors without modifying their contents.

    Use the same full-population X for every N in a dimensional sweep. Reusing
    a returned record is explicit: this function does not silently retain a
    stale calibration when its input or host execution environment changes.
    """
    x = np.asarray(X)
    if (isinstance(d, bool) or not isinstance(d, (int, np.integer)) or d < 1
            or x.ndim != 2 or x.shape[1] != d or x.shape[0] < 1
            or x.dtype != np.float32):
        raise ValueError("Calibration requires the original nonempty float32 (N,d) descriptors")
    if not np.isfinite(x).all() or np.any(np.max(np.abs(x), axis=1) == 0):
        raise ValueError("Calibration descriptors must be finite and nonzero")
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("Calibration seed must be a nonnegative integer")
    raw = np.ascontiguousarray(x, dtype="<f4").tobytes()
    input_hash = hashlib.sha256(
        b"overhead-calibration-raw-float32-v1\0" +
        np.asarray(x.shape, dtype="<i8").tobytes() + raw).hexdigest()
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), 27191]))
    sample_count = min(len(x), CALIBRATION_SAMPLE_LIMIT)
    sample_rows = rng.choice(len(x), sample_count, replace=False)
    sample = x[sample_rows].copy()
    pair_rows = rng.integers(0, sample_count, size=COSINE_BATCH_PAIRS)
    right_rows = rng.integers(0, sample_count, size=COSINE_BATCH_PAIRS)
    left, right = sample[pair_rows], sample[right_rows]
    hash_rows = np.arange(HASH_BATCH_DESCRIPTORS) % sample_count
    proof_values = [int(hashlib.sha256(np.ascontiguousarray(row, dtype="<f4").tobytes()).hexdigest(), 16)
                    for row in sample]
    scalar_values = [proof_values[index % sample_count] for index in range(SCALAR_BATCH_ITEMS)]
    expected = proof_values[0]
    vector_source = rng.integers(0, 2, VECTOR_BATCH_BYTES, dtype=np.uint8)

    # One untimed warmup per class; no adaptive repetitions or best-of selection.
    cosine_aligned(left[:8], right[:8])
    hashlib.sha256(np.ascontiguousarray(sample[0], dtype="<f4").tobytes()).hexdigest()
    _ = (proof_values[0] == expected, proof_values[0] % SCALAR_MODULUS)
    np.packbits(vector_source[:64].copy() == vector_source[:64], bitorder="big").tobytes()

    def hash_batch():
        return [hashlib.sha256(np.ascontiguousarray(sample[index], dtype="<f4").tobytes()).hexdigest()
                for index in hash_rows]

    def vector_batch():
        copied = vector_source.copy()
        return np.packbits(copied == vector_source, bitorder="big").tobytes()

    cosine_ms, cosine_result = _timed_ms(lambda: cosine_aligned(left, right))
    hash_ms, hash_result = _timed_ms(hash_batch)
    compare_ms, comparison_result = _timed_ms(lambda: [value == expected for value in scalar_values])
    modulo_ms, modulo_result = _timed_ms(lambda: [value % SCALAR_MODULUS for value in scalar_values])
    vector_ms, vector_result = _timed_ms(vector_batch)
    # Retain and validate actual kernel outputs. No expected performance ordering
    # is checked, and no simulated prediction is produced by this calibration.
    if (cosine_result.shape != (COSINE_BATCH_PAIRS,) or not np.isfinite(cosine_result).all()
            or len(hash_result) != HASH_BATCH_DESCRIPTORS
            or len(comparison_result) != SCALAR_BATCH_ITEMS
            or len(modulo_result) != SCALAR_BATCH_ITEMS
            or len(vector_result) != (VECTOR_BATCH_BYTES + 7) // 8):
        raise AssertionError("Calibration kernel did not execute its declared workload")
    timings = dict(cosine=cosine_ms, sha256=hash_ms, scalar_compare=compare_ms,
                   scalar_modulo=modulo_ms, vector=vector_ms)
    costs = dict(cosine_per_pair_ms=cosine_ms / COSINE_BATCH_PAIRS,
                 sha256_per_descriptor_ms=hash_ms / HASH_BATCH_DESCRIPTORS,
                 scalar_compare_ms=compare_ms / SCALAR_BATCH_ITEMS,
                 scalar_modulo_ms=modulo_ms / SCALAR_BATCH_ITEMS,
                 vector_per_byte_ms=vector_ms / VECTOR_BATCH_BYTES)
    return KernelCalibration(costs, int(seed), int(d), tuple(x.shape), input_hash,
                             sample_count, sum(timings.values()), timings)


def _operation_count(operation: Mapping, field: str) -> int:
    value = operation.get(field, 0)
    if (isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 0):
        raise ValueError("Operation byte and comparison counts must be nonnegative integers")
    return int(value)


def operation_duration(operation: Mapping, node_factor: float,
                       calibration: KernelCalibration) -> float:
    """Charge the operation class with the shared calibration and resource factor.

    Hash cost scales above one descriptor; even a shorter hash retains one
    descriptor's measured fixed setup cost. The cosine cost already includes
    both input rows, so their bytes are not charged a second time. Generic
    vector and scalar work can coexist in a single operation record.
    """
    factor = float(node_factor)
    if not math.isfinite(factor) or factor <= 0:
        raise ValueError("Node compute factor must be positive and finite")
    if not isinstance(calibration, KernelCalibration):
        raise TypeError("Shared operation costs require a KernelCalibration record")
    size = _operation_count(operation, "input_bytes")
    comparisons = _operation_count(operation, "comparisons")
    name = str(operation.get("operation", "")).lower()
    costs = calibration.costs_ms
    if "cosine" in name:
        cost = costs["cosine_per_pair_ms"] * max(1, comparisons)
    elif "hash" in name or "sha" in name:
        cost = costs["sha256_per_descriptor_ms"] * max(1.0, size / calibration.descriptor_bytes)
        cost += comparisons * costs["scalar_compare_ms"]
    else:
        scalar_key = ("scalar_modulo_ms" if operation.get("arithmetic") == "modulo"
                      or "modulo" in name else "scalar_compare_ms")
        cost = size * costs["vector_per_byte_ms"] + comparisons * costs[scalar_key]
    duration = factor * cost
    if not math.isfinite(duration) or duration < 0:
        raise ValueError("Calibrated operation duration is outside finite nonnegative costs")
    return duration
