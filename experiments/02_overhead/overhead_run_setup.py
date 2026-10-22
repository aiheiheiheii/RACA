"""Persisted shared calibration for one controlled Group 02 generation.

Calibration is not repeated when resuming N points. The policy, all immutable
unit-cost records, and their joint SHA256 are bound into the run identity.
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping

from .overhead_calibration import (
    CALIBRATION_SCHEMA, COSINE_BATCH_PAIRS, HASH_BATCH_DESCRIPTORS,
    SCALAR_BATCH_ITEMS, VECTOR_BATCH_BYTES, CALIBRATION_SAMPLE_LIMIT,
    KernelCalibration, calibrate_kernel_costs,
)

CALIBRATION_POPULATION_N = 5000


def calibration_key(seed: int, d: int) -> str:
    return f"seed{seed}_d{d}"


def calibration_policy() -> dict:
    """Return the fixed public batch policy; no target rank enters it."""
    return dict(schema=CALIBRATION_SCHEMA, population_n=CALIBRATION_POPULATION_N,
                scope="one maximum-domain raw descriptor population per seed/d; shared across N and methods",
                clock="perf_counter_ns elapsed host kernel time", threadpool_limit=1,
                timed_batches_per_kernel=1, untimed_warmups_per_kernel=1,
                cosine_batch_pairs=COSINE_BATCH_PAIRS,
                hash_batch_descriptors=HASH_BATCH_DESCRIPTORS,
                scalar_batch_items=SCALAR_BATCH_ITEMS,
                vector_batch_input_bytes=VECTOR_BATCH_BYTES,
                maximum_descriptor_samples=CALIBRATION_SAMPLE_LIMIT,
                predictions="actual protocol kernels; calibration sets costs only")


def calibration_snapshot_sha256(records: Mapping, policy: Mapping) -> str:
    serialized = json.dumps(dict(policy=dict(policy), records=dict(records)),
                            sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def validate_calibration_snapshot(records, policy, seeds, dimensions) -> dict:
    """Reject missing, extra, corrupted, or mismatched calibration records."""
    if not isinstance(policy, Mapping) or dict(policy) != calibration_policy():
        raise ValueError("Different or unsupported kernel calibration policy")
    if not isinstance(records, Mapping):
        raise ValueError("Kernel calibration snapshot must be an object")
    expected = {(int(seed), int(d)) for seed in seeds for d in dimensions}
    if set(records) != {calibration_key(seed, d) for seed, d in expected}:
        raise ValueError("Kernel calibration snapshot does not match the declared seed/d grid")
    calibrations = {}
    for seed, d in sorted(expected):
        calibration = KernelCalibration.from_dict(records[calibration_key(seed, d)])
        if (calibration.seed != seed or calibration.dimension != d
                or calibration.input_shape != (CALIBRATION_POPULATION_N, d)):
            raise ValueError("Kernel calibration record has a different seed/d/full population")
        calibrations[(seed, d)] = calibration
    return calibrations


def build_calibration_snapshot(loader, dataset_root, config, seeds, dimensions) -> tuple[dict, dict]:
    """Build the full snapshot once before saving any trial in a generation."""
    records, calibrations = {}, {}
    for seed in seeds:
        for d in dimensions:
            group = loader(dataset_root=dataset_root, n=CALIBRATION_POPULATION_N,
                           corruption_rate_percent=config.corruption_rate_percent,
                           dim=d, H=config.H, seed=seed)
            calibration = calibrate_kernel_costs(group["X"], seed, d)
            records[calibration_key(seed, d)] = calibration.to_dict()
            calibrations[(seed, d)] = calibration
    validate_calibration_snapshot(records, calibration_policy(), seeds, dimensions)
    return records, calibrations
