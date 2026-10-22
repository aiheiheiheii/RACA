"""Raw-content helpers for the three exact hash/digest replica baselines.

A common original descriptor is not provided by this dataset. I-EDI/CTCV
therefore choose a pool reference by exact SHA256 multiplicity, then history
and temporary identity. This is an explicit same-content replica adaptation;
no tolerance metric, ground-truth labels or generator prototype is used.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import numpy as np
from experiments.common.dataset_adapter import AlgorithmInput


def readonly(values: np.ndarray, dtype=None) -> np.ndarray:
    result = np.array(values, dtype=dtype, copy=True)
    result.setflags(write=False)
    return result


def canonical_descriptor_bytes(row: np.ndarray) -> bytes:
    """Serialize stored values verbatim as little-endian float32 C-order bytes.

    No normalization, rounding, tolerance or content canonicalization is done.
    The fixed storage dtype and byte order only make hash inputs reproducible.
    """
    values = np.asarray(row)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("Hash payload must be a finite nonempty descriptor row")
    if values.dtype.kind != "f" or values.dtype.itemsize != 4:
        raise ValueError("Hash baselines require the stored float32 descriptors")
    return np.ascontiguousarray(values, dtype="<f4").tobytes(order="C")


def descriptor_hash(row: np.ndarray) -> str:
    return hashlib.sha256(canonical_descriptor_bytes(row)).hexdigest()


def exact_hash_anchor(data: AlgorithmInput) -> tuple[int, tuple[str, ...]]:
    """Choose the most frequent full-content hash; deterministic observable ties.

    Even a singleton hash group is allowed. This does not certify that the
    chosen model is correct and does not reuse RACA-HIM's reference selection.
    """
    pool = np.asarray(data.reference_indices, dtype=np.int64)
    if not len(pool):
        raise ValueError("A hash reference requires a nonempty reference pool")
    hashes = tuple(descriptor_hash(data.X[index]) for index in pool)
    counts = Counter(hashes)
    order = sorted(range(len(pool)), key=lambda position: (
        -counts[hashes[position]], float(data.history[pool[position]]),
        int(data.model_ids[pool[position]])))
    return int(pool[order[0]]), hashes
