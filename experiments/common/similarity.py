"""Stable float64 cosine calculations with the declared inclusive interval."""

from __future__ import annotations

import numpy as np


def _unit_rows(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim == 1:
        values = values.reshape(1, -1)
    if values.ndim != 2 or values.shape[1] == 0:
        raise ValueError("Descriptors must be a matrix with at least one feature")
    if not np.all(np.isfinite(values)):
        raise ValueError("Descriptors contain NaN or Inf")
    # Scaling first avoids overflow/underflow when a finite vector's magnitude
    # is much larger/smaller than that of the stored unit descriptors.
    scale = np.max(np.abs(values), axis=1, keepdims=True)
    if np.any(scale == 0.0):
        raise ValueError("Cosine similarity is undefined for a zero vector")
    scaled = values / scale
    return scaled / np.linalg.norm(scaled, axis=1, keepdims=True)


def cosine_pairwise(X: np.ndarray, Y: np.ndarray | None = None) -> np.ndarray:
    """Return all row-pair cosines, rejecting non-finite and zero vectors."""
    left = _unit_rows(X)
    right = left if Y is None else _unit_rows(Y)
    if left.shape[1] != right.shape[1]:
        raise ValueError("Descriptor feature dimensions differ")
    return np.clip(left @ right.T, -1.0, 1.0)


def cosine_aligned(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Return one cosine per aligned pair without allocating an N-by-N matrix."""
    left = _unit_rows(X)
    right = _unit_rows(Y)
    if left.shape != right.shape:
        raise ValueError("Aligned descriptor arrays must have the same shape")
    return np.clip(np.einsum("ij,ij->i", left, right), -1.0, 1.0)


def within_interval(sim: np.ndarray, lower: float, upper: float) -> np.ndarray:
    """Use inclusive bounds exactly; clipping belongs to the cosine routine.

    No tolerance widens the scientific threshold. NaN/Inf are errors rather
    than silently producing a corrupted prediction.
    """
    values = np.asarray(sim, dtype=np.float64)
    if not np.all(np.isfinite(values)):
        raise ValueError("Similarity values contain NaN or Inf")
    if not np.isfinite(lower) or not np.isfinite(upper) or not -1 <= lower <= upper <= 1:
        raise ValueError("Require finite -1 <= lower <= upper <= 1")
    return (values >= lower) & (values <= upper)

