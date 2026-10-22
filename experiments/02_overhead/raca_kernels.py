"""Exact local RACA-HIM kernels that reuse repeated reference work.

These helpers change neither descriptors nor verification thresholds. The
cosine helper returns the same clipped scores as the shared aligned kernel;
only repeated normalization of an unchanged row is avoided. History updates
retain the single-event formulas and require one event per target.
"""
from __future__ import annotations

import math

import numpy as np

from experiments.common.similarity import _unit_rows


def _row_indices(values, n: int, name: str, *, unique: bool = False) -> np.ndarray:
    raw = np.asarray(values)
    if raw.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional integer vector")
    # Empty Python sequences have NumPy's default float dtype; they contain no
    # invalid identities and are useful for a no-event update.
    if not len(raw):
        return np.empty(0, dtype=np.int64)
    if not np.issubdtype(raw.dtype, np.integer):
        raise ValueError(f"{name} must contain integer row identities")
    if np.any(raw < 0) or np.any(raw >= n):
        raise ValueError(f"{name} contains an out-of-range row identity")
    rows = raw.astype(np.int64, copy=False)
    if unique and len(np.unique(rows)) != len(rows):
        raise ValueError(f"{name} must contain unique row identities")
    return rows


def cached_reference_cosine(X, target_rows, verifier_rows) -> np.ndarray:
    """Return unchanged aligned cosine scores, normalizing each used row once.

    Repeated targets and verifiers are permitted, and output positions follow
    the supplied aligned pairs. Unused rows are not inspected, matching
    ``cosine_aligned(X[verifier_rows], X[target_rows])``. No normalization is
    written back to the original descriptors or to any hash input.
    """
    values = np.asarray(X)
    if values.ndim != 2 or values.shape[1] == 0:
        raise ValueError("Descriptors must be a matrix with at least one feature")
    targets = _row_indices(target_rows, len(values), "target_rows")
    verifiers = _row_indices(verifier_rows, len(values), "verifier_rows")
    if targets.shape != verifiers.shape:
        raise ValueError("Target and verifier rows must have matching aligned shapes")
    if not len(targets):
        return np.empty(0, dtype=np.float64)
    unique_rows, inverse = np.unique(
        np.concatenate((verifiers, targets)), return_inverse=True,
    )
    units = _unit_rows(values[unique_rows])
    count = len(targets)
    left, right = units[inverse[:count]], units[inverse[count:]]
    return np.clip(np.einsum("ij,ij->i", left, right), -1.0, 1.0)


def history_update_vectorized(history, target_rows, predictions, gt_rows, rho) -> np.ndarray:
    """Copy and update unique targets plus disjoint trusted references.

    Target verdicts must be integer/bool zero or one. A normal verdict and
    accepted GT membership apply ``(1-rho)*h``; an abnormal verdict applies
    ``h+rho*(1-h)``. The original history is never mutated.
    """
    state = np.asarray(history)
    if state.ndim != 1 or not np.issubdtype(state.dtype, np.floating):
        raise ValueError("History must be a one-dimensional floating-point vector")
    if not np.isfinite(state).all() or np.any((state < 0) | (state > 1)):
        raise ValueError("History must be finite and in [0,1]")
    if isinstance(rho, (bool, np.bool_)):
        raise ValueError("rho must be finite and in (0,1]")
    try:
        rate = float(rho)
    except (TypeError, ValueError) as exc:
        raise ValueError("rho must be finite and in (0,1]") from exc
    if not math.isfinite(rate) or not 0.0 < rate <= 1.0:
        raise ValueError("rho must be finite and in (0,1]")
    targets = _row_indices(target_rows, len(state), "target_rows", unique=True)
    references = _row_indices(gt_rows, len(state), "gt_rows", unique=True)
    if np.intersect1d(targets, references, assume_unique=True).size:
        raise ValueError("Targets and GT rows must be disjoint")
    verdicts = np.asarray(predictions)
    if verdicts.ndim != 1 or len(verdicts) != len(targets):
        raise ValueError("Predictions must align with target rows")
    if len(verdicts) and (not (
            np.issubdtype(verdicts.dtype, np.integer)
            or np.issubdtype(verdicts.dtype, np.bool_))
            or np.any((verdicts != 0) & (verdicts != 1))):
        raise ValueError("Predictions must be binary integer or bool verdicts")
    # The experiment's history state uses float64. Preserving that precision
    # also makes these elementwise operations identical to its scalar loop.
    updated = np.array(state, dtype=np.float64, copy=True)
    current = updated[targets]
    updated[targets] = np.where(
        verdicts.astype(bool, copy=False),
        current + rate * (1.0 - current),
        current * (1.0 - rate),
    )
    updated[references] *= 1.0 - rate
    return updated
