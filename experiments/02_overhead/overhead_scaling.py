"""Anonymous arrival identities and nested workloads for a controlled N sweep.

These helpers know only row indices and budgets.  They never inspect descriptor
contents or labels.  Earlier arrival cohorts have lower anonymous identities,
so adding nodes does not replace a valid initial reference cohort.  Work grows
by taking the budget increment from each new arrival cohort.
"""
from __future__ import annotations

from typing import Callable

import numpy as np

_IDENTITY_STREAM = 920105
_WORKLOAD_STREAM = 920205


def _integer(name: str, value: int, minimum: int) -> int:
    if (isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer)) or value < minimum):
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _cohorts(n: int, base_size: int, cohort_size: int) -> tuple[tuple[int, int], ...]:
    n = _integer("n", n, 1)
    base_size = _integer("base_size", base_size, 1)
    cohort_size = _integer("cohort_size", cohort_size, 1)
    if n < base_size or (n - base_size) % cohort_size:
        raise ValueError("n must include the complete base and complete arrival cohorts")
    return ((0, base_size),) + tuple(
        (start, start + cohort_size) for start in range(base_size, n, cohort_size)
    )


def _immutable_int64(values) -> np.ndarray:
    values = np.ascontiguousarray(values, dtype="<i8")
    return np.frombuffer(values.tobytes(), dtype="<i8")


def stable_model_ids(n: int, seed: int = 2026, *, base_size: int = 1000,
                     cohort_size: int = 500) -> np.ndarray:
    """Assign reproducible anonymous IDs independently inside arrival cohorts.

    A row keeps the same ID at every larger N.  The base uses IDs [0,base_size),
    and each later cohort uses the next higher interval in a random order.
    Neither the RNG stream nor the existing cohort size depends on current N.
    """
    cohorts = _cohorts(n, base_size, cohort_size)
    seed = _integer("seed", seed, 0)
    pieces = []
    for start, stop in cohorts:
        rng = np.random.default_rng(np.random.SeedSequence([
            seed, _IDENTITY_STREAM, start,
        ]))
        pieces.append(rng.permutation(stop - start) + start)
    return _immutable_int64(np.concatenate(pieces))


def nested_fixed_targets(n: int, model_ids, excluded_rows,
                         budget_fn: Callable[[int], int], seed: int = 2026, *,
                         base_size: int = 1000,
                         cohort_size: int = 500) -> np.ndarray:
    """Choose a content-blind incremental workload from each arrival cohort.

    The base contributes B(base_size); later cohorts contribute B(end)-B(start).
    Prefix nesting holds when existing IDs and excluded reference rows remain
    fixed.  If GT expansion introduces new reference rows, the exclusions can
    legitimately change the workload; callers must disclose that condition
    rather than claim the unchanged-reference scaling control still holds.
    """
    cohorts = _cohorts(n, base_size, cohort_size)
    n = int(n)
    seed = _integer("seed", seed, 0)
    if not callable(budget_fn):
        raise ValueError("budget_fn must be callable")
    ids = np.asarray(model_ids)
    if (ids.shape != (n,) or ids.dtype.kind not in "iu"
            or np.any(ids < 0) or np.any(ids > np.iinfo(np.int64).max)
            or len(np.unique(ids)) != n):
        raise ValueError("model_ids must contain n unique nonnegative int64 identities")
    ids = ids.astype(np.int64, copy=False)
    excluded = np.asarray(excluded_rows)
    if excluded.size == 0 and excluded.ndim == 1:
        excluded = np.empty(0, dtype=np.int64)
    if (excluded.ndim != 1 or excluded.dtype.kind not in "iu"
            or np.any(excluded < 0) or np.any(excluded >= n)):
        raise ValueError("excluded_rows must be valid integer row indices")
    excluded = np.unique(excluded.astype(np.int64, copy=False))

    selected, previous_budget = [], 0
    for start, stop in cohorts:
        budget = _integer("budget", budget_fn(stop), 0)
        increment = budget - previous_budget
        if increment < 0:
            raise ValueError("The scaling workload budget must be nondecreasing")
        candidates = np.setdiff1d(np.arange(start, stop), excluded, assume_unique=True)
        if increment > len(candidates):
            raise ValueError("The cohort lacks enough non-reference rows for its budget increment")
        # Sorting before sampling ties the selection to stable anonymous IDs,
        # without looking at X, label, bank identity, or the current N.
        candidates = candidates[np.argsort(ids[candidates], kind="stable")]
        rng = np.random.default_rng(np.random.SeedSequence([
            seed, _WORKLOAD_STREAM, start,
        ]))
        selected.append(candidates[rng.permutation(len(candidates))[:increment]])
        previous_budget = budget
    result = np.concatenate(selected)
    if len(result) != previous_budget or len(np.unique(result)) != len(result):
        raise ValueError("Nested workload cardinality does not match the declared budget")
    return _immutable_int64(result)
