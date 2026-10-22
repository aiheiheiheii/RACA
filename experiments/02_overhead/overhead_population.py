"""Offline stratified, nested descriptor populations for the Group 02 N sweep.

Labels are used only while constructing the deployment fixture, in the same
role as the dataset's stratified sampling quotas.  They are discarded before
returning a group to an algorithm.  Every prefix preserves the original 3%
corruption quota and 80/20 subtype splits while keeping existing rows intact.
No sample ID, prototype, or label is supplied to verification decisions.
"""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_UP
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np

from dataset.load_dataset import ROOT as DATASET_ROOT, load_group

SCALING_POPULATION_SCHEMA = "overhead-nested-stratified-population-v1"
SCALING_N_VALUES = tuple(range(1000, 5001, 500))
SUBTYPE_NAMES = (
    "common_normal", "high_personalized_normal", "severe_corrupt", "mild_corrupt",
)
_COHORT_SHUFFLE_STREAM = 729103


def _integer(name: str, value: int, minimum: int) -> int:
    if (isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer)) or value < minimum):
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _corruption_rate(value: float) -> float:
    if (isinstance(value, (bool, np.bool_)) or not math.isfinite(value)
            or not 0.0 <= value <= 50.0):
        raise ValueError("corruption_rate_percent must be finite and in [0,50]")
    return float(value)


def _round_count(value: float) -> int:
    return int(Decimal(str(value)).quantize(Decimal("1"), rounding=ROUND_HALF_UP))


def population_quotas(n: int, corruption_rate_percent: float = 3.0) -> tuple[int, ...]:
    """Use exactly the dataset loader's HALF_UP cumulative subtype quotas."""
    n = _integer("n", n, 1)
    rate = _corruption_rate(corruption_rate_percent)
    corrupt = _round_count(n * rate / 100.0)
    normal = n - corrupt
    high = _round_count(normal * 0.20)
    mild = _round_count(corrupt * 0.20)
    return normal - high, high, corrupt - mild, mild


def _validated_grid(grid: Iterable[int]) -> tuple[int, ...]:
    values = tuple(_integer("grid entry", value, 1) for value in grid)
    if not values or any(later <= first for first, later in zip(values, values[1:])):
        raise ValueError("grid must be nonempty and strictly increasing")
    return values


@dataclass(frozen=True)
class NestedPopulation:
    """Immutable deployment descriptors; only aggregate counts remain."""

    X: np.ndarray
    grid: tuple[int, ...]
    seed: int
    corruption_rate_percent: float
    counts_by_n: tuple[tuple[int, tuple[int, ...]], ...]
    population_sha256: str

    def prefix(self, n: int) -> dict:
        """Return a detached group dictionary without any per-row labels."""
        n = _integer("n", n, 1)
        if n not in self.grid:
            raise ValueError("n must be a declared nested population grid point")
        counts = dict(zip(SUBTYPE_NAMES, dict(self.counts_by_n)[n]))
        increments = tuple(b - a for a, b in zip(self.grid, self.grid[1:]))
        cohort_size = increments[0] if increments and len(set(increments)) == 1 else None
        prefix = self.X[:n]
        metadata = {
            "schema": SCALING_POPULATION_SCHEMA,
            "population_sha256": self.population_sha256,
            "prefix_sha256": hashlib.sha256(prefix.tobytes()).hexdigest(),
            "base_size": self.grid[0],
            "cohort_size": cohort_size,
            "domain_max": self.grid[-1],
            "grid": list(self.grid),
            "seed": self.seed,
            "dimension": int(self.X.shape[1]),
            "corruption_rate_percent": self.corruption_rate_percent,
            "counts": dict(counts),
            "prefix_counts": {
                str(point): dict(zip(SUBTYPE_NAMES, values))
                for point, values in self.counts_by_n
            },
            "construction": "offline stratified cohort construction; labels discarded",
        }
        return {
            "X": prefix,
            "N": n,
            "dimension": int(self.X.shape[1]),
            "seed": self.seed,
            "corruption_rate_percent": self.corruption_rate_percent,
            "counts": counts,
            "scaling_fixture": metadata,
        }


def build_population_from_group(group: dict, *, grid=SCALING_N_VALUES,
                                seed: int = 2026,
                                corruption_rate_percent: float = 3.0) -> NestedPopulation:
    """Construct the fixture from max-domain X and offline subtype strata.

    A source row enters once, in the first arrival cohort needing its subtype.
    Independent anonymous shuffling inside cohorts removes subtype block order.
    The constructor never reads sample_id, prototype, cosine, or binary labels.
    """
    grid = _validated_grid(grid)
    seed = _integer("seed", seed, 0)
    rate = _corruption_rate(corruption_rate_percent)
    x = np.asarray(group["X"])
    labels = np.asarray(group["subtype_label"])
    if (x.dtype != np.float32 or x.ndim != 2 or x.shape[0] != grid[-1]
            or x.shape[1] < 1 or not np.isfinite(x).all()):
        raise ValueError("Expected finite float32 max-domain descriptors")
    if (labels.shape != (grid[-1],) or labels.dtype.kind not in "iu"
            or np.any(labels < 0) or np.any(labels > 3)):
        raise ValueError("Offline subtype labels must be integer strata 0..3")
    queues = tuple(np.flatnonzero(labels == subtype) for subtype in range(4))
    if tuple(len(queue) for queue in queues) != population_quotas(grid[-1], rate):
        raise ValueError("Source population does not match the declared subtype quotas")

    source_order, records = [], []
    previous_n = 0
    previous_quotas = (0, 0, 0, 0)
    for n in grid:
        quotas = population_quotas(n, rate)
        cohort = np.concatenate([
            queue[start:stop] for queue, start, stop in zip(queues, previous_quotas, quotas)
        ])
        if len(cohort) != n - previous_n:
            raise ValueError("Grid quotas cannot form the requested arrival cohort")
        rng = np.random.default_rng(np.random.SeedSequence([
            seed, _COHORT_SHUFFLE_STREAM, previous_n,
        ]))
        cohort = cohort[rng.permutation(len(cohort))]
        source_order.append(cohort)
        # This is an offline sanity check. Only these aggregate counts survive.
        accumulated = np.concatenate(source_order)
        actual = tuple(int(np.count_nonzero(labels[accumulated] == subtype))
                       for subtype in range(4))
        if actual != quotas:
            raise ValueError("Nested population violates cumulative subtype quotas")
        records.append((n, actual))
        previous_n, previous_quotas = n, quotas

    ordered = np.concatenate(source_order)
    if len(np.unique(ordered)) != grid[-1]:
        raise ValueError("A source descriptor cannot occur in multiple cohorts")
    raw = np.ascontiguousarray(x[ordered], dtype="<f4").tobytes()
    immutable = np.frombuffer(raw, dtype="<f4").reshape(x.shape)
    identity = {
        "schema": SCALING_POPULATION_SCHEMA,
        "grid": grid,
        "seed": seed,
        "corruption_rate_percent": rate,
        "dimension": x.shape[1],
        "counts_by_n": records,
    }
    digest = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
        + b"\n" + raw
    ).hexdigest()
    return NestedPopulation(immutable, grid, seed, rate, tuple(records), digest)


@lru_cache(maxsize=16)
def _load_population(root_text: str, rate: float, dim: int,
                     H: float, seed: int) -> NestedPopulation:
    maximum = load_group(dataset_root=Path(root_text), n=SCALING_N_VALUES[-1],
                         corruption_rate_percent=rate, dim=dim, H=H, seed=seed)
    return build_population_from_group(maximum, seed=seed, corruption_rate_percent=rate)


def load_scaling_group(dataset_root=DATASET_ROOT, n: int = 3000,
                       corruption_rate_percent: float = 3.0,
                       dim: int = 30, H: float = 0.5, seed: int = 2026) -> dict:
    """Load one maximum-domain population and return its controlled N prefix."""
    n = _integer("n", n, 1)
    if n not in SCALING_N_VALUES:
        raise ValueError("n must be a declared Group 02 scaling grid point")
    seed = _integer("seed", seed, 0)
    dim = _integer("dim", dim, 1)
    rate = _corruption_rate(corruption_rate_percent)
    if isinstance(H, (bool, np.bool_)) or not math.isfinite(H):
        raise ValueError("H must be finite")
    population = _load_population(str(Path(dataset_root).resolve()), rate, dim, float(H), seed)
    result = population.prefix(n)
    result["heterogeneity_H"] = float(H)
    result["scaling_fixture"]["heterogeneity_H"] = float(H)
    return result
