"""Read-only data loading with a strict observations/evaluation-label boundary."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any

import numpy as np

from .config import DATASET_ROOT, DetectionConfig
from .reproducibility import seed_rng

# Importing an unmodified dataset module must not add dataset/__pycache__ files.
# The entry point also disables bytecode before importing any project code.
sys.dont_write_bytecode = True
from dataset.load_dataset import load_group  # noqa: E402


def _readonly(values: Any, dtype: Any = None) -> np.ndarray:
    array = np.array(values, dtype=dtype, copy=True)
    array.setflags(write=False)
    return array


@dataclass(frozen=True)
class AlgorithmInput:
    """Only observable descriptors, temporary identities, history and partitions.

    No labels, source sample IDs, prototype, or cosine-to-prototype fields are
    present. Indices refer to full-N arrays, while labels are held separately
    by ``EvaluationData`` for metric calculation and posthoc GT diagnostics.
    """

    X: np.ndarray
    model_ids: np.ndarray
    history: np.ndarray
    reference_indices: np.ndarray
    evaluation_indices: np.ndarray

    def __post_init__(self) -> None:
        if np.asarray(self.X).dtype != np.float32:
            raise ValueError("Algorithm descriptors must preserve dataset float32 precision")
        x = _readonly(self.X)
        if x.ndim != 2 or x.shape[0] == 0 or x.shape[1] == 0:
            raise ValueError("X must have a nonempty (N, d) shape")
        if not np.all(np.isfinite(x)) or np.any(np.max(np.abs(x), axis=1) == 0):
            raise ValueError("Descriptors must be finite, nonzero vectors")
        n = len(x)
        raw_ids = np.asarray(self.model_ids)
        if raw_ids.shape != (n,) or not np.issubdtype(raw_ids.dtype, np.integer):
            raise ValueError("Temporary model IDs must be an integer vector of length N")
        ids = _readonly(raw_ids, np.int64)
        if len(np.unique(ids)) != n:
            raise ValueError("Temporary model IDs must be unique")
        history = _readonly(self.history, np.float64)
        if history.shape != (n,) or not np.all(np.isfinite(history)) or np.any((history < 0) | (history > 1)):
            raise ValueError("History must be finite in [0,1] with one value per model")
        partitions = []
        for name in ("reference_indices", "evaluation_indices"):
            raw = np.asarray(getattr(self, name))
            if raw.ndim != 1 or not np.issubdtype(raw.dtype, np.integer):
                raise ValueError(f"{name} must be a one-dimensional integer vector")
            array = _readonly(raw, np.int64)
            if len(np.unique(array)) != len(array) or np.any((array < 0) | (array >= n)):
                raise ValueError(f"{name} must contain unique, valid row indices")
            partitions.append(array)
        joined = np.concatenate(partitions)
        if len(joined) != n or len(np.unique(joined)) != n:
            raise ValueError("Reference and evaluation partitions must be disjoint and cover N rows")
        for name, value in zip(
            ("X", "model_ids", "history", "reference_indices", "evaluation_indices"),
            (x, ids, history, *partitions),
        ):
            object.__setattr__(self, name, value)


@dataclass(frozen=True)
class EvaluationData:
    """Full-N evaluation-only labels; slice using evaluation_indices for scores.

    Full arrays also support posthoc reference/GT composition diagnostics.
    Algorithms receive only ``observations``, never this evaluation container.
    """

    observations: AlgorithmInput
    binary_label: np.ndarray
    subtype_label: np.ndarray

    def __post_init__(self) -> None:
        truth = np.asarray(self.binary_label)
        subtypes = np.asarray(self.subtype_label)
        shape = (len(self.observations.X),)
        if truth.shape != shape or not np.all(np.isin(truth, (0, 1))):
            raise ValueError("Evaluation binary labels must be 0/1 and match all descriptor rows")
        if subtypes.shape != shape or not np.all(np.isin(subtypes, (0, 1, 2, 3))):
            raise ValueError("Evaluation subtype labels must be 0/1/2/3 and match all descriptor rows")
        object.__setattr__(self, "binary_label", _readonly(truth, np.int8))
        object.__setattr__(self, "subtype_label", _readonly(subtypes, np.int8))


def prepare_evaluation_data(
    group: Mapping[str, Any], seed: int, config: DetectionConfig
) -> EvaluationData:
    """Split once without observing labels or provenance; never resample on failure.

    Independent random streams govern the shared reference/evaluation split
    and temporary identities. Algorithm observations are consequently unchanged
    by perturbing labels, sample_id, prototype or cosine_to_prototype.
    """
    x = np.asarray(group["X"])
    if x.ndim != 2 or x.dtype != np.float32 or x.shape[1] != config.dim:
        raise ValueError(f"Expected float32 descriptors with shape (N, {config.dim})")
    n = len(x)
    if int(group.get("N", n)) != n:
        raise ValueError("Declared dataset N differs from descriptor row count")
    if config.reference_pool_size >= n:
        raise ValueError("Reference pool must leave at least one evaluation target")
    cr = float(group["corruption_rate_percent"])
    split_rng = seed_rng(seed, 1101, n, cr)
    id_rng = seed_rng(seed, 1102, n, cr)
    partition = split_rng.permutation(n)
    # Numeric IDs are independent temporary identities, not source-bank IDs.
    # Their permutation also provides a reproducible neutral tie-break rule.
    model_ids = id_rng.permutation(n).astype(np.int64) + 1_000_000
    observations = AlgorithmInput(
        X=x,
        model_ids=model_ids,
        history=np.full(n, config.h0, dtype=np.float64),
        reference_indices=partition[: config.reference_pool_size],
        evaluation_indices=partition[config.reference_pool_size :],
    )
    # Labels are accessed only after the complete label-blind observations exist.
    truth = np.asarray(group["binary_label"])
    subtypes = np.asarray(group["subtype_label"])
    if truth.shape != (n,) or subtypes.shape != (n,):
        raise ValueError("Full dataset labels must match descriptor row count")
    if not np.all(np.isin(truth, (0, 1))) or not np.all(np.isin(subtypes, (0, 1, 2, 3))):
        raise ValueError("Dataset binary/subtype label encoding is invalid")
    return EvaluationData(observations, truth, subtypes)


def load_evaluation_data(
    N: int,
    cr: float,
    seed: int,
    config: DetectionConfig,
    dataset_root: str | Path = DATASET_ROOT,
) -> EvaluationData:
    """Load actual descriptor banks through the provided dataset API."""
    group = load_group(
        dataset_root=Path(dataset_root).resolve(),
        n=N,
        corruption_rate_percent=cr,
        dim=config.dim,
        H=config.H,
        seed=seed,
    )
    if group["X"].shape != (N, config.dim):
        raise ValueError("Loaded descriptor shape does not match the requested N/d")
    return prepare_evaluation_data(group, seed, config)
