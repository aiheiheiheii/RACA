"""Offline controlled scenes with an explicit observations/labels boundary.

The dataset's labels control only the composition of the 100-member workload.
Methods receive ``Scene.algorithm_input`` and an availability mask, never the
scene, its source rows, its labels, or a dataset generation prototype. One
scene is reused for all five failure thresholds and all four methods.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

# The common adapter disables bytecode before loading the read-only dataset.
from experiments.common.dataset_adapter import AlgorithmInput
from experiments.common.reproducibility import sha256_array
from dataset.load_dataset import draw_candidate_set, load_group

from .config_ground_truth import (
    D, DATASET_ROOT, DELTA_K, FAILURE_VALUES, GLOBAL_CR_PERCENT, H, H0,
    INITIAL_K, MAX_CANDIDATE_POOL_SIZE, N, Q_VALUES, TAU_LOWER, TAU_UPPER,
)

# Dataset candidate sampling already uses SeedSequence([seed, 1147]). Separate
# fixed streams prevent failure uniforms and Random-M from consuming its state.
DATASET_CANDIDATE_STREAM = 1147
FAILURE_STREAM = 4401
RANDOM_METHOD_STREAM = 4402


def _readonly(values: Any, dtype: Any = None) -> np.ndarray:
    array = np.array(values, dtype=dtype, copy=True)
    array.setflags(write=False)
    return array


def _seed_value(seed: int) -> int:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("Trial seed must be a nonnegative integer")
    return int(seed)


def _grid_index(value: float, grid: tuple[float, ...], name: str) -> int:
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a declared probability") from exc
    if not np.isfinite(numeric):
        raise ValueError(f"{name} must be finite")
    matches = [index for index, point in enumerate(grid) if abs(numeric - point) <= 1e-12]
    if len(matches) != 1:
        raise ValueError(f"{name} must belong to {grid}")
    return matches[0]


def _validate_group(group: Mapping[str, Any], seed: int | None = None) -> None:
    """Quality checks use evaluation labels without changing any descriptors."""
    x = np.asarray(group["X"])
    if x.shape != (N, D) or x.dtype != np.float32:
        raise ValueError(f"Expected float32 descriptors with shape ({N}, {D})")
    if not np.all(np.isfinite(x)) or np.any(np.max(np.abs(x), axis=1) == 0):
        raise ValueError("Dataset descriptors must be finite nonzero vectors")
    truth = np.asarray(group["binary_label"])
    subtypes = np.asarray(group["subtype_label"])
    if (truth.shape != (N,) or not np.issubdtype(truth.dtype, np.integer)
            or not np.isin(truth, (0, 1)).all()):
        raise ValueError("Dataset binary labels must be an integer 0/1 vector of length N")
    if (subtypes.shape != (N,) or not np.issubdtype(subtypes.dtype, np.integer)
            or set(np.unique(subtypes).tolist()) != {0, 1, 2, 3}):
        raise ValueError("All four original dataset subtypes must be retained")
    if not np.array_equal(truth, (subtypes >= 2).astype(truth.dtype)):
        raise ValueError("Dataset binary and subtype evaluation labels disagree")
    expected_bad = int(round(N * GLOBAL_CR_PERCENT / 100.0))
    if np.count_nonzero(truth) != expected_bad or expected_bad < MAX_CANDIDATE_POOL_SIZE // 2:
        raise ValueError("The global 3% group must contain 90 corrupted rows, sufficient for q_C=50%")
    for key, expected in (
        ("N", N), ("dimension", D), ("corruption_rate_percent", GLOBAL_CR_PERCENT),
        ("heterogeneity_H", H), ("tau_lower", TAU_LOWER), ("tau_upper", TAU_UPPER),
    ):
        if key not in group or float(group[key]) != float(expected):
            raise ValueError(f"Dataset metadata {key} must equal {expected}")
    if seed is not None and int(group.get("seed", -1)) != seed:
        raise ValueError("Loaded group metadata does not match its requested seed")
    ids = np.asarray(group["sample_id"])
    if (ids.shape != (N,) or not np.issubdtype(ids.dtype, np.integer)
            or len(np.unique(ids)) != N):
        raise ValueError("Dataset source sample IDs must be unique integer identities")
    # Source IDs remain in the offline group; they are never copied to model_ids.


def load_trial_group(seed: int, dataset_root: str | Path = DATASET_ROOT) -> dict[str, Any]:
    """Load one fixed N/cr/d/H logical group, to reuse across all eleven q_C."""
    trial_seed = _seed_value(seed)
    group = load_group(
        dataset_root=Path(dataset_root), n=N,
        corruption_rate_percent=GLOBAL_CR_PERCENT, dim=D, H=H, seed=trial_seed,
    )
    _validate_group(group, trial_seed)
    return group


@dataclass(frozen=True)
class Scene:
    """Shared offline workload; ``algorithm_input`` is the sole method input.

    ``source_indices`` are row indices in the seed's original logical group,
    retained solely for auditing. The IDs visible to methods are 0..99 in the
    already shuffled candidate order. Evaluation fields are posthoc only.
    """

    seed: int
    q_C_requested: float
    q_C_realized: float
    algorithm_input: AlgorithmInput
    evaluation_labels: np.ndarray
    evaluation_subtypes: np.ndarray
    source_indices: np.ndarray
    failure_uniform: np.ndarray
    random_method_seed: int
    candidate_fingerprint: str
    random_streams: dict[str, Any]

    def __post_init__(self) -> None:
        size = MAX_CANDIDATE_POOL_SIZE
        truth = np.asarray(self.evaluation_labels)
        subtypes = np.asarray(self.evaluation_subtypes)
        source = np.asarray(self.source_indices)
        uniform = np.asarray(self.failure_uniform)
        if truth.shape != (size,) or not np.isin(truth, (0, 1)).all():
            raise ValueError("Scene evaluation labels must match the 100-member candidate pool")
        if subtypes.shape != (size,) or not np.isin(subtypes, (0, 1, 2, 3)).all():
            raise ValueError("Scene evaluation subtypes must match the candidate pool")
        if not np.array_equal(truth, (subtypes >= 2).astype(truth.dtype)):
            raise ValueError("Scene evaluation label and subtype values disagree")
        if (source.shape != (size,) or not np.issubdtype(source.dtype, np.integer)
                or len(np.unique(source)) != size or np.any((source < 0) | (source >= N))):
            raise ValueError("Scene source rows must be distinct valid logical-group indices")
        if (uniform.shape != (size,) or not np.all(np.isfinite(uniform))
                or np.any((uniform < 0.0) | (uniform >= 1.0))):
            raise ValueError("Failure uniforms must be finite values in [0, 1), one per candidate")
        expected = Q_VALUES[_grid_index(self.q_C_requested, Q_VALUES, "q_C")]
        if (np.count_nonzero(truth) != int(round(size * expected))
                or abs(float(self.q_C_realized) - expected) > 1e-12):
            raise ValueError("Scene composition must realize the requested full-pool q_C exactly")
        data = self.algorithm_input
        order = np.arange(size, dtype=np.int64)
        if (data.X.shape != (size, D) or data.X.dtype != np.float32
                or not np.array_equal(data.model_ids, order)
                or not np.array_equal(data.reference_indices, order)
                or len(data.evaluation_indices) != 0 or not np.all(data.history == H0)):
            raise ValueError("Scene observations must preserve the shuffled order, temporary IDs and common history")
        for name, value, dtype in (
            ("evaluation_labels", truth, np.int8), ("evaluation_subtypes", subtypes, np.int8),
            ("source_indices", source, np.int64), ("failure_uniform", uniform, np.float64),
        ):
            object.__setattr__(self, name, _readonly(value, dtype))

    @property
    def prefix_corruption_ratios(self) -> dict[int, float]:
        """Evaluation-only realized compositions of the nested 50/75/100 pools."""
        return {
            size: float(self.evaluation_labels[:size].mean())
            for size in range(INITIAL_K, MAX_CANDIDATE_POOL_SIZE + 1, DELTA_K)
        }

    def availability(self, p_fail: float) -> np.ndarray:
        """Apply one of five thresholds to the same uniforms; never redraw."""
        probability = FAILURE_VALUES[_grid_index(p_fail, FAILURE_VALUES, "p_fail")]
        return _readonly(self.failure_uniform >= probability, np.bool_)


def draw_scenario(group: Mapping[str, Any], seed: int, q_C: float) -> Scene:
    """Draw the exact 100-member controlled pool once without subtype filters.

    ``draw_candidate_set`` already shuffles its output. Its order is retained;
    expansion later reveals successive prefixes instead of resampling models.
    Only after observable input construction are labels copied for evaluation.
    """
    trial_seed = _seed_value(seed)
    q_index = _grid_index(q_C, Q_VALUES, "q_C")
    q_requested = float(Q_VALUES[q_index])
    if int(group.get("seed", -1)) != trial_seed:
        raise ValueError("Scenario seed must match the loaded logical group's seed")
    # load_trial_group performs full-group quality checks once per seed. The
    # scene checks below verify only the draw, avoiding eleven scans of full X.
    source, q_realized = draw_candidate_set(
        group=group, K=MAX_CANDIDATE_POOL_SIZE, q_C=q_requested, seed=trial_seed,
        use_severe_only=False, use_common_only=False,
    )
    source = np.asarray(source)
    size = MAX_CANDIDATE_POOL_SIZE
    if (source.shape != (size,) or not np.issubdtype(source.dtype, np.integer)
            or len(np.unique(source)) != size or np.any((source < 0) | (source >= N))):
        raise ValueError("Dataset sampler did not return 100 distinct logical-group rows")
    order = np.arange(size, dtype=np.int64)
    observations = AlgorithmInput(
        X=np.asarray(group["X"])[source], model_ids=order,
        history=np.full(size, H0, dtype=np.float64), reference_indices=order,
        evaluation_indices=np.empty(0, dtype=np.int64),
    )
    failure_entropy = [trial_seed, FAILURE_STREAM, q_index]
    random_entropy = [trial_seed, RANDOM_METHOD_STREAM, q_index]
    uniforms = np.random.default_rng(np.random.SeedSequence(failure_entropy)).random(size)
    random_seed = int(np.random.SeedSequence(random_entropy).generate_state(1, dtype=np.uint64)[0])
    identity = {
        "seed": trial_seed, "q_grid_index": q_index,
        "descriptor_sha256": sha256_array(observations.X),
        "temporary_id_sha256": sha256_array(observations.model_ids),
        "history_sha256": sha256_array(observations.history),
        "reference_order_sha256": sha256_array(observations.reference_indices),
        "source_row_sha256": sha256_array(source.astype(np.int64)),
    }
    fingerprint = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return Scene(
        seed=trial_seed, q_C_requested=q_requested, q_C_realized=float(q_realized),
        algorithm_input=observations,
        evaluation_labels=np.asarray(group["binary_label"])[source],
        evaluation_subtypes=np.asarray(group["subtype_label"])[source],
        source_indices=source, failure_uniform=uniforms,
        random_method_seed=random_seed, candidate_fingerprint=fingerprint,
        random_streams={
            "candidate_sampling": {"entropy": [trial_seed, DATASET_CANDIDATE_STREAM],
                                   "source": "dataset.load_dataset.draw_candidate_set"},
            "failure_uniform": {"entropy": failure_entropy, "bit_generator": "PCG64"},
            "random_method": {"entropy": random_entropy, "generated_seed": random_seed,
                              "bit_generator": "PCG64"},
        },
    )
