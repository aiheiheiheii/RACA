"""Predeclared settings and measured configuration grids for group 01."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_ROOT = PROJECT_ROOT / "dataset"
RESULTS_ROOT = PROJECT_ROOT / "results" / "01_detection"
DETECTION_IMPLEMENTATION_VERSION = "hash-baselines-v4"

N_VALUES = tuple(range(1000, 5001, 500))
CR_VALUES = tuple(i / 2.0 for i in range(2, 11))
D1_CR_VALUES = tuple(float(i) for i in range(1, 6))
D2_N_VALUES = tuple(range(1000, 5001, 1000))
SEEDS = tuple(range(2026, 2056))
SUBTYPE_NAMES = (
    "common_normal",
    "high_personalized_normal",
    "severe_corrupt",
    "mild_corrupt",
)


@dataclass(frozen=True)
class DetectionConfig:
    """Fixed algorithm parameters, independent of evaluation labels."""

    K: int = 50
    M: int = 25
    reference_pool_size: int = 100
    expansion_step: int = 25
    h0: float = 0.10
    tau_lower: float = 0.997
    tau_upper: float = 1.0
    dim: int = 30
    H: float = 0.5
    p_fail: float = 0.0
    # Fixed descriptor adaptations; these are not parameters fitted to labels.
    iedi_budget_fraction: float = 0.10  # gamma minimum-audit fraction, not a maximum budget
    iedi_control_V: float = 1.0
    iedi_tag_modulus: int = 7919  # I-EDI Section V-A experimental setting
    baseline_block_count: int = 5
    ctcv_collusion_bound: int = 1
    ctcv_time_limit: float = 1.0
    ctcv_block_count: int = 0  # 0 => min(replica count, available raw bytes)
    ctcv_proof_vector_length: int = 0  # 0 => source experiment's ell = 1000 * replica count
    dyna_quorum_fraction: float = 2.0 / 3.0

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        """Reject incompatible settings before a run starts."""
        for name in ("K", "M", "reference_pool_size", "expansion_step", "dim", "baseline_block_count", "iedi_tag_modulus"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.M > self.K or self.K > self.reference_pool_size:
            raise ValueError("Require M <= K <= reference_pool_size")
        if self.baseline_block_count > self.dim:
            raise ValueError("baseline_block_count must not exceed descriptor dimension")
        if self.iedi_tag_modulus < 2:
            raise ValueError("iedi_tag_modulus must be at least 2")
        if (isinstance(self.ctcv_collusion_bound, bool)
                or not isinstance(self.ctcv_collusion_bound, int)
                or not 0 <= self.ctcv_collusion_bound < self.reference_pool_size):
            raise ValueError("Require 0 <= ctcv_collusion_bound < reference_pool_size")
        if (isinstance(self.ctcv_proof_vector_length, bool)
                or not isinstance(self.ctcv_proof_vector_length, int)
                or self.ctcv_proof_vector_length < 0):
            raise ValueError("ctcv_proof_vector_length must be a nonnegative integer")
        if (isinstance(self.ctcv_block_count, bool) or not isinstance(self.ctcv_block_count, int)
                or self.ctcv_block_count < 0):
            raise ValueError("ctcv_block_count must be a nonnegative integer")
        for name in ("iedi_budget_fraction", "dyna_quorum_fraction"):
            value = getattr(self, name)
            if not math.isfinite(value) or not 0.0 < value <= 1.0:
                raise ValueError(f"{name} must be finite and in (0, 1]")
        if math.ceil(self.dyna_quorum_fraction * self.reference_pool_size) >= self.reference_pool_size:
            raise ValueError("Strict DynaEDI+ inconsistency quorum must be reachable")
        for name in ("iedi_control_V", "ctcv_time_limit"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be positive and finite")
        for name in ("h0", "H", "p_fail"):
            value = getattr(self, name)
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be finite and in [0, 1]")
        if not (
            math.isfinite(self.tau_lower)
            and math.isfinite(self.tau_upper)
            and -1.0 <= self.tau_lower <= self.tau_upper <= 1.0
        ):
            raise ValueError("Require -1 <= tau_lower <= tau_upper <= 1")
        if self.p_fail != 0.0:
            raise ValueError("Group 01 requires p_fail=0; failures belong to later experiments")


def d1_configurations() -> tuple[tuple[int, float], ...]:
    """D1's complete 45 real (N, cr) points."""
    return tuple((n, cr) for n in N_VALUES for cr in D1_CR_VALUES)


def d2_configurations() -> tuple[tuple[int, float], ...]:
    """D2's complete 45 real (N, cr) points."""
    return tuple((n, cr) for n in D2_N_VALUES for cr in CR_VALUES)


def configurations() -> tuple[tuple[int, float], ...]:
    """Sorted union: 65 full points; overlapping D1/D2 points run once."""
    return tuple(sorted(set(d1_configurations()) | set(d2_configurations())))

