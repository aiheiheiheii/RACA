"""Predeclared Group 02 grid and protocol assumptions, independent of labels."""
from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path

from experiments.common.method_names import RACA_HIM

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_ROOT = PROJECT_ROOT / "dataset"
RESULT_ROOT = PROJECT_ROOT / "results" / "02_overhead"
OVERHEAD_IMPLEMENTATION_VERSION = "overhead-core-v4"
N_VALUES = tuple(range(1000, 5001, 500))
D_PANEL_VALUES = (10, 20, 30, 40, 50)
SEEDS = (2026,)
METHODS = (RACA_HIM, "DVA-P-Adapted", "I-EDI-Adapted", "CTCV-Adapted")
EVALUATION_MODES = ("fixed_workload",)
MAIN_EVALUATION_MODE = "fixed_workload"


@dataclass(frozen=True)
class OverheadConfig:
    """The 800 ms constraint applies to the verification stage.

    Message field lengths are disclosed simulation assumptions. SHA256 is
    executed locally; unexecuted cryptographic primitives require an explicit
    cost model in the corresponding adapter rather than a fabricated benchmark.
    """

    K: int = 50
    M: int = 25
    # Zero means the full N-instance candidate domain, not a fixed 100-row pool.
    reference_pool_size: int = 0
    expansion_step: int = 25
    dim: int = 30
    H: float = 0.5
    corruption_rate_percent: float = 3.0
    tau_lower: float = 0.997
    tau_upper: float = 1.0
    h0: float = 0.10
    alpha: float = 0.5
    lambda_: float = 0.5
    A_max: int = 20
    rho: float = 0.5
    eta: float = 0.5
    beta_B: float = 0.10
    sigma: float = 0.5
    p_fail: float = 0.0
    T_round_max_ms: float = 800.0
    round_id: int = 1
    resource_rounds: int = 1
    request_bytes: int = 64
    result_bytes: int = 32
    assignment_bytes: int = 64
    acknowledgment_bytes: int = 32
    digest_bytes: int = 32
    hash_throughput_mib_s: float = 200.0
    memory_throughput_mib_s: float = 1000.0
    scalar_compare_ms: float = 0.001
    iedi_budget_fraction: float = 0.10
    iedi_control_V: float = 1.0
    iedi_tag_modulus: int = 7919
    baseline_block_count: int = 5
    ctcv_collusion_bound: int = 1
    ctcv_time_limit: float = 0.8
    ctcv_block_count: int = 0
    ctcv_proof_vector_length: int = 0

    def __post_init__(self) -> None:
        for name in ("K", "M", "expansion_step", "dim",
                     "A_max", "round_id", "resource_rounds", "request_bytes",
                     "result_bytes", "assignment_bytes", "acknowledgment_bytes",
                     "digest_bytes", "iedi_tag_modulus", "baseline_block_count"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.M > self.K:
            raise ValueError("Require M <= K")
        if (isinstance(self.reference_pool_size, bool)
                or not isinstance(self.reference_pool_size, int)
                or self.reference_pool_size < 0
                or (self.reference_pool_size and self.reference_pool_size < self.K)):
            raise ValueError("reference_pool_size must be 0 (full domain) or >= K")
        if self.dim not in D_PANEL_VALUES or self.baseline_block_count > 4 * self.dim:
            raise ValueError("Unsupported dimension or more blocks than descriptor bytes")
        if self.round_id != 1 or self.resource_rounds < self.round_id:
            raise ValueError("This single-round experiment uses round_id=1")
        for name in ("H", "p_fail", "sigma"):
            value = getattr(self, name)
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be finite and in [0,1]")
        for name in ("h0", "alpha", "eta"):
            value = getattr(self, name)
            if not math.isfinite(value) or not 0.0 < value < 1.0:
                raise ValueError(f"{name} must be finite and in (0,1)")
        if not math.isfinite(self.rho) or not 0.0 < self.rho <= 1.0:
            raise ValueError("rho must be finite and in (0,1]")
        for name in ("beta_B", "iedi_budget_fraction"):
            value = getattr(self, name)
            if not math.isfinite(value) or not 0.0 < value <= 1.0:
                raise ValueError(f"{name} must be finite and in (0,1]")
        for name in ("lambda_", "T_round_max_ms", "iedi_control_V", "ctcv_time_limit",
                     "hash_throughput_mib_s", "memory_throughput_mib_s", "scalar_compare_ms"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be positive and finite")
        if not (math.isfinite(self.tau_lower) and math.isfinite(self.tau_upper)
                and -1.0 <= self.tau_lower <= self.tau_upper <= 1.0):
            raise ValueError("Require -1 <= tau_lower <= tau_upper <= 1")
        if self.p_fail != 0.0:
            raise ValueError("Group 02 requires p_fail=0")
        if not math.isfinite(self.corruption_rate_percent) or not 0.0 <= self.corruption_rate_percent <= 50.0:
            raise ValueError("Invalid corruption percentage")
        if self.iedi_tag_modulus < 2:
            raise ValueError("iedi_tag_modulus must be at least two")
        if self.digest_bytes != 32:
            raise ValueError("A full SHA256 digest is exactly 32 bytes")
        for name in ("ctcv_collusion_bound", "ctcv_block_count", "ctcv_proof_vector_length"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if self.ctcv_collusion_bound >= self.M:
            raise ValueError("CTCV collectors must fit in the verifier pool")

    def budget(self, n: int) -> int:
        if isinstance(n, bool) or not isinstance(n, int) or n <= self.M:
            raise ValueError("N must be an integer greater than M")
        return math.ceil(self.beta_B * (n - self.M))


def configurations() -> tuple[tuple[int, int, int], ...]:
    """Exactly 9 N values × 5 dimensions × one paired seed."""
    return tuple((n, d, seed) for n in N_VALUES for d in D_PANEL_VALUES for seed in SEEDS)
