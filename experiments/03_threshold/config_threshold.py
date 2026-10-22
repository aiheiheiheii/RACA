"""Predeclared full Group 03 protocol; no reduced or quick execution mode."""
from __future__ import annotations

from pathlib import Path

from experiments.common.config import DetectionConfig

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_ROOT = PROJECT_ROOT / "dataset"
RESULT_ROOT = PROJECT_ROOT / "results" / "03_threshold"
EXPERIMENT = "03_threshold"
METHOD = "RACA-HIM"
IMPLEMENTATION_VERSION = "threshold-sensitivity-v1"
N = 3000
D = 30
H = 0.5
CR_VALUES = (1.0, 2.0, 3.0, 4.0, 5.0)
CR_PANEL_VALUES = CR_VALUES
TAU_VALUES = (0.990, 0.991, 0.992, 0.993, 0.994, 0.995,
              0.996, 0.997, 0.998, 0.999, 0.9995)
DEFAULT_TAU = 0.997
TAU_UPPER = 1.0
K = 50
M = 25
DELTA_K = 25
H0 = 0.10
P_FAIL = 0.0
REFERENCE_POOL_SIZE = 100
REPETITIONS = 30
SEEDS = tuple(range(2026, 2056))
METRICS = ("accuracy", "precision", "recall", "f1", "fpr", "fnr")
PLOT_METRICS = ("accuracy", "f1", "fpr", "fnr")
COUNTS = ("TP", "TN", "FP", "FN")
EXPECTED_TRIALS = len(CR_VALUES) * REPETITIONS
EXPECTED_RUNS = EXPECTED_TRIALS * len(TAU_VALUES)
BASE_CONFIG = DetectionConfig(K=K, M=M, reference_pool_size=REFERENCE_POOL_SIZE,
                              expansion_step=DELTA_K, h0=H0, tau_lower=DEFAULT_TAU,
                              tau_upper=TAU_UPPER, dim=D, H=H, p_fail=P_FAIL)


def validate_protocol() -> None:
    """Check the declared grid without loading data or invoking a detector."""
    if len(TAU_VALUES) != 11 or tuple(sorted(set(TAU_VALUES))) != TAU_VALUES:
        raise ValueError("Require eleven distinct ascending threshold values")
    if not all(-1.0 <= tau <= TAU_UPPER == 1.0 for tau in TAU_VALUES):
        raise ValueError("Thresholds must be valid inclusive cosine intervals")
    if DEFAULT_TAU not in TAU_VALUES or len(SEEDS) != REPETITIONS:
        raise ValueError("Default threshold or repetition protocol is inconsistent")
    if EXPECTED_TRIALS != 150 or EXPECTED_RUNS != 1650:
        raise ValueError("Formal protocol must contain 150 trials and 1650 runs")
    BASE_CONFIG.validate()


def protocol_metadata() -> dict:
    """Only the parameters used by this static sensitivity experiment."""
    return {
        "N": N, "d": D, "H": H, "cr_values_percent": list(CR_VALUES),
        "tau_values": list(TAU_VALUES), "default_tau": DEFAULT_TAU,
        "tau_upper": TAU_UPPER, "K": K, "M": M, "delta_K": DELTA_K,
        "h0": H0, "p_fail": P_FAIL, "reference_pool_size": REFERENCE_POOL_SIZE,
        "evaluation_count": N - REFERENCE_POOL_SIZE,
        "repetitions": REPETITIONS, "seeds": list(SEEDS),
        "expected_trials": EXPECTED_TRIALS, "expected_runs": EXPECTED_RUNS,
        "split_rng_stream": 1101, "temporary_id_rng_stream": 1102,
        "candidate_order": "ascending (history, seeded temporary model ID)",
        "assignment": "one GT verifier per target; balanced cyclic Group 01 rule",
        "ci_method": "two-sided Student-t interval on finite seed-level metrics",
        "undefined_metrics": "zero denominator => NaN with explicit metric name",
        "failure_policy": "no resampling; absent predictions/metrics retained",
    }
