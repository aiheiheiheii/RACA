"""Predeclared formal protocol. No reduced-run or quick mode is provided."""
from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_ROOT = PROJECT_ROOT / "dataset"
RESULT_ROOT = PROJECT_ROOT / "results" / "06_long_term"
EXPERIMENT = "06_long_term"
DATASET_NAME = "TNSE-SimSimple-v1"
ALGORITHM_VERSION = "long-term-causal-manuscript-v1"
CODE_VERSION = ALGORITHM_VERSION
BASELINE_ADAPTATION_VERSION = "long-term-native-feedback-adapters-v1"
METHODS = ("RACA-HIM", "I-EDI-Adapted", "DVA-P-Adapted", "Age-Only", "Random")
DYNAMIC_CR_VALUES = (1, 2, 3, 4, 5)
SEEDS = tuple(range(2026, 2056))
REPETITIONS = 30
N = 1000
D = 30
H = 0.5
TOTAL_ROUNDS = 200
PLOT_ROUNDS = tuple(range(0, 201, 20))
TAU_LOWER = 0.997
TAU_UPPER = 1.0
K = 50
M = 25
DELTA_K = 25
GT_QUORUM = (2, 3)
ALPHA = 0.5
LAMBDA = 0.5
H0 = 0.10
A_MAX = 20
RHO = 0.5
ETA = 0.5
BETA_B = 0.10
SIGMA = 0.5
P_FAIL = 0.05
T_ROUND_MAX_MS = 800.0
IEDI_CONTROL_V = 1.0
IEDI_TAG_MODULUS = 7919
IEDI_QOS_WEIGHT = 0.5
IEDI_MINIMUM_FRACTION = 0.10
IEDI_QOS_DECAY_FLOOR = 0.5
IEDI_QOS_DECAY_SCALE = 20
DVAP_CONTROL_LAMBDA = 1.0
EXPECTED_SCENARIOS = 150
EXPECTED_TRAJECTORIES = 750
EXPECTED_METHOD_ROUNDS = 150000
REFERENCE_DOCUMENTS_SHA256 = {
    "Prompt - Codex.txt": "a987be2fcf4141fc74bc219cf691d4d76c2649c014430177cf84ebc36bcd108a",
    "Latest version of the paper.md": "61de825be020a3196d3f175fedfa7a82c6c87ddc6ad58caa3a5e0aaf4e9ac2c3",
    "TMC.pdf": "320cf6280cd643841e4ac1ace1f20ed7d51b8ffa95c26cd956d4bc34f134ddcb",
    "I-EDI.pdf": "c07a2116294b725384ea21575a68cfd81e534c06d9727a367dc155ddc35e3256",
    "DVA-P.pdf": "8066c88e06d528f81da2058137f6ccd5ce870056bb29f9486d6dfbcc61ddb11a",
}


def protocol_metadata() -> dict:
    values = {name: value for name, value in globals().items()
              if name.isupper() and isinstance(value, (str, int, float, tuple))}
    return {
        "parameters": values,
        "reference_documents_sha256": REFERENCE_DOCUMENTS_SHA256,
        "dynamic_ratio": "distinct ever-corrupted identities / N over all 200 rounds",
        "metric": "CDR(t)=first correctly detected active-window events through t / all events",
        "gt_refresh": "reconfirm each round from that method's observed candidates and history",
        "gt_source": "experiments.01_detection.proposed_detection.build_ground_truth",
        "gt_policy": "ceil(2K/3); history_consensus ranking; retained candidates, +25 expansion",
        "joint_source": "experiments.05_joint_decision.joint_methods.raca_him_greedy",
        "joint_policy": "manuscript marginal J, best nonempty feasible prefix; no v2 refinement",
        "resource_source": "dataset.load_dataset.resource_profiles/resource_costs",
        "trace_source": "dataset.load_dataset.load_dynamic_case/generate_dynamic_trace",
        "native_enrollment": "t=0 SHA256 of each model's own observed little-endian float32 bytes",
        "native_processors": "25 temporary-ID processors rotate each round; disjoint current targets",
        "failure_model": "either interacting endpoint fails if its shared fail_uniform[t-1,i] < .05",
        "timeout_model": "failed attempts consume their reserved full pair time; no free retries",
        "deadline_scope": "verification stage only; GT request and peer-comparison counts separately recorded",
        "ci": "Student t 95%; ddof=1; independent complete seed trajectories, not rounds/models",
        "failure_population": "exclude only failed method-seed trajectory from its summary; preserve raw",
        "figure": "Fig_L1_long_term_detection; 5 panels, 11 observed points, no error bands",
        "baseline_adaptation_version": BASELINE_ADAPTATION_VERSION,
    }


def validate_protocol() -> None:
    """Future runtime guard; never invoked during this code-writing request."""
    expected = (1000, 30, .5, 200, 50, 25, 25, .997, 1., .5, .5,
                .10, 20, .5, .5, .10, .5, .05, 800.)
    actual = (N, D, H, TOTAL_ROUNDS, K, M, DELTA_K, TAU_LOWER, TAU_UPPER,
              ALPHA, LAMBDA, H0, A_MAX, RHO, ETA, BETA_B, SIGMA, P_FAIL,
              T_ROUND_MAX_MS)
    if actual != expected or GT_QUORUM != (2, 3):
        raise ValueError("Formal Group 06 protocol was changed")
    if SEEDS != tuple(range(2026, 2056)) or REPETITIONS != 30:
        raise ValueError("Require all thirty seeds, 2026..2055")
    if DYNAMIC_CR_VALUES != (1, 2, 3, 4, 5) or PLOT_ROUNDS != tuple(range(0, 201, 20)):
        raise ValueError("Require all five ratios and eleven displayed rounds")
    if METHODS != ("RACA-HIM", "I-EDI-Adapted", "DVA-P-Adapted", "Age-Only", "Random"):
        raise ValueError("Require exactly the five declared methods")
    if (IEDI_CONTROL_V, IEDI_TAG_MODULUS, IEDI_QOS_WEIGHT, IEDI_MINIMUM_FRACTION,
            IEDI_QOS_DECAY_FLOOR, IEDI_QOS_DECAY_SCALE, DVAP_CONTROL_LAMBDA) != (
            1.0, 7919, .5, .10, .5, 20, 1.0):
        raise ValueError("Declared baseline adaptation parameters were changed")
    if (len(DYNAMIC_CR_VALUES) * len(SEEDS) != EXPECTED_SCENARIOS
            or EXPECTED_SCENARIOS * len(METHODS) != EXPECTED_TRAJECTORIES
            or EXPECTED_TRAJECTORIES * TOTAL_ROUNDS != EXPECTED_METHOD_ROUNDS):
        raise ValueError("Incomplete formal grid")
