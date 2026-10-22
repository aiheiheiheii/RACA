"""Predeclared Group 05 protocol. Importing this file performs no experiment."""
from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_ROOT = PROJECT_ROOT / "dataset"
RESULT_ROOT = PROJECT_ROOT / "results" / "05_joint_decision"
EXPERIMENT = "05_joint_decision"
DATASET_NAME = "TNSE-SimSimple-v1"
ALGORITHM_VERSION = "joint-decision-incumbent-refinement-v2"
RACA_ALGORITHM_VERSION = "raca-him-candidate-refinement-v2"
RACA_LOCAL_SEARCH_MAX_STEPS = 25
RACA_REPLACEMENT_POOL_LIMIT = 64
RACA_IMPROVEMENT_EPS = 1e-12
DVA_ADAPTATION_VERSION = "dva-p-single-round-pair-subset-v1"
N_VALUES = tuple(range(1000, 5001, 500))
BETA_PANEL_VALUES = (0.05, 0.10, 0.15, 0.20, 0.25)
SIGMA_PANEL_VALUES = (0.00, 0.25, 0.50, 0.75, 1.00)
J1_SIGMA = 0.50
J2_BETA_B = 0.10
METHODS = ("RACA-HIM", "DVA-P-Adapted", "RF-TS", "Cost-First", "Random")
REPETITIONS = 30
SEEDS = tuple(range(2026, 2056))
CORRUPTION_RATE_PERCENT = 3.0
D = 30
H = 0.5
K = 50
M = 25
DELTA_K = 25
TAU_LOWER = 0.997
TAU_UPPER = 1.0
ALPHA = 0.5
LAMBDA = 0.5
H0 = 0.10
A_MAX = 20
RHO = 0.5
ETA = 0.5
P_FAIL = 0.0
T_ROUND_MAX_MS = 800.0  # Explicit override of the dataset's legacy 400 ms default.
ROUND_ID = 1
HISTORY_ROUNDS = 20
HISTORY_VERIFICATION_PROB = 0.10
HISTORY_ANOMALY_PROB = 0.03
HISTORY_INVALID_PROB = 0.02
MIN_PAIRED_TRIALS = 2
EXPECTED_CONFIGURATIONS = 81
EXPECTED_TRIALS = 2430
EXPECTED_METHOD_RECORDS = 12150
SUMMARY_POLICY = "paired_all_five_methods"
REFERENCE_DOCUMENTS_SHA256 = {
    "Prompt - Codex.txt": "292d744df9f416060bd12cca39c4c1189bc8e1ea52b40d14e6e13fe3d4fdfca1",
    "Latest version of the paper.md": "61de825be020a3196d3f175fedfa7a82c6c87ddc6ad58caa3a5e0aaf4e9ac2c3",
    "TMC.pdf": "320cf6280cd643841e4ac1ace1f20ed7d51b8ffa95c26cd956d4bc34f134ddcb",
    "DVA-P.pdf": "8066c88e06d528f81da2058137f6ccd5ce870056bb29f9486d6dfbcc61ddb11a",
}


def parameter_pairs() -> tuple[tuple[float, float], ...]:
    """Nine unique pairs; the J1/J2 intersection is computed exactly once."""
    return tuple(sorted({(beta, 0.50) for beta in BETA_PANEL_VALUES}
                        | {(0.10, sigma) for sigma in SIGMA_PANEL_VALUES}))


def figures_for(beta: float, sigma: float) -> tuple[str, ...]:
    return tuple(name for name, included in (
        ("J1", sigma == 0.50 and beta in BETA_PANEL_VALUES),
        ("J2", beta == 0.10 and sigma in SIGMA_PANEL_VALUES),
    ) if included)


def validate_protocol() -> None:
    """Runtime guards for the fixed formal grid, without a reduced-run mode."""
    if N_VALUES != tuple(range(1000, 5001, 500)) or SEEDS != tuple(range(2026, 2056)):
        raise ValueError("Require all nine N values and thirty declared seeds")
    if METHODS != ("RACA-HIM", "DVA-P-Adapted", "RF-TS", "Cost-First", "Random") or REPETITIONS != 30:
        raise ValueError("Require the five declared methods and thirty repetitions")
    if BETA_PANEL_VALUES != (0.05, 0.10, 0.15, 0.20, 0.25):
        raise ValueError("Budget panel grid changed")
    if SIGMA_PANEL_VALUES != (0.00, 0.25, 0.50, 0.75, 1.00):
        raise ValueError("Resource panel grid changed")
    if (CORRUPTION_RATE_PERCENT, D, H, K, M, DELTA_K) != (3.0, 30, 0.5, 50, 25, 25):
        raise ValueError("Dataset and GT protocol changed")
    if (ALPHA, LAMBDA, H0, A_MAX, RHO, ETA, P_FAIL, ROUND_ID,
            T_ROUND_MAX_MS, TAU_LOWER, TAU_UPPER) != (
            0.5, 0.5, 0.10, 20, 0.5, 0.5, 0.0, 1, 800.0, 0.997, 1.0):
        raise ValueError("Require the manuscript-aligned single-round settings")
    if (HISTORY_ROUNDS, HISTORY_VERIFICATION_PROB, HISTORY_ANOMALY_PROB,
            HISTORY_INVALID_PROB) != (20, 0.10, 0.03, 0.02):
        raise ValueError("Synthetic history assumptions changed")
    if (RACA_LOCAL_SEARCH_MAX_STEPS, RACA_REPLACEMENT_POOL_LIMIT, RACA_IMPROVEMENT_EPS) != (25, 64, 1e-12):
        raise ValueError("Require the declared v2 local improvement bounds")
    if (len(parameter_pairs()) * len(N_VALUES),
            len(parameter_pairs()) * len(N_VALUES) * len(SEEDS),
            len(parameter_pairs()) * len(N_VALUES) * len(SEEDS) * len(METHODS)) != (
            EXPECTED_CONFIGURATIONS, EXPECTED_TRIALS, EXPECTED_METHOD_RECORDS):
        raise ValueError("Incomplete formal configuration union")


def protocol_metadata() -> dict:
    return {
        "N_values": N_VALUES, "parameter_pairs": parameter_pairs(), "seeds": SEEDS,
        "methods": METHODS, "cr": CORRUPTION_RATE_PERCENT, "d": D, "H": H,
        "K": K, "M": M, "delta_K": DELTA_K, "gt_quorum": [2, 3],
        "gt_selection": "history, descending peer support, descending mean support cosine, temporary ID",
        "gt_candidate_domain": "all N models; grow by 25 until success or exhaustion",
        "alpha": ALPHA, "lambda": LAMBDA, "deadline_ms": T_ROUND_MAX_MS,
        "round_id": ROUND_ID, "p_fail": P_FAIL, "h0": H0, "A_max": A_MAX,
        "rho": RHO, "eta": ETA,
        "history": {"rounds": HISTORY_ROUNDS, "verification_probability": HISTORY_VERIFICATION_PROB,
                    "anomaly_given_event": HISTORY_ANOMALY_PROB,
                    "invalid_given_event": HISTORY_INVALID_PROB,
                    "normal_given_event": 1 - HISTORY_ANOMALY_PROB - HISTORY_INVALID_PROB,
                    "independent_of_current_corruption": True},
        "resource": {"source": "dataset.load_dataset.resource_costs",
                     "fixture": "experiments.02_overhead.overhead_resources.shared_resource_profiles",
                     "compute_ms_at_d30": 4.0, "bandwidth_Mbps": 50.0,
                     "network_ms": 10.0, "descriptor_precision": "float32"},
        "objective": "sum(selected risk)/B - lambda*max(verifier loads)/deadline_ms",
        "nonempty_required": True,
        "raca_prefix_policy": "retain v1 greedy best prefix and RF full/best prefix as candidates; final answer may be locally refined and need not be a prefix",
        "raca_algorithm_version": RACA_ALGORITHM_VERSION,
        "raca_local_search": {
            "max_steps": RACA_LOCAL_SEARCH_MAX_STEPS,
            "replacement_enter_pool_limit": RACA_REPLACEMENT_POOL_LIMIT,
            "replacement_leave_pool_limit": RACA_REPLACEMENT_POOL_LIMIT,
            "strict_improvement_threshold": RACA_IMPROVEMENT_EPS,
            "acceptance": "same independent objective evaluator, original feasibility constraints",
            "universal_strict_superiority": False,
        },
        "manuscript_alignment": "v2 extends the supplied manuscript's joint-decision stage; manuscript revision required",
        "mode": "Native Joint Decision Mode", "summary_policy": SUMMARY_POLICY,
        "method_definitions": {
            "RACA-HIM": "v2: independently construct v1 marginal-J greedy and RF full/best-prefix incumbents; bounded strictly improving add/remove/reassign/replacement; retain all original candidates and return maximum independently evaluated J",
            "DVA-P-Adapted": "source Eq13/Eq21 and Algorithm1 priority/add/single swap with Q(0)=0; risk traversal; minimum intrinsic legal GT cost adapter",
            "RF-TS": "descending visible risk then minimum load+pair-cost assignment; temporary ID ties",
            "Cost-First": "minimum marginal makespan; ties by intrinsic cost then temporary target/verifier ID; risk ignored for ranking",
            "Random": "independent seeded uniform permutation of initially feasible pairs; accept unused targets within current GT capacity",
        },
        "min_paired_trials": MIN_PAIRED_TRIALS,
        "ci": "two-sided Student t 95%, ddof=1, paired-all-five trial population",
        "dva_adaptation_version": DVA_ADAPTATION_VERSION,
        "reference_documents_sha256": REFERENCE_DOCUMENTS_SHA256,
        "expected_trials": EXPECTED_TRIALS, "expected_method_records": EXPECTED_METHOD_RECORDS,
    }
