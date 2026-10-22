"""User-selected single-seed native experiment, without repetition averaging."""
from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_ROOT = PROJECT_ROOT / "dataset"
RESULT_ROOT = PROJECT_ROOT / "results" / "06_long_term"
EXPERIMENT = "06_long_term_native_single"
DATASET_NAME = "TNSE-SimSimple-v1"
ALGORITHM_VERSION = "long-term-native-iedi-dvap-ediv-v1"
CODE_VERSION = "long-term-native-single-run-v1"
PROTOCOL_FAMILY = ALGORITHM_VERSION
BASELINE_ADAPTATION_VERSION = "native-iedi-alg2-dvap-ediv-v1"
METHODS = ("RACA-HIM", "I-EDI", "DVA-P+EDI-V")
DYNAMIC_CR_VALUES = (1, 2, 3, 4, 5)
SEEDS = (2026,)
REPETITIONS = 1
AGGREGATION = "single_run"
N, D, H = 1000, 30, 0.5
TOTAL_ROUNDS = 200
PLOT_ROUNDS = tuple(range(0, 201, 20))
TAU_LOWER, TAU_UPPER = 0.997, 1.0
K, M, DELTA_K = 50, 25, 25
GT_QUORUM = (2, 3)
ALPHA, LAMBDA = 0.5, 0.5
H0, A_MAX = 0.10, 20
RHO, ETA = 0.5, 0.5
BETA_B, SIGMA, P_FAIL = 0.10, 0.5, 0.05
T_ROUND_MAX_MS = 800.0
IEDI_CONTROL_V = 1.0
IEDI_QOS_WEIGHT = 0.5
IEDI_MINIMUM_FRACTION = 0.10
IEDI_QOS_DECAY_FLOOR = 0.5
IEDI_QOS_DECAY_SCALE = 1.0
IEDI_NO_HISTORY_RATE_PRIOR = 0.5
IEDI_NO_HISTORY_QOS_PRIOR = 0.5
IEDI_CHALLENGE_MODE = "algorithm2-random-modulus"
IEDI_ETA_MIN = 4096
IEDI_ETA_MAX = 8191
IEDI_FIXED_ETA = 7919
DVAP_CONTROL_LAMBDA = 1.0
DVAP_EDIV_BLOCK_BYTES = 4
DVAP_EDIV_DETECTION_PROBABILITY = 0.999
DVAP_EDIV_MIN_CORRUPTED_BLOCKS = 1
EXPECTED_SCENARIOS = 5
EXPECTED_TRAJECTORIES = 15
EXPECTED_METHOD_ROUNDS = 3000
PROTOCOL_SETTINGS = {
    "family": PROTOCOL_FAMILY,
    "iedi_challenge_mode": IEDI_CHALLENGE_MODE,
    "iedi_eta_min": IEDI_ETA_MIN,
    "iedi_eta_max": IEDI_ETA_MAX,
    "iedi_fixed_eta": IEDI_FIXED_ETA,
    "dvap_ediv_block_bytes": DVAP_EDIV_BLOCK_BYTES,
    "dvap_ediv_detection_probability": DVAP_EDIV_DETECTION_PROBABILITY,
    "dvap_ediv_min_corrupted_blocks": DVAP_EDIV_MIN_CORRUPTED_BLOCKS,
}
REFERENCE_DOCUMENTS_SHA256 = {
    "Prompt - Codex.txt": "a987be2fcf4141fc74bc219cf691d4d76c2649c014430177cf84ebc36bcd108a",
    "Latest version of the paper.md": "61de825be020a3196d3f175fedfa7a82c6c87ddc6ad58caa3a5e0aaf4e9ac2c3",
    "TMC.pdf": "320cf6280cd643841e4ac1ace1f20ed7d51b8ffa95c26cd956d4bc34f134ddcb",
    "I-EDI.pdf": "c07a2116294b725384ea21575a68cfd81e534c06d9727a367dc155ddc35e3256",
    "DVA-P.pdf": "8066c88e06d528f81da2058137f6ccd5ce870056bb29f9486d6dfbcc61ddb11a",
}


def protocol_metadata() -> dict:
    return {
        "protocol_family": PROTOCOL_FAMILY,
        "parameters": {name: value for name, value in globals().items()
                       if name.isupper() and isinstance(value, (str, int, float, tuple))},
        "protocol_settings": dict(PROTOCOL_SETTINGS),
        "aggregation": AGGREGATION,
        "repetition_policy": "User requested one run only: fixed seed 2026, no across-seed means, standard deviations or confidence intervals",
        "reference_documents_sha256": dict(REFERENCE_DOCUMENTS_SHA256),
        "method_protocols": {
            "RACA-HIM": "current-round two-thirds collaborative GT and descriptor cosine",
            "I-EDI": "source Algorithm2/Eq39 per-target per-round random-modulus tag challenge/response",
            "DVA-P+EDI-V": "DVA-P subset scheduling plus fresh-nonce EDI-V VMHT challenge/proof verification",
        },
        "gt_source": "experiments.01_detection.proposed_detection.build_ground_truth",
        "joint_source": "experiments.05_joint_decision.joint_methods.raca_him_greedy",
        "joint_policy": "manuscript best nonempty marginal-J prefix; no v2 refinement",
        "trace_source": "dataset.load_dataset.load_dynamic_case/generate_dynamic_trace",
        "resource_source": "dataset.load_dataset.resource_profiles/resource_costs",
        "dynamic_ratio": "distinct ever-corrupted identities / N over the full 200-round horizon",
        "metric": "first correct active-window event detections through t / all horizon events",
        "native_registration": "method-private trusted t=0 payload registration before any event; no future descriptor access",
        "native_processors": "M temporary-ID processors rotate each round, disjoint current target set",
        "protocol_random_streams": "SeedSequence([seed,6107]) for I-EDI challenges; SeedSequence([seed,6108]) for EDI-V nonce/inspection; independent of selection RNGs",
        "iedi_eta_domain_assumption": "uniform independent integers on inclusive [4096,8191]; source does not specify a random domain",
        "iedi_qos_observations": "own past response finish_ms and descriptor bytes/total_ms; historical population extrema; failed response uses deadline and zero throughput; no external QWS logs",
        "iedi_qos_time_adapter": "source alpha=.5,C=1; this workload uses round as time unit",
        "iedi_rate_observations": "SR=received responses/attempts including mismatched tags; ER=failed interactions/attempts; no-history rates and QoS=.5",
        "iedi_selection_scope": "native reliability/temporary-ID order; no Age ranking and no capacity-shortfall queue correction",
        "ediv_workload_assumptions": "4-byte blocks; requested detection probability .999; >=1 changed block; descriptor workload can require full-root coverage",
        "native_workload_boundary": "each heterogeneous model has its own trusted t0 original; the source papers' identical replicas of one original are not claimed as the same workload",
        "resource_timing_boundary": "all protocols use shared dataset pair costs for scheduling/deadline; actual native proof/hash operations are recorded, but these costs are not measured cryptographic wallclock times",
        "failure_model": "shared per-node exogenous failures exposed only by actual interactions",
        "deadline_scope": "verification stage max serialized verifier load <=800ms; GT preparation diagnosed separately",
        "ci": "not computed for a single run",
        "legacy_results_policy": "never load adapted five-method checkpoints or relabel their results as native protocols",
        "figure": "one Figure L1, five ratios, three methods, eleven actual single-run points, no averaging or CI",
        "baseline_adaptation_version": BASELINE_ADAPTATION_VERSION,
    }


def validate_protocol() -> None:
    if METHODS != ("RACA-HIM", "I-EDI", "DVA-P+EDI-V"):
        raise ValueError("Native formal run requires exactly RACA-HIM, I-EDI and DVA-P+EDI-V")
    if (N, D, H, TOTAL_ROUNDS, K, M, DELTA_K, TAU_LOWER, TAU_UPPER,
        ALPHA, LAMBDA, H0, A_MAX, RHO, ETA, BETA_B, SIGMA, P_FAIL, T_ROUND_MAX_MS) != (
        1000, 30, .5, 200, 50, 25, 25, .997, 1., .5, .5, .10, 20, .5, .5, .10, .5, .05, 800.):
        raise ValueError("Native formal common workload parameters changed")
    if (GT_QUORUM != (2, 3) or SEEDS != (2026,) or REPETITIONS != 1
            or AGGREGATION != "single_run"
            or DYNAMIC_CR_VALUES != (1, 2, 3, 4, 5)
            or PLOT_ROUNDS != tuple(range(0, 201, 20))):
        raise ValueError("Native formal grid changed")
    if (IEDI_QOS_DECAY_SCALE != 1.0 or IEDI_QOS_DECAY_FLOOR != .5
            or IEDI_CHALLENGE_MODE != "algorithm2-random-modulus"
            or (IEDI_ETA_MIN, IEDI_ETA_MAX, IEDI_FIXED_ETA) != (4096, 8191, 7919)
            or (DVAP_EDIV_BLOCK_BYTES, DVAP_EDIV_DETECTION_PROBABILITY,
                DVAP_EDIV_MIN_CORRUPTED_BLOCKS) != (4, .999, 1)):
        raise ValueError("Native baseline protocol settings changed")
    if (EXPECTED_SCENARIOS, EXPECTED_TRAJECTORIES, EXPECTED_METHOD_ROUNDS) != (5, 15, 3000):
        raise ValueError("Native formal output cardinality changed")
    if RESULT_ROOT.resolve() != (PROJECT_ROOT / "results" / "06_long_term").resolve():
        raise ValueError("All Group 06 outputs must use results/06_long_term")
