"""The complete predeclared Group 04 protocol; no quick or reduced mode."""
from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_ROOT = PROJECT_ROOT / "dataset"
RESULT_ROOT = PROJECT_ROOT / "results" / "04_ground_truth"
FIGURE_ROOT = RESULT_ROOT / "figures"
EXPERIMENT = "04_ground_truth"
DATASET_NAME = "TNSE-SimSimple-v1"
LEGACY_IMPLEMENTATION_VERSION = "ground-truth-reliability-v1"
IMPLEMENTATION_VERSION = "ground-truth-supermajority-v2"
READABLE_IMPLEMENTATION_VERSIONS = (LEGACY_IMPLEMENTATION_VERSION, IMPLEMENTATION_VERSION)
DIAGNOSTICS_VERSION = "admission-and-final-k-v2"
# Fixed conservative quorum, declared before the revised experiment is run.
# This changes the assurance/availability tradeoff, not the saved v1 results.
GT_QUORUM_NUMERATOR = 2
GT_QUORUM_DENOMINATOR = 3
GT_SELECTION_POLICY = "history_consensus"
REVISED_MANUSCRIPT_SHA256 = "61de825be020a3196d3f175fedfa7a82c6c87ddc6ad58caa3a5e0aaf4e9ac2c3"
N = 3000
GLOBAL_CR_PERCENT = 3.0
D = 30
H = 0.5
TAU_LOWER = 0.997
TAU_UPPER = 1.0
INITIAL_K = 50
GT_SIZE_M = 25
DELTA_K = 25
MAX_CANDIDATE_POOL_SIZE = 100
H0 = 0.10
Q_VALUES = tuple(i / 100 for i in range(0, 51, 5))
FAILURE_VALUES = tuple(i / 100 for i in range(0, 21, 5))
REPETITIONS = 1000
SEEDS = tuple(range(2026, 3026))
METHODS = ("RACA-HIM", "Fixed-K Majority", "Top-M", "Random-M")
K_VALUES = (50, 75, 100)
EXPECTED_POOLS = len(SEEDS) * len(Q_VALUES)
EXPECTED_SCENARIOS = EXPECTED_POOLS * len(FAILURE_VALUES)
EXPECTED_RUNS = EXPECTED_SCENARIOS * len(METHODS)
GT_KERNEL_SOURCE = "experiments.01_detection.proposed_detection.build_ground_truth"
# Content identities of the supplied implementation references; runtime does
# not need the user's Desktop paths or access to those original attachments.
REFERENCE_DOCUMENTS_SHA256 = {
    "Prompt - Codex.txt": "5adc4ff89618037240329416659c2278e16505fb1244a99020e06d52b077f1fa",
    "Latest version of the paper.md": "9f900d7b254a8fce89885a9bb50e7c06bcb28f866ec9fe0be019dc1151949d5f",
    "TMC.pdf": "320cf6280cd643841e4ac1ace1f20ed7d51b8ffa95c26cd956d4bc34f134ddcb",
}


def validate_protocol() -> None:
    if (N, GLOBAL_CR_PERCENT, D, H) != (3000, 3.0, 30, 0.5):
        raise ValueError("Group 04 requires the predeclared N/cr/d/H values")
    if Q_VALUES != tuple(i / 100 for i in range(0, 51, 5)):
        raise ValueError("Require eleven q_C points from 0 to 0.50")
    if FAILURE_VALUES != (0.0, 0.05, 0.10, 0.15, 0.20):
        raise ValueError("Require five responder-failure probabilities")
    if (INITIAL_K, GT_SIZE_M, DELTA_K, MAX_CANDIDATE_POOL_SIZE) != (50, 25, 25, 100):
        raise ValueError("Require M=25 and the nested K=50/75/100 protocol")
    if (TAU_LOWER, TAU_UPPER, H0) != (0.997, 1.0, 0.10):
        raise ValueError("Require the declared similarity interval and history")
    if SEEDS != tuple(range(2026, 3026)) or len(SEEDS) != REPETITIONS:
        raise ValueError("Require all 1000 independent seed-level trials")
    if (EXPECTED_POOLS, EXPECTED_SCENARIOS, EXPECTED_RUNS) != (11000, 55000, 220000):
        raise ValueError("Formal experimental grid is incomplete")
    if (GT_QUORUM_NUMERATOR, GT_QUORUM_DENOMINATOR, GT_SELECTION_POLICY) != (2, 3, "history_consensus"):
        raise ValueError("Revised GT requires a fixed two-thirds quorum and consensus tie breaking")


def protocol_metadata(version=IMPLEMENTATION_VERSION) -> dict:
    """Keep v1 evidence readable under its original exact protocol identity."""
    if version not in READABLE_IMPLEMENTATION_VERSIONS:
        raise ValueError(f"Unsupported Group 04 algorithm version: {version}")
    protocol = {
        "N": N, "global_cr_percent": GLOBAL_CR_PERCENT, "d": D, "H": H,
        "q_values": list(Q_VALUES), "failure_values": list(FAILURE_VALUES),
        "tau_lower": TAU_LOWER, "tau_upper": TAU_UPPER,
        "initial_K": INITIAL_K, "M": GT_SIZE_M, "delta_K": DELTA_K,
        "max_candidate_pool": MAX_CANDIDATE_POOL_SIZE, "K_values": list(K_VALUES),
        "h0": H0, "repetitions": REPETITIONS, "seeds": list(SEEDS),
        "methods": list(METHODS), "expected_pools": EXPECTED_POOLS,
        "expected_scenarios": EXPECTED_SCENARIOS, "expected_runs": EXPECTED_RUNS,
        "method_configuration": {
            "RACA-HIM": {"majority": True, "expansion": True, "history_rank": True},
            "Fixed-K Majority": {"majority": True, "expansion": False, "history_rank": True},
            "Top-M": {"majority": False, "expansion": True, "history_rank": True},
            "Random-M": {"majority": False, "expansion": True, "history_rank": False},
        },
        "candidate_sampler": "dataset.load_dataset.draw_candidate_set(K=100, both restrictions=False)",
        "candidate_sampler_rng": "SeedSequence([seed, 1147]); loader shuffles the complete pool",
        "candidate_order": "loader's shuffled order; history ties use ascending temporary slot ID",
        "failure_model": "one U(0,1) per candidate; available iff uniform >= p_fail; fixed within trial",
        "majority": "theta_K=ceil((K+1)/2); failed slots remain in K; self only if available",
        "ci_method": "two-sided Wilson 95% interval; seed-level binary outcomes; std uses ddof=1",
        "failure_denominator": "all 1000 seeds, including unformed and contaminated GTs",
        "cache_accounting": "one exact 100x100 cosine cache per seed/q; separate physical dot-product and logical peer counts",
        "early_stopping": False,
        "use_severe_only": False, "use_common_only": False,
        "data_provenance": "synthetic descriptors from stored banks; no real model/network measurements",
        "reference_documents_sha256": dict(REFERENCE_DOCUMENTS_SHA256),
    }
    if version == IMPLEMENTATION_VERSION:
        protocol.update(
            majority="theta_K=ceil(2*K/3); failed slots remain in K; self only if available",
            gt_quorum_numerator=GT_QUORUM_NUMERATOR,
            gt_quorum_denominator=GT_QUORUM_DENOMINATOR,
            quorum_fraction={"numerator": GT_QUORUM_NUMERATOR, "denominator": GT_QUORUM_DENOMINATOR},
            gt_selection_policy=GT_SELECTION_POLICY,
            gt_consensus_tie_break="ascending history; descending support count; descending mean supporting peer cosine; ascending temporary ID",
            revised_theorem_scope="responsive legal normal clique at least max(theta_K,M); detectable corrupted subset excluded; full purity requires all corruptions detectable",
            revised_manuscript_sha256=REVISED_MANUSCRIPT_SHA256,
        )
        for method in ("RACA-HIM", "Fixed-K Majority"):
            protocol["method_configuration"][method].update(
                quorum_numerator=GT_QUORUM_NUMERATOR, quorum_denominator=GT_QUORUM_DENOMINATOR,
                selection_policy=GT_SELECTION_POLICY,
            )
    return protocol
