"""Synthetic, label-independent observations before the single decision round.

This module deliberately accepts only a population size, an independent seed and
public configuration. Current descriptors and ground-truth labels cannot enter
the historical observation generator through its API.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class HistoryConfig:
    rounds: int = 20
    verification_probability: float = 0.10
    anomalous_probability: float = 0.03
    invalid_probability: float = 0.02
    initial_anomaly: float = 0.10
    max_age: float = 20.0
    rho: float = 0.5
    eta: float = 0.5
    alpha: float = 0.5
    decision_round: int = 1

    def validate(self) -> None:
        if not isinstance(self.rounds, int) or self.rounds < 1:
            raise ValueError("History rounds must be a positive integer")
        probabilities = (
            self.verification_probability, self.anomalous_probability,
            self.invalid_probability, self.initial_anomaly, self.rho,
            self.eta, self.alpha,
        )
        if not all(np.isfinite(p) and 0.0 <= p <= 1.0 for p in probabilities):
            raise ValueError("History probabilities and weights must lie in [0,1]")
        if self.anomalous_probability + self.invalid_probability > 1.0:
            raise ValueError("Conditional history event probabilities exceed one")
        if not np.isfinite(self.max_age) or self.max_age <= 0.0:
            raise ValueError("max_age must be positive and finite")


@dataclass(frozen=True)
class HistoricalState:
    anomaly_state: np.ndarray
    age: np.ndarray
    risk_scores: np.ndarray
    last_verified: np.ndarray
    event_counts: dict[str, int]
    decision_round: int
    seed: int
    config: HistoryConfig

    def to_manifest(self) -> dict[str, Any]:
        return {
            "kind": "synthetic_label_independent_prehistory",
            "seed": self.seed,
            "config": asdict(self.config),
            "prehistory_rounds": [
                self.decision_round - self.config.rounds,
                self.decision_round - 1,
            ],
            "event_counts": dict(self.event_counts),
            "risk_min": float(np.min(self.risk_scores)) if self.risk_scores.size else None,
            "risk_max": float(np.max(self.risk_scores)) if self.risk_scores.size else None,
            "risk_std": float(np.std(self.risk_scores)) if self.risk_scores.size else None,
            "valid_last_verified_count": int(np.isfinite(self.last_verified).sum()),
            "interpretation": "Coverage of observable synthetic risk, not prediction of current corruption labels",
        }


def build_historical_state(
    n: int, seed: int, config: HistoryConfig | None = None,
) -> HistoricalState:
    """Generate 20 independent historical observation rounds by default.

    With decision round t=1, historical rounds are -19,...,0. The negative
    numbering keeps the formal decision round at one without simulating a long
    experiment. Only normal/anomalous events refresh last_verified; invalid
    events increase h but leave verification age unchanged.
    """
    settings = config if config is not None else HistoryConfig()
    settings.validate()
    if not isinstance(n, (int, np.integer)) or n < 0:
        raise ValueError("n must be a nonnegative integer")
    rng = np.random.default_rng(int(seed))
    h = np.full(int(n), settings.initial_anomaly, dtype=np.float64)
    last = np.full(int(n), np.nan, dtype=np.float64)
    counts = {"normal": 0, "anomalous": 0, "timeout_or_invalid": 0, "not_observed": 0}
    start = settings.decision_round - settings.rounds
    for historical_round in range(start, settings.decision_round):
        observed = rng.random(int(n)) < settings.verification_probability
        event = rng.random(int(n))
        anomalous = observed & (event < settings.anomalous_probability)
        invalid = observed & (event >= settings.anomalous_probability) & (
            event < settings.anomalous_probability + settings.invalid_probability
        )
        normal = observed & ~(anomalous | invalid)
        h[normal] *= 1.0 - settings.rho
        h[anomalous] += settings.rho * (1.0 - h[anomalous])
        h[invalid] += settings.rho * settings.eta * (1.0 - h[invalid])
        last[normal | anomalous] = historical_round
        counts["normal"] += int(normal.sum())
        counts["anomalous"] += int(anomalous.sum())
        counts["timeout_or_invalid"] += int(invalid.sum())
        counts["not_observed"] += int((~observed).sum())
    age = np.ones(int(n), dtype=np.float64)
    valid = np.isfinite(last)
    age[valid] = np.minimum(
        (settings.decision_round - last[valid]) / settings.max_age, 1.0,
    )
    risk = settings.alpha * age + (1.0 - settings.alpha) * h
    h, last, age, risk = (
        np.frombuffer(array.tobytes(), dtype=np.float64).reshape(array.shape)
        for array in (h, last, age, risk)
    )
    state = HistoricalState(h, age, risk, last, counts, settings.decision_round, int(seed), settings)
    validate_historical_state(state)
    return state


generate_historical_state = build_historical_state


def validate_historical_state(state: HistoricalState) -> None:
    """Runtime invariants; called only when the user executes the experiment."""
    shape = state.risk_scores.shape
    if len(shape) != 1 or any(
        array.shape != shape for array in (state.anomaly_state, state.age, state.last_verified)
    ):
        raise ValueError("Historical state vectors must have a shared one-dimensional shape")
    for array in (state.anomaly_state, state.age, state.risk_scores):
        if not np.all(np.isfinite(array) & (array >= 0.0) & (array <= 1.0)):
            raise ValueError("Historical anomaly, age and risk must lie in [0,1]")
    valid = np.isfinite(state.last_verified)
    if np.any(state.last_verified[valid] >= state.decision_round):
        raise ValueError("History includes a current or future observation")
    expected = state.config.alpha * state.age + (1.0 - state.config.alpha) * state.anomaly_state
    if not np.array_equal(state.risk_scores, expected):
        raise ValueError("Risk scores do not match the manuscript formula")
