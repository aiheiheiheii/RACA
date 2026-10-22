"""A shared, immutable heterogeneous resource fixture for the Group 02 N sweep.

Drawing ``resource_profiles(seed, n)`` separately at each N changes the random
round effects: the generator draws n node rows before the round rows.  This
module draws the largest declared node domain once, then exposes a prefix.
The same row IDs and round therefore keep the same costs across N, without
removing heterogeneity or changing the dataset's cost formula.
"""
from __future__ import annotations

from functools import lru_cache
import hashlib
import json
from types import MappingProxyType
from typing import Mapping

import numpy as np

from dataset.load_dataset import resource_profiles

RESOURCE_FIXTURE_SCHEMA = "overhead-shared-resource-v1"
RESOURCE_DOMAIN_MAX = 5000


def _integer(name: str, value: int, minimum: int) -> int:
    if (isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer)) or value < minimum):
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _immutable_float32(values: np.ndarray) -> np.ndarray:
    # A bytes buffer is genuinely immutable; setting flags.writeable=True on
    # this array or on a returned prefix cannot expose a mutable cached array.
    values = np.ascontiguousarray(values, dtype="<f4")
    return np.frombuffer(values.tobytes(), dtype="<f4").reshape(values.shape)


@lru_cache(maxsize=16)
def _snapshot(seed: int, n_max: int, rounds: int) -> Mapping:
    profiles = resource_profiles(seed=seed, n=n_max, rounds=rounds)
    node_z = _immutable_float32(profiles["node_z"])
    round_z = _immutable_float32(profiles["round_z"])
    metadata = {
        "resource_fixture_schema": RESOURCE_FIXTURE_SCHEMA,
        "resource_domain_max": n_max,
        "resource_rounds": rounds,
        "resource_seed": seed,
    }
    digest = hashlib.sha256(
        json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode("utf-8")
        + b"\n" + node_z.tobytes() + round_z.tobytes()
    ).hexdigest()
    return MappingProxyType({
        "node_z": node_z,
        "round_z": round_z,
        "seed": seed,
        **metadata,
        "resource_fixture_sha256": digest,
    })


def shared_resource_profiles(seed: int = 2026, n: int = RESOURCE_DOMAIN_MAX,
                             rounds: int = 1, *,
                             n_max: int = RESOURCE_DOMAIN_MAX) -> Mapping:
    """Return an immutable N-prefix of one declared maximum-size snapshot.

    Small synthetic N values use the same fixture and are supported for tests.
    ``n_max`` is explicit so a different declared sweep has a different fixture
    hash.  All methods and all N points of a sweep must use the same value.
    The hash describes the *full* snapshot and is consequently constant across
    prefixes, making this control visible in checkpoints and run provenance.
    """
    seed = _integer("seed", seed, 0)
    n = _integer("n", n, 1)
    rounds = _integer("rounds", rounds, 1)
    n_max = _integer("n_max", n_max, 1)
    if n > n_max:
        raise ValueError("n cannot exceed the declared resource domain maximum")
    snapshot = _snapshot(seed, n_max, rounds)
    return MappingProxyType({**snapshot, "node_z": snapshot["node_z"][:n]})


def resource_fixture_metadata(profiles: Mapping) -> dict:
    """Return the JSON-serializable resource provenance for a result record."""
    return {key: profiles[key] for key in (
        "resource_fixture_schema", "resource_domain_max", "resource_rounds",
        "resource_seed", "resource_fixture_sha256",
    )}
