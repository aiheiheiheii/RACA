"""Controlled-scale resource tests; no paper workloads or outputs are run."""
from __future__ import annotations

import importlib
from pathlib import Path
import sys

sys.dont_write_bytecode = True
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pytest

from dataset.load_dataset import resource_costs, resource_profiles

resources = importlib.import_module("experiments.02_overhead.overhead_resources")


@pytest.mark.parametrize("n", [6, 12, 1000, 1500, 2000, 2500, 3000, 5000])
def test_every_n_uses_the_same_node_prefix_and_round(n):
    prefix = resources.shared_resource_profiles(seed=2026, n=n)
    full = resource_profiles(seed=2026, n=5000, rounds=1)
    assert np.array_equal(prefix["node_z"], full["node_z"][:n])
    assert np.array_equal(prefix["round_z"], full["round_z"])
    assert prefix["node_z"].shape == (n, 3)
    assert prefix["node_z"].dtype == np.dtype("float32")
    assert prefix["resource_domain_max"] == 5000
    assert prefix["resource_fixture_sha256"] == resources.shared_resource_profiles(
        seed=2026, n=5000)["resource_fixture_sha256"]


@pytest.mark.parametrize("d", [10, 20, 30, 40, 50])
def test_same_heterogeneous_pair_has_identical_cost_across_n(d):
    targets, verifiers = np.array([30, 90, 900]), np.array([0, 3, 23])
    costs = [resource_costs(targets, verifiers, d=d, sigma=0.5,
                           profiles=resources.shared_resource_profiles(n=n))
             for n in (1000, 2000, 2500, 5000)]
    for key in costs[0]:
        for later in costs[1:]:
            assert np.array_equal(costs[0][key], later[key])
    # This fix preserves the heterogeneous fixture rather than setting sigma=0.
    assert np.ptp(costs[0]["compute_ms"]) > 0.0
    assert np.ptp(costs[0]["network_ms"]) > 0.0


def test_cached_snapshot_cannot_be_mutated_through_any_prefix():
    first = resources.shared_resource_profiles(n=6)
    before = first["node_z"].copy()
    with pytest.raises(TypeError):
        first["seed"] = 123
    for values in (first["node_z"], first["round_z"], first["node_z"].base):
        with pytest.raises(ValueError):
            values.flags.writeable = True
    with pytest.raises(ValueError):
        first["node_z"][0, 0] = 10.0
    second = resources.shared_resource_profiles(n=12)
    assert np.array_equal(second["node_z"][:6], before)
    assert np.shares_memory(first["node_z"], second["node_z"])


def test_resource_metadata_is_serializable_and_changes_with_fixture():
    import json
    base = resources.shared_resource_profiles(n=6)
    metadata = resources.resource_fixture_metadata(base)
    assert json.loads(json.dumps(metadata)) == metadata
    assert len(metadata["resource_fixture_sha256"]) == 64
    for changed in ({"seed": 2027}, {"n_max": 5001}, {"rounds": 2}):
        assert resources.shared_resource_profiles(n=6, **changed)[
            "resource_fixture_sha256"] != base["resource_fixture_sha256"]


@pytest.mark.parametrize("change", [
    {"n": 0}, {"n": -1}, {"n": 5001}, {"n": True}, {"n": 5.0},
    {"seed": -1}, {"seed": False}, {"seed": 2026.0},
    {"rounds": 0}, {"rounds": True}, {"rounds": 1.0},
    {"n_max": 0}, {"n_max": False}, {"n_max": 5000.0},
    {"n": 6, "n_max": 5},
])
def test_invalid_resource_arguments_rejected(change):
    with pytest.raises(ValueError):
        resources.shared_resource_profiles(**change)


def test_numpy_integer_arguments_are_normalized():
    profiles = resources.shared_resource_profiles(
        seed=np.int64(2026), n=np.int64(6), rounds=np.int64(1), n_max=np.int64(5000))
    assert profiles["seed"] == 2026
    assert isinstance(profiles["resource_domain_max"], int)


def test_original_varying_n_changes_round_fixture_but_shared_fixture_does_not():
    legacy_small = resource_profiles(seed=2026, n=1000, rounds=1)
    legacy_mid = resource_profiles(seed=2026, n=2500, rounds=1)
    assert not np.array_equal(legacy_small["round_z"], legacy_mid["round_z"])
    shared_small = resources.shared_resource_profiles(seed=2026, n=1000)
    shared_mid = resources.shared_resource_profiles(seed=2026, n=2500)
    assert np.array_equal(shared_small["round_z"], shared_mid["round_z"])
