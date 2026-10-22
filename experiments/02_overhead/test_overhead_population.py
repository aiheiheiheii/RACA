"""Nested fixture tests using small synthetic populations only."""
from __future__ import annotations

import ast
import importlib
from pathlib import Path
import sys

sys.dont_write_bytecode = True
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pytest

population = importlib.import_module("experiments.02_overhead.overhead_population")
GRID = (100, 150, 200)


def _group(maximum=200, rate=3.0):
    quotas = population.population_quotas(maximum, rate)
    labels = np.concatenate([np.full(count, subtype, dtype=np.uint8)
                             for subtype, count in enumerate(quotas)])
    x = np.zeros((maximum, 10), dtype=np.float32)
    x[:, 0] = np.arange(1, maximum + 1, dtype=np.float32)
    x[:, 1] = labels
    order = np.random.default_rng(123).permutation(maximum)
    return {"X": x[order], "subtype_label": labels[order]}


def test_nested_cohorts_preserve_exact_loader_quotas_and_all_source_rows():
    group = _group()
    fixture = population.build_population_from_group(group, grid=GRID)
    for n in GRID:
        result = fixture.prefix(n)
        actual = tuple(int(np.count_nonzero(result["X"][:, 1] == subtype))
                       for subtype in range(4))
        assert actual == population.population_quotas(n)
        assert result["counts"] == dict(zip(population.SUBTYPE_NAMES, actual))
        assert np.array_equal(result["X"], fixture.X[:n])
        assert np.shares_memory(result["X"], fixture.X)
    assert set(fixture.X[:, 0]) == set(group["X"][:, 0])
    assert len(np.unique(fixture.X[:, 0])) == GRID[-1]


def test_anonymous_shuffle_is_reproducible_and_independent_of_labels_at_runtime():
    group = _group()
    first = population.build_population_from_group(group, grid=GRID, seed=2026)
    second = population.build_population_from_group(group, grid=GRID, seed=2026)
    other = population.build_population_from_group(group, grid=GRID, seed=2027)
    assert np.array_equal(first.X, second.X)
    assert first.population_sha256 == second.population_sha256
    assert not np.array_equal(first.X, other.X)
    assert first.population_sha256 != other.population_sha256
    for start, stop in ((0, 100), (100, 150), (150, 200)):
        assert not np.array_equal(first.X[start:stop, 1], np.sort(first.X[start:stop, 1]))


def test_population_and_all_prefixes_are_truly_immutable():
    group = _group()
    fixture = population.build_population_from_group(group, grid=GRID)
    original = fixture.X.copy()
    group["X"][:] = 0
    group["subtype_label"][:] = 0
    assert np.array_equal(fixture.X, original)
    for values in (fixture.X, fixture.X.base, fixture.prefix(100)["X"]):
        with pytest.raises(ValueError):
            values.flags.writeable = True
    with pytest.raises(ValueError):
        fixture.X[0, 0] = 0


def test_runtime_group_discards_all_label_and_provenance_arrays():
    fixture = population.build_population_from_group(_group(), grid=GRID)
    result = fixture.prefix(150)
    assert set(result) == {"X", "N", "dimension", "seed", "corruption_rate_percent",
                           "counts", "scaling_fixture"}
    assert not any(key in result for key in (
        "binary_label", "subtype_label", "sample_id", "prototype", "cosine_to_prototype"))
    result["counts"]["common_normal"] = -1
    result["scaling_fixture"]["prefix_counts"]["100"]["common_normal"] = -1
    fresh = fixture.prefix(150)
    assert fresh["counts"]["common_normal"] > 0
    assert fresh["scaling_fixture"]["prefix_counts"]["100"]["common_normal"] > 0


def test_population_metadata_is_serializable_and_full_hash_shared_across_prefixes():
    import json
    fixture = population.build_population_from_group(_group(), grid=GRID)
    small = fixture.prefix(100)["scaling_fixture"]
    large = fixture.prefix(200)["scaling_fixture"]
    assert json.loads(json.dumps(small)) == small
    assert small["population_sha256"] == large["population_sha256"]
    assert small["prefix_sha256"] != large["prefix_sha256"]
    assert small["base_size"] == 100
    assert small["cohort_size"] == 50
    assert small["domain_max"] == 200
    assert small["prefix_counts"]["150"] == fixture.prefix(150)["counts"]


@pytest.mark.parametrize("rate", [0.0, 3.0, 5.0, 50.0])
def test_quota_counts_and_construction_at_supported_rates(rate):
    fixture = population.build_population_from_group(
        _group(rate=rate), grid=GRID, corruption_rate_percent=rate)
    for n, counts in fixture.counts_by_n:
        assert sum(counts) == n
        assert counts == population.population_quotas(n, rate)


@pytest.mark.parametrize("grid", [(), (100, 100, 200), (200, 100), (0, 200),
                                 (100.0, 200), (True, 200)])
def test_invalid_grid_rejected(grid):
    with pytest.raises(ValueError):
        population.build_population_from_group(_group(), grid=grid)


@pytest.mark.parametrize("change", [
    {"seed": -1}, {"seed": True}, {"seed": 2026.0},
    {"corruption_rate_percent": -1}, {"corruption_rate_percent": 51},
    {"corruption_rate_percent": float("nan")}, {"corruption_rate_percent": True},
])
def test_invalid_constructor_parameters_rejected(change):
    with pytest.raises(ValueError):
        population.build_population_from_group(_group(), grid=GRID, **change)


@pytest.mark.parametrize("fault", ["float64", "nan", "wrong_n", "unknown_stratum",
                                  "float_labels", "wrong_quotas", "missing_labels"])
def test_invalid_source_population_rejected(fault):
    group = _group()
    if fault == "float64":
        group["X"] = group["X"].astype(np.float64)
    elif fault == "nan":
        group["X"][0, 0] = np.nan
    elif fault == "wrong_n":
        group["X"] = group["X"][:-1]
    elif fault == "unknown_stratum":
        group["subtype_label"][0] = 4
    elif fault == "float_labels":
        group["subtype_label"] = group["subtype_label"].astype(float)
    elif fault == "wrong_quotas":
        group["subtype_label"][:] = 0
    else:
        group.pop("subtype_label")
    with pytest.raises((ValueError, KeyError)):
        population.build_population_from_group(group, grid=GRID)


@pytest.mark.parametrize("n", [0, 99, 101, 201, True, 100.0])
def test_prefix_requires_declared_integer_grid_value(n):
    fixture = population.build_population_from_group(_group(), grid=GRID)
    with pytest.raises(ValueError):
        fixture.prefix(n)


def test_wrapper_loads_one_maximum_population_then_reuses_it(monkeypatch, tmp_path):
    calls = []
    def fake_load_group(**kwargs):
        calls.append(kwargs)
        return _group(maximum=5000)
    monkeypatch.setattr(population, "load_group", fake_load_group)
    population._load_population.cache_clear()
    for n in (1000, 2500, 5000, 1000):
        result = population.load_scaling_group(tmp_path, n=n, dim=10, seed=2026)
        assert result["X"].shape == (n, 10)
        assert result["heterogeneity_H"] == 0.5
        assert result["scaling_fixture"]["base_size"] == 1000
        assert result["scaling_fixture"]["cohort_size"] == 500
        assert result["scaling_fixture"]["domain_max"] == 5000
    assert len(calls) == 1
    assert calls[0]["n"] == 5000
    population._load_population.cache_clear()


def test_fixture_constructor_never_reads_sample_or_prototype_information():
    source = Path(population.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    keys = {node.slice.value for node in ast.walk(tree)
            if isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Constant)
            and isinstance(node.slice.value, str)}
    assert "subtype_label" in keys
    assert not keys.intersection({"sample_id", "prototype", "binary_label", "cosine_to_prototype"})
