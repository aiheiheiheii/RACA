"""Anonymous controlled-scale identities and workload tests without protocols."""
from __future__ import annotations

import ast
import importlib
import math
from pathlib import Path
import sys

sys.dont_write_bytecode = True
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pytest

scaling = importlib.import_module("experiments.02_overhead.overhead_scaling")


def _small_ids(n):
    return scaling.stable_model_ids(n, seed=2026, base_size=100, cohort_size=50)


def _small_targets(n, excluded=(0, 1, 2, 3, 4), *, seed=2026):
    return scaling.nested_fixed_targets(n, _small_ids(n), excluded,
                                        lambda size: math.ceil(.1 * (size - 5)), seed,
                                        base_size=100, cohort_size=50)


def test_anonymous_ids_are_stable_prefixes_and_keep_new_cohorts_higher():
    full = _small_ids(300)
    for n in (100, 150, 200, 300):
        ids = _small_ids(n)
        assert np.array_equal(ids, full[:n])
        assert np.array_equal(np.sort(ids), np.arange(n))
    for start, stop in ((0, 100), (100, 150), (150, 200), (200, 250), (250, 300)):
        assert np.array_equal(np.sort(full[start:stop]), np.arange(start, stop))
        assert not np.array_equal(full[start:stop], np.arange(start, stop))
    assert np.max(full[:100]) < np.min(full[100:])
    assert np.array_equal(np.argsort(full[:100])[:25], np.argsort(full)[:25])


def test_seed_changes_only_the_anonymous_order_and_remains_deterministic():
    first = _small_ids(200)
    other = scaling.stable_model_ids(200, seed=2027, base_size=100, cohort_size=50)
    assert not np.array_equal(first, other)
    assert np.array_equal(first, _small_ids(200))
    assert np.array_equal(np.sort(first), np.sort(other))


def test_targets_are_exact_ordered_prefixes_and_exclude_both_reference_sets():
    excluded = (0, 1, 2, 3, 4, 10, 11, 12, 13, 14)
    full = _small_targets(300, excluded)
    for n in (100, 150, 200, 300):
        targets = _small_targets(n, excluded)
        assert np.array_equal(targets, full[:len(targets)])
        assert len(targets) == math.ceil(.1 * (n - 5))
        assert len(np.unique(targets)) == len(targets)
        assert not set(targets).intersection(excluded)
        assert np.all(targets < n)
    counts = [int(np.count_nonzero((full >= start) & (full < stop)))
              for start, stop in ((0, 100), (100, 150), (150, 200), (200, 250), (250, 300))]
    assert counts == [10, 5, 5, 5, 5]


def test_formal_budget_work_is_distributed_over_the_arrival_cohorts():
    ids = scaling.stable_model_ids(5000)
    budget = lambda n: math.ceil(.1 * (n - 25))
    excluded = np.arange(50)
    full = scaling.nested_fixed_targets(5000, ids, excluded, budget)
    assert len(full) == 498
    assert np.count_nonzero(full < 1000) == 98
    for start in range(1000, 5000, 500):
        assert np.count_nonzero((full >= start) & (full < start + 500)) == 50
    for n in (1000, 2000, 5000):
        targets = scaling.nested_fixed_targets(n, ids[:n], excluded, budget)
        assert np.array_equal(targets, full[:budget(n)])


def test_targets_are_reproducible_and_seed_sensitive():
    assert np.array_equal(_small_targets(200), _small_targets(200))
    assert not np.array_equal(_small_targets(200), _small_targets(200, seed=2027))


def test_returned_id_and_target_arrays_are_genuinely_immutable():
    for values in (_small_ids(200), _small_targets(200)):
        for array in (values, values[:5]):
            with pytest.raises(ValueError):
                array.flags.writeable = True
        with pytest.raises(ValueError):
            values[0] = 0


def test_zero_budget_and_duplicate_exclusions_are_supported():
    targets = scaling.nested_fixed_targets(100, _small_ids(100), [0, 0, 1],
                                           lambda n: 0, base_size=100, cohort_size=50)
    assert targets.dtype == np.int64
    assert len(targets) == 0
    assert not targets.flags.writeable


def test_later_cohort_references_are_excluded_without_using_content():
    excluded = np.array([0, 1, 2, 110, 160])
    targets = _small_targets(200, excluded)
    assert not set(targets).intersection(excluded)
    assert len(targets) == 20


@pytest.mark.parametrize("change", [
    {"n": 0}, {"n": 99}, {"n": 101}, {"n": 175}, {"n": 100.0}, {"n": True},
    {"seed": -1}, {"seed": 2026.0}, {"seed": False},
    {"base_size": 0}, {"base_size": True}, {"cohort_size": 0}, {"cohort_size": 50.0},
])
def test_invalid_identity_arguments_rejected(change):
    kwargs = {"n": 100, "seed": 2026, "base_size": 100, "cohort_size": 50, **change}
    with pytest.raises(ValueError):
        scaling.stable_model_ids(**kwargs)


@pytest.mark.parametrize("fault", ["short", "duplicate", "float", "negative", "overflow"])
def test_invalid_model_id_arrays_rejected(fault):
    ids = np.array(_small_ids(100), copy=True)
    if fault == "short":
        ids = ids[:-1]
    elif fault == "duplicate":
        ids[0] = ids[1]
    elif fault == "float":
        ids = ids.astype(float)
    elif fault == "negative":
        ids[0] = -1
    else:
        ids = ids.astype(np.uint64)
        ids[0] = np.iinfo(np.uint64).max
    with pytest.raises(ValueError):
        scaling.nested_fixed_targets(100, ids, [], lambda n: 10,
                                     base_size=100, cohort_size=50)


@pytest.mark.parametrize("excluded", [[-1], [100], [0.0], [True], [[0, 1]]])
def test_invalid_excluded_rows_rejected(excluded):
    with pytest.raises(ValueError):
        _small_targets(100, excluded)


@pytest.mark.parametrize("budget", [lambda n: -1, lambda n: 1.0, lambda n: True,
                                   lambda n: 101, lambda n: 10 if n == 100 else 9, None])
def test_invalid_or_infeasible_budgets_rejected(budget):
    with pytest.raises(ValueError):
        scaling.nested_fixed_targets(150, _small_ids(150), [], budget,
                                     base_size=100, cohort_size=50)


def test_cohort_budget_must_fit_that_cohort_not_only_the_whole_population():
    with pytest.raises(ValueError, match="cohort lacks"):
        scaling.nested_fixed_targets(150, _small_ids(150), [],
                                     lambda n: 0 if n == 100 else 51,
                                     base_size=100, cohort_size=50)


def test_helpers_have_no_descriptor_or_label_dependencies():
    tree = ast.parse(Path(scaling.__file__).read_text(encoding="utf-8"))
    imports = {node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)}
    imports.update(alias.name for node in ast.walk(tree) if isinstance(node, ast.Import)
                   for alias in node.names)
    assert not any(name.startswith("dataset") for name in imports)
    names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
    assert not names.intersection({"X", "labels", "subtype_label", "sample_id", "prototype"})
