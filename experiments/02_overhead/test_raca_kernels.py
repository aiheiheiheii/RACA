"""Local cache/update equivalence checks; no experiment results are written."""
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

from experiments.common.similarity import cosine_aligned, within_interval

kernels = importlib.import_module("experiments.02_overhead.raca_kernels")


@pytest.mark.parametrize("d", (10, 30, 50))
@pytest.mark.parametrize("seed", (0, 2026))
def test_cached_cosines_and_predictions_are_bit_identical(d, seed):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(32, d)).astype(np.float32)
    targets = np.array([10, 7, 29, 10, 20, 8, 7, 31], dtype=np.int64)
    verifiers = np.array([1, 2, 1, 2, 2, 1, 3, 1], dtype=np.int64)
    original = x.copy()
    x.setflags(write=False)
    expected = cosine_aligned(x[verifiers], x[targets])
    actual = kernels.cached_reference_cosine(x, targets, verifiers)
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(
        within_interval(actual, .997, 1.), within_interval(expected, .997, 1.),
    )
    # Exact inclusive boundaries use the same observed score, without adding
    # a tolerance that would widen the declared interval.
    lower, upper = float(expected.min()), float(expected.max())
    np.testing.assert_array_equal(
        within_interval(actual, lower, upper), within_interval(expected, lower, upper),
    )
    np.testing.assert_array_equal(x, original)
    assert not x.flags.writeable


def test_each_unique_row_is_normalized_once_including_overlapping_roles(monkeypatch):
    x = np.array([[1., 0.], [1., .001], [0., 1.], [.001, 1.]], np.float32)
    targets, verifiers = np.array([1, 2, 1, 3, 2]), np.array([0, 0, 1, 0, 1])
    calls = []
    original = kernels._unit_rows

    def counted(values):
        calls.append(values.copy())
        return original(values)

    monkeypatch.setattr(kernels, "_unit_rows", counted)
    actual = kernels.cached_reference_cosine(x, targets, verifiers)
    np.testing.assert_array_equal(actual, cosine_aligned(x[verifiers], x[targets]))
    assert len(calls) == 1
    np.testing.assert_array_equal(calls[0], x[np.unique(np.r_[targets, verifiers])])
    assert len(calls[0]) < 2 * len(targets)


def test_stable_normalization_retains_extreme_finite_values():
    x = np.array([[1e300, 1e299], [1e-300, 1e-301], [-1e300, -1e299]], np.float64)
    targets, verifiers = np.array([0, 1, 2]), np.array([1, 0, 1])
    np.testing.assert_array_equal(
        kernels.cached_reference_cosine(x, targets, verifiers),
        cosine_aligned(x[verifiers], x[targets]),
    )


@pytest.mark.parametrize("invalid", (0., np.nan, np.inf))
def test_selected_invalid_descriptor_is_rejected(invalid):
    x = np.array([[1., 1.], [invalid, invalid]], np.float32)
    with pytest.raises(ValueError):
        kernels.cached_reference_cosine(x, [1], [0])


def test_invalid_unused_row_is_not_read():
    x = np.array([[1., 1.], [1., .001], [np.nan, np.nan]], np.float32)
    np.testing.assert_array_equal(
        kernels.cached_reference_cosine(x, [1], [0]), cosine_aligned(x[[0]], x[[1]]),
    )


@pytest.mark.parametrize("targets,verifiers", (
    ([0., 1.], [1, 0]), ([-1], [0]), ([2], [0]), ([0], [2]),
    ([[0]], [[1]]), ([0, 1], [1]), ([True], [0]),
))
def test_invalid_aligned_row_identities_are_rejected(targets, verifiers):
    with pytest.raises(ValueError):
        kernels.cached_reference_cosine(np.eye(2), targets, verifiers)


def test_empty_aligned_cosine_has_valid_float64_shape():
    actual = kernels.cached_reference_cosine(np.eye(2), [], [])
    assert actual.shape == (0,) and actual.dtype == np.float64


@pytest.mark.parametrize("rho", (.1, .5, 1.))
def test_vectorized_history_is_bit_identical_to_original_scalar_updates(rho):
    state = np.array([0., .1, .3, .7, 1., .8, .9, .4], np.float64)
    targets, predictions, gt = np.array([4, 0, 7, 3]), np.array([0, 1, 1, 0]), np.array([2, 5])
    expected = state.copy()
    for row, prediction in zip(targets, predictions):
        if prediction:
            expected[row] += rho * (1 - expected[row])
        else:
            expected[row] *= 1 - rho
    expected[gt] *= 1 - rho
    original = state.copy()
    state.setflags(write=False)
    actual = kernels.history_update_vectorized(state, targets, predictions, gt, rho)
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(state, original)
    assert not np.shares_memory(actual, state)
    assert not state.flags.writeable


@pytest.mark.parametrize("targets,predictions,gt,rho", (
    ([0, 0], [0, 1], [2], .5), ([0], [0], [2, 2], .5),
    ([0], [0], [0], .5), ([0], [2], [2], .5), ([0], [-1], [2], .5),
    ([0], [.5], [2], .5), ([0, 1], [0], [2], .5), ([0.], [0], [2], .5),
    ([-1], [0], [2], .5), ([3], [0], [2], .5), ([0], [0], [3], .5),
    ([0], [0], [2], 0.), ([0], [0], [2], 1.1),
    ([0], [0], [2], np.nan), ([0], [0], [2], True),
))
def test_invalid_history_event_inputs_are_rejected(targets, predictions, gt, rho):
    with pytest.raises(ValueError):
        kernels.history_update_vectorized(np.full(3, .1), targets, predictions, gt, rho)


@pytest.mark.parametrize("state", (
    np.array([np.nan]), np.array([np.inf]), np.array([-0.1]),
    np.array([1.1]), np.array([[.1]]), np.array([0]),
))
def test_invalid_history_state_is_rejected(state):
    with pytest.raises(ValueError):
        kernels.history_update_vectorized(state, [], [], [], .5)


def test_empty_history_update_returns_independent_unchanged_state():
    state = np.array([.1, .2])
    actual = kernels.history_update_vectorized(state, [], [], [], .5)
    np.testing.assert_array_equal(actual, state)
    assert not np.shares_memory(actual, state)
