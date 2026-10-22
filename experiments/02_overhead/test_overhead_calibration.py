"""Shared calibration invariants; these tests produce no experiment results."""
from dataclasses import FrozenInstanceError
import importlib
import json

import numpy as np
import pytest

calibration_module = importlib.import_module("experiments.02_overhead.overhead_calibration")


def descriptors(n=300, d=10):
    rng = np.random.default_rng(917)
    return rng.normal(size=(n, d)).astype(np.float32)


def fixed_calibration(d=10):
    return calibration_module.KernelCalibration(
        dict(cosine_per_pair_ms=.12, sha256_per_descriptor_ms=.08,
             scalar_compare_ms=.01, scalar_modulo_ms=.03, vector_per_byte_ms=.001),
        seed=2026, dimension=d, input_shape=(50, d), input_sha256="a" * 64,
        sample_count=50, benchmark_cpu_ms=10., kernel_batch_cpu_ms=dict(cosine=3., sha256=7.))


@pytest.fixture(scope="module")
def measured_calibration():
    x = descriptors()
    original = x.copy()
    record = calibration_module.calibrate_kernel_costs(x, 2026, 10)
    assert np.array_equal(x, original)
    return record


def test_actual_calibration_executes_complete_predeclared_batches(measured_calibration):
    record = measured_calibration
    payload = record.to_dict()
    assert record.schema == calibration_module.CALIBRATION_SCHEMA
    assert record.input_shape == (300, 10)
    assert record.sample_count == calibration_module.CALIBRATION_SAMPLE_LIMIT
    assert record.descriptor_bytes == 40
    assert record.benchmark_cpu_ms == pytest.approx(sum(record.kernel_batch_cpu_ms.values()))
    assert len(record.input_sha256) == 64
    assert set(record.kernel_batch_cpu_ms) == {"cosine", "sha256", "scalar_compare", "scalar_modulo", "vector"}
    assert all(np.isfinite(value) and value > 0 for value in record.costs_ms.values())
    assert all(np.isfinite(value) and value > 0 for value in record.kernel_batch_cpu_ms.values())
    assert payload["cosine_batch_pairs"] == calibration_module.COSINE_BATCH_PAIRS
    assert payload["hash_batch_descriptors"] == calibration_module.HASH_BATCH_DESCRIPTORS
    assert payload["scalar_batch_items"] == calibration_module.SCALAR_BATCH_ITEMS
    assert payload["vector_batch_input_bytes"] == calibration_module.VECTOR_BATCH_BYTES
    assert json.loads(json.dumps(payload)) == payload


def test_calibration_is_deeply_immutable_and_exported_metadata_is_detached(measured_calibration):
    record = measured_calibration
    with pytest.raises(TypeError):
        record.costs_ms["cosine_per_pair_ms"] = 0
    with pytest.raises(TypeError):
        record.kernel_batch_cpu_ms["cosine"] = 0
    with pytest.raises(FrozenInstanceError):
        record.seed = 0
    exported = record.to_dict()
    exported["costs_ms"]["cosine_per_pair_ms"] = 0
    exported["kernel_batch_cpu_ms"]["cosine"] = 0
    exported["input_shape"][0] = 1
    assert record.costs_ms["cosine_per_pair_ms"] > 0
    assert record.kernel_batch_cpu_ms["cosine"] > 0
    assert record.input_shape == (300, 10)


def test_saved_calibration_roundtrip_preserves_all_costs_and_provenance(measured_calibration):
    saved = json.loads(json.dumps(measured_calibration.to_dict()))
    restored = calibration_module.KernelCalibration.from_dict(saved)
    assert restored.to_dict() == saved
    assert dict(restored.costs_ms) == dict(measured_calibration.costs_ms)


@pytest.mark.parametrize("mutation", ["unit_cost", "batch_policy", "total_time", "schema", "input_hash", "sample_count"])
def test_rehydration_rejects_corrupted_snapshot_or_policy(measured_calibration, mutation):
    payload = measured_calibration.to_dict()
    if mutation == "unit_cost":
        payload['costs_ms']['cosine_per_pair_ms'] *= 2
    elif mutation == "batch_policy":
        payload['cosine_batch_pairs'] += 1
    elif mutation == "total_time":
        payload['benchmark_cpu_ms'] *= 2
    elif mutation == "schema":
        payload['schema'] = 'invented-schema'
    elif mutation == "input_hash":
        payload['input_sha256'] = 'x' * 64
    else:
        payload['sample_count'] -= 1
    with pytest.raises(ValueError):
        calibration_module.KernelCalibration.from_dict(payload)


def test_constructor_owns_its_cost_and_timing_maps():
    costs = dict(fixed_calibration().costs_ms)
    times = dict(cosine=1.)
    record = calibration_module.KernelCalibration(costs, 1, 10, (2, 10), "a" * 64, 2, 1., times)
    costs["cosine_per_pair_ms"] = 1000
    times["cosine"] = 1000
    assert record.costs_ms["cosine_per_pair_ms"] == .12
    assert record.kernel_batch_cpu_ms["cosine"] == 1.


def test_calibration_provenance_is_input_sensitive_and_has_no_hidden_cache():
    first_x = descriptors(n=12)
    second_x = first_x.copy()
    second_x[0, 0] += .125
    first = calibration_module.calibrate_kernel_costs(first_x, 2026, 10)
    second = calibration_module.calibrate_kernel_costs(second_x, 2026, 10)
    assert first.input_sha256 != second.input_sha256
    assert first.input_shape == second.input_shape == (12, 10)
    assert first.sample_count == second.sample_count == 12
    assert first is not second


def test_shared_unit_costs_do_not_read_or_depend_on_method_name():
    record = fixed_calibration()
    operation = dict(operation="descriptor_cosine", input_bytes=80, comparisons=2)
    charges = [calibration_module.operation_duration(dict(operation, method=method), 1.5, record)
               for method in ("RACA-HIM", "DVA-P-Adapted", "I-EDI-Adapted", "CTCV-Adapted")]
    assert charges == [.36] * 4
    # Ensure the implementation never consults a method field, even indirectly.
    class MethodBlindOperation(dict):
        def get(self, key, default=None):
            if key == "method":
                raise AssertionError("Method-specific cost lookup")
            return super().get(key, default)
    assert calibration_module.operation_duration(MethodBlindOperation(operation), 1.5, record) == .36


@pytest.mark.parametrize("operation", [
    dict(operation="descriptor_cosine", input_bytes=80, comparisons=4),
    dict(operation="sha256", input_bytes=40),
    dict(operation="hash", input_bytes=80),
    dict(operation="compare", comparisons=9),
    dict(operation="compare", comparisons=5, arithmetic="modulo"),
    dict(operation="vector", input_bytes=256),
    dict(operation="memory_copy", input_bytes=80, comparisons=2),
])
def test_every_operation_class_uses_the_same_node_resource_factor(operation):
    record = fixed_calibration()
    base = calibration_module.operation_duration(operation, 1., record)
    assert calibration_module.operation_duration(operation, 2.5, record) == pytest.approx(base * 2.5)


def test_cosine_input_bytes_are_not_charged_again_after_full_raw_pair_kernel():
    record = fixed_calibration()
    charge = calibration_module.operation_duration(
        dict(operation="descriptor_cosine", input_bytes=80, comparisons=1), 1., record)
    assert charge == record.costs_ms["cosine_per_pair_ms"]


def test_hash_uses_full_descriptor_cost_and_retains_fixed_setup_for_short_inputs():
    record = fixed_calibration()
    descriptor = calibration_module.operation_duration(dict(operation="hash", input_bytes=40), 1., record)
    assert descriptor == record.costs_ms["sha256_per_descriptor_ms"]
    assert calibration_module.operation_duration(dict(operation="hash", input_bytes=80), 1., record) == 2 * descriptor
    assert calibration_module.operation_duration(dict(operation="hash", input_bytes=2), 1., record) == descriptor


def test_scalar_arithmetic_and_vector_work_are_charged_from_their_shared_classes():
    record = fixed_calibration()
    assert calibration_module.operation_duration(dict(operation="compare", comparisons=4), 1., record) == .04
    assert calibration_module.operation_duration(dict(operation="compare", comparisons=4, arithmetic="modulo"), 1., record) == .12
    assert calibration_module.operation_duration(dict(operation="vector", input_bytes=200, comparisons=4), 1., record) == pytest.approx(.24)
    assert calibration_module.operation_duration(dict(operation="vector", input_bytes=0), 1., record) == 0


@pytest.mark.parametrize("factor", [0, -1, float("nan"), float("inf")])
def test_invalid_resource_factor_is_rejected(factor):
    with pytest.raises(ValueError):
        calibration_module.operation_duration(dict(operation="hash", input_bytes=40), factor, fixed_calibration())


def test_a_finite_resource_factor_cannot_overflow_the_operation_cost():
    with pytest.raises(ValueError, match="duration"):
        calibration_module.operation_duration(
            dict(operation="descriptor_cosine", input_bytes=80, comparisons=100),
            1e308, fixed_calibration())


@pytest.mark.parametrize("field,value", [("input_bytes", -1), ("comparisons", -1),
                                       ("input_bytes", 1.5), ("comparisons", True)])
def test_invalid_operation_counts_are_rejected(field, value):
    with pytest.raises(ValueError):
        calibration_module.operation_duration(dict(operation="vector", **{field: value}), 1., fixed_calibration())


@pytest.mark.parametrize("value", [0, -1, float("nan"), float("inf")])
def test_invalid_calibrated_unit_cost_is_rejected(value):
    costs = dict(fixed_calibration().costs_ms)
    costs["cosine_per_pair_ms"] = value
    with pytest.raises(ValueError):
        calibration_module.KernelCalibration(costs, 2026, 10, (20, 10), "a" * 64, 20, 1., {"cosine": 1.})


@pytest.mark.parametrize("problem", ["float64", "empty", "zero", "nan", "wrongdim", "negative_seed"])
def test_invalid_raw_input_fails_before_calibration(problem):
    x, seed, d = descriptors(n=5), 2026, 10
    if problem == "float64":
        x = x.astype(np.float64)
    elif problem == "empty":
        x = np.empty((0, 10), dtype=np.float32)
    elif problem == "zero":
        x[0] = 0
    elif problem == "nan":
        x[0, 0] = np.nan
    elif problem == "wrongdim":
        d = 20
    else:
        seed = -1
    with pytest.raises(ValueError):
        calibration_module.calibrate_kernel_costs(x, seed, d)


def test_unresolved_clock_is_rejected_instead_of_fabricating_a_positive_cost(monkeypatch):
    monkeypatch.setattr(calibration_module.time, "perf_counter_ns", lambda: 100)
    with pytest.raises(RuntimeError, match="clock"):
        calibration_module.calibrate_kernel_costs(descriptors(n=2), 2026, 10)
