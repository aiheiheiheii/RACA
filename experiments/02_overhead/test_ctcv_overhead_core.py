"""Small synthetic tests for compact CTCV requests and retained proof rules."""
from __future__ import annotations

import hashlib
import importlib
from pathlib import Path
import struct
import sys

sys.dont_write_bytecode = True
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pytest

baseline = importlib.import_module("experiments.02_overhead.overhead_baselines")
cfg = importlib.import_module("experiments.02_overhead.config_overhead")


def _case(count=120, dim=10, *, corrupt=False):
    x = np.zeros((count + 4, dim), dtype=np.float32)
    x[:, 0] = 1.0
    targets = np.arange(4, len(x), dtype=np.int64)
    if corrupt:
        x[targets[::7], 1] = 0.02
    ids = np.arange(1_000_000, 1_000_000 + len(x), dtype=np.int64)
    pool = np.arange(4, dtype=np.int64)
    reference = baseline.prepare_reference(x, pool, ids)
    assignments = tuple((int(row), 2) for row in targets)
    return x, ids, pool, targets, reference, assignments


def _trace(count=120, dim=10, *, corrupt=False, ell=0):
    case = _case(count, dim, corrupt=corrupt)
    x, ids, pool, targets, reference, assignments = case
    trace = baseline.verify_ctcv(
        x, targets, reference, cfg.OverheadConfig(dim=dim, ctcv_proof_vector_length=ell),
        assignments=assignments, model_ids=ids, verifiers=pool, collector_rows=np.array([0, 1]))
    return case, trace


@pytest.mark.parametrize("dim", [10, 30, 50])
def test_compact_request_reconstructs_every_original_tag(dim):
    (x, ids, _, targets, reference, _), trace = _trace(120, dim, corrupt=True)
    diag = trace.diagnostics
    for index, row in enumerate(targets):
        wire = struct.pack("<QIII", diag["request_seed"], diag["requested_block_count"],
                           len(reference.payload), int(ids[row]) % diag["block_count"])
        seed, blocks, size, start = struct.unpack("<QIII", wire)
        boundaries, shuffled = baseline._ctcv_partition(seed, blocks, size)
        assert boundaries.tolist() == diag["block_boundaries_bytes"]
        assert shuffled.tolist() == diag["shuffled_block_order"]
        order = np.roll(shuffled, -start)
        raw = baseline.canonical_descriptor_bytes(x[row])
        reordered = b"".join(raw[boundaries[b]:boundaries[b + 1]] for b in order)
        tag = int(hashlib.sha256(reordered).hexdigest(), 16) % diag["proof_vector_length"]
        assert tag == diag["actual_proof_tags"][index]
        expected_payload = b"".join(reference.payload[boundaries[b]:boundaries[b + 1]] for b in order)
        expected = int(hashlib.sha256(expected_payload).hexdigest(), 16) % diag["proof_vector_length"]
        assert expected == diag["correct_proof_tags"][index]
        request = next(e for e in trace.events
                       if e["message_type"] == "ctcv_request" and e["target_row"] == row)
        assert request["payload_bytes"] == 20
        assert request["payload_sha256"] == hashlib.sha256(wire).hexdigest()


def test_request_local_cache_only_reuses_identical_correct_challenges():
    (_, ids, _, targets, _, _), trace = _trace(123, 10, corrupt=True)
    unique_starts = len(set((ids[targets] % trace.diagnostics["block_count"]).tolist()))
    correct_ops = [op for op in trace.operations if op["node"] == baseline.AV
                   and op["operation"] == "hash" and op["stage"] == "aggregate"]
    response_ops = [op for op in trace.operations if op["operation"] == "hash"
                    and op["stage"] == "task"]
    assert len(correct_ops) == unique_starts == 40
    assert len(response_ops) == len(targets)
    assert trace.diagnostics["correct_challenge_cache_hits"] == len(targets) - unique_starts
    assert all(op["stage"] == "aggregate" for op in correct_ops)
    # Independent requests rebuild correct proofs; this is not a global cache.
    (_, second_trace) = _trace(123, 10, corrupt=True)
    assert second_trace.diagnostics["correct_challenges_computed"] == unique_starts


@pytest.mark.parametrize("count", [40, 100, 200])
def test_source_ell_and_three_complete_packed_combinations_are_retained(count):
    _, trace = _trace(count, 10)
    assert trace.diagnostics["proof_vector_length"] == 1000 * count
    combinations = [e for e in trace.events
                    if e["message_type"] in ("ctcv_correct_combination", "ctcv_proof_combination")]
    assert len(combinations) == 3  # AV.Inquiry CR and k+1 complete collector reports.
    assert all(e["payload_bytes"] == 125 * count for e in combinations)
    assert trace.diagnostics["flag"] == 0
    assert np.array_equal(trace.predictions, np.zeros(count, dtype=np.int8))


def test_complete_communication_scales_with_tasks_without_repeated_full_partition_tables():
    totals = []
    for count in (100, 200, 300):
        _, trace = _trace(count, 10)
        totals.append(sum(e["payload_bytes"] + e["control_bytes"] for e in trace.events))
    assert totals[1] - totals[0] == totals[2] - totals[1]
    # The adapter preserves source ell; byte dimensionality changes the real
    # response work but no longer multiplies duplicated request metadata.
    dimension_totals = []
    for dim in (10, 30, 50):
        _, trace = _trace(120, dim)
        dimension_totals.append(sum(e["payload_bytes"] + e["control_bytes"] for e in trace.events))
    assert len(set(dimension_totals)) == 1


def test_modulo_collision_remains_a_protocol_collision():
    _, trace = _trace(25, 10, corrupt=True, ell=1)
    assert trace.diagnostics["full_hash_equal_count"] < 25
    assert trace.diagnostics["flag"] == 0
    assert np.array_equal(trace.predictions, np.zeros(25, dtype=np.int8))
    assert trace.diagnostics["combination_or_modulus_masked_hash_mismatch_count"] > 0

