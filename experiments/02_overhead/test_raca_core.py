"""Focused synthetic checks for Group 02's declared RACA-HIM core adapter."""
from __future__ import annotations

from dataclasses import replace
import importlib
from pathlib import Path
from types import SimpleNamespace
import sys

sys.dont_write_bytecode = True
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pytest

from experiments.common.config import DetectionConfig
from experiments.common.dataset_adapter import AlgorithmInput

core = importlib.import_module("experiments.02_overhead.raca_core")
original = importlib.import_module("experiments.01_detection.proposed_detection")
counter = importlib.import_module("experiments.02_overhead.communication_counter")


def _input(x, *, ids=None, history=None, reference=None):
    x = np.asarray(x, dtype=np.float32)
    n = len(x)
    reference = np.arange(n, dtype=np.int64) if reference is None else reference
    return AlgorithmInput(
        X=x, model_ids=np.arange(n, dtype=np.int64) if ids is None else ids,
        history=np.full(n, .10) if history is None else history,
        reference_indices=reference,
        evaluation_indices=np.setdiff1d(np.arange(n, dtype=np.int64), reference),
    )


def _config(n, k=3, m=3, step=3, dim=10):
    return DetectionConfig(K=k, M=m, reference_pool_size=max(k, n),
                           expansion_step=step, dim=dim, baseline_block_count=1)


def _assert_same(left, right):
    assert left.success == right.success
    assert left.threshold == right.threshold
    assert left.reason == right.reason
    assert left.expansion_trace == right.expansion_trace
    for field in ("selected_indices", "candidate_indices", "support_counts"):
        np.testing.assert_array_equal(getattr(left, field), getattr(right, field))
        assert not getattr(right, field).flags.writeable


@pytest.mark.parametrize("seed", range(16))
def test_cached_majority_matches_original_without_labels(seed):
    rng = np.random.default_rng(seed)
    centers = np.eye(3, 10)
    x = (centers[rng.integers(0, 3, size=24)] + rng.normal(0, .0002, (24, 10))).astype(np.float32)
    data = _input(x, ids=rng.permutation(24), history=rng.choice([.05, .1, .5], size=24))
    config = _config(24)
    result = core.build_ground_truth_core(data, config)
    _assert_same(original.build_ground_truth(data, config), result)
    final_size = len(result.candidate_indices)
    assert result.unique_pair_count == final_size * (final_size - 1) // 2
    assert result.normalized_row_count == final_size
    assert np.unique(np.concatenate([s.new_candidate_indices for s in result.core_steps])).size == final_size


def test_expansion_recomputes_threshold_and_acceptance_using_new_support():
    x = np.zeros((9, 10), dtype=np.float32)
    x[:, 0] = 1
    x[1] = np.eye(10, dtype=np.float32)[1]
    x[2] = np.eye(10, dtype=np.float32)[2]
    data, config = _input(x), _config(9)
    result = core.build_ground_truth_core(data, config)
    _assert_same(original.build_ground_truth(data, config), result)
    assert result.success
    assert [step.candidate_size for step in result.expansion_trace] == [3, 6]
    assert [step.threshold for step in result.expansion_trace] == [2, 4]
    assert [step.new_pair_count for step in result.core_steps] == [3, 12]
    assert [step.kernel_dot_count for step in result.core_steps] == [9, 18]
    np.testing.assert_array_equal(result.selected_indices, [0, 3, 4])


def test_insufficient_majority_exhausts_full_candidate_domain():
    data = _input(np.eye(11, dtype=np.float32))
    config = _config(11, k=3, m=2, step=3, dim=11)
    result = core.build_ground_truth_core(data, config)
    _assert_same(original.build_ground_truth(data, config), result)
    assert not result.success
    assert [step.candidate_size for step in result.expansion_trace] == [3, 6, 9, 11]
    assert result.normalized_row_count == 11
    assert result.unique_pair_count == 55
    assert result.selected_indices.size == 0


def test_reference_subset_and_temporary_id_history_order_are_preserved():
    x = np.zeros((8, 10), dtype=np.float32)
    x[:, 0] = 1
    data = _input(x, ids=np.array([7, 6, 5, 4, 3, 2, 1, 0]),
                  history=np.array([.1, .4, .2, .2, .1, .1, .1, .1]),
                  reference=np.array([0, 2, 3, 5, 7]))
    config = _config(5)
    result = core.build_ground_truth_core(data, config)
    _assert_same(original.build_ground_truth(data, config), result)
    np.testing.assert_array_equal(result.selected_indices, [7, 5, 0])


def test_reference_pool_smaller_than_initial_k_is_unavailable():
    x = np.zeros((5, 10), dtype=np.float32)
    x[:, 0] = 1
    data = _input(x, reference=np.array([0, 1]))
    config = _config(5)
    result = core.build_ground_truth_core(data, config)
    _assert_same(original.build_ground_truth(data, config), result)
    assert result.core_steps == ()


@pytest.mark.parametrize("lower, upper", [(1., 1.), (.997, 1.), (-1., 0.), (-.5, .5)])
def test_inclusive_interval_never_adds_an_extra_tolerance(lower, upper):
    x = np.zeros((7, 10), dtype=np.float32)
    x[:, 0] = np.array([1, 1, -1, -1, 1, -1, 1])
    data = _input(x)
    config = replace(_config(7), tau_lower=lower, tau_upper=upper)
    _assert_same(original.build_ground_truth(data, config), core.build_ground_truth_core(data, config))


class _Simulator:
    def __init__(self, d=10):
        self.d = d
        self.config = SimpleNamespace(request_bytes=64, assignment_bytes=64)
        self.node_ready = {}
        self.runtime = []
        self.counter = counter.CommunicationCounter()

    def send(self, event, phase, start):
        fields = dict(event)
        sender, receiver = fields.pop("sender"), fields.pop("receiver")
        kind = fields.pop("message_type")
        payload, control = fields.pop("payload_bytes", 0), fields.pop("control_bytes", 0)
        transfer, lag = (payload + control) / 100., 1.
        end = start + transfer + lag
        self.counter.add_message(sender, receiver, phase, kind, payload, control,
                                 **fields, start_ms=start, end_ms=end,
                                 transfer_ms=transfer, network_ms=lag)
        return end


def _expanded_case():
    x = np.zeros((9, 10), dtype=np.float32)
    x[:, 0] = 1
    x[1] = np.eye(10, dtype=np.float32)[1]
    x[2] = np.eye(10, dtype=np.float32)[2]
    gt = core.build_ground_truth_core(_input(x), _config(9))
    return SimpleNamespace(gt=gt, gt_cpu_ms=10.)


def test_core_protocol_uploads_each_actual_candidate_once_and_retains_membership():
    case, sim = _expanded_case(), _Simulator()
    end = core.execute_ground_truth_core(case, sim)
    sim.counter.validate()
    events = sim.counter.events
    uploads = [event for event in events if event["message_type"] == "gt_candidate_upload"]
    requests = [event for event in events if event["message_type"] == "gt_candidate_request"]
    memberships = [event for event in events if event["message_type"] == "gt_membership"]
    assert len(uploads) == len(requests) == 6
    assert len(memberships) == 3
    assert len({event["sender"] for event in uploads}) == 6
    assert all(event["payload_bytes"] == 4 * sim.d for event in uploads)
    assert {event["receiver"] for event in memberships} == set(case.gt.selected_indices)
    assert sum(operation["cpu_ms"] for operation in sim.runtime) == pytest.approx(10.)
    assert all(operation["measurement_kind"] == "measured_cpu" for operation in sim.runtime)
    assert sim.counter.totals()["total_bytes"] == 6 * (64 + 4 * sim.d) + 3 * 64
    assert end == max(event["end_ms"] for event in memberships)


def test_membership_can_be_carried_by_later_batch_without_losing_gt_barriers():
    case, sim = _expanded_case(), _Simulator()
    end = core.execute_ground_truth_core(case, sim, notify_membership=False)
    sim.counter.validate()
    assert not any(event['message_type']=='gt_membership' for event in sim.counter.events)
    assert sim.counter.totals()['total_bytes'] == 6*(64+4*sim.d)
    assert sum(operation['cpu_ms'] for operation in sim.runtime) == pytest.approx(10.)
    assert end == max(operation['end_ms'] for operation in sim.runtime)
    # A subsequent batch starts after all GT uploads and the majority decision.
    batch_end = sim.send(dict(sender=-1,receiver=int(case.gt.selected_indices[0]),
        message_type='batch_assignment',control_bytes=64+4,role='gt_verifier',
        membership_in_header=True), 'verification', end)
    assert batch_end > end
    assert sim.counter.events[-1]['start_ms'] == end
    assert sim.counter.events[-1]['membership_in_header']


def test_expansion_upload_waits_for_previous_majority_cpu_and_kernel_waits_for_uploads():
    case, sim = _expanded_case(), _Simulator()
    core.execute_ground_truth_core(case, sim, cpu_ms=20.)
    kernels = [operation for operation in sim.runtime
               if operation["operation"] == "gt_incremental_vectorized_majority"]
    for step_id, operation in enumerate(kernels):
        uploads = [event for event in sim.counter.events
                   if event["message_type"] == "gt_candidate_upload" and event["expansion_step"] == step_id]
        assert operation["start_ms"] >= max(event["end_ms"] for event in uploads)
        if step_id:
            requests = [event for event in sim.counter.events
                        if event["message_type"] == "gt_candidate_request" and event["expansion_step"] == step_id]
            assert min(event["start_ms"] for event in requests) >= kernels[step_id - 1]["end_ms"]
    first_requests = [event for event in sim.counter.events
                      if event["message_type"] == "gt_candidate_request" and event["expansion_step"] == 0]
    assert min(event["start_ms"] for event in first_requests) >= sim.runtime[0]["end_ms"]
    assert sum(operation["cpu_ms"] for operation in sim.runtime) == pytest.approx(20.)


def test_failed_gt_retains_all_uploads_without_membership():
    data = _input(np.eye(7, dtype=np.float32))
    gt = core.build_ground_truth_core(data, _config(7, k=3, m=2, dim=7))
    sim = _Simulator(d=7)
    end = core.execute_ground_truth_core(SimpleNamespace(gt=gt, gt_cpu_ms=1.), sim)
    assert len(sim.counter.events) == 14
    assert not any(event["message_type"] == "gt_membership" for event in sim.counter.events)
    assert end == sim.runtime[-1]["end_ms"]


@pytest.mark.parametrize("bad_cpu", [float("nan"), float("inf"), -1.])
def test_invalid_measured_cpu_is_rejected(bad_cpu):
    with pytest.raises(ValueError):
        core.execute_ground_truth_core(_expanded_case(), _Simulator(), cpu_ms=bad_cpu)


def test_network_adapter_rejects_old_all_to_all_result():
    x = np.zeros((5, 10), dtype=np.float32)
    x[:, 0] = 1
    gt = original.build_ground_truth(_input(x), _config(5))
    with pytest.raises(TypeError):
        core.execute_ground_truth_core(SimpleNamespace(gt=gt, gt_cpu_ms=1.), _Simulator())


def test_small_reference_pool_failure_still_charges_measured_gt_cpu():
    x = np.zeros((5, 10), dtype=np.float32)
    x[:, 0] = 1
    gt = core.build_ground_truth_core(_input(x, reference=np.array([0, 1])), _config(5))
    sim = _Simulator()
    end = core.execute_ground_truth_core(SimpleNamespace(gt=gt, gt_cpu_ms=2.), sim)
    assert end == 2.
    assert not sim.counter.events
    assert sum(operation["cpu_ms"] for operation in sim.runtime) == 2.
