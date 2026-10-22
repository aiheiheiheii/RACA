"""Core protocol invariants; synthetic workloads never become paper results."""
from dataclasses import replace
import importlib
import json

import numpy as np
import pytest

simulator = importlib.import_module('experiments.02_overhead.overhead_simulator')
core = importlib.import_module('experiments.02_overhead.overhead_core_simulator')
counter_module = importlib.import_module('experiments.02_overhead.communication_counter')
cfg = importlib.import_module('experiments.02_overhead.config_overhead')
runner = importlib.import_module('experiments.02_overhead.overhead_experiment')


def descriptors(n=80, d=10):
    x = np.zeros((n, d), dtype=np.float32)
    x[:, 0] = 1
    x[:, 1] = np.arange(n)*.0001
    return x


def test_compact_counter_has_exact_same_bytes_and_message_counts_as_full_ledger():
    full = counter_module.CommunicationCounter()
    compact = counter_module.CommunicationCounter(retain_events=False)
    for index in range(8):
        for counter in (full, compact):
            counter.add_message(index, -1, 'gt' if index%2 else 'verification',
                                'descriptor', payload_bytes=index*4, control_bytes=index,
                                start_ms=0., end_ms=1., transfer_ms=.2, network_ms=.8)
    full.validate()
    compact.validate()
    assert full.totals() == compact.totals()
    assert full.phase_totals() == compact.phase_totals()
    assert not compact.events and len(full.events) == 8


def test_compact_counter_still_rejects_invalid_timing():
    counter = counter_module.CommunicationCounter(retain_events=False)
    with pytest.raises(AssertionError):
        counter.add_message(1, -1, 'gt', 'bad', start_ms=2., end_ms=1.)
    assert counter.totals()['message_count'] == 0


@pytest.mark.parametrize('d', cfg.D_PANEL_VALUES)
def test_core_executes_same_entire_budget_with_auditable_phase_totals(d):
    x = descriptors(d=d)
    before = x.copy()
    config = cfg.OverheadConfig(K=4, M=2, expansion_step=2, dim=d)
    results = core.simulate_core_configuration({'X':x}, len(x), d, 2026, config)
    assert {result.row['method'] for result in results} == set(cfg.METHODS)
    assert len(results) == 4
    assert np.array_equal(x, before)
    assert len({result.row['fixed_workload_sha256'] for result in results}) == 1
    assert len({result.diagnostics['selected_target_ids_sha256'] for result in results}) == 1
    for result in results:
        row = result.row
        assert row['status'] == 'success', row['reason']
        assert row['selected_count'] == row['attempted_count'] == row['completed_count'] == config.budget(len(x))
        assert row['within_deadline_count']+row['deadline_violations'] == row['completed_count']
        ledger = json.loads(row['communication_phase_totals_json'])
        assert row['total_bytes'] == sum(item['total_bytes'] for item in ledger)
        assert row['message_count'] == sum(item['message_count'] for item in ledger)
        assert row['total_bytes'] == row['gt_bytes']+row['verification_bytes']+row['control_bytes']
        assert row['total_time_ms'] == pytest.approx(row['gt_time_ms']+row['decision_cpu_ms']+
                            row['verification_time_ms']+row['history_cpu_ms'])
        assert all(np.isfinite(row[field]) and row[field]>=0 for field in
                   ('total_time_ms','verification_time_ms','gt_time_ms','decision_cpu_ms'))
        assert not result.messages and not result.runtime and not result.tasks
        assert result.diagnostics['all_fixed_workload_protocol_steps_executed']
        if row['method'] == 'RACA-HIM':
            types = {item['message_type']:item for item in ledger}
            assert 'gt_membership' not in types
            assert result.diagnostics['gt_membership_notification'].startswith('merged into')
            assert types['gt_candidate_upload']['message_count'] == result.diagnostics['candidate_count']
            assert types['gt_candidate_upload']['payload_bytes'] == 4*d*result.diagnostics['candidate_count']
            assert types['descriptor_response']['payload_bytes'] == 4*d*config.budget(len(x))
        if row['method'] == 'CTCV-Adapted':
            assert result.diagnostics['ell'] == 1000*config.budget(len(x))


def test_deadline_exceedance_does_not_remove_a_completed_fixed_workload():
    x = descriptors()
    config = cfg.OverheadConfig(K=4, M=2, expansion_step=2, dim=10, T_round_max_ms=.0001)
    results = core.simulate_core_configuration({'X':x}, len(x), 10, 2026, config)
    for result in results:
        assert result.row['status'] == 'success', result.row.get('reason')
        assert result.row['completed_count'] == config.budget(len(x))
        assert result.row['deadline_violations'] == config.budget(len(x))
        assert result.row['within_deadline_count'] == 0
        assert not result.row['deadline_met']
        assert runner._complete_fixed(result.row)


def test_core_pair_decisions_keep_resource_penalty_when_deadline_is_diagnostic():
    targets, verifiers = np.array([2,3]), np.array([0,1])
    costs = np.array([[100., 120.], [110., 90.]])
    pairs, plan = simulator.joint_schedule(targets, verifiers, np.ones(2), costs,
        2, 1., fixed_workload=True, enforce_deadline=False)
    assert pairs == [(3,1), (2,0)]
    assert plan['generated_count'] == 2 and not plan['enforce_deadline']
    assert all(value>0 for value in plan['prefix_makespans_ms'])


def test_batch_controls_have_explicit_identity_and_bitmap_byte_costs():
    x = descriptors()
    config = cfg.OverheadConfig(K=4, M=2, expansion_step=2, dim=10)
    results = core.simulate_core_configuration({'X':x}, len(x), 10, 2026, config)
    for result in results:
        ledger = json.loads(result.row['communication_phase_totals_json'])
        batches = [item for item in ledger if item['message_type']=='batch_assignment']
        assert len(batches) == 1
        item = batches[0]
        assert item['control_bytes'] == item['message_count']*config.assignment_bytes+4*config.budget(len(x))
        assert not any(item['message_type'] in ('assignment_control','verification_result') for item in ledger)
        if result.row['method'] != 'CTCV-Adapted':
            assert any(item['message_type']=='batch_verification_result' for item in ledger)
        else:
            assert any(item['message_type']=='ctcv_correct_combination' for item in ledger)
            assert any(item['message_type']=='ctcv_co_verification_result' for item in ledger)
