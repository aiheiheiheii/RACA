"""Fixed-workload key-step protocols, with exact compact byte ledgers.

This explicitly adapted experiment charges measured preparation/decision
kernels and simulated protocol transfers/verification kernels. It runs every
one of the same B targets; deadline exceedances are diagnostics, not deletions.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
import time
import traceback

import numpy as np

from dataset.load_dataset import resource_costs
from experiments.common.method_names import RACA_HIM
from experiments.common.similarity import within_interval
from . import overhead_baselines as baseline
from .config_overhead import METHODS, MAIN_EVALUATION_MODE
from .overhead_simulator import (
    EventSimulator, RunResult, prepare_case, joint_schedule, _baseline_stage,
    _baseline_cost_matrix, _trace_tasks,
)
from .raca_core import CORE_PROTOCOL, execute_ground_truth_core
from .raca_kernels import cached_reference_cosine, history_update_vectorized
from .overhead_calibration import calibrate_kernel_costs
from .overhead_resources import resource_fixture_metadata


def _core_costs(method, sim, targets, verifiers):
    target, verifier = np.broadcast_arrays(targets[:, None], verifiers[None, :])
    if method == RACA_HIM:
        pair = resource_costs(target, verifier, d=sim.d, sigma=sim.config.sigma,
                              round_id=1, profiles=sim.profiles)
        # One request, one raw descriptor, one cosine. Batch controls are
        # separately transmitted once per participating verifier.
        local = resource_costs(verifiers, verifiers, d=sim.d, sigma=sim.config.sigma,
                               round_id=1, profiles=sim.profiles)
        cosine_ms = local['compute_ms']/(4.0*sim.d/30)*sim.calibration.costs_ms['cosine_per_pair_ms']
        return (cosine_ms[None, :] + pair['transfer_ms'] + 2*pair['network_ms']
                + pair['transfer_ms']*sim.config.request_bytes/(4*sim.d))
    costs = _baseline_cost_matrix(method, sim, targets, verifiers)
    local = resource_costs(verifiers, verifiers, d=sim.d, sigma=sim.config.sigma,
                           round_id=1, profiles=sim.profiles)
    if method == 'CTCV-Adapted':
        controls = local['network_ms'] + local['transfer_ms']*sim.config.assignment_bytes/(4*sim.d)
    else:
        controls = (2*local['network_ms'] + local['transfer_ms']*
                    (sim.config.assignment_bytes+sim.config.result_bytes)/(4*sim.d))
    return costs-controls[None, :]


def _reference_preparation(case, sim, verifiers):
    tick = time.perf_counter()
    reference = baseline.prepare_reference(case.data.X, verifiers, case.data.model_ids,
                                          history=case.data.history, config=sim.config)
    cpu_ms = (time.perf_counter()-tick)*1000
    # Hash/reduction kernels are already measured above, so their simulated
    # operations must not be added again. Every actual reference upload stays.
    network_end = _baseline_stage(sim, list(reference.events), [], 'gt', 0.0)
    sim.node_ready[-1] = network_end+cpu_ms
    sim.record_operation(dict(node=-1, operation='reference_hash_reduction', phase='gt',
        start_ms=network_end, end_ms=network_end+cpu_ms, cpu_ms=cpu_ms,
        input_bytes=sum(op.get('input_bytes', 0) for op in reference.operations),
        measurement_kind='measured_cpu'))
    return reference, network_end+cpu_ms, cpu_ms


def _raca_verification(case, sim, assignments, preparation):
    """Parallel primary queues; one dispatch and one bitmap result per queue."""
    counts = Counter(verifier for _, verifier in assignments)
    starts = {verifier:sim.send(dict(sender=-1, receiver=verifier,
        message_type='batch_assignment', control_bytes=sim.config.assignment_bytes+4*count,
        task_count=count, role='gt_verifier', membership_in_header=True),
        'verification', preparation) for verifier, count in counts.items()}
    targets = np.asarray([target for target, _ in assignments], np.int64)
    verifiers = np.asarray([verifier for _, verifier in assignments], np.int64)
    scores = cached_reference_cosine(case.data.X, targets, verifiers)
    predictions = (~within_interval(scores, sim.config.tau_lower, sim.config.tau_upper)).astype(np.int8)
    tasks, loads = [], dict(starts)
    for position, (target, verifier) in enumerate(assignments):
        begin = loads[verifier]
        cursor = sim.send(dict(sender=verifier, receiver=target, message_type='descriptor_request',
            control_bytes=sim.config.request_bytes, target_row=target), 'verification', begin)
        cursor = sim.send(dict(sender=target, receiver=verifier, message_type='descriptor_response',
            payload_bytes=4*sim.d, target_row=target), 'verification', cursor)
        cursor = sim.compute(dict(node=verifier, operation='descriptor_cosine', input_bytes=8*sim.d,
            comparisons=1, target_row=target), 'verification', cursor)
        loads[verifier] = cursor
        tasks.append(dict(target_row=target, verifier_row=verifier,
            start_ms=begin-preparation, response_end_ms=cursor-preparation,
            end_ms=cursor-preparation, prediction=int(predictions[position])))
    reports = {verifier:sim.send(dict(sender=verifier, receiver=-1,
        message_type='batch_verification_result', control_bytes=sim.config.result_bytes+(count+7)//8,
        task_count=count), 'verification', loads[verifier]) for verifier, count in counts.items()}
    for task in tasks:
        task['end_ms'] = reports[task['verifier_row']]-preparation
    return tasks, max(reports.values(), default=preparation)


def _history_update(case, tasks, config):
    tick = time.perf_counter()
    history = history_update_vectorized(case.data.history,
        [task['target_row'] for task in tasks], [task['prediction'] for task in tasks],
        case.gt.selected_indices, config.rho)
    return (time.perf_counter()-tick)*1000, hashlib.sha256(history.astype('<f8').tobytes()).hexdigest()


def simulate_core_configuration(group, n, d, seed, config, *, calibration=None):
    case = prepare_case(group, n, d, seed, config, core_protocol=True)
    if calibration is None:
        # Direct synthetic/unit calls have no run-level persistent setup. The
        # real runner supplies one saved full-population calibration for all N.
        calibration = calibrate_kernel_costs(case.data.X, seed, d)
    budget, results = config.budget(n), []
    for method in METHODS:
        sim = EventSimulator(case.profiles, d, config, retain_events=False, calibration=calibration)
        targets = case.fixed_targets[np.argsort(case.data.model_ids[case.fixed_targets], kind='stable')]
        own = case.verifiers if method == RACA_HIM else case.baseline_verifiers
        verifiers = own[np.argsort(case.data.model_ids[own], kind='stable')]
        context = dict(seed=seed, N=n, d=d, cr=config.corruption_rate_percent, H=config.H,
            method=method, evaluation_mode=MAIN_EVALUATION_MODE, B=budget,
            fixed_workload_sha256=case.workload_sha256)
        assignments, tasks, preparation, decision_cpu, prep_cpu = [], [], None, None, None
        diagnostics = dict(core_transport='batched dispatch/result for every method',
            preparation_clock='measured actual preparation kernels + simulated actual preparation transfers',
            fixed_workload='execute all common B targets; deadline diagnostic only',
            candidate_count=len(case.gt.candidate_indices) if method==RACA_HIM else config.M,
            resource_sigma=config.sigma)
        diagnostics['kernel_calibration'] = calibration.to_dict()
        diagnostics['resource_fixture'] = resource_fixture_metadata(case.profiles)
        if 'scaling_fixture' in group:
            diagnostics['scaling_fixture'] = group['scaling_fixture']
            diagnostics['reference_rows'] = verifiers.tolist()
            diagnostics['workload_rows'] = case.fixed_targets.tolist()
            diagnostics['workload_design'] = 'content-blind budget increments per arrival cohort; references excluded'
        try:
            if method == RACA_HIM:
                preparation = execute_ground_truth_core(case, sim, notify_membership=False)
                prep_cpu = case.gt_cpu_ms
                diagnostics['gt_protocol'] = CORE_PROTOCOL
                diagnostics['gt_membership_notification'] = 'merged into participating verifier batch assignment header'
                diagnostics['gt_expansion_sizes'] = [step.candidate_size for step in case.gt.core_steps]
                if not case.gt.success:
                    raise RuntimeError(case.gt.reason)
            else:
                reference, preparation, prep_cpu = _reference_preparation(case, sim, verifiers)
                diagnostics['reference_anchor_sha256'] = reference.sha256
                diagnostics['reference_anchor_authenticated'] = False
            tick = time.perf_counter()
            costs = _core_costs(method, sim, targets, verifiers)
            if method == RACA_HIM:
                # Cold start has age=1 and h=h0, exactly as the existing round.
                risks = config.alpha*np.ones(n)+(1-config.alpha)*case.data.history
                assignments, plan = joint_schedule(targets, verifiers, risks[targets], costs,
                    budget, config.T_round_max_ms, config.lambda_, fixed_workload=True,
                    enforce_deadline=False)
            else:
                choose = {'DVA-P-Adapted':baseline.choose_dvap, 'I-EDI-Adapted':baseline.choose_iedi,
                          'CTCV-Adapted':baseline.choose_ctcv}[method]
                kwargs = dict(fixed_workload=True)
                if method == 'CTCV-Adapted':
                    kwargs.update(collusion_bound=config.ctcv_collusion_bound, seed=seed)
                if method == 'I-EDI-Adapted':
                    kwargs.update(control_V=config.iedi_control_V, minimum_fraction=config.iedi_budget_fraction)
                selected = choose(targets, verifiers, costs, budget, config.T_round_max_ms, **kwargs)
                assignments, plan = list(selected.assignments), selected.diagnostics
            decision_cpu = (time.perf_counter()-tick)*1000
            diagnostics['decision_plan'] = plan
            if len(assignments)!=budget or {target for target, _ in assignments}!=set(targets):
                raise AssertionError('Core overhead benchmark must execute every common B target exactly once')
            diagnostics['selected_target_ids_sha256'] = hashlib.sha256(
                np.sort(case.data.model_ids[[t for t, _ in assignments]]).astype('<i8').tobytes()).hexdigest()
            verify_tick = time.perf_counter()
            if method == RACA_HIM:
                tasks, finish = _raca_verification(case, sim, assignments, preparation)
            else:
                verify = {'DVA-P-Adapted':baseline.verify_dvap, 'I-EDI-Adapted':baseline.verify_iedi,
                          'CTCV-Adapted':baseline.verify_ctcv}[method]
                trace = verify(case.data.X, targets, reference, config, assignments=assignments,
                    model_ids=case.data.model_ids, verifiers=verifiers, seed=seed,
                    collector_rows=plan.get('collector_rows'))
                tasks, finish = _trace_tasks(sim, trace, assignments, preparation, batch_controls=True)
                if len(trace.predictions)!=len(tasks):
                    raise AssertionError('Missing actual proof verdict')
                predictions = dict(zip(trace.diagnostics.get('target_rows', targets.tolist()), trace.predictions))
                for task in tasks:
                    task['prediction'] = int(predictions[task['target_row']])
                for key in ('proof_vector_length', 'packed_report_bytes', 'collector_count', 'request_partition_bytes',
                            'flag', 'correct_challenges_computed', 'correct_challenge_cache_hits', 'block_count'):
                    if key in trace.diagnostics:
                        diagnostics[key] = trace.diagnostics[key]
                if 'proof_vector_length' in trace.diagnostics:
                    diagnostics['ell'] = trace.diagnostics['proof_vector_length']
            verification_cpu = (time.perf_counter()-verify_tick)*1000
            history_cpu = 0.0
            if method == RACA_HIM:
                history_cpu, state_hash = _history_update(case, tasks, config)
                diagnostics['updated_history_sha256'] = state_hash
            late = [task['end_ms']>config.T_round_max_ms+1e-9 or
                (method=='CTCV-Adapted' and task.get('response_end_ms', task['end_ms'])>
                 config.ctcv_time_limit*1000+1e-9) for task in tasks]
            sim.counter.validate()
            phase_ledger = sim.counter.phase_totals()
            row = dict(context, selected_count=budget, attempted_count=budget, completed_count=len(tasks),
                within_deadline_count=sum(not value for value in late), deadline_violations=sum(late),
                deadline_met=not any(late), gt_success=bool(case.gt.success) if method==RACA_HIM else None,
                gt_applicable=method==RACA_HIM, gt_time_ms=preparation, preparation_time_ms=preparation,
                gt_cpu_ms=prep_cpu, decision_cpu_ms=decision_cpu, verification_cpu_ms=verification_cpu,
                history_cpu_ms=history_cpu, verification_time_ms=finish-preparation,
                total_time_ms=finish+decision_cpu+history_cpu,
                time_definition='core hybrid: measured preparation/decision/history kernels + simulated protocol transfers and verification kernels',
                status='success', reason='', **sim.counter.totals())
            row['communication_phase_totals_json'] = json.dumps(phase_ledger, separators=(',', ':'))
            row['runtime_phase_totals_json'] = json.dumps(list(sim.operation_totals.values()), separators=(',', ':'))
            row['core_phase_counts_json'] = json.dumps(dict(communication=phase_ledger,
                runtime=list(sim.operation_totals.values())), separators=(',', ':'))
            row['corrupted_verdict_count'] = sum(task['prediction'] for task in tasks)
            row['target_ids_sha256'] = case.workload_sha256
            diagnostics['all_fixed_workload_protocol_steps_executed'] = True
            diagnostics['deadline_scope'] = 'diagnostic; success means all B protocol verdicts obtained'
            results.append(RunResult(row, [], [], [], diagnostics))
        except Exception as exc:
            row = dict(context, selected_count=len(assignments), attempted_count=None, completed_count=None,
                gt_applicable=method==RACA_HIM, gt_success=bool(case.gt.success) if method==RACA_HIM else None,
                gt_time_ms=preparation, gt_cpu_ms=prep_cpu, decision_cpu_ms=decision_cpu,
                verification_time_ms=None, total_time_ms=None, status='failure',
                reason=f'{type(exc).__name__}: {exc}', **sim.counter.totals())
            diagnostics['exception_traceback'] = traceback.format_exc()
            results.append(RunResult(row, [], [], [], diagnostics))
    return results
