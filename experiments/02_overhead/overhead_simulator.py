"""Actual decisions/proofs with an explicit, event-derived network simulation.

The controller is a virtual endpoint (-1); the CTCV contract is -2. Their links
mirror the participating edge node's resource profile. Links are full duplex,
independent and uncontended. A primary verifier handles whole tasks serially;
different primaries run concurrently. Additional collector CPU work is also
serialized at its actual node. These are disclosed simulation assumptions.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import heapq
import importlib
import math
import time

import numpy as np

from dataset.load_dataset import resource_costs, resource_profiles
from experiments.common.config import DetectionConfig
from experiments.common.dataset_adapter import AlgorithmInput
from experiments.common.method_names import RACA_HIM
from experiments.common.similarity import cosine_aligned, within_interval

if __package__:
    from .communication_counter import CommunicationCounter
    from .config_overhead import METHODS, EVALUATION_MODES, OverheadConfig
else:
    from communication_counter import CommunicationCounter
    from config_overhead import METHODS, EVALUATION_MODES, OverheadConfig

gt_module = importlib.import_module("experiments.01_detection.proposed_detection")


@dataclass
class RunResult:
    row: dict
    messages: list[dict]
    runtime: list[dict]
    tasks: list[dict]
    diagnostics: dict


@dataclass
class Case:
    data: AlgorithmInput
    gt: object
    gt_cpu_ms: float
    profiles: dict
    targets: np.ndarray
    verifiers: np.ndarray
    baseline_targets: np.ndarray
    baseline_verifiers: np.ndarray
    fixed_targets: np.ndarray
    workload_sha256: str


def prepare_case(group: dict, n: int, d: int, seed: int, config: OverheadConfig,
                 *, core_protocol=False) -> Case:
    """Only X crosses this boundary: no labels, prototype or sample provenance."""
    x = np.asarray(group["X"])
    if x.dtype != np.float32 or x.shape != (n, d):
        raise ValueError("Expected the dataset's original float32 (N,d) matrix")
    fixture = group.get("scaling_fixture") if core_protocol else None
    if fixture is not None:
        from .overhead_scaling import stable_model_ids
        ids = stable_model_ids(n, seed, base_size=fixture["base_size"],
                               cohort_size=fixture["cohort_size"])
    else:
        rng = np.random.default_rng(np.random.SeedSequence([seed, n, 9201]))
        ids = rng.permutation(n).astype(np.int64)
    domain = np.arange(n, dtype=np.int64)
    data = AlgorithmInput(x, ids, np.full(n, config.h0), domain, np.empty(0, np.int64))
    dc = DetectionConfig(K=config.K, M=config.M, reference_pool_size=n,
                         expansion_step=config.expansion_step, dim=d, h0=config.h0,
                         tau_lower=config.tau_lower, tau_upper=config.tau_upper,
                         H=config.H, p_fail=0.0)
    start = time.perf_counter()
    if core_protocol:
        from .raca_core import build_ground_truth_core
        gt = build_ground_truth_core(data, dc)
    else:
        gt = gt_module.build_ground_truth(data, dc)
    gt_cpu = (time.perf_counter() - start) * 1000.0
    # Baselines choose their own neutral identities before looking at content.
    # They never reuse even the identities of cosine-selected RACA-HIM GT.
    verifiers = gt.selected_indices if gt.success else np.argsort(ids, kind="stable")[:config.M]
    targets = np.setdiff1d(domain, verifiers, assume_unique=True)
    baseline_verifiers = np.argsort(ids, kind="stable")[:config.M]
    baseline_targets = np.setdiff1d(domain, baseline_verifiers, assume_unique=True)
    common_targets = np.intersect1d(targets, baseline_targets, assume_unique=True)
    if len(common_targets) < config.budget(n):
        raise ValueError("Insufficient common targets after excluding independent reference sets")
    if fixture is not None:
        from .overhead_scaling import nested_fixed_targets
        fixed = nested_fixed_targets(n, ids, np.union1d(verifiers, baseline_verifiers),
            config.budget, seed, base_size=fixture["base_size"], cohort_size=fixture["cohort_size"])
    else:
        fixed_rng = np.random.default_rng(np.random.SeedSequence([seed, n, 9202]))
        fixed = fixed_rng.permutation(common_targets)[:config.budget(n)]
    digest = hashlib.sha256(fixed.astype("<i8").tobytes()).hexdigest()
    if core_protocol:
        from .overhead_resources import shared_resource_profiles
        profiles = shared_resource_profiles(seed=seed, n=n, rounds=1)
    else:
        profiles = resource_profiles(seed=seed, n=n, rounds=1)
    return Case(data, gt, gt_cpu, profiles,
                targets, np.asarray(verifiers), baseline_targets, baseline_verifiers, fixed, digest)


def joint_schedule(targets: np.ndarray, verifiers: np.ndarray, risks: np.ndarray,
                   cost_ms: np.ndarray, budget: int, deadline_ms: float,
                   lambda_: float = .5, *, fixed_workload: bool = False,
                   enforce_deadline: bool = True) -> tuple[list, dict]:
    """TNSE greedy: generate all feasible prefixes, then select the best one.

    Fixed mode bypasses the native prefix selection ONLY to compare an identical
    workload. It never increases the native decision's number of tasks in raw
    native_budget records. Pair ties use supplied temporary-identity order.
    """
    costs = np.asarray(cost_ms, np.float64)
    risks = np.asarray(risks, np.float64)
    if costs.shape != (len(targets), len(verifiers)) or risks.shape != (len(targets),):
        raise ValueError("Incompatible scheduling array shapes")
    if (isinstance(budget, bool) or not isinstance(budget, (int,np.integer)) or budget < 1
            or not math.isfinite(deadline_ms) or deadline_ms <= 0
            or not math.isfinite(lambda_) or lambda_ <= 0):
        raise ValueError("Invalid budget/deadline")
    if len(np.unique(targets)) != len(targets) or len(np.unique(verifiers)) != len(verifiers):
        raise ValueError("Target and verifier rows must be unique")
    if not len(targets) or not len(verifiers):
        return [], dict(generated_count=0, chosen_prefix=0, prefix_objectives=[], prefix_makespans_ms=[], marginal_gains=[])
    if not np.isfinite(costs).all() or np.any(costs <= 0) or not np.isfinite(risks).all():
        raise ValueError("Scheduling costs must be positive and finite")
    if np.any((risks < 0) | (risks > 1)) or set(targets.tolist()) & set(verifiers.tolist()):
        raise ValueError("Invalid risks or target/verifier intersection")
    loads = np.zeros(len(verifiers))
    used = np.zeros(len(targets), bool)
    assignments, objectives, spans, gains = [], [], [], []
    span = benefit = 0.0
    finish = np.empty_like(costs)
    delta = np.empty_like(costs)
    gain = np.empty_like(costs)
    feasible = np.empty(costs.shape, dtype=bool)
    risk_per_task = risks / budget
    for _ in range(min(budget, len(targets))):
        np.add(costs, loads[None, :], out=finish)
        if enforce_deadline:
            np.less_equal(finish, deadline_ms, out=feasible)
            feasible &= ~used[:, None]
        else:
            feasible[:] = ~used[:, None]
        np.maximum(span, finish, out=delta)
        np.subtract(delta, span, out=delta)
        np.multiply(delta, lambda_, out=gain)
        np.divide(gain, deadline_ms, out=gain)
        np.subtract(risk_per_task[:, None], gain, out=gain)
        gain[~feasible] = -np.inf
        position = int(np.argmax(gain))
        ti, vi = np.unravel_index(position, gain.shape)
        if not np.isfinite(gain[ti, vi]):
            break
        used[ti] = True
        loads[vi] = finish[ti, vi]
        span = float(loads.max())
        benefit += float(risks[ti]) / budget
        assignments.append((int(targets[ti]), int(verifiers[vi])))
        objectives.append(benefit - lambda_ * span / deadline_ms)
        spans.append(span)
        gains.append(float(gain[ti, vi]))
    chosen = len(assignments) if fixed_workload else (int(np.argmax(objectives)) + 1 if objectives else 0)
    return assignments[:chosen], dict(generated_count=len(assignments), chosen_prefix=chosen,
                                     prefix_objectives=objectives, prefix_makespans_ms=spans,
                                     marginal_gains=gains, fixed_prefix_override=fixed_workload,
                                     enforce_deadline=enforce_deadline,
                                     tie_rule="supplied temporary target/verifier identity order")


class EventSimulator:
    def __init__(self, profiles: dict, d: int, config: OverheadConfig, *, retain_events=True, calibration=None):
        self.profiles, self.d, self.config = profiles, d, config
        self.retain_events = retain_events
        self.calibration = calibration
        self.counter = CommunicationCounter(retain_events=retain_events)
        self.runtime: list[dict] = []
        self.operation_totals: dict[tuple[str, str], dict] = {}
        self.node_ready: dict[int, float] = {}
        self.pair_cache: dict[tuple[int, int], dict] = {}

    def pair(self, target: int, verifier: int) -> dict:
        n = len(self.profiles["node_z"])
        if not (0 <= target < n and 0 <= verifier < n):
            raise IndexError("Use full-N row indices for resource IDs")
        key = (int(target), int(verifier))
        if key not in self.pair_cache:
            self.pair_cache[key] = {k: float(np.asarray(v).item()) for k, v in resource_costs(
                np.array([target]), np.array([verifier]), d=self.d, sigma=self.config.sigma,
                round_id=1, profiles=self.profiles).items()}
        return self.pair_cache[key]

    def endpoints(self, sender: int, receiver: int) -> tuple[int, int]:
        if sender < 0 and receiver < 0:
            return 0, 0  # controller-contract link mirrors node 0, disclosed
        if sender < 0:
            return receiver, receiver
        if receiver < 0:
            return sender, sender
        return sender, receiver

    def message_duration(self, sender: int, receiver: int, size: int) -> tuple[float, float]:
        t, g = self.endpoints(sender, receiver)
        pair = self.pair(t, g)
        return pair["transfer_ms"] * size / (4 * self.d), pair["network_ms"]

    def send(self, event: dict, phase: str, start: float) -> float:
        event = dict(event)
        sender, receiver = int(event.pop("sender")), int(event.pop("receiver"))
        kind = event.pop("message_type")
        payload = int(event.pop("payload_bytes", 0))
        control = int(event.pop("control_bytes", 0))
        event.pop("total_bytes", None)
        event.pop("phase", None)
        transfer, lag = self.message_duration(sender, receiver, payload + control)
        end = start + transfer + lag
        self.counter.add_message(sender, receiver, phase, kind, payload, control,
                                 **event, start_ms=start, end_ms=end, transfer_ms=transfer, network_ms=lag)
        return end

    def operation_duration(self, op: dict) -> float:
        node = max(0, int(op["node"]))
        factor = self.pair(node, node)["compute_ms"] / (4.0 * self.d / 30)
        if self.calibration is not None:
            from .overhead_calibration import operation_duration
            return operation_duration(op, factor, self.calibration)
        name = str(op["operation"]).lower()
        size = int(op.get("input_bytes", 0))
        comparisons = int(op.get("comparisons", 0))
        if size < 0 or comparisons < 0:
            raise ValueError("Operation counts cannot be negative")
        if "cosine" in name:
            return self.pair(node, node)["compute_ms"] * max(1, comparisons)
        rate = (self.config.hash_throughput_mib_s if "hash" in name or "sha" in name
                else self.config.memory_throughput_mib_s)
        return factor * (size / (rate * 1024**2) * 1000.0 + comparisons * self.config.scalar_compare_ms)

    def compute(self, op: dict, phase: str, start: float) -> float:
        node = int(op["node"])
        begin = max(start, self.node_ready.get(node, 0.0))
        duration = self.operation_duration(op)
        end = begin + duration
        self.node_ready[node] = end
        self.record_operation(dict(op, phase=phase, start_ms=begin, end_ms=end,
                                   simulated_ms=duration, measurement_kind="simulated_operation"))
        return end

    def record_operation(self, operation):
        key = (operation['phase'], operation['operation'])
        aggregate = self.operation_totals.setdefault(key, dict(phase=key[0], operation=key[1],
            operation_count=0, input_bytes=0, comparisons=0, simulated_ms=0.0, cpu_ms=0.0))
        aggregate['operation_count'] += 1
        for field in ('input_bytes', 'comparisons', 'simulated_ms', 'cpu_ms'):
            aggregate[field] += operation.get(field, 0)
        if self.retain_events:
            self.runtime.append(operation)


def _gt_preparation(case: Case, sim: EventSimulator) -> float:
    ordered = np.lexsort((case.data.model_ids, case.data.history))
    clock = 0.0
    for step_id, step in enumerate(case.gt.expansion_trace):
        candidates = ordered[:step.candidate_size]
        ends = []
        for g in candidates:
            cursor = clock
            for t in candidates:
                if t == g:
                    continue
                cursor = sim.send(dict(sender=int(g), receiver=int(t), message_type="gt_request",
                                       control_bytes=sim.config.request_bytes, expansion_step=step_id,
                                       target_row=int(t), verifier_row=int(g)), "gt", cursor)
                cursor = sim.send(dict(sender=int(t), receiver=int(g), message_type="gt_descriptor",
                                       payload_bytes=4 * sim.d, expansion_step=step_id,
                                       target_row=int(t), verifier_row=int(g)), "gt", cursor)
                cursor = sim.compute(dict(node=int(g), operation="gt_cosine", input_bytes=8 * sim.d,
                                          comparisons=1, target_row=int(t), expansion_step=step_id), "gt", cursor)
            cursor = sim.send(dict(sender=int(g), receiver=-1, message_type="gt_support_report",
                                   control_bytes=sim.config.result_bytes, expansion_step=step_id), "gt", cursor)
            ends.append(cursor)
        clock = max(ends, default=clock)
    if case.gt.success:
        # All selected members are notified concurrently after majority counts.
        clock = max((sim.send(dict(sender=-1, receiver=int(g), message_type="gt_membership",
                                   control_bytes=sim.config.assignment_bytes), "gt", clock)
                     for g in case.verifiers), default=clock)
    return clock


def _proposed_cost_matrix(sim: EventSimulator, targets: np.ndarray, verifiers: np.ndarray) -> np.ndarray:
    t, g = np.broadcast_arrays(targets[:, None], verifiers[None, :])
    cost = resource_costs(t, g, d=sim.d, sigma=sim.config.sigma, round_id=1, profiles=sim.profiles)
    # Descriptor payload is counted once. Requests and result/control links each
    # add their own single propagation and byte-proportional transfer.
    result = cost["compute_ms"] + cost["transfer_ms"] + cost["network_ms"]
    result += cost["network_ms"] + cost["transfer_ms"] * sim.config.request_bytes / (4 * sim.d)
    controller = np.array([sum(sim.message_duration(-1, int(v), sim.config.assignment_bytes))
                           + sum(sim.message_duration(int(v), -1, sim.config.result_bytes)) for v in verifiers])
    return result + controller[None, :]


def _baseline_cost_matrix(method: str, sim: EventSimulator, targets: np.ndarray,
                          verifiers: np.ndarray) -> np.ndarray:
    t, g = np.broadcast_arrays(targets[:, None], verifiers[None, :])
    pair = resource_costs(t, g, d=sim.d, sigma=sim.config.sigma, round_id=1, profiles=sim.profiles)
    if method == "DVA-P-Adapted":
        proof = sim.config.digest_bytes
    elif method == "CTCV-Adapted":
        ell = sim.config.ctcv_proof_vector_length or 1000 * min(len(targets), sim.config.budget(len(sim.profiles['node_z'])))
        proof = max(1, ((ell - 1).bit_length()+7)//8)
    else:
        proof = max(1, ((sim.config.iedi_tag_modulus - 1).bit_length()+7)//8)
    tx = pair["transfer_ms"] * proof / (4 * sim.d) + pair["network_ms"]
    # Target response generation is a real hash, charged on that target's node.
    local_t = resource_costs(targets, targets, d=sim.d, sigma=sim.config.sigma, round_id=1, profiles=sim.profiles)
    local_g = resource_costs(verifiers, verifiers, d=sim.d, sigma=sim.config.sigma, round_id=1, profiles=sim.profiles)
    hash_times = local_t['compute_ms'] / (4.0*sim.d/30) * (4*sim.d/(sim.config.hash_throughput_mib_s*1024**2)*1000)
    scalar_times = local_g['compute_ms']/(4.0*sim.d/30)*sim.config.scalar_compare_ms
    if sim.calibration is not None:
        units = sim.calibration.costs_ms
        hash_times = local_t['compute_ms']/(4.0*sim.d/30)*units['sha256_per_descriptor_ms']
        scalar_times = local_g['compute_ms']/(4.0*sim.d/30)*units['scalar_compare_ms']
        if method in ('I-EDI-Adapted', 'CTCV-Adapted'):
            hash_times += local_t['compute_ms']/(4.0*sim.d/30)*units['scalar_modulo_ms']
    request_size = sim.config.request_bytes
    if method == "CTCV-Adapted":
        from .overhead_baselines import CTCV_COMPACT_REQUEST_BYTES
        request_size += CTCV_COMPACT_REQUEST_BYTES
    request = local_t['network_ms'] + local_t['transfer_ms']*request_size/(4*sim.d)
    control = (2*local_g['network_ms'] + local_g['transfer_ms']*(sim.config.assignment_bytes+sim.config.result_bytes)/(4*sim.d)
               + scalar_times)
    if method == 'CTCV-Adapted':
        collectors = np.random.default_rng(int(sim.profiles['seed'])).permutation(np.sort(verifiers))[:sim.config.ctcv_collusion_bound+1]
        producer, collector = np.broadcast_arrays(verifiers[:,None],collectors[None,:])
        relay = resource_costs(producer,collector,d=sim.d,sigma=sim.config.sigma,round_id=1,profiles=sim.profiles)
        forward = np.where(producer==collector,0.0,relay['network_ms']+relay['transfer_ms']*proof/(4*sim.d)).max(axis=1)
        control = (local_g['network_ms']+local_g['transfer_ms']*sim.config.assignment_bytes/(4*sim.d)
                   +scalar_times+forward)
    return tx + (hash_times + request)[:, None] + control[None, :]


def _baseline_stage(sim: EventSimulator, events: list, operations: list, phase: str,
                    start: float) -> float:
    """Independent preparation flows are parallel per edge node; reduction follows."""
    ready: dict[int, float] = {}
    for event in events:
        edge = event["receiver"] if event["receiver"] >= 0 else event["sender"]
        ready[edge] = sim.send(event, phase, ready.get(edge, start))
    for op in operations:
        node = int(op["node"])
        ready[node] = sim.compute(op, phase, ready.get(node, start) if node >= 0 else max(ready.values(), default=start))
    return max(ready.values(), default=start)


def _execute_event_dag(sim: EventSimulator, vertices: list[dict], start: float) -> list[float]:
    """Run dependency-ready events chronologically; CPU is FIFO per node.

    Links run independently. A CPU reservation may postpone another CPU
    operation on that same node, but never unrelated network sends. Recording
    an event's future completion only releases its children at that completion.
    """
    children = [[] for _ in vertices]
    remaining = [len(vertex['parents']) for vertex in vertices]
    ends = [start] * len(vertices)
    queue = []
    for index, vertex in enumerate(vertices):
        for parent in vertex['parents']:
            if parent >= index or parent < 0:
                raise ValueError('Protocol dependencies must reference prior vertices')
            children[parent].append(index)
        if not remaining[index]:
            heapq.heappush(queue,(start,index))
    visited = 0
    while queue:
        ready,index = heapq.heappop(queue)
        vertex = vertices[index]
        if vertex['kind']=='message':
            end=sim.send(vertex['payload'],'verification',ready)
        elif vertex['kind']=='operation':
            end=sim.compute(vertex['payload'],'verification',ready)
        elif vertex['kind']=='barrier':
            end=ready
        else:
            raise ValueError('Unknown protocol event type')
        ends[index]=end
        visited+=1
        for child in children[index]:
            remaining[child]-=1
            if not remaining[child]:
                child_ready=max((ends[parent] for parent in vertices[child]['parents']),default=start)
                heapq.heappush(queue,(child_ready,child))
    if visited != len(vertices):
        raise RuntimeError('Protocol dependency graph could not be completed')
    return ends


def _trace_tasks(sim: EventSimulator, trace, assignments: list, start: float,
                 *, batch_controls=False) -> tuple[list, float]:
    setup_events=[event for event in trace.events if event.get('stage')=='setup']
    setup_ops=[op for op in trace.operations if op.get('stage')=='setup']
    setup_end=_baseline_stage(sim,[],setup_ops,'verification',start)
    clock=_baseline_stage(sim,setup_events,[],'verification',setup_end)
    vertices=[]
    def add(kind, payload=None, parents=()):
        vertices.append(dict(kind=kind,payload=payload,parents=tuple(dict.fromkeys(parents))))
        return len(vertices)-1
    def barrier(parents):
        return add('barrier',parents=parents)
    task_events={int(t):[] for t,g in assignments}
    task_ops={int(t):[] for t,g in assignments}
    for event in trace.events:
        if event.get('stage')=='task':
            task_events[int(event['target_row'])].append(event)
    for op in trace.operations:
        if op.get('stage')=='task':
            task_ops[int(op['target_row'])].append(op)
    previous_primary={}
    task_vertices=[]
    counts = {}
    for _, verifier in assignments:
        counts[verifier] = counts.get(verifier, 0) + 1
    dispatched = {}
    if batch_controls:
        for verifier, count in counts.items():
            dispatched[verifier] = add('message', dict(sender=-1, receiver=verifier,
                message_type='batch_assignment', control_bytes=sim.config.assignment_bytes + 4*count,
                task_count=count))
    for task_id,(target,verifier) in enumerate(assignments):
        parents = ([previous_primary[verifier]] if verifier in previous_primary
                   else ([dispatched[verifier]] if batch_controls else []))
        entry=barrier(parents)
        cursor=entry
        events,ops=task_events[target],task_ops[target]
        if batch_controls:
            events = [event for event in events if event['message_type'] not in
                      ('assignment_control', 'verification_result')]
        def is_proof(event):
            return any(token in event['message_type'] for token in ('proof','tag','digest'))
        pre=[event for event in events if not is_proof(event) and 'result' not in event['message_type']]
        proofs=[event for event in events if is_proof(event)]
        post=[event for event in events if 'result' in event['message_type']]
        for event in pre:
            cursor=add('message',event,[cursor])
        for op in ops:
            if int(op['node'])==target:
                cursor=add('operation',op,[cursor])
        last_proof_arrivals={}
        for hop in sorted({int(event.get('proof_hop',0)) for event in proofs}):
            hop_ends=[]
            for event in proofs:
                if int(event.get('proof_hop',0))==hop:
                    message=add('message',event,[cursor])
                    last_proof_arrivals[int(event['receiver'])]=message
                    hop_ends.append(message)
            cursor=barrier(hop_ends)
            hop_operations=[]
            for op in ops:
                if int(op['node'])!=target and op.get('proof_hop')==hop:
                    hop_operations.append(add('operation',op,[cursor]))
            if hop_operations:
                cursor=barrier(hop_operations)
        response=cursor
        receiver_operations=[]
        for op in ops:
            if int(op['node'])!=target and 'proof_hop' not in op:
                receiver_operations.append(add('operation',op,[last_proof_arrivals.get(int(op['node']),cursor)]))
        if receiver_operations:
            cursor=barrier([cursor,*receiver_operations])
        for event in post:
            cursor=add('message',event,[cursor])
        previous_primary[verifier]=cursor
        task_vertices.append((task_id,target,verifier,entry,response,cursor))
    result_vertices = []
    if batch_controls and not any(op.get('stage')=='aggregate' for op in trace.operations):
        for verifier, count in counts.items():
            result_vertices.append(add('message', dict(sender=verifier, receiver=-1,
                message_type='batch_verification_result', control_bytes=sim.config.result_bytes + (count+7)//8,
                task_count=count), [previous_primary[verifier]]))
    tasks_done=barrier([task[-1] for task in task_vertices] + result_vertices)
    aggregate_ops=[op for op in trace.operations if op.get('stage')=='aggregate']
    aggregate_events=[event for event in trace.events if event.get('stage')=='aggregate']
    finish=tasks_done
    if aggregate_ops or aggregate_events:
        source_done={}
        for op in aggregate_ops:
            node=int(op['node'])
            if node==-1:
                source_done[node]=add('operation',op,[source_done.get(node,tasks_done)])
        # Original AV.Inquiry supplies CR before SC asks its collectors.
        correct_ends=[]
        for event in aggregate_events:
            if event['message_type']=='ctcv_correct_combination':
                correct_ends.append(add('message',event,[source_done.get(int(event['sender']),tasks_done)]))
        inquiry_arrived=barrier(correct_ends or [tasks_done])
        requested={}
        for event in aggregate_events:
            if 'request' in event['message_type']:
                requested[int(event['receiver'])]=add('message',event,[inquiry_arrived])
        for op in aggregate_ops:
            node=int(op['node'])
            if node>=0:
                source_done[node]=add('operation',op,[source_done.get(node,requested.get(node,tasks_done))])
        report_ends=[]
        for event in aggregate_events:
            if 'request' not in event['message_type'] and 'result' not in event['message_type'] and event['message_type']!='ctcv_correct_combination':
                report_ends.append(add('message',event,[source_done.get(int(event['sender']),tasks_done)]))
        finish=barrier([inquiry_arrived,*report_ends])
        for op in aggregate_ops:
            if int(op['node'])==-2:
                finish=add('operation',op,[finish])
        for event in aggregate_events:
            if 'result' in event['message_type']:
                finish=add('message',event,[finish])
    ends=_execute_event_dag(sim,vertices,clock)
    tasks=[]
    for task_id,target,verifier,entry,response,end in task_vertices:
        task=dict(task_id=task_id,target_row=target,verifier_row=verifier,
                  start_ms=ends[entry]-start,response_end_ms=ends[response]-start,
                  producer_completion_ms=ends[end]-start,end_ms=ends[end]-start,
                  task_ms=ends[end]-ends[entry])
        if aggregate_ops or aggregate_events:
            task['end_ms']=ends[finish]-start
        elif batch_controls:
            # A completed protocol includes the one final report for its batch.
            report = result_vertices[list(counts).index(verifier)]
            task['end_ms']=ends[report]-start
        tasks.append(task)
    return tasks,ends[finish]


def simulate_configuration(group: dict, n: int, d: int, seed: int,
                           config: OverheadConfig) -> list[RunResult]:
    """Load once outside this function; collect both O1/O2 and both modes here."""
    baseline = importlib.import_module("experiments.02_overhead.overhead_baselines")
    case = prepare_case(group, n, d, seed, config)
    budget = config.budget(n)
    results = []
    for mode in EVALUATION_MODES:
        for method in METHODS:
            sim = EventSimulator(case.profiles, d, config)
            assignments, tasks, diagnostics = [], [], {}
            preparation = decision_cpu = prep_cpu = None
            try:
                sim = EventSimulator(case.profiles, d, config)
                native_targets = case.targets if method == RACA_HIM else case.baseline_targets
                targets = case.fixed_targets if mode == "fixed_workload" else native_targets
                # Stable tie ordering uses neutral identities, never original-bank IDs.
                targets = targets[np.argsort(case.data.model_ids[targets], kind="stable")]
                own_verifiers = case.verifiers if method == RACA_HIM else case.baseline_verifiers
                verifiers = own_verifiers[np.argsort(case.data.model_ids[own_verifiers], kind="stable")]
                diagnostics = dict(eligible_verifier_rows=verifiers.tolist(),
                                   verifier_fixture=("TNSE majority GT" if method==RACA_HIM else "independent content-blind temporary-ID-ranked M rows; no RACA-HIM GT reuse"),
                                   native_candidate_count=len(native_targets),
                                   native_candidate_rows_sha256=hashlib.sha256(native_targets.astype('<i8').tobytes()).hexdigest(),
                                   cold_start="no last-success records; h=.1, A=1; no invented observed history",
                                   fixed_workload_note="target/prefix selection bypassed in fixed mode; native decision separately recorded")
                start_cpu = time.perf_counter()
                if method == RACA_HIM:
                    preparation = _gt_preparation(case, sim)
                    prep_cpu = case.gt_cpu_ms
                else:
                    reference = baseline.prepare_reference(case.data.X, verifiers, case.data.model_ids,
                                                           history=case.data.history, config=config)
                    prep_cpu = (time.perf_counter()-start_cpu)*1000
                    preparation = _baseline_stage(sim, list(reference.events), list(reference.operations), "gt", 0.0)
                    diagnostics["reference"] = reference.diagnostics
                risk_cpu = 0.0
                decision_start = time.perf_counter()
                plan_diag = {}
                if method == RACA_HIM and not case.gt.success:
                    assignments = []
                    plan_diag["reason"] = case.gt.reason
                elif method == RACA_HIM:
                    risk_start = time.perf_counter()
                    age = np.ones(n, dtype=np.float64)
                    risks = config.alpha*age+(1-config.alpha)*case.data.history
                    risk_cpu = (time.perf_counter()-risk_start)*1000
                    costs = _proposed_cost_matrix(sim, targets, verifiers)
                    assignments, plan_diag = joint_schedule(targets, verifiers, risks[targets], costs,
                                                            budget, config.T_round_max_ms, config.lambda_,
                                                            fixed_workload=mode=="fixed_workload")
                else:
                    costs = _baseline_cost_matrix(method, sim, targets, verifiers)
                    choose = {"DVA-P-Adapted": baseline.choose_dvap, "I-EDI-Adapted": baseline.choose_iedi,
                              "CTCV-Adapted": baseline.choose_ctcv}[method]
                    kwargs = dict(fixed_workload=mode=="fixed_workload")
                    if method == "I-EDI-Adapted":
                        kwargs["control_V"] = config.iedi_control_V
                        kwargs["minimum_fraction"] = config.iedi_budget_fraction
                    if method == "CTCV-Adapted":
                        kwargs.update(collusion_bound=config.ctcv_collusion_bound, seed=seed)
                    plan = choose(targets, verifiers, costs, budget, config.T_round_max_ms, **kwargs)
                    assignments, plan_diag = list(plan.assignments), plan.diagnostics
                decision_cpu = (time.perf_counter()-decision_start)*1000
                diagnostics["decision"] = plan_diag
                if len(assignments)>budget or len({t for t,g in assignments})!=len(assignments):
                    raise AssertionError("Duplicate target or exceeded budget")
                if any(t not in set(targets.tolist()) or g not in set(verifiers.tolist()) for t,g in assignments):
                    raise AssertionError("Assignment outside shared target/verifier domains")
                verify_cpu_start = time.perf_counter()
                tasks = []
                finish = preparation
                if method == RACA_HIM:
                    load = {}
                    assigned_t = np.array([t for t,g in assignments], np.int64)
                    assigned_g = np.array([g for t,g in assignments], np.int64)
                    similarities = cosine_aligned(case.data.X[assigned_g],case.data.X[assigned_t]) if assignments else np.empty(0)
                    predictions = (~within_interval(similarities,config.tau_lower,config.tau_upper)).astype(np.int8)
                    for pos,(target,verifier) in enumerate(assignments):
                        begin = load.get(verifier, preparation)
                        cursor = sim.send(dict(sender=-1,receiver=verifier,message_type="assignment_control",control_bytes=config.assignment_bytes,target_row=target),"verification",begin)
                        cursor = sim.send(dict(sender=verifier,receiver=target,message_type="descriptor_request",control_bytes=config.request_bytes,target_row=target),"verification",cursor)
                        cursor = sim.send(dict(sender=target,receiver=verifier,message_type="descriptor_response",payload_bytes=4*d,target_row=target),"verification",cursor)
                        cursor = sim.compute(dict(node=verifier,operation="descriptor_cosine",input_bytes=8*d,comparisons=1,target_row=target),"verification",cursor)
                        cursor = sim.send(dict(sender=verifier,receiver=-1,message_type="verification_result",control_bytes=config.result_bytes,target_row=target),"verification",cursor)
                        load[verifier]=cursor
                        tasks.append(dict(target_row=target,verifier_row=verifier,start_ms=begin-preparation,end_ms=cursor-preparation,task_ms=cursor-begin,prediction=int(predictions[pos])))
                    finish=max(load.values(),default=preparation)
                elif assignments:
                    verify = {"DVA-P-Adapted":baseline.verify_dvap,"I-EDI-Adapted":baseline.verify_iedi,"CTCV-Adapted":baseline.verify_ctcv}[method]
                    selected = np.array([t for t,g in assignments],np.int64)
                    trace=verify(case.data.X,selected,reference,config,assignments=assignments,
                                 model_ids=case.data.model_ids,verifiers=verifiers,seed=seed,
                                 collector_rows=plan_diag.get("collector_rows"))
                    tasks,finish=_trace_tasks(sim,trace,assignments,preparation)
                    if len(trace.predictions)!=len(tasks):
                        raise AssertionError("Missing actual baseline proof verdict")
                    for task,prediction in zip(tasks,trace.predictions):
                        task["prediction"]=int(prediction)
                    diagnostics["protocol"]=trace.diagnostics
                verification_cpu=(time.perf_counter()-verify_cpu_start)*1000
                ver_time=finish-preparation
                for task in tasks:
                    task["completed"]=task["end_ms"]<=config.T_round_max_ms+1e-9
                    if method == 'CTCV-Adapted':
                        task['response_deadline_violation'] = task.get('response_end_ms',task['end_ms']) > config.ctcv_time_limit*1000+1e-9
                        task['completed'] = task['completed'] and not task['response_deadline_violation']
                    task["deadline_violation"]=not task["completed"]
                completed=sum(task["completed"] for task in tasks)
                violations=len(tasks)-completed
                status=("gt_unavailable" if method==RACA_HIM and not case.gt.success else
                        "deadline_failure" if violations else "no_tasks" if not tasks else
                        "partial_workload" if mode=="fixed_workload" and completed!=budget else "success")
                sim.counter.validate()
                common=dict(seed=seed,N=n,d=d,cr=config.corruption_rate_percent,H=config.H,method=method,
                            evaluation_mode=mode,B=budget,fixed_workload_sha256=case.workload_sha256)
                row=dict(common,selected_count=len(assignments),attempted_count=len(tasks),completed_count=completed,
                         gt_success=bool(case.gt.success) if method==RACA_HIM else None,
                         gt_applicable=method==RACA_HIM,deadline_violations=violations,
                         gt_time_ms=preparation,decision_cpu_ms=decision_cpu,risk_cpu_ms=risk_cpu,
                         gt_cpu_ms=prep_cpu,verification_cpu_ms=verification_cpu,
                         verification_time_ms=ver_time,total_time_ms=preparation+decision_cpu+ver_time,
                         time_definition="hybrid: simulated preparation + measured decision CPU + simulated verification",
                         status=status,reason=plan_diag.get("reason",""),**sim.counter.totals())
                row["bytes_per_completed"]=row["total_bytes"]/completed if completed else None
                row["time_per_completed_ms"]=row["total_time_ms"]/completed if completed else None
                row["target_ids_sha256"]=hashlib.sha256(np.array([t for t,g in assignments],dtype="<i8").tobytes()).hexdigest()
                diagnostics["selected_target_rows"]=[t for t,g in assignments]
                if mode == "fixed_workload":
                    diagnostics["required_fixed_target_rows"]=case.fixed_targets.tolist()
                    selected_set={t for t,g in assignments}
                    diagnostics["unscheduled_fixed_target_rows"]=[int(t) for t in case.fixed_targets if int(t) not in selected_set]
                diagnostics["gt_expansion_trace"]=[vars(step) for step in case.gt.expansion_trace] if method==RACA_HIM else None
                diagnostics["operation_model"]={"sha256_mib_s":config.hash_throughput_mib_s,"memory_mib_s":config.memory_throughput_mib_s,"scalar_compare_ms":config.scalar_compare_ms,"source":"explicit simulator assumptions; not measured cryptographic benchmarks"}
                if any(not math.isfinite(row[k]) or row[k]<0 for k in ("gt_time_ms","decision_cpu_ms","verification_time_ms","total_time_ms")):
                    raise AssertionError("Invalid nonfinite or negative duration")
                sim.runtime.extend([dict(component="preparation_cpu",cpu_ms=prep_cpu,measurement_kind="measured_cpu"),
                                    dict(component="risk_cpu",cpu_ms=risk_cpu,measurement_kind="measured_cpu",included_in="decision_including_risk_cpu"),
                                    dict(component="decision_including_risk_cpu",cpu_ms=decision_cpu,measurement_kind="measured_cpu"),
                                    dict(component="verification_simulation_cpu",cpu_ms=verification_cpu,measurement_kind="measured_cpu")])
                results.append(RunResult(row,[dict(common,**e) for e in sim.counter.events],
                                         [dict(common,**r) for r in sim.runtime],[dict(common,**t) for t in tasks],diagnostics))
            except Exception as exc:
                # Retain already observed events from a failed method. Other
                # methods/modes still run; missing completion evidence is NA.
                import traceback
                common = dict(seed=seed,N=n,d=d,cr=config.corruption_rate_percent,H=config.H,
                              method=method,evaluation_mode=mode,B=budget,
                              fixed_workload_sha256=case.workload_sha256)
                row = dict(common,selected_count=len(assignments),attempted_count=None,
                           completed_count=None,deadline_violations=None,
                           gt_success=bool(case.gt.success) if method==RACA_HIM else None,
                           gt_applicable=method==RACA_HIM,gt_time_ms=preparation,
                           gt_cpu_ms=prep_cpu,decision_cpu_ms=decision_cpu,
                           verification_time_ms=None,total_time_ms=None,status='failure',
                           reason=f'{type(exc).__name__}: {exc}',**sim.counter.totals())
                diagnostics.update(exception_traceback=traceback.format_exc(),
                                   execution_evidence='partial method transcript; unobserved completion counts/times are missing')
                results.append(RunResult(row,[dict(common,**e) for e in sim.counter.events],
                                         [dict(common,**r) for r in sim.runtime],
                                         [dict(common,**t) for t in tasks],diagnostics))
    return results


def simulate_core_configuration(group, n, d, seed, config, *, calibration=None):
    """The single-seed, fixed-workload key-step experiment entry point."""
    from .overhead_core_simulator import simulate_core_configuration as simulate
    return simulate(group, n, d, seed, config, calibration=calibration)
