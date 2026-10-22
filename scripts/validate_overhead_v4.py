"""Read-only, one-dimensional code check; never writes publication results."""
from __future__ import annotations

from dataclasses import replace
import hashlib
import importlib
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
from threadpoolctl import threadpool_limits

from experiments.common.reproducibility import dataset_snapshot, verify_manifest

cfg = importlib.import_module('experiments.02_overhead.config_overhead')
population = importlib.import_module('experiments.02_overhead.overhead_population')
simulator = importlib.import_module('experiments.02_overhead.overhead_simulator')
calibration_module = importlib.import_module('experiments.02_overhead.overhead_calibration')


def validate():
    d, seed = 30, 2026
    config = replace(cfg.OverheadConfig(), dim=d)
    before = dataset_snapshot(cfg.DATASET_ROOT)
    integrity = verify_manifest(cfg.DATASET_ROOT)
    if integrity['missing'] or integrity['mismatch']:
        raise AssertionError('Dataset manifest verification failed')
    rows, previous_workload, previous_references = [], None, None
    with threadpool_limits(limits=1):
        full = population.load_scaling_group(n=5000, dim=d, seed=seed)
        calibration = calibration_module.calibrate_kernel_costs(full['X'], seed, d)
        for n in cfg.N_VALUES:
            group = population.load_scaling_group(n=n, dim=d, seed=seed)
            original = hashlib.sha256(group['X'].tobytes()).hexdigest()
            records = simulator.simulate_core_configuration(group, n, d, seed, config,
                                                             calibration=calibration)
            assert hashlib.sha256(group['X'].tobytes()).hexdigest() == original
            workload = records[0].diagnostics['workload_rows']
            references = {r.row['method']:r.diagnostics['reference_rows'] for r in records}
            if previous_workload is not None and references == previous_references:
                assert workload[:len(previous_workload)] == previous_workload
            previous_workload, previous_references = workload, references
            assert len({r.row['fixed_workload_sha256'] for r in records}) == 1
            assert len({r.diagnostics['selected_target_ids_sha256'] for r in records}) == 1
            for record in records:
                row = record.row
                assert row['status'] == 'success', row['reason']
                assert row['completed_count'] == row['attempted_count'] == config.budget(n)
                ledger = json.loads(row['communication_phase_totals_json'])
                assert sum(item['total_bytes'] for item in ledger) == row['total_bytes']
                if row['method'] == 'RACA-HIM':
                    assert not any(item['message_type'] == 'gt_membership' for item in ledger)
                    assert sum(item['payload_bytes'] for item in ledger
                               if item['message_type']=='descriptor_response') == 4*d*config.budget(n)
                rows.append(dict(N=n, d=d, seed=seed, method=row['method'], B=row['B'],
                    total_time_ms=row['total_time_ms'], total_bytes=row['total_bytes'],
                    preparation_time_ms=row['gt_time_ms'], decision_cpu_ms=row['decision_cpu_ms'],
                    verification_time_ms=row['verification_time_ms'],
                    deadline_violations=row['deadline_violations'],
                    gt_candidate_count=record.diagnostics['candidate_count'],
                    resource_fixture=record.diagnostics['resource_fixture'],
                    population_sha256=group['scaling_fixture']['population_sha256'],
                    reference_rows=references[row['method']],
                    workload_sha256=row['fixed_workload_sha256']))
            print(f'Validation only: N={n}, d={d}; all four methods completed B={config.budget(n)}.', flush=True)
    assert dataset_snapshot(cfg.DATASET_ROOT) == before
    monotonic = {}
    for method in cfg.METHODS:
        own = [row for row in rows if row['method']==method]
        monotonic[method] = {metric:all(a[metric] <= b[metric] for a,b in zip(own,own[1:]))
                             for metric in ('total_time_ms','total_bytes')}
    report = dict(scope='One d=30 real-data sweep in memory for code validation only; '
                  'no formal full-grid run, result files or publication figures',
                  implementation_version=cfg.OVERHEAD_IMPLEMENTATION_VERSION,
                  all_four_methods_complete_B=True, exact_byte_conservation=True,
                  raw_X_and_dataset_unchanged=True, unchanged_reference_workloads_nested=True,
                  kernel_calibration=calibration.to_dict(), observed_monotonicity=monotonic, rows=rows)
    path = ROOT/'reports'/'02_overhead_v4_validation.json'
    path.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(monotonic,ensure_ascii=False,indent=2))
    print(f'Code validation report: {path}',flush=True)


if __name__ == '__main__':
    validate()
