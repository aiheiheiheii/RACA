"""Single-seed paired core-overhead runs with compact atomic checkpoints.

Algorithm execution belongs to overhead_simulator. This module only loads one
descriptor instance, persists four method records, and summarizes their exact
core-phase counters. O1 and O2 share every underlying simulation.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import ast
from dataclasses import asdict, replace
from datetime import datetime, timezone
import gzip
import hashlib
import json
import math
from pathlib import Path
import time
import traceback

import numpy as np
from scipy.stats import t
from threadpoolctl import threadpool_limits

from .overhead_population import load_scaling_group as load_group
from experiments.common.method_names import RACA_HIM
from experiments.common.reproducibility import (
    dataset_snapshot, environment_metadata, verify_manifest,
)
from .overhead_storage import (
    atomic_write_json, atomic_write_csv, atomic_write_text,
    output_directory_lock, retry_io,
)
from .overhead_identity import computation_source_fingerprint, verified_checkpoint_sources, checkpoint_identity_source
from .overhead_run_setup import (
    calibration_policy, calibration_snapshot_sha256,
    validate_calibration_snapshot, build_calibration_snapshot,
)
from .config_overhead import (
    DATASET_ROOT, PROJECT_ROOT, RESULT_ROOT, N_VALUES, D_PANEL_VALUES, SEEDS,
    METHODS, EVALUATION_MODES, MAIN_EVALUATION_MODE, OVERHEAD_IMPLEMENTATION_VERSION,
    OverheadConfig, configurations,
)

RUN_FIELDS = (
    "implementation_version", "run_fingerprint", "configuration_id", "seed", "N", "d", "cr", "H",
    "method", "evaluation_mode", "B", "selected_count", "attempted_count", "completed_count",
    "gt_success", "gt_applicable", "deadline_violations", "gt_time_ms", "decision_cpu_ms",
    "verification_time_ms", "total_time_ms", "gt_bytes", "verification_bytes", "control_bytes",
    "total_bytes", "message_count", "fixed_workload_sha256", "status", "reason", "diagnostics_json",
    "within_deadline_count", "deadline_met", "core_phase_counts_json",
)


def _json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Unsupported serialized value: {type(value).__name__}")


def _atomic_json(path: Path, payload, compressed=False):
    """A stopped run can never expose a partially written usable checkpoint."""
    atomic_write_json(path,payload,compressed=compressed,default=_json_default)


def write_csv(path, rows, fieldnames=None):
    records=[{key:_csv_value(value) for key,value in row.items()} for row in rows]
    atomic_write_csv(path,records,fieldnames)


def _configuration_id(n, d, seed):
    return f"N{n}_d{d}_seed{seed}"


def source_fingerprint():
    """Hash all local algorithm/runner/style source used by this experiment."""
    sources = set((PROJECT_ROOT / "experiments" / "02_overhead").glob("*.py"))
    sources.update((PROJECT_ROOT / "experiments" / "common").glob("*.py"))
    sources.update((PROJECT_ROOT / "experiments" / "01_detection").glob("*.py"))
    sources.add(DATASET_ROOT / "load_dataset.py")
    digest = hashlib.sha256()
    records = {}
    for path in sorted(sources, key=lambda item: item.as_posix()):
        relative = path.relative_to(PROJECT_ROOT).as_posix()
        file_digest = hashlib.sha256(path.read_bytes()).hexdigest()
        records[relative] = file_digest
        digest.update(relative.encode("utf-8") + b"\0" + file_digest.encode("ascii") + b"\n")
    return digest.hexdigest(), records


def run_identity(config, snapshot, source_hash, environment, *,
                 kernel_calibration_sha256=None, kernel_calibration_policy=None):
    material = {
        "implementation_version": OVERHEAD_IMPLEMENTATION_VERSION,
        "config": asdict(config), "grid": configurations(), "dataset": snapshot,
        "source_sha256": source_hash,
        # Measured decision CPU runtimes from different environments must not mix.
        "environment": environment, "threadpool_limit": 1,
    }
    if kernel_calibration_sha256 is not None or kernel_calibration_policy is not None:
        if kernel_calibration_sha256 is None or kernel_calibration_policy is None:
            raise ValueError("Calibration snapshot hash and policy must be supplied together")
        material["kernel_calibration_sha256"] = kernel_calibration_sha256
        material["kernel_calibration_policy"] = kernel_calibration_policy
    encoded = json.dumps(material, sort_keys=True, separators=(",", ":"),
                         default=_json_default, allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _orchestration_contract():
    """Bind the entire driver, including input bindings and control flow.

    A reviewed legacy migration permits this specific repair. Later driver
    changes require a fresh compatibility audit, even if intended as IO fixes.
    """
    tree = ast.parse(Path(__file__).read_text(encoding='utf-8-sig'))
    return hashlib.sha256(ast.dump(tree, include_attributes=False).encode()).hexdigest()


def _checkpoint(path, fingerprint, n, d, seed, compatible_fingerprints=()):
    if not path.exists():
        return None
    try:
        def read():
            with gzip.open(path,"rt",encoding="utf-8") as stream:
                return json.load(stream)
        record=retry_io(read,description=f'Read checkpoint {path}')
        origin=record.get('run_fingerprint')
        results = record["results"]
        expected = {(method, mode) for method in METHODS for mode in EVALUATION_MODES}
        actual = {(result["row"]["method"], result["row"]["evaluation_mode"]) for result in results}
        if (origin not in {fingerprint,*compatible_fingerprints} or not record.get("complete")
                or record.get("configuration") != {"N": n, "d": d, "seed": seed}
                or len(results) != len(expected) or actual != expected):
            return None
        for result in results:
            row = result["row"]
            if (row.get("implementation_version") != OVERHEAD_IMPLEMENTATION_VERSION
                    or row.get("run_fingerprint") != origin
                    or row.get("N") != n or row.get("d") != d or row.get("seed") != seed
                    or row.get("configuration_id") != _configuration_id(n, d, seed)
                    or not all(isinstance(result.get(key), list) for key in ("messages", "runtime", "tasks"))
                    or not isinstance(result.get("diagnostics"), dict)):
                return None
        if _systemic_execution_error(results):
            # Exceptions for every method are evidence of a shared program or
            # loading error, not completed measurements to skip forever.
            return None
        if origin != fingerprint:
            # Normalize an independently verified compatible record in memory
            # only. Its on-disk bytes/fingerprint remain original audit evidence.
            record['checkpoint_origin_fingerprint']=origin
            record['run_fingerprint']=fingerprint
            for result in results:
                row=result['row']
                row['checkpoint_origin_fingerprint']=origin
                row['run_fingerprint']=fingerprint
        return record
    except (FileNotFoundError, gzip.BadGzipFile, EOFError, ValueError, KeyError, TypeError):
        return None
    # Other I/O errors propagate: an inaccessible checkpoint must not be
    # mistaken for an invalid measurement and overwritten by recomputation.


def _result_record(result, n, d, seed, config, fingerprint):
    row = dict(result.row)
    context = {"configuration_id": _configuration_id(n, d, seed), "seed": seed, "N": n, "d": d,
               "method": row["method"], "evaluation_mode": row["evaluation_mode"]}
    row.update({**context, "implementation_version": OVERHEAD_IMPLEMENTATION_VERSION,
                "run_fingerprint": fingerprint, "cr": config.corruption_rate_percent, "H": config.H})
    if row["method"] not in METHODS or row["evaluation_mode"] not in EVALUATION_MODES:
        raise ValueError("Simulator returned an undeclared method or evaluation mode")
    diagnostics = dict(result.diagnostics)
    row["diagnostics_json"] = json.dumps(diagnostics, ensure_ascii=False, default=_json_default,
                                          separators=(",", ":"))
    # Exact phase/count totals remain in the row. Per-message, CPU-operation,
    # and task objects are deliberately not serialized in the core protocol.
    return {"row": row, "messages": [], "runtime": [], "tasks": [],
            "diagnostics": diagnostics}


def _failure_records(n, d, seed, config, fingerprint, exc):
    """Keep whole-configuration failures visible rather than losing their seeds."""
    reason = f"{type(exc).__name__}: {exc}"
    evidence = {"exception_traceback": traceback.format_exc()}
    results = []
    for mode in EVALUATION_MODES:
        for method in METHODS:
            row = {field: None for field in RUN_FIELDS}
            row.update({"implementation_version": OVERHEAD_IMPLEMENTATION_VERSION,
                        "run_fingerprint": fingerprint, "configuration_id": _configuration_id(n, d, seed),
                        "seed": seed, "N": n, "d": d, "cr": config.corruption_rate_percent, "H": config.H,
                        "method": method, "evaluation_mode": mode, "B": config.budget(n),
                        "selected_count": 0, "attempted_count": 0, "completed_count": 0,
                        "status": "failure", "reason": reason,
                        "diagnostics_json": json.dumps(evidence, ensure_ascii=False)})
            results.append({"row": row, "messages": [], "runtime": [], "tasks": [], "diagnostics": evidence})
    return results


def _number(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def _known_count_total(trials, field):
    values = [_number(row.get(field)) for row in trials]
    return sum(value for value in values if math.isfinite(value))


def _workload_hash(row):
    return (row.get("fixed_workload_sha256") or row.get("target_ids_sha256")
            or row.get("workload_sha256"))


def _complete_fixed(row):
    budget = _number(row.get("B"))
    return (row.get("status") == "success" and math.isfinite(budget) and budget > 0
            and all(_number(row.get(field)) == budget for field in
                    ("selected_count", "attempted_count", "completed_count")))


def summarize_runs(rows, fingerprint):
    """Core observations from the same paired seed; failures remain counted.

    The main fixed-workload estimand includes a seed only when every method
    completed its same predeclared B targets. Deadline exceedances remain
    separately recorded; they do not truncate the common overhead workload.
    Native-budget summaries are diagnostic and have no equal-workload claim.
    """
    grouped = defaultdict(list)
    paired = defaultdict(list)
    for row in rows:
        grouped[(int(row["N"]), int(row["d"]), row["method"], row["evaluation_mode"])].append(row)
        if row["evaluation_mode"] == MAIN_EVALUATION_MODE:
            paired[(int(row["N"]), int(row["d"]), int(row["seed"]))].append(row)
    eligible = set()
    pairing_reasons = Counter()
    for key, trials in paired.items():
        if len(trials) != len(METHODS) or {row["method"] for row in trials} != set(METHODS):
            pairing_reasons["missing_method"] += 1
        elif not all(_complete_fixed(row) for row in trials):
            pairing_reasons["incomplete_or_failed_method"] += 1
        elif len({_number(row["B"]) for row in trials}) != 1:
            pairing_reasons["different_budget"] += 1
        elif any(not _workload_hash(row) for row in trials) or len({_workload_hash(row) for row in trials}) != 1:
            pairing_reasons["missing_or_different_fixed_target_fingerprint"] += 1
        elif not all(math.isfinite(_number(row.get(metric))) and _number(row.get(metric)) >= 0
                     for row in trials for metric in ("total_time_ms", "total_bytes")):
            pairing_reasons["invalid_overhead_metric"] += 1
        else:
            eligible.add(key)
    summaries = {"time": [], "communication": []}
    for n in N_VALUES:
        for d in D_PANEL_VALUES:
            for mode in EVALUATION_MODES:
                for method in METHODS:
                    trials = grouped[(n, d, method, mode)]
                    if mode == MAIN_EVALUATION_MODE:
                        valid = [row for row in trials if (n, d, int(row["seed"])) in eligible]
                        scope = ("single paired core-workload observation; same complete B targets; deadline exceedances included"
                                 if len(SEEDS) == 1 else
                                 "complete paired fixed-workload seeds, common B-target fingerprint")
                    else:
                        valid = [row for row in trials if row.get("status") == "success"
                                 and _number(row.get("completed_count")) > 0]
                        scope = "successful native-budget seeds; different workloads, diagnostics only"
                    base = {"N": n, "d": d, "cr": trials[0]["cr"] if trials else None,
                            "H": trials[0]["H"] if trials else None, "method": method,
                            "evaluation_mode": mode, "implementation_version": OVERHEAD_IMPLEMENTATION_VERSION,
                            "run_fingerprint": fingerprint, "run_count": len(trials),
                            "n_expected": len(SEEDS), "n_missing": len(SEEDS) - len(trials),
                            "n_failed": len(trials) - len(valid),
                            "n_method_failed": sum(row.get("status") != "success" for row in trials),
                            "n_incomplete_workload": sum(not _complete_fixed(row) for row in trials)
                                if mode == MAIN_EVALUATION_MODE else None,
                            "selected_count_total": _known_count_total(trials, "selected_count"),
                            "attempted_count_total": _known_count_total(trials, "attempted_count"),
                            "completed_count_total": _known_count_total(trials, "completed_count"),
                            "selected_count_unknown": sum(not math.isfinite(_number(row.get("selected_count"))) for row in trials),
                            "attempted_count_unknown": sum(not math.isfinite(_number(row.get("attempted_count"))) for row in trials),
                            "completed_count_unknown": sum(not math.isfinite(_number(row.get("completed_count"))) for row in trials),
                            "count_totals_scope": "sum of observed counts only; unknown counts are separate",
                            "deadline_violations_total": sum(_number(row.get("deadline_violations"))
                                for row in trials if math.isfinite(_number(row.get("deadline_violations")))),
                            "status_counts_json": json.dumps(dict(Counter(row["status"] for row in trials)), sort_keys=True),
                            "statistics_scope": scope}
                    for kind, metric, unit in (("time", "total_time_ms", "ms"),
                                               ("communication", "total_bytes", "bytes")):
                        values = np.asarray([_number(row.get(metric)) for row in valid], dtype=float)
                        values = values[np.isfinite(values) & (values >= 0)]
                        count = len(values)
                        mean = float(values[0]) if count == 1 else float(values.mean()) if count else math.nan
                        std = float(values.std(ddof=1)) if count > 1 else math.nan
                        half = float(t.ppf(.975, count - 1) * std / math.sqrt(count)) if count > 1 else math.nan
                        summaries[kind].append({**base, "metric": metric, "unit": unit,
                                                "n_valid": count, "value": mean, "mean": mean, "std": std,
                                                "estimator": "single paired observation" if len(SEEDS) == 1 else "seed mean",
                                                "ci95_lower": mean - half, "ci95_upper": mean + half})
    diagnostics = {"fixed_workload_eligible_paired_seeds": len(eligible),
                   "fixed_workload_excluded_paired_seeds": dict(pairing_reasons),
                   "fixed_workload_seed_pair_count": len(paired),
                   "fixed_workload_expected_seed_pairs": len(configurations()),
                   "seed_policy": f"{len(SEEDS)} paired seed(s) per N/d point",
                   "confidence_intervals": ("not estimated: one paired seed per point; value is the original observation"
                                            if len(SEEDS) == 1 else
                                            "Student-t 95%, seed-level samples; n<2 => undefined interval")}
    return summaries, diagnostics


def _csv_value(value):
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False, default=_json_default, separators=(",", ":"))
    if isinstance(value, np.generic):
        return value.item()
    return value


def materialize_outputs(output, fingerprint, compatible_fingerprints=()):
    """Read each compact checkpoint once; export rows and the two summaries."""
    output = Path(output)
    checkpoint_paths = [(n, d, seed, output / "checkpoints" / f"{_configuration_id(n, d, seed)}.json.gz")
                        for n, d, seed in configurations()]
    rows = []
    diagnostics_rows = []
    print('Reading compact checkpoints and preparing the two saved summaries.',flush=True)
    for index,(n,d,seed,path) in enumerate(checkpoint_paths,1):
        record = _checkpoint(path, fingerprint, n, d, seed,compatible_fingerprints)
        if record is None:
            continue
        for result in record["results"]:
            row = result["row"]
            rows.append(row)
            context = {key: row[key] for key in ("configuration_id", "seed", "N", "d", "method", "evaluation_mode")}
            diagnostics_rows.append({**context, "status": row["status"],
                                     "diagnostics_json": row["diagnostics_json"]})
        if index%15==0:
            print(f'Preparing summaries: {index}/{len(checkpoint_paths)} checkpoints.',flush=True)
    raw = output / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    row_keys = set(RUN_FIELDS) | {key for row in rows for key in row}
    write_csv(raw / "overhead_runs.csv", rows,
              list(RUN_FIELDS) + sorted(row_keys - set(RUN_FIELDS)))
    write_csv(raw / "protocol_diagnostics.csv", diagnostics_rows)
    summaries, statistics_metadata = summarize_runs(rows, fingerprint)
    for kind, records in summaries.items():
        write_csv(output / "summary" / f"{kind}_overhead_summary.csv", records)
    _atomic_json(output / "logs" / "statistics_policy.json", statistics_metadata)
    return rows, summaries, statistics_metadata


def _progress_log(path, payload):
    with retry_io(lambda:path.open('a',encoding='utf-8'),description=f'Open progress log {path}') as stream:
        stream.write(json.dumps(payload, ensure_ascii=False, default=_json_default) + "\n")
        stream.flush()


def _save_progress_manifest(output,manifest):
    """A volatile progress log must not abort durably saved computations."""
    try:
        _atomic_json(output/'logs'/'run_manifest.json',manifest)
        return True
    except OSError as exc:
        warning=f'{type(exc).__name__}: {exc}'
        warnings=manifest.setdefault('io_warnings',[])
        if len(warnings)<100:
            warnings.append(warning)
        print(f'Progress manifest could not be updated; saved checkpoints are intact. {warning}',flush=True)
        return False


def _systemic_execution_error(results):
    """Stop shared execution errors; genuine protocol deadlines stay recorded."""
    if results and all(result['row'].get('status')=='failure' for result in results):
        return results[0]['row'].get('reason') or 'All methods raised execution exceptions'
    return None


def run_experiment(output_dir=RESULT_ROOT, *, config=None, resume=True, force=False):
    """Own one result root; compatible IO-only repairs retain saved trials."""
    with output_directory_lock(output_dir) as output:
        return _run_experiment(output,config=config,resume=resume,force=force)


def _read_manifest(path):
    if not path.is_file():
        return None
    def read():
        with path.open('r',encoding='utf-8') as stream:
            return json.load(stream)
    try:
        return retry_io(read,description=f'Read run manifest {path}')
    except (json.JSONDecodeError,UnicodeError):
        return None


def _run_experiment(output, *, config=None, resume=True, force=False):
    from .overhead_simulator import simulate_core_configuration
    from .plot_overhead import plot_results
    config=config or OverheadConfig()
    output=Path(output).resolve()
    for name in ('raw','summary','figures','logs','checkpoints'):
        (output/name).mkdir(parents=True,exist_ok=True)
    start=time.perf_counter()
    previous=_read_manifest(output/'logs'/'run_manifest.json')
    before=dataset_snapshot(DATASET_ROOT)
    integrity=verify_manifest(DATASET_ROOT)
    if integrity['missing'] or integrity['mismatch']:
        raise RuntimeError('Dataset manifest verification failed: '+json.dumps(integrity))
    environment=environment_metadata()
    source_hash,sources=source_fingerprint()
    computation_hash,computation_sources=computation_source_fingerprint(PROJECT_ROOT)
    contract_hash=_orchestration_contract()
    manifest={
        'identity_schema':'computation-v1',
        'implementation_version':OVERHEAD_IMPLEMENTATION_VERSION,
        'started_utc':datetime.now(timezone.utc).isoformat(),'status':'running',
        'config':asdict(config),'N_values':N_VALUES,'d_panel_values':D_PANEL_VALUES,'seeds':SEEDS,
        'methods':METHODS,'evaluation_modes':EVALUATION_MODES,'main_evaluation_mode':MAIN_EVALUATION_MODE,
        'grid':configurations(),'expected_configurations':len(configurations()),
        'expected_method_mode_runs':len(configurations())*len(METHODS)*len(EVALUATION_MODES),
        'dataset_manifest':integrity,'dataset_snapshot':before,
        'source_sha256':source_hash,'source_files_sha256':sources,
        'computation_source_sha256':computation_hash,'computation_source_files_sha256':computation_sources,
        'orchestration_contract_sha256':contract_hash,
        'environment':environment,'threadpool_limit':1,'resume':resume,'force':force,
        'measurement_scope':'core online steps on one common fixed B-target workload',
        'checkpoint_payload_schema':'core-summary-v1',
        'trace_policy':'exact phase totals and counts; per-event objects and CSV exports omitted',
        'deadline_scope':'verification stage diagnostics only; complete fixed B tasks are not truncated',
        'confidence_interval_unit':'not estimated; one paired seed per point',
        'executed_configurations_this_invocation':0,'resumed_configurations':0,
    }
    calibration_start=time.perf_counter()
    calibrations=None
    policy=calibration_policy()
    if resume and not force and isinstance(previous,dict):
        try:
            saved_records=previous['kernel_calibrations']
            saved_policy=previous['kernel_calibration_policy']
            candidate_calibrations=validate_calibration_snapshot(saved_records,saved_policy,SEEDS,D_PANEL_VALUES)
            candidate_hash=calibration_snapshot_sha256(saved_records,saved_policy)
            candidate_context=dict(manifest,kernel_calibrations=saved_records,
                                   kernel_calibration_policy=saved_policy,
                                   kernel_calibration_sha256=candidate_hash)
            verified,_=verified_checkpoint_sources(previous,candidate_context)
            if verified:
                calibrations=candidate_calibrations
                records=saved_records
        except (KeyError,TypeError,ValueError):
            # A malformed or mismatched generation cannot supply unit costs.
            # A new snapshot also gives the new trial generation a new identity.
            pass
    reused_calibration=calibrations is not None
    if calibrations is None:
        with threadpool_limits(limits=1):
            records,calibrations=build_calibration_snapshot(load_group,DATASET_ROOT,config,SEEDS,D_PANEL_VALUES)
    calibration_hash=calibration_snapshot_sha256(records,policy)
    fingerprint=run_identity(config,before,checkpoint_identity_source(computation_hash,contract_hash),environment,
                             kernel_calibration_sha256=calibration_hash,kernel_calibration_policy=policy)
    manifest.update(run_fingerprint=fingerprint,kernel_calibrations=records,
                    kernel_calibration_policy=policy,kernel_calibration_sha256=calibration_hash,
                    kernel_calibration_reused=reused_calibration,
                    kernel_calibration_setup_wall_seconds=time.perf_counter()-calibration_start,
                    derived_outputs={'state':'pending','run_fingerprint':fingerprint})
    compatible,rejected=verified_checkpoint_sources(previous,manifest) if resume and not force else ([],{})
    manifest['compatible_checkpoint_sources']=compatible
    manifest['checkpoint_compatibility_rejections']=rejected
    compatible_fingerprints=tuple(source['run_fingerprint'] for source in compatible)
    # Persist identity before any computation: a log may lag counts, but it
    # must carry enough provenance to verify every saved trial after a crash.
    _atomic_json(output/'logs'/'run_manifest.json',manifest)
    print(f'Group 02 core overhead: {len(configurations())} paired configurations, '
          f'{len(SEEDS)} seed per point, {len(EVALUATION_MODES)} fixed-workload mode, '
          f'{len(METHODS)} methods.',flush=True)
    print('Effective configuration:\n'+json.dumps(asdict(config),indent=2,ensure_ascii=False),flush=True)
    print(f'Verification deadline diagnostic = {config.T_round_max_ms:g} ms; '
          f'all fixed B tasks execute; output = {output}.',flush=True)
    print(f'Computation identity: {fingerprint}',flush=True)
    print('Shared kernel calibration: '+('verified saved snapshot reused' if reused_calibration else
          'one declared calibration batch per seed/d; reused across all N and methods')+
          f'; snapshot SHA256 = {calibration_hash}.',flush=True)
    if compatible_fingerprints:
        print('Verified compatible prior checkpoint identities: '+', '.join(compatible_fingerprints),flush=True)
    if rejected:
        print('Rejected prior identities: '+json.dumps(rejected,ensure_ascii=False),flush=True)
    fatal_reason=None
    current_phase='simulation'
    try:
        with threadpool_limits(limits=1):
            for index,(n,d,seed) in enumerate(configurations(),1):
                path=output/'checkpoints'/f'{_configuration_id(n,d,seed)}.json.gz'
                if resume and not force and _checkpoint(path,fingerprint,n,d,seed,compatible_fingerprints) is not None:
                    manifest['resumed_configurations']+=1
                    if index==1 or index%5==0:
                        print(f'[{index}/{len(configurations())}] Validated/resumed N={n}, d={d}, seed={seed}.',flush=True)
                    continue
                tick=time.perf_counter()
                point_config=replace(config,dim=d)
                try:
                    group=load_group(dataset_root=DATASET_ROOT,n=n,corruption_rate_percent=config.corruption_rate_percent,
                                     dim=d,H=config.H,seed=seed)
                    results=[_result_record(result,n,d,seed,point_config,fingerprint)
                             for result in simulate_core_configuration(group,n,d,seed,point_config,
                                                                        calibration=calibrations[(seed,d)])]
                    expected={(method,mode) for method in METHODS for mode in EVALUATION_MODES}
                    actual={(result['row']['method'],result['row']['evaluation_mode']) for result in results}
                    if len(results)!=len(expected) or actual!=expected:
                        raise ValueError('Simulator must return every declared method/mode, including failures')
                except Exception as exc:
                    results=_failure_records(n,d,seed,point_config,fingerprint,exc)
                    print(f'[{index}/{len(configurations())}] N={n}, d={d}, seed={seed}: failed ({exc}).',flush=True)
                elapsed=time.perf_counter()-tick
                # A checkpoint commit is critical. It precedes all volatile
                # progress logs, and any failure leaves its recovery file intact.
                _atomic_json(path,{'run_fingerprint':fingerprint,'complete':True,
                                   'checkpoint_payload_schema':'core-summary-v1',
                                   'configuration':{'N':n,'d':d,'seed':seed},
                                   'wall_seconds':elapsed,'results':results},compressed=True)
                manifest['executed_configurations_this_invocation']+=1
                statuses=dict(Counter(result['row']['status'] for result in results))
                progress={'configuration_index':index,'N':n,'d':d,'seed':seed,
                          'run_fingerprint':fingerprint,'wall_seconds':elapsed,'statuses':statuses}
                try:
                    _progress_log(output/'logs'/'progress.jsonl',progress)
                except OSError as exc:
                    print(f'Progress append failed; checkpoint {path.name} is already saved: {exc}',flush=True)
                print(f'[{index}/{len(configurations())}] N={n}, d={d}, seed={seed}: {statuses}; {elapsed:.2f} s.',flush=True)
                if index%5==0:
                    _save_progress_manifest(output,manifest)
                fatal_reason=_systemic_execution_error(results)
                if fatal_reason:
                    break
        saved=manifest['resumed_configurations']+manifest['executed_configurations_this_invocation']
        manifest.update(saved_configurations=saved,missing_configurations=len(configurations())-saved,
                        saved_method_mode_runs=saved*len(METHODS)*len(EVALUATION_MODES),
                        elapsed_wall_seconds=time.perf_counter()-start)
        manifest['dataset_unchanged']=before==dataset_snapshot(DATASET_ROOT)
        if not manifest['dataset_unchanged']:
            raise RuntimeError('Dataset changed during execution; saved evidence must be audited')
        if fatal_reason:
            manifest['status']='execution_failed'
            manifest['execution_error']=fatal_reason
            _save_progress_manifest(output,manifest)
            raise RuntimeError(f'{fatal_reason}. Saved checkpoints are intact; no bulk export was attempted.')
        current_phase='materialization'
        manifest['status']='exporting'
        manifest['derived_outputs']={'state':'building','run_fingerprint':fingerprint}
        _atomic_json(output/'logs'/'run_manifest.json',manifest)
        print('Simulation finished. Exporting compact rows and two summaries, then drawing O1/O2.',flush=True)
        rows,summaries,statistics_metadata=materialize_outputs(output,fingerprint,compatible_fingerprints)
        records_per_configuration=len(METHODS)*len(EVALUATION_MODES)
        manifest['saved_configurations']=len(rows)//records_per_configuration
        manifest['missing_configurations']=len(configurations())-len(rows)//records_per_configuration
        manifest['saved_method_mode_runs']=len(rows)
        manifest['statuses']=dict(Counter(row['status'] for row in rows))
        manifest['statistics']=statistics_metadata
        manifest['derived_outputs']={'state':'summaries_ready','run_fingerprint':fingerprint}
        _atomic_json(output/'logs'/'run_manifest.json',manifest)
        current_phase='plotting'
        metadata=plot_results(output)
        _atomic_json(output/'logs'/'plot_style.json',metadata)
        manifest['derived_outputs']['state']='figures_ready'
        manifest['figures_generated']=metadata['figures_generated']
        manifest['status']='complete' if not manifest['missing_configurations'] else 'incomplete'
        manifest['elapsed_wall_seconds']=time.perf_counter()-start
        manifest['finished_utc']=datetime.now(timezone.utc).isoformat()
        current_phase='report'
        generate_report(output,rows,manifest,metadata)
        manifest['derived_outputs']['state']='complete'
        _atomic_json(output/'logs'/'run_manifest.json',manifest)
        print(f'Saved {len(rows)} method/mode records. Paired fixed-workload seeds: '
              f"{statistics_metadata['fixed_workload_eligible_paired_seeds']}/{len(configurations())}.",flush=True)
        print('Figures generated: '+str(metadata['figures_generated']),flush=True)
        return manifest
    except KeyboardInterrupt:
        manifest['status']='interrupted'
        manifest['interrupted_phase']=current_phase
        manifest['dataset_unchanged']=before==dataset_snapshot(DATASET_ROOT)
        manifest['saved_configurations']=manifest['resumed_configurations']+manifest['executed_configurations_this_invocation']
        manifest['missing_configurations']=len(configurations())-manifest['saved_configurations']
        _save_progress_manifest(output,manifest)
        print('Interrupted. Checkpoints retained; no bulk export is started during exception cleanup. Run normally to resume.',flush=True)
        return manifest
    except Exception as exc:
        # Preserve the original exception. Cleanup must not attempt several GB
        # of exports or throw a second file error over the useful traceback.
        manifest['status']='output_io_failed' if isinstance(exc,OSError) else 'execution_failed'
        manifest['failed_phase']=current_phase
        manifest['last_exception']=f'{type(exc).__name__}: {exc}'
        manifest['exception_traceback']=traceback.format_exc()
        manifest['saved_configurations']=manifest['resumed_configurations']+manifest['executed_configurations_this_invocation']
        manifest['missing_configurations']=len(configurations())-manifest['saved_configurations']
        _save_progress_manifest(output,manifest)
        print(f'Failed during {current_phase}. Saved checkpoints are retained; resume after resolving the reported error.',flush=True)
        raise


def generate_report(output, rows, manifest, plot_metadata):
    """Report single-seed observations and the scope of the core protocol."""
    lines = ["# 第二组核心步骤 Overhead 实际运行报告", "",
             f"实现版本：`{OVERHEAD_IMPLEMENTATION_VERSION}`。运行身份：`{manifest['run_fingerprint']}`。",
             f"配置记录状态：`{manifest['status']}`；已保存 {manifest['saved_configurations']}/{len(configurations())} 个配置，"
             f"尚未运行 {manifest['missing_configurations']} 个配置。此状态描述记录是否齐全，不表示全部协议成功。", "",
             f"数据集运行前后完全一致：{manifest['dataset_unchanged']}。",
             f"完整网格为 N={list(N_VALUES)}、d={list(D_PANEL_VALUES)}、seed={list(SEEDS)}，cr=3%、H=.5。",
             "默认每个 N/d 仅执行一次配对试验；每点四方法执行相同 B=ceil(.1(N−25)) 个固定目标。",
             f"{len(configurations())} 个配置共 {len(configurations())*len(METHODS)*len(EVALUATION_MODES)} 条方法记录；不运行 native_budget，不做多轮平均。",
             "验证阶段 800 ms 只作 deadline 诊断；所有 B 任务执行到最终结果，超时任务计数单独保存。", "",
             "## 实际工作量", "",
             "| 方法 | 模式 | 已记录 trials | selected 总数 | attempted 总数 | completed 总数 | completed 未知 trials | 状态 |",
             "|---|---|---:|---:|---:|---:|---:|---|"]
    for mode in EVALUATION_MODES:
        for method in METHODS:
            subset = [r for r in rows if r['method']==method and r['evaluation_mode']==mode]
            totals = [_known_count_total(subset, field) for field in ('selected_count','attempted_count','completed_count')]
            unknown = sum(not math.isfinite(_number(r.get('completed_count'))) for r in subset)
            states = json.dumps(dict(Counter(r['status'] for r in subset)),ensure_ascii=False)
            lines.append(f"| {method} | {mode} | {len(subset)} | {totals[0]:g} | {totals[1]:g} | {totals[2]:g} | {unknown} | {states} |")
    statistics = manifest['statistics']
    lines.extend(["", "计数总数只加已观测记录；异常中断后的未知完成数不能解释为零。core_phase_counts_json 保存各阶段精确消息、字节和操作汇总。", "",
                  "## 时间、通信和可比性", "",
                  "O1 = 实测核心准备 CPU + 模拟准备通信时延 + 实测核心决策 CPU + 模拟核心验证完成时延 + 实测必要历史更新 CPU。",
                  "数据加载、离线状态构造和文件输出不属于核心在线时间；各方法采用同一记账边界。",
                  "同一主 verifier 的完整任务串行，其他主 verifier 并行；CTCV 按批次执行原有块核验和聚合流程。",
                  "O2 等于核心逻辑消息精确字节总数：gt_bytes 和 verification_bytes 只计 payload，control_bytes 计两阶段控制字段，三项互斥且相加为 total_bytes。",
                  "网络带宽改变传输时间，不改变消息字节数。KB=1024 bytes；MB=1024² bytes。", "",
                  "验证 CPU 单位费用由每个 seed/d 的完整5000实例人口校准一次；四方法及各N共同复用同一不可变标尺。",
                  "续跑只复用严格核验代码、配置、数据、环境、驱动契约和snapshot身份后的原校准，不跨N重新计时或改变方法倍率。",
                  f"校准snapshot SHA256：`{manifest['kernel_calibration_sha256']}`；本次复用已有标尺：{manifest['kernel_calibration_reused']}。", "",
                  f"进入主图的完整配对点：{statistics['fixed_workload_eligible_paired_seeds']}/{len(configurations())}。",
                  "被排除配对 seed 的原因：`"+json.dumps(statistics['fixed_workload_excluded_paired_seeds'],ensure_ascii=False)+"`。",
                  "只有四方法都得到同一 B 个目标最终结果、目标指纹一致且时间/字节有效才进入主图；超时不会改变共同任务量。",
                  "value 与兼容列 mean 都保存单次原始观测，std/CI 为未定义；本组没有重复测量，不能作为均值或置信区间解释。",
                  "固定任务模式不评估原生 subset/prefix 的选择优势，比较的是同一任务量下的核心流程开销。", "",
                  "## 方法与适配限制", "",
                  f"{RACA_HIM} 保留多数支持、必要扩展、风险/联合分配、描述符验证及历史更新；GT采用一次候选上传和协调器增量比较的公开适配。",
                  "各方法共用批量assignment/result控制规则；CTCV完整分块排列由20字节seed请求重建，保留实际hash/modulo、k+1 collectors及完整CR/reports。",
                  "DVA-P、I-EDI、CTCV 采用同一核心在线边界，保留各自分配、标签/块核验和必要消息。",
                  "本组采用公开适配流程及中立 vendor 内容代理；数据集没有 authenticated vendor 原内容。",
                  "全链广播、区块链共识、签名和认证封装未模拟；实际SHA256/cosine等微内核校准后按共享节点资源倍率收费，并非完整密码学部署实测。",
                  "完整来源和不能继承的安全/长期保证见 reports/02_overhead_baseline_adaptation.md。", "",
                  "## 保存结果", "",
                  f"输出目录：`{Path(output).resolve()}`。", "",
                  "- raw/overhead_runs.csv：180 条单次核心步骤观测及精确阶段汇总。",
                  "- raw/protocol_diagnostics.csv：状态、deadline 诊断与核心协议计数。",
                  "- summary/time_overhead_summary.csv、communication_overhead_summary.csv：单次配对观测。",
                  "- checkpoints/：紧凑原子保存配置；按新版本代码/配置/数据/环境指纹续跑，不复用 v2 结果。",
                  "- figures/Fig_O1_time_overhead 与 Fig_O2_communication_overhead：仅两幅图，各保存 PDF/PNG。",
                  "不导出逐消息、逐CPU操作或逐任务大体积轨迹。", "",
                  "本次实际生成的 Figure：`"+json.dumps(plot_metadata.get('figures_generated',[]))+"`。",
                  "没有有效配对样本时不补画曲线；旧同名文件不证明当前运行生成了图片。图片需要人工打开后确认裁剪和可读性；代码未冒充视觉检查。"])
    text = '\n'.join(lines)+'\n'
    atomic_write_text(Path(output)/'report.md',text)
    if Path(output).resolve() == RESULT_ROOT.resolve():
        atomic_write_text(PROJECT_ROOT/'reports'/'02_overhead_report.md',text)
