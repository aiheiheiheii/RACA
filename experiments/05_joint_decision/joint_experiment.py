"""Complete formal grid with common exogenous inputs and atomic trial resume.

Importing this module performs no experiment. A checkpoint commits all five
outcomes together, including common GT failure and individual method errors.
"""
from __future__ import annotations

from datetime import datetime, timezone
import importlib
import json
import logging
import math
import sys
from time import perf_counter
import traceback
import uuid

import numpy as np
from threadpoolctl import threadpool_info, threadpool_limits

from experiments.common.reproducibility import environment_metadata
from . import config_joint_decision as cfg
from .joint_inputs import prepare_trial_inputs, resource_cost_components_for_sigma
from .joint_methods import run_method
from .joint_objective import JointProblem, evaluate_joint_objective
from .joint_storage import (
    DIAGNOSTIC_COLUMNS, archive_previous_run, atomic_json, checkpoint_path, dataset_identity, digest_json,
    export_checkpoints, file_identity, read_checkpoint, sha256_file, source_identity,
    trial_key, validate_saved_decision, write_checkpoint,
)

IO_PATCH_VERSION = "windows-sharing-retry-stdout-v1"
# Exact, audited predecessor of this I/O-only patch. Other algorithm or source
# changes still require a fresh run. Preserve checkpoint scientific identity.
IO_PATCH_PREDECESSOR_CODE_VERSION = "7394ebcd142afdcb6a9f79aa11c8eecfe2d6436c058f098049eeaa142c448691"
IO_PATCH_PREDECESSOR_HASHES = {
    "experiments/05_joint_decision/joint_experiment.py": "a124e0521caf41a0bc933a803a695f459f0ec62a70d4944a7124ae73a0c168f3",
    "experiments/05_joint_decision/joint_storage.py": "7b3d78ceeecff37d6314f6afbc7c91c37b1e143d066525d261205c3766dedc61",
    "experiments/05_joint_decision/validate_joint_decision.py": "3b9160ea0d2a21f3aeab48162b2c6ad444adc879e6048022a0f2fb4274e1bdc9",
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _jsonable(value):
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _resume_io_patch(saved: dict, identity: dict, source: dict) -> dict | None:
    """Allow this exact I/O repair once, without relabelling old trial results.

    Dataset, scientific protocol, algorithm and numerical environment must all
    match. After migration, only the exact recorded runtime source is accepted;
    this is not a permanent whitelist for arbitrary later edits to I/O files.
    """
    if any(digest_json(saved.get(field)) != digest_json(identity[field]) for field in (
            "dataset_fingerprint", "protocol", "algorithm_version", "numerical_environment")):
        return None
    previous_hashes = saved.get("source_sha256", {})
    current_hashes = source.get("source_sha256", {})
    history = saved.get("runtime_patch_history", [])
    if history:
        latest = history[-1]
        if (latest.get("patch_version") == IO_PATCH_VERSION
                and latest.get("runtime_source_sha256") == current_hashes
                and latest.get("runtime_code_version") == source["code_version"]
                and latest.get("runtime_run_fingerprint") == digest_json(identity)):
            return saved
        return None
    if (saved.get("code_version") != IO_PATCH_PREDECESSOR_CODE_VERSION
            or digest_json(previous_hashes) != saved["code_version"]
            or set(previous_hashes) != set(current_hashes)
            or any(previous_hashes.get(name) != expected for name, expected in IO_PATCH_PREDECESSOR_HASHES.items())):
        return None
    changed = sorted(name for name in current_hashes if current_hashes[name] != previous_hashes[name])
    if not changed or not set(changed).issubset(IO_PATCH_PREDECESSOR_HASHES):
        return None
    saved["runtime_patch_history"] = [{
        "patch_version": IO_PATCH_VERSION, "applied_at_utc": _now(),
        "runtime_code_version": source["code_version"],
        "runtime_source_sha256": current_hashes,
        "runtime_run_fingerprint": digest_json(identity),
        "changed_source_files": changed,
        "scientific_identity_policy": "retain original run_id/code_version/source_sha256/run_fingerprint",
    }]
    return saved


def _base_record(manifest, n, seed, beta, sigma, shared) -> dict:
    gt = shared.ground_truth
    size = len(shared.target_rows) if gt.success else None
    budget = math.ceil(beta * size) if size is not None else None
    return {
        **{k: manifest[k] for k in ("run_id", "code_version", "algorithm_version", "dataset_fingerprint")},
        "experiment": cfg.EXPERIMENT, "trial_key": trial_key(n, seed, beta, sigma),
        "seed": seed, "N": n, "cr": cfg.CORRUPTION_RATE_PERCENT, "d": cfg.D, "H": cfg.H,
        "beta_B": beta, "sigma": sigma, "K": cfg.K, "M": cfg.M, "B": budget,
        "gt_success": bool(gt.success), "gt_size": len(gt.selected_indices),
        "gt_final_K": int(gt.expansion_trace[-1].candidate_size) if gt.expansion_trace else None,
        "gt_expansion_count": max(0, len(gt.expansion_trace) - 1),
        "target_pool_size": size, "descriptor_fingerprint": shared.descriptor_fingerprint,
        "deadline_ms": cfg.T_ROUND_MAX_MS, "lambda_value": cfg.LAMBDA,
        "selected_count": None, "completed_count": None, "risk_reward": None,
        "normalized_risk_reward": None, "makespan_ms": None, "verification_makespan_ms": None,
        "normalized_delay_penalty": None, "joint_utility": None, "deadline_satisfied": None,
        "budget_utilization": None, "unused_budget": None, "feasible": False,
        "status": "pending", "failure_reason": "", "algorithm_seconds": None,
    }


def _compact_decision(evaluation, method_result) -> dict:
    return _jsonable({
        "columns": [[task[field] for task in evaluation.assignment_diagnostics]
                    for field in DIAGNOSTIC_COLUMNS],
        "verifier_loads": evaluation.verifier_loads,
        "proposed_assignments": method_result.assignments,
        "method_diagnostics": method_result.diagnostics,
    })


def _execute_trial(manifest, shared, n, seed, beta, sigma, costs, logger) -> dict:
    base = _base_record(manifest, n, seed, beta, sigma, shared)
    evidence = _jsonable(shared.to_manifest())
    evidence["gt_verifier_ids"] = shared.verifier_ids.tolist()
    evidence["target_pool_size"] = len(shared.target_rows) if shared.ground_truth.success else None
    payload = {"trial_key": base["trial_key"], "run_fingerprint": manifest["run_fingerprint"],
               "shared": evidence, "records": [], "decisions": {}}
    if not shared.ground_truth.success:
        payload["records"] = [{**base, "method": m, "status": "gt_unavailable",
                               "failure_reason": shared.ground_truth.reason} for m in cfg.METHODS]
        return payload
    problem = JointProblem(
        target_ids=shared.target_ids, verifier_ids=shared.verifier_ids,
        risk_scores=shared.history.risk_scores[shared.target_rows], costs=costs.total_ms,
        budget=base["B"], lambda_value=cfg.LAMBDA, deadline_ms=cfg.T_ROUND_MAX_MS,
        compute_costs=costs.compute_ms, transfer_costs=costs.transfer_ms,
        network_costs=costs.network_ms,
    )
    evidence["resource_costs"] = costs.to_manifest()
    for method in cfg.METHODS:
        row = {**base, "method": method}
        start = perf_counter()
        try:
            result = run_method(method, problem, seed=shared.streams["random_baseline_seed"])
            evaluation = evaluate_joint_objective(problem, result.assignments)
            row.update(_jsonable(evaluation.to_dict()))
            row["status"] = "success" if row["feasible"] else evaluation.status
            row["method_status"] = result.status
            row["verifier_loads_json"] = json.dumps(_jsonable(evaluation.verifier_loads), sort_keys=True)
            row["failure_reason"] = row.get("failure_reason", "") or (
                "" if row["feasible"] else str(result.diagnostics.get("reason", result.status)))
            row["algorithm_seconds"] = perf_counter() - start
            row["budget_utilization"] = row["selected_count"] / base["B"] if base["B"] else None
            row["unused_budget"] = base["B"] - row["selected_count"] if base["B"] is not None else None
            payload["decisions"][method] = _compact_decision(evaluation, result)
            if row["feasible"]:
                validate_saved_decision(row, payload["decisions"][method], evidence)
        except Exception as error:
            row = {**base, "method": method, "status": "algorithm_exception",
                   "failure_reason": f"{type(error).__name__}: {error}",
                   "algorithm_seconds": perf_counter() - start}
            payload["decisions"][method] = {"columns": [[] for _ in DIAGNOSTIC_COLUMNS],
                                            "verifier_loads": {}, "traceback": traceback.format_exc()}
            logger.exception("Method failed: %s %s", base["trial_key"], method)
        payload["records"].append(row)
    return _jsonable(payload)


def _prepare_manifest(root, force) -> dict:
    source, dataset = source_identity(), dataset_identity()
    environment = environment_metadata()
    numerical_environment = {
        "python": environment["python"], "platform": environment["platform"],
        "packages": {key: environment["packages"][key] for key in ("numpy", "scipy")},
        "blas": [{k: entry.get(k) for k in ("internal_api", "version", "threading_layer", "architecture")}
                 for entry in threadpool_info()], "thread_limit": 1,
    }
    identity = {"code_version": source["code_version"],
                "dataset_fingerprint": dataset["dataset_fingerprint"],
                "protocol": cfg.protocol_metadata(), "algorithm_version": cfg.ALGORITHM_VERSION,
                "numerical_environment": numerical_environment}
    fingerprint = digest_json(identity)
    path = root / "logs" / "run_manifest.json"
    previous_run = None
    if path.exists():
        saved = json.loads(path.read_text(encoding="utf-8-sig"))
        if not force and saved.get("run_fingerprint") == fingerprint:
            return saved
        if not force:
            patched = _resume_io_patch(saved, identity, source)
            if patched is not None:
                print(f"Group 05: I/O-only repair; resuming {len(saved.get('checkpoints', {}))} "
                      "saved trials under their original scientific identity.", flush=True)
                return patched
        changed = [field for field, value in identity.items()
                   if digest_json(saved.get(field)) != digest_json(value)]
        print("Group 05: archiving previous results before starting a new complete run "
              f"({', '.join(changed) if changed else 'explicit --force'}).", flush=True)
        previous_run = archive_previous_run(root, saved, "forced_restart" if force else "identity_changed")
        previous_run["changed_identity_fields"] = changed
    return {
        **source, **dataset, **identity, "run_fingerprint": fingerprint,
        "run_id": uuid.uuid4().hex, "started_at_utc": _now(), "status": "initialized",
        "methods": list(cfg.METHODS), "seeds": list(cfg.SEEDS),
        "summary_policy": cfg.SUMMARY_POLICY,
        "environment": {**environment,
                        "rng": "PCG64; dataset_seed=trial seed; other seeds=SeedSequence([trial_seed, stream_code])",
                        "stream_codes": {"history": 5101, "resource": 5102,
                                         "temporary_ids": 5103, "random_baseline": 5104}},
        "checkpoints": {}, "files": {}, "completed_trials": 0,
        "previous_run": previous_run,
        "raw_result_source": "atomic complete five-method configuration-trial checkpoints",
        "history_interpretation": "synthetic past observations independent of current labels; risk coverage only",
    }


def _execution_report(root, manifest, records) -> None:
    counts = {m: sum(r["feasible"] and r["method"] == m for r in records) for m in cfg.METHODS}
    text = ("# Group 05 execution report\n\n"
            f"Run: `{manifest['run_id']}`\n\n"
            f"Committed trials: {manifest['completed_trials']}; method records: {len(records)}.\n\n"
            f"Dataset fingerprint: `{manifest['dataset_fingerprint']}`\n\n"
            "Valid method records (before paired filtering):\n\n"
            + "\n".join(f"- {m}: {counts[m]} / {cfg.EXPECTED_TRIALS}" for m in cfg.METHODS)
            + "\n\nThe main figures use seeds where all five methods are valid. "
            "Missing or illegal outcomes are excluded explicitly; negative utility is retained. "
            "Budget panels have different B denominators; compare methods within each panel. "
            "Synthetic history does not establish prediction of current corruption.\n")
    (root / "logs" / "execution_report.md").write_text(text, encoding="utf-8")


def run_experiment(*, force: bool = False) -> dict:
    cfg.validate_protocol()
    root = cfg.RESULT_ROOT
    for name in ("raw", "summary", "figures", "logs"):
        (root / name).mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("tnse.group05")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    handlers = []
    manifest_path = root / "logs" / "run_manifest.json"
    manifest = None
    try:
        manifest = _prepare_manifest(root, force)
        # Archive old logs before opening a new FileHandler. A resumed run
        # appends, while a new run receives its own log after archival.
        log_mode = "w" if manifest.get("status") == "initialized" else "a"
        handlers = [logging.FileHandler(root / "logs" / "execution.log", mode=log_mode, encoding="utf-8"),
                    logging.StreamHandler(sys.stdout)]
        for handler in handlers:
            handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
            logger.addHandler(handler)
        if log_mode == "w" and manifest.get("previous_run"):
            logger.info("Previous results archived at %s; starting a new complete run (%s)",
                        manifest["previous_run"]["archive_path"], manifest["previous_run"]["reason"])
        manifest["status"] = "running"
        manifest.pop("error", None)
        manifest.pop("figure_status", None)
        manifest.pop("figure_error", None)
        atomic_json(manifest_path, manifest)
        with threadpool_limits(limits=1):
            for n in cfg.N_VALUES:
                for seed in cfg.SEEDS:
                    pending = []
                    for beta, sigma in cfg.parameter_pairs():
                        key = trial_key(n, seed, beta, sigma)
                        path = checkpoint_path(root, manifest["run_id"], key)
                        if path.exists():
                            if key in manifest["checkpoints"] and sha256_file(path) != manifest["checkpoints"][key]["sha256"]:
                                raise ValueError(f"Saved trial was modified: {key}")
                            read_checkpoint(path, manifest["run_fingerprint"], key, manifest)
                            manifest["checkpoints"][key] = file_identity(path, root)
                        else:
                            if key in manifest["checkpoints"]:
                                raise ValueError(f"A committed trial checkpoint is missing: {key}")
                            pending.append((beta, sigma))
                    manifest["completed_trials"] = len(manifest["checkpoints"])
                    if not pending:
                        logger.info("Reused complete common scenario N=%d seed=%d", n, seed)
                        continue
                    shared = prepare_trial_inputs(n, seed, dataset_root=cfg.DATASET_ROOT)
                    cost_cache = {}
                    for beta, sigma in pending:
                        if shared.ground_truth.success and sigma not in cost_cache:
                            cost_cache[sigma] = resource_cost_components_for_sigma(shared, sigma)
                        payload = _execute_trial(manifest, shared, n, seed, beta, sigma,
                                                 cost_cache.get(sigma), logger)
                        key = payload["trial_key"]
                        path = checkpoint_path(root, manifest["run_id"], key)
                        write_checkpoint(path, payload)
                        manifest["checkpoints"][key] = file_identity(path, root)
                        manifest["completed_trials"] = len(manifest["checkpoints"])
                        manifest["updated_at_utc"] = _now()
                        atomic_json(manifest_path, manifest)
                        logger.info("Committed %d/%d: %s", manifest["completed_trials"], cfg.EXPECTED_TRIALS, key)
        if len(manifest["checkpoints"]) != cfg.EXPECTED_TRIALS:
            raise ValueError("Formal trial grid is incomplete")
        manifest["completed_trials"] = len(manifest["checkpoints"])
        records = export_checkpoints(root, manifest)
        if dataset_identity()["dataset_fingerprint"] != manifest["dataset_fingerprint"]:
            raise ValueError("Dataset changed during execution")
        manifest["status"] = "complete"
        manifest["completed_at_utc"] = _now()
        manifest["method_records"] = len(records)
        manifest["valid_method_records"] = sum(r["feasible"] for r in records)
        _execution_report(root, manifest, records)
        atomic_json(manifest_path, manifest)
        plotter = importlib.import_module("experiments.05_joint_decision.plot_joint_decision")
        try:
            plotter.plot_joint_decision(root)
            (root / "logs" / "figure_error.json").unlink(missing_ok=True)
        except Exception as error:
            try:
                atomic_json(root / "logs" / "figure_error.json", {
                    "run_id": manifest["run_id"], "status": "failed",
                    "error": f"{type(error).__name__}: {error}", "at_utc": _now(),
                })
            except Exception as save_error:
                error.add_note(f"Could not save figure failure status: {type(save_error).__name__}: {save_error}")
            raise
        # The plotter pins the completed run manifest's bytes. Do not rewrite
        # it after plotting; figure status/provenance lives in figure_manifest.
        logger.info("Group 05 complete: %d method records", len(records))
        return manifest
    except BaseException as error:
        if manifest is not None and manifest.get("status") != "complete":
            manifest["status"] = "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
            manifest["error"] = f"{type(error).__name__}: {error}"
            try:
                atomic_json(manifest_path, manifest)
            except Exception as save_error:
                error.add_note(f"Could not save failure status: {type(save_error).__name__}: {save_error}")
                for note in getattr(save_error, "__notes__", ()):
                    error.add_note(note)
        raise
    finally:
        for handler in handlers:
            logger.removeHandler(handler)
            handler.close()
