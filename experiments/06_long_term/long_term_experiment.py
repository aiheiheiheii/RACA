"""Single-seed orchestration, authenticated checkpoints and streaming exports.

Only this layer owns future events. Methods receive a causal interaction port;
offline truth is never placed in their configuration or state. All work starts
inside run_experiment(), which the user runs later from run_long_term.py.
"""
from __future__ import annotations

from collections import Counter
import csv
from datetime import datetime, timedelta, timezone
import gzip
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
from time import perf_counter
import traceback
import uuid
import zlib

import numpy as np

from dataset.load_dataset import load_dynamic_case, resource_profiles
from experiments.common.reproducibility import (
    dataset_snapshot, environment_metadata, sha256_array, verify_manifest,
)

from . import config_native_long_term as config
from .dynamic_simulator import DynamicScenario, validate_dynamic_case
from .long_term_methods import LongTermMethod
from .long_term_metrics import (
    OfflineEventEvaluator, summarize_single_run_rounds, summarize_single_run_events,
)

_io = importlib.import_module("experiments.05_joint_decision.joint_storage")
RESULT_ROOT = config.RESULT_ROOT
CHECKPOINT_SCHEMA = "long-term-native-method-trajectory-v1"
IDENTITY_FIELDS = ("run_id", "code_version", "algorithm_version", "dataset_fingerprint")
ROUND_FIELDS = (
    "experiment", "method", "seed", "N", "d", "H", "cr_dynamic", "round",
    "budget", "selected_count", "attempted_count", "completed_count", "missed_count",
    "gt_success", "gt_size", "gt_observation_count", "mean_verification_age",
    "mean_historical_anomaly", "mean_risk", "round_new_detected_events",
    "cumulative_detected_events", "total_corruption_events", "cumulative_detection_rate",
    "round_false_positives", "round_true_positives", "verification_makespan_ms", "deadline_satisfied", "status",
    "gt_failed_round", "virtual_number_queue", "virtual_delay_queue_sum_seconds", "method_algorithm_version",
    "failure_reason", "diagnostics_json", "scenario_fingerprint", *IDENTITY_FIELDS,
)
EVENT_FIELDS = (
    "seed", "cr_dynamic", "method", "event_id", "model_id", "onset_round", "recovery_round",
    "detected", "first_detection_round", "detection_delay", "status", "trajectory_status", "censored",
    "censor_round_exclusive", "observed_active_rounds", "evaluation_last_round", *IDENTITY_FIELDS,
)
DIAGNOSTIC_FIELDS = (
    "seed", "cr_dynamic", "method", "round", "phase", "target_id", "verifier_id",
    "temporary_target_id", "temporary_verifier_id", "outcome", "attempted", "completed", "age_before", "h_before", "risk_before",
    "h_after", "last_valid_round_before", "last_valid_round_after", "age_after",
    "similarity", "digest", "tag", "compute_ms", "transfer_ms", "network_ms", "total_ms",
    "start_ms", "finish_ms", "deadline_satisfied", "failure_reason", "extra_json",
    *IDENTITY_FIELDS,
)
# Only files imported by this group's simulation/aggregation path define its
# computational identity. Other groups' plots, tests and reports are not used.
# Figure code authenticates itself separately in figure_manifest.json.
SOURCE_SCOPE_VERSION = "long-term-native-protocol-runtime-dependencies-v1"
SOURCE_FILES = (
    "experiments/__init__.py",
    "experiments/common/__init__.py",
    "experiments/common/config.py",
    "experiments/common/dataset_adapter.py",
    "experiments/common/method_names.py",
    "experiments/common/reproducibility.py",
    "experiments/common/similarity.py",
    "experiments/01_detection/__init__.py",
    "experiments/01_detection/adaptation_utils.py",
    "experiments/01_detection/iedi_adapted.py",
    "experiments/01_detection/proposed_detection.py",
    "experiments/05_joint_decision/__init__.py",
    "experiments/05_joint_decision/config_joint_decision.py",
    "experiments/05_joint_decision/joint_methods.py",
    "experiments/05_joint_decision/joint_objective.py",
    "experiments/05_joint_decision/joint_storage.py",
    "experiments/06_long_term/__init__.py",
    "experiments/06_long_term/run_long_term.py",
    "experiments/06_long_term/config_native_long_term.py",
    "experiments/06_long_term/native_iedi.py",
    "experiments/06_long_term/native_edi_v.py",
    "experiments/06_long_term/dynamic_simulator.py",
    "experiments/06_long_term/long_term_methods.py",
    "experiments/06_long_term/long_term_metrics.py",
    "experiments/06_long_term/long_term_experiment.py",
    "requirements.txt",
    "requirements-lock.txt",
)
PERFORMANCE_PATCH_VERSION = "pinned-native-dvap-engineering-resume-v1"
PINNED_RUN_FINGERPRINT = "b3faba42ff223edc85288adcdd49232cbba2411b69409c229cf842ae140719ae"
PINNED_MANIFEST_SHA256 = "283822385bbb28cf09335de53aae1738edb36f53cf8b8c29c2b6db79f50da391"
PINNED_DRIVER_PATH = "experiments/06_long_term/long_term_experiment.py"
PINNED_METHODS_PATH = "experiments/06_long_term/long_term_methods.py"
PINNED_DRIVER_BEFORE = "1a1b76a60e769338c8c83871fdf9ca61c8ea9a44d4cebe1a6642d9639cb44064"
PINNED_METHODS_BEFORE = "e82a7e5312e5a853dd14e31116cfe02817723677775eb423b225ac5768cf5e24"
PINNED_METHODS_AFTER = "26a15e29f8dfc6b7a66641d37c5fa717d792e12052999ea0a783310e2a23fad7"
PINNED_TRAJECTORY_COUNT = 275
EXPORT_PATHS = {
    "long_term_rounds": "raw/long_term_rounds.csv",
    "corruption_events": "raw/corruption_events.csv",
    "verification_diagnostics": "raw/verification_diagnostics.csv.gz",
    "long_term_summary": "summary/long_term_summary.csv",
    "event_detection_summary": "summary/event_detection_summary.csv",
}


def _now() -> str:
    return datetime.now(timezone(timedelta(hours=8))).isoformat(timespec="seconds")


def _clean(value):
    if isinstance(value, np.ndarray):
        return _clean(value.tolist())
    if isinstance(value, np.generic):
        return _clean(value.item())
    if isinstance(value, dict):
        return {str(key): _clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Path):
        return value.as_posix()
    return value


def _encoded(value) -> bytes:
    return json.dumps(_clean(value), ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False).encode("utf-8")


def _digest(value) -> str:
    return hashlib.sha256(_encoded(value)).hexdigest()


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for part in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(part)
    return digest.hexdigest()


def _atomic_bytes(path: Path, payload: bytes) -> None:
    _io.retry_io(lambda: path.parent.mkdir(parents=True, exist_ok=True), description=f"Create {path.parent}")
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    completed = False
    try:
        with temporary.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        completed = True
        # Reuse the existing bounded Windows sharing-lock repair. A completed
        # temporary file is retained if commit fails; never rerun algorithms.
        _io.replace_file(temporary, path)
    finally:
        if not completed:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass


def _save_manifest(manifest: dict) -> None:
    _atomic_bytes(RESULT_ROOT / "logs" / "run_manifest.json", _encoded(manifest))


def _preserve_previous_manifest(saved: dict, saved_bytes: bytes) -> dict:
    """Keep the exact prior run manifest in the existing logs directory.

    Old checkpoint directories remain untouched. A prior snapshot with the
    same run ID is never overwritten with different bytes.
    """
    digest = hashlib.sha256(saved_bytes).hexdigest()
    run_id = str(saved.get("run_id", ""))
    if len(run_id) != 24 or any(c not in "0123456789abcdef" for c in run_id):
        run_id = digest[:24]
    snapshot = RESULT_ROOT / "logs" / f"run_manifest_{run_id}.json"
    if snapshot.is_file() and snapshot.read_bytes() != saved_bytes:
        snapshot = RESULT_ROOT / "logs" / f"run_manifest_{run_id}_{digest[:16]}.json"
    if snapshot.is_file():
        if snapshot.read_bytes() != saved_bytes:
            raise ValueError("Previous-run manifest snapshot differs from its preserved bytes")
    else:
        _atomic_bytes(snapshot, saved_bytes)
    if _file_digest(snapshot) != digest:
        raise ValueError("Previous-run manifest snapshot failed its byte-identity check")
    return _identity_file(snapshot)


def _single_run_mode() -> bool:
    return tuple(config.SEEDS) == (2026,) and config.REPETITIONS == 1


def _source_snapshot() -> dict[str, str]:
    """Hash the explicit runtime dependency closure, never unrelated groups."""
    # Dataset Python modules are additionally covered by dataset_snapshot.
    return {name: _file_digest(config.PROJECT_ROOT / name) for name in SOURCE_FILES}


def _source_changes(before: dict, after: dict) -> list[dict]:
    return [{"path": name, "before": before.get(name), "after": after.get(name)}
            for name in sorted(set(before) | set(after)) if before.get(name) != after.get(name)]


def _reject_pinned_resume(reason: str) -> None:
    raise ValueError("The interrupted native run cannot accept this engineering migration: "
                     + reason + ". Its original manifest/checkpoints have not been replaced. "
                     "Do not use --force to bypass this provenance check.")


def _saved_metadata(saved: dict) -> tuple[dict, dict]:
    """Validate small scenario/trajectory records before any fast-path reuse."""
    scenarios, trajectories = {}, {}
    allowed = {(cr, seed) for cr in config.DYNAMIC_CR_VALUES for seed in config.SEEDS}
    for item in saved.get("scenarios", []):
        key = (item.get("cr_dynamic"), item.get("seed"))
        if (key not in allowed or key in scenarios
                or item.get("event_count") != key[0] * config.N // 100
                or not isinstance(item.get("arrays_sha256"), dict)
                or item.get("scenario_fingerprint") != _digest(item["arrays_sha256"])
                or not isinstance(item.get("streams"), dict)
                or item["streams"] != _streams(key[1])
                or len(str(item.get("temporary_ids_sha256", ""))) != 64):
            raise ValueError(f"Invalid saved exogenous scenario metadata: {key}")
        scenarios[key] = item
    for item in saved.get("trajectories", []):
        key = (item.get("cr_dynamic"), item.get("seed"), item.get("method"))
        if (key[:2] not in scenarios or key[2] not in config.METHODS or key in trajectories
                or not isinstance(item.get("checkpoint"), dict)
                or sum(item.get("round_status_counts", {}).values()) != config.TOTAL_ROUNDS):
            raise ValueError(f"Invalid saved trajectory metadata: {key}")
        expected_path = _checkpoint_path(saved["run_fingerprint"], *key).relative_to(RESULT_ROOT).as_posix()
        if item["checkpoint"].get("path") != expected_path:
            raise ValueError(f"Saved checkpoint path does not match its exact run/scenario/method: {key}")
        trajectories[key] = item
    return scenarios, trajectories


def _authenticate_saved_trajectories(saved: dict) -> tuple[dict, dict]:
    """Authenticate compressed bytes, without decompressing large diagnostics.

    Envelope identity, all 200 rounds and the payload checksum are still
    verified by _read_checkpoint during final export; this metadata fast path
    never treats a checksum as an unexecuted scientific trajectory.
    """
    scenarios, candidates = _saved_metadata(saved)
    verified = {}
    print(f"Authenticating saved checkpoint bytes: {len(candidates)} files", flush=True)
    for number, (key, item) in enumerate(candidates.items(), 1):
        path = _checkpoint_path(saved["run_fingerprint"], *key)
        if _identity_file(path) != item["checkpoint"]:
            raise ValueError(f"Checkpoint SHA256/size differs from the recorded run: {path.name}")
        if item.get("failure") is None and item.get("valid_round_count") == config.TOTAL_ROUNDS:
            verified[key] = item
        if number % 25 == 0 or number == len(candidates):
            print(f"Authenticated saved checkpoint bytes: {number}/{len(candidates)}", flush=True)
    return scenarios, verified


def _compatible_saved_run(saved: dict, *, saved_bytes: bytes, sources: dict, dataset: dict,
                          environment: dict, runtime_fingerprint: str) -> dict | None:
    """Resume exact native identities or this one audited, pinned predecessor.

    The scientific run identity and original source hashes are never rewritten.
    A performance patch records a separate exact runtime identity. No general
    'same algorithm version' or arbitrary-source compatibility exception exists.
    """
    pinned = saved.get("run_fingerprint") == PINNED_RUN_FINGERPRINT
    common_matches = (saved.get("dataset_files_sha256") == dataset
            and saved.get("dataset_fingerprint") == _digest(dataset)
            and _digest(saved.get("protocol")) == _digest(config.protocol_metadata())
            and _digest(saved.get("environment")) == _digest(environment)
            and saved.get("algorithm_version") == config.ALGORITHM_VERSION
            and saved.get("code_version") == config.CODE_VERSION
            and saved.get("protocol_family") == config.PROTOCOL_FAMILY)
    original_sources = saved.get("sources_sha256")
    identity_matches = (isinstance(original_sources, dict)
            and saved.get("run_fingerprint") == _digest({"sources": original_sources,
                "dataset": dataset, "protocol": saved.get("protocol"), "environment": environment})
            and saved.get("run_id") == str(saved.get("run_fingerprint", ""))[:24])
    if _single_run_mode():
        # The user changed the experiment population, not its scientific
        # algorithms. Old multi-seed identities remain separate; the pinned
        # 275-trajectory migration does not apply to this new 15-trajectory run.
        if (common_matches and identity_matches
                and original_sources == sources
                and saved["run_fingerprint"] == runtime_fingerprint
                and saved.get("aggregation") == "single_run"
                and saved.get("repetitions") == 1
                and saved.get("single_seed") == 2026):
            return saved
        return None
    if not common_matches or not identity_matches:
        if pinned:
            _reject_pinned_resume("dataset, protocol, environment or original scientific identity changed")
        return None
    if original_sources == sources and saved["run_fingerprint"] == runtime_fingerprint:
        return saved
    if not pinned:
        return None
    if (len(original_sources) != 27 or set(original_sources) != set(sources)
            or original_sources.get(PINNED_DRIVER_PATH) != PINNED_DRIVER_BEFORE
            or original_sources.get(PINNED_METHODS_PATH) != PINNED_METHODS_BEFORE
            or sources.get(PINNED_METHODS_PATH) != PINNED_METHODS_AFTER
            or any(original_sources[name] != sources[name] for name in sources
                   if name not in {PINNED_DRIVER_PATH, PINNED_METHODS_PATH})):
        _reject_pinned_resume("source changes differ from the exact approved driver/methods engineering patch")
    history = saved.get("runtime_patch_history", [])
    if history:
        latest = history[-1]
        snapshot = RESULT_ROOT / "logs/recovery" / f"{saved['run_id']}_{PINNED_MANIFEST_SHA256}.json"
        if (len(history) != 1 or latest.get("patch_version") != PERFORMANCE_PATCH_VERSION
                or latest.get("runtime_sources_sha256") != sources
                or latest.get("runtime_run_fingerprint") != runtime_fingerprint
                or latest.get("original_manifest_sha256") != PINNED_MANIFEST_SHA256
                or latest.get("verified_checkpoint_count") != PINNED_TRAJECTORY_COUNT
                or not snapshot.is_file() or _file_digest(snapshot) != PINNED_MANIFEST_SHA256):
            _reject_pinned_resume("the recorded runtime patch or original manifest snapshot does not match")
        original = json.loads(snapshot.read_bytes().decode("utf-8-sig"))
        _, old_records = _saved_metadata(original)
        _, current_records = _saved_metadata(saved)
        bundle = [{"cr_dynamic": key[0], "seed": key[1], "method": key[2],
                   "checkpoint": item["checkpoint"]} for key, item in old_records.items()]
        if (len(old_records) != PINNED_TRAJECTORY_COUNT
                or latest.get("verified_checkpoint_identities_sha256") != _digest(bundle)
                or latest.get("original_manifest_snapshot") != _identity_file(snapshot)
                or any(key not in current_records
                       or current_records[key]["checkpoint"] != item["checkpoint"]
                       for key, item in old_records.items())):
            _reject_pinned_resume("the original 275 checkpoint identities do not match their immutable recovery snapshot")
        return saved
    if hashlib.sha256(saved_bytes).hexdigest() != PINNED_MANIFEST_SHA256:
        _reject_pinned_resume("initial manifest bytes differ from the audited interrupted predecessor")
    scenarios, verified = _authenticate_saved_trajectories(saved)
    expected = [(cr, seed, method) for cr in config.DYNAMIC_CR_VALUES
                for seed in config.SEEDS for method in config.METHODS][:PINNED_TRAJECTORY_COUNT]
    if (list(verified) != expected or len(saved.get("trajectories", [])) != PINNED_TRAJECTORY_COUNT
            or len(scenarios) != 92 or saved.get("failures")):
        _reject_pinned_resume("the interrupted predecessor does not contain its 275 complete audited trajectories")
    snapshot = RESULT_ROOT / "logs/recovery" / f"{saved['run_id']}_{PINNED_MANIFEST_SHA256}.json"
    if snapshot.is_file():
        if snapshot.read_bytes() != saved_bytes:
            _reject_pinned_resume("the original manifest recovery snapshot was modified")
    else:
        _atomic_bytes(snapshot, saved_bytes)
    checkpoint_bundle = [{"cr_dynamic": key[0], "seed": key[1], "method": key[2],
                          "checkpoint": item["checkpoint"]} for key, item in verified.items()]
    patched = dict(saved)
    patched["runtime_patch_history"] = [{
        "patch_version": PERFORMANCE_PATCH_VERSION, "applied_at": _now(),
        "original_manifest_sha256": PINNED_MANIFEST_SHA256,
        "original_manifest_snapshot": _identity_file(snapshot),
        "original_run_fingerprint": PINNED_RUN_FINGERPRINT,
        "original_driver_sha256": PINNED_DRIVER_BEFORE,
        "original_methods_sha256": PINNED_METHODS_BEFORE,
        "approved_methods_sha256": PINNED_METHODS_AFTER,
        "runtime_sources_sha256": sources, "runtime_run_fingerprint": runtime_fingerprint,
        "verified_checkpoint_count": len(verified),
        "verified_checkpoint_identities_sha256": _digest(checkpoint_bundle),
        "identity_policy": "retain original run_id/run_fingerprint/sources_sha256 and every existing checkpoint byte",
        "equivalence_rationale": (
            "DVA-P caches intrinsic cost/temporary-ID ordering only when its global minimizer is already "
            "capacity-feasible for every replacement; original scores, ties and diagnostics remain unchanged. "
            "Driver changes only progress logging, authenticated checkpoint metadata reuse and object lifetimes; "
            "dataset, random streams, protocols, budgets, feedback and statistics are unchanged."),
    }]
    print(f"Authenticated engineering migration: preserving all {len(verified)} original checkpoints "
          f"under scientific run {saved['run_id']}.", flush=True)
    return patched


def _assert_inputs_unchanged(dataset: dict, sources: dict, manifest: dict) -> None:
    current_dataset = dataset_snapshot(config.DATASET_ROOT)
    current_sources = _source_snapshot()
    dataset_changes = _source_changes(dataset, current_dataset)
    source_changes = _source_changes(sources, current_sources)
    if dataset_changes or source_changes:
        manifest["input_changes"] = {"dataset": dataset_changes, "sources": source_changes}
        changed = dataset_changes + source_changes
        raise RuntimeError("Experiment inputs changed during the run: "
                           + ", ".join(item["path"] for item in changed)
                           + ". Outputs are not accepted as complete.")


def _reuse_saved_exports(manifest: dict) -> bool:
    """Authenticate exports without loading cases or executing algorithms.

    This driver verified every checkpoint envelope while exporting. A
    matching whole-file SHA256 here proves those already-validated bytes have
    not changed. This avoids parsing/recompressing gigabytes just to finalize.
    Partial/missing exports fall back to normal per-method checkpoint resume.
    """
    scenarios, trajectories = manifest.get("scenarios", []), manifest.get("trajectories", [])
    expected_scenarios = {(cr, seed) for cr in config.DYNAMIC_CR_VALUES for seed in config.SEEDS}
    expected_trajectories = {(cr, seed, method) for cr, seed in expected_scenarios for method in config.METHODS}
    if (len(scenarios) != len(expected_scenarios)
            or {(item["cr_dynamic"], item["seed"]) for item in scenarios} != expected_scenarios
            or len(trajectories) != len(expected_trajectories)
            or {(item["cr_dynamic"], item["seed"], item["method"]) for item in trajectories}
               != expected_trajectories
            or manifest.get("failures")
            or any(item.get("failure") is not None or item.get("valid_round_count") != config.TOTAL_ROUNDS
                   or sum(item.get("round_status_counts", {}).values()) != config.TOTAL_ROUNDS
                   for item in trajectories)):
        return False
    try:
        for number, item in enumerate(trajectories, 1):
            path = _checkpoint_path(manifest["run_fingerprint"], item["cr_dynamic"], item["seed"], item["method"])
            if _identity_file(path) != item.get("checkpoint"):
                raise ValueError(f"Saved checkpoint file identity mismatch: {path.name}")
            if number % 50 == 0:
                print(f"Authenticating saved trajectories: {number}/{len(trajectories)}", flush=True)
        for name, relative in EXPORT_PATHS.items():
            path = RESULT_ROOT / relative
            if _identity_file(path) != manifest.get("files", {}).get(name):
                raise ValueError(f"Saved export file identity mismatch: {relative}")
            print(f"Authenticated saved export: {relative}", flush=True)
    except (OSError, KeyError, TypeError, ValueError) as error:
        manifest["export_reuse_rejection"] = str(error)
        print(f"Rebuilding exports from method checkpoints: {error}", flush=True)
        return False
    manifest["saved_export_recovery"] = {
        "verified_at": _now(), "checkpoint_files_verified": len(trajectories),
        "export_files_verified": len(EXPORT_PATHS), "simulation_reexecuted": False,
        "method_round_records": config.EXPECTED_METHOD_ROUNDS,
        "verification": "whole-file SHA256 and size against the same native manifest; export verified checkpoint envelope digests",
    }
    return True


def _finish_run(manifest: dict, *, method_round_records: int, trajectory_count: int) -> None:
    manifest.update(status="complete", completed_at=_now(), dataset_unchanged=True,
                    method_round_records=method_round_records, trajectory_count=trajectory_count,
                    runtime_failure_count=len(manifest["failures"]))
    _save_manifest(manifest)
    _write_run_report(manifest)
    try:
        plot = importlib.import_module("experiments.06_long_term.plot_long_term")
        plot.plot_long_term(RESULT_ROOT)
    except Exception as error:
        manifest["plot_error"] = traceback.format_exc()
        try:
            _save_manifest(manifest)
        except OSError as manifest_error:
            error.add_note(f"Could not save plot failure detail: {manifest_error}")
        raise
    if manifest["runtime_failure_count"]:
        print(f"WARNING: {manifest['runtime_failure_count']} method trajectories failed. "
              "Their raw records are retained and n_failed is reported; inspect logs/run_manifest.json.", flush=True)
    print(f"Completed Group 06. Results: {RESULT_ROOT}", flush=True)


def _streams(seed: int) -> dict[str, int]:
    # Retain the existing common exogenous stream identities. Protocol
    # challenge randomness occupies new independent streams after them.
    names = ("resource", "gt_tiebreak", "raca", "iedi", "dva", "age_only", "random",
             "iedi_protocol", "dva_protocol")
    return {"trace": int(seed), **{
        name: int(np.random.SeedSequence([seed, 6100 + i]).generate_state(1)[0])
        for i, name in enumerate(names)}}


def _checkpoint_path(run_fingerprint: str, cr: int, seed: int, method: str) -> Path:
    return (RESULT_ROOT / "logs" / "checkpoints" / run_fingerprint /
            f"cr{cr:02d}_seed{seed}_{method}.json.gz")


def _save_checkpoint(path: Path, identity: dict, payload: dict) -> None:
    envelope = {"schema": CHECKPOINT_SCHEMA, "identity": identity,
                "payload_sha256": _digest(payload), "payload": payload}
    _atomic_bytes(path, gzip.compress(_encoded(envelope), compresslevel=6, mtime=0))


def _read_checkpoint(path: Path, identity: dict) -> dict:
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        envelope = json.load(stream)
    if envelope.get("schema") != CHECKPOINT_SCHEMA or envelope.get("identity") != identity:
        raise ValueError(f"Incompatible checkpoint identity: {path}")
    payload = envelope["payload"]
    if _digest(payload) != envelope.get("payload_sha256"):
        raise ValueError(f"Checkpoint content checksum mismatch: {path}")
    rounds = payload.get("rounds", [])
    if [row["round"] for row in rounds] != list(range(1, config.TOTAL_ROUNDS + 1)):
        raise ValueError(f"Checkpoint must contain all 200 consecutive rounds: {path}")
    if any(row["method"] != identity["method"] or row["seed"] != identity["seed"]
           or row["cr_dynamic"] != identity["cr_dynamic"] for row in rounds):
        raise ValueError(f"Mixed checkpoint method/scenario identity: {path}")
    return payload


def _run_method(scenario: DynamicScenario, case: dict, *, method: str, seed: int,
                cr: int, streams: dict, temporary_ids: np.ndarray, identity: dict) -> dict:
    started = perf_counter()
    context = f"cr={cr}% seed={seed} {method}"
    print(f"Starting trajectory: {context}; phase=initialize", flush=True)
    evaluator = OfflineEventEvaluator(case, method=method, seed=seed, cr_dynamic=cr)
    rows, diagnostics = [], []
    failure, algorithm = None, None
    try:
        env = scenario.for_method(method)
        enrollment_tags = env.trusted_enrollment() if method in ("I-EDI", "DVA-P+EDI-V") else None
        method_seed = streams[{"RACA-HIM": "raca", "I-EDI": "iedi", "DVA-P+EDI-V": "dva"}[method]]
        algorithm = LongTermMethod(method, config, method_seed, temporary_ids,
                                   enrollment_tags=enrollment_tags)
    except Exception:
        failure = {"round": 0, "traceback": traceback.format_exc()}
    for t in range(1, config.TOTAL_ROUNDS + 1):
        if t == 1 or (t - 1) % 20 == 0:
            print(f"Trajectory progress: {context}; phase=verification; "
                  f"round={t}/{config.TOTAL_ROUNDS}; elapsed={perf_counter() - started:.1f}s", flush=True)
        base = {"experiment": config.EXPERIMENT, "method": method, "seed": seed,
                "N": config.N, "d": config.D, "H": config.H, "cr_dynamic": cr,
                "round": t, **{field: identity[field] for field in IDENTITY_FIELDS},
                "scenario_fingerprint": identity["scenario_fingerprint"]}
        if failure is None:
            try:
                env.begin_round(t)
                result = algorithm.step(env)
                statistics = result.round_stats()
                diagnostics_json = _encoded(result.diagnostics).decode("utf-8")
                staged_diagnostics = [{"method": method, "seed": seed, "cr_dynamic": cr,
                                       "round": t, **diagnostic,
                                       **{field: identity[field] for field in IDENTITY_FIELDS}}
                                      for diagnostic in result.verification_diagnostics]
            except Exception:
                failure = {"round": t, "traceback": traceback.format_exc()}
        if failure is None:
            # Consume exactly once, after all fallible method/serialization work.
            # An evaluator invariant failure interrupts export rather than
            # silently converting a broken scientific metric into a valid row.
            measured = evaluator.consume_round(t, result.outcomes)
            row = {**statistics, **measured, **base, "diagnostics_json": diagnostics_json}
            diagnostics.extend(staged_diagnostics)
        else:
            # The method is stopped after an unexpected error: its partly
            # mutated state must never masquerade as a valid continuation.
            measured = evaluator.snapshot()
            row = {**base, **measured, "budget": math.ceil(config.BETA_B * (config.N - config.M)),
                   "selected_count": 0, "attempted_count": 0, "completed_count": 0,
                   "missed_count": 0, "gt_success": None, "gt_size": 0,
                   "gt_observation_count": 0, "verification_makespan_ms": 0.0,
                   "deadline_satisfied": None, "status": "runtime_error",
                   "failure_reason": failure["traceback"], "diagnostics_json": "{}"}
        rows.append(row)
        if t % 20 == 0 or t == config.TOTAL_ROUNDS:
            print(f"Trajectory progress: {context}; completed={t}/{config.TOTAL_ROUNDS}; "
                  f"status={row['status']}; elapsed={perf_counter() - started:.1f}s", flush=True)
    events = [{**row, **{field: identity[field] for field in IDENTITY_FIELDS}}
              for row in evaluator.event_rows(trajectory_status="runtime_error" if failure else "ok")]
    print(f"Finished trajectory execution: {context}; phase=prepare-checkpoint; "
          f"elapsed={perf_counter() - started:.1f}s", flush=True)
    payload = _clean({"rounds": rows, "events": events, "diagnostics": diagnostics,
                      "failure": failure, "state": algorithm.state.snapshot() if algorithm is not None else None})
    print(f"Trajectory payload ready: {context}; elapsed={perf_counter() - started:.1f}s", flush=True)
    return payload


def _csv_export(path: Path, rows, fields: tuple[str, ...] | list[str], *, extra: bool = False) -> None:
    """Stream large diagnostic files; atomic replacement preserves prior exports."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    opener = gzip.open if path.suffix == ".gz" else open
    completed = False
    try:
        with opener(temporary, "wt", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
            writer.writeheader()
            known = set(fields)
            for record in rows:
                row = dict(record)
                extras = {key: item for key, item in row.items() if key not in known}
                if extras and extra:
                    row["extra_json"] = _encoded(extras).decode("utf-8")
                elif extras:
                    raise ValueError(f"Unexpected export columns for {path.name}: {sorted(extras)}")
                writer.writerow({key: _clean(value) for key, value in row.items() if key in known})
        completed = True
        _io.replace_file(temporary, path)
    finally:
        if not completed:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass


def _identity_file(path: Path) -> dict:
    return {"path": path.relative_to(RESULT_ROOT).as_posix(),
            "size_bytes": path.stat().st_size, "sha256": _file_digest(path)}


def _payloads(checkpoints: list[tuple[Path, dict]]):
    for number, (path, identity) in enumerate(checkpoints, 1):
        if number == 1 or number % 25 == 0 or number == len(checkpoints):
            print(f"Reading and validating trajectory data: {number}/{len(checkpoints)}", flush=True)
        yield _read_checkpoint(path, identity)


def _record_metadata(items: list[dict], value: dict, key_fields: tuple[str, ...]) -> None:
    """Replace one small metadata record in place, retaining the saved inventory.

    During resume the original 275 records must stay represented even if the
    user interrupts while the fast path is still walking earlier scenarios.
    """
    key = tuple(value[field] for field in key_fields)
    for position, previous in enumerate(items):
        if tuple(previous[field] for field in key_fields) == key:
            items[position] = value
            return
    items.append(value)


def _restore_verified_trajectory(manifest: dict, checkpoints: list,
                                 item: dict, scenario_record: dict,
                                 common_identity: dict) -> None:
    """Restore authenticated metadata only; final export still checks payload."""
    cr, seed, method = item["cr_dynamic"], item["seed"], item["method"]
    identity = {**common_identity, "cr_dynamic": cr, "seed": seed, "method": method,
                "scenario_fingerprint": scenario_record["scenario_fingerprint"]}
    path = _checkpoint_path(common_identity["run_fingerprint"], cr, seed, method)
    checkpoints.append((path, identity))
    _record_metadata(manifest["trajectories"], {**item, "resumed": True,
        "resume_validation": "recorded compressed-file SHA256 and size; full envelope rechecked at export"},
        ("cr_dynamic", "seed", "method"))
    print(f"[{len(checkpoints)}/{config.EXPECTED_TRAJECTORIES}] cr={cr}% seed={seed} "
          f"{method}: resumed authenticated checkpoint metadata; no simulation", flush=True)


def run_experiment(*, force: bool = False) -> None:
    """Run seed 2026 once for each ratio/method; resume exact single-run identities."""
    config.validate_protocol()
    if not _single_run_mode():
        raise ValueError("This entry requires one repetition with seed 2026")
    print(f"Single-run experiment: seed=2026; repetitions=1; "
          f"{config.EXPECTED_SCENARIOS} scenarios x {len(config.METHODS)} methods = "
          f"{config.EXPECTED_TRAJECTORIES} trajectories; "
          f"{config.EXPECTED_METHOD_ROUNDS} method-round records; no across-seed averaging", flush=True)
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    dataset_config = json.loads((config.DATASET_ROOT / "config.json").read_text(encoding="utf-8-sig"))
    if dataset_config.get("dataset_name") != config.DATASET_NAME:
        raise ValueError("The configured dataset must be TNSE-SimSimple-v1")
    manifest_check = verify_manifest(config.DATASET_ROOT)
    if manifest_check["missing"] or manifest_check["mismatch"]:
        raise ValueError(f"Dataset manifest integrity check failed: {manifest_check}")
    before = dataset_snapshot(config.DATASET_ROOT)
    sources = _source_snapshot()
    dataset_fingerprint = _digest(before)
    environment = environment_metadata()
    # Package and Python changes invalidate simulation checkpoints conservatively.
    fingerprint = _digest({"sources": sources, "dataset": before,
                           "protocol": config.protocol_metadata(), "environment": environment})
    common_identity = {"run_id": fingerprint[:24], "run_fingerprint": fingerprint,
                       "dataset_fingerprint": dataset_fingerprint,
                       "code_version": config.CODE_VERSION, "algorithm_version": config.ALGORITHM_VERSION}
    manifest = {**common_identity, "status": "running", "started_at": _now(),
                "source_scope_version": SOURCE_SCOPE_VERSION,
                "protocol_family": config.PROTOCOL_FAMILY,
                "aggregation": "single_run", "repetitions": 1, "single_seed": config.SEEDS[0],
                "methods": config.METHODS, "seeds": config.SEEDS,
                "dynamic_cr_values": config.DYNAMIC_CR_VALUES,
                "total_rounds": config.TOTAL_ROUNDS, "plot_rounds": config.PLOT_ROUNDS,
                "dataset_name": config.DATASET_NAME, "dataset_files_sha256": before,
                "dataset_manifest_verification": manifest_check,
                "sources_sha256": sources, "protocol": config.protocol_metadata(),
                "environment": environment, "force_requested": force,
                "scenarios": [], "trajectories": [], "files": {}, "failures": []}
    saved_path = RESULT_ROOT / "logs" / "run_manifest.json"
    resuming = False
    saved_scenarios, verified_saved = {}, {}
    if saved_path.is_file() and force and not _single_run_mode():
        protected = json.loads(saved_path.read_bytes().decode("utf-8-sig"))
        if protected.get("run_fingerprint") == PINNED_RUN_FINGERPRINT:
            _reject_pinned_resume("--force would bypass preservation of the 275 existing scientific checkpoints; omit --force")
    if saved_path.is_file():
        saved_bytes = saved_path.read_bytes()
        saved = json.loads(saved_bytes.decode("utf-8-sig"))
        compatible = None if force else _compatible_saved_run(
            saved, saved_bytes=saved_bytes, sources=sources, dataset=before,
            environment=environment, runtime_fingerprint=fingerprint)
        if compatible is not None:
            # Every rejection can still abort before the first manifest write.
            # Never replace the interrupted predecessor merely to announce a
            # resume whose source/checkpoint provenance is incompatible.
            saved_scenarios, verified_saved = _authenticate_saved_trajectories(compatible)
            manifest = compatible
            prior_status = manifest.get("status")
            manifest.setdefault("resume_history", []).append({
                "resumed_at": _now(), "previous_status": prior_status,
                "previous_interrupted_at": manifest.get("interrupted_at"),
                "previous_driver_error": manifest.get("driver_error"),
                "previous_plot_error": manifest.get("plot_error"),
            })
            manifest.update(status="running", resumed_at=_now(), force_requested=False,
                            runtime_source_scope_version=SOURCE_SCOPE_VERSION)
            manifest.pop("driver_error", None)
            manifest.pop("interrupted_at", None)
            manifest.pop("plot_error", None)
            common_identity = {field: manifest[field] for field in (*IDENTITY_FIELDS, "run_fingerprint")}
            fingerprint = manifest["run_fingerprint"]
            resuming = True
            print(f"Resuming native-protocol run {manifest['run_id']} with exact saved identities.", flush=True)
        else:
            snapshot = _preserve_previous_manifest(saved, saved_bytes)
            manifest["previous_run_manifest"] = snapshot
            manifest["previous_run_policy"] = (
                "previous manifest bytes preserved in existing logs; all old checkpoints retained; "
                "no multi-seed payload or summary reused for the new single-seed population")
            print(f"Preserved previous manifest: {snapshot['path']}; "
                  "old checkpoint files remain under their original fingerprint. "
                  "Starting the independent seed-2026 single run.", flush=True)
    checkpoints = []
    _save_manifest(manifest)
    try:
        if resuming and _reuse_saved_exports(manifest):
            _assert_inputs_unchanged(before, sources, manifest)
            print(f"All {config.EXPECTED_TRAJECTORIES} saved native trajectories and five exports authenticated. "
                  "Finalizing and drawing Figure L1 without repeating simulation.", flush=True)
            _finish_run(manifest, method_round_records=config.EXPECTED_METHOD_ROUNDS,
                        trajectory_count=config.EXPECTED_TRAJECTORIES)
            return
        # Missing/partial exports use the original method-level resume path.
        # Do not append a second copy of saved scenarios or trajectory entries.
        # Preserve the full saved metadata inventory throughout resume. Each
        # visited record is replaced in place; an interruption while restoring
        # the first scenes must not erase evidence of later saved checkpoints.
        if not resuming:
            manifest.update(scenarios=[], trajectories=[])
        manifest.update(failures=[], files={})
        manifest.pop("saved_export_recovery", None)
        _save_manifest(manifest)
        for cr in config.DYNAMIC_CR_VALUES:
            for seed in config.SEEDS:
                saved_scene = saved_scenarios.get((cr, seed))
                if (saved_scene is not None
                        and all((cr, seed, method) in verified_saved for method in config.METHODS)):
                    _record_metadata(manifest["scenarios"], dict(saved_scene), ("cr_dynamic", "seed"))
                    for method in config.METHODS:
                        _restore_verified_trajectory(manifest, checkpoints,
                            verified_saved[(cr, seed, method)], saved_scene, common_identity)
                    _save_manifest(manifest)
                    print(f"Restored complete scenario cr={cr}% seed={seed}; "
                          "skipped dynamic-case/resource preparation.", flush=True)
                    continue
                streams = _streams(seed)
                print(f"Preparing dynamic scenario: cr={cr}% seed={seed}", flush=True)
                # Exactly one case and resource trace for all three methods.
                case = load_dynamic_case(dataset_root=config.DATASET_ROOT, cr_percent=cr,
                                         seed=seed, n=config.N, d=config.D, H=config.H,
                                         rounds=config.TOTAL_ROUNDS)
                validate_dynamic_case(case, n=config.N, d=config.D, rounds=config.TOTAL_ROUNDS,
                                      cr_percent=cr)
                profiles = resource_profiles(seed=streams["resource"], n=config.N,
                                             rounds=config.TOTAL_ROUNDS)
                scenario_hashes = {key: sha256_array(np.asarray(case[key])) for key in (
                    "normal_X", "corrupted_event_X", "event_model_ids", "corrupted_mask",
                    "fail_uniform", "onset_round", "recovery_round")}
                scenario_hashes.update({"resource_" + key: sha256_array(value)
                                        for key, value in profiles.items() if isinstance(value, np.ndarray)})
                scenario_fingerprint = _digest(scenario_hashes)
                scenario = DynamicScenario(case, profiles, p_fail=config.P_FAIL,
                                           sigma=config.SIGMA, deadline_ms=config.T_ROUND_MAX_MS,
                                           beta_budget=config.BETA_B, validate=False,
                                           protocol_settings={**config.PROTOCOL_SETTINGS,
                                               "iedi_seed": streams["iedi_protocol"],
                                               "dvap_ediv_seed": streams["dva_protocol"]})
                temporary_ids = np.random.default_rng(streams["gt_tiebreak"]).permutation(config.N)
                temporary_hash = sha256_array(temporary_ids)
                if saved_scene is not None and (
                        scenario_fingerprint != saved_scene["scenario_fingerprint"]
                        or temporary_hash != saved_scene["temporary_ids_sha256"]):
                    raise ValueError("Recreated partial scenario differs from its saved exogenous/resource identity")
                _record_metadata(manifest["scenarios"], {"seed": seed, "cr_dynamic": cr,
                    "event_count": len(case["event_model_ids"]), "streams": streams,
                    "scenario_fingerprint": scenario_fingerprint, "arrays_sha256": scenario_hashes,
                    "temporary_ids_sha256": temporary_hash}, ("cr_dynamic", "seed"))
                for method in config.METHODS:
                    if (cr, seed, method) in verified_saved:
                        _restore_verified_trajectory(manifest, checkpoints,
                            verified_saved[(cr, seed, method)], saved_scene, common_identity)
                        _save_manifest(manifest)
                        continue
                    identity = {**common_identity, "seed": seed, "cr_dynamic": cr,
                                "method": method, "scenario_fingerprint": scenario_fingerprint}
                    path = _checkpoint_path(fingerprint, cr, seed, method)
                    payload, resumed = None, False
                    if path.is_file() and not force:
                        try:
                            candidate = _read_checkpoint(path, identity)
                            if candidate["failure"] is None:
                                payload, resumed = candidate, True
                            del candidate
                        except (ValueError, KeyError, OSError, EOFError, zlib.error) as error:
                            candidate = None  # Do not retain a rejected large decoded payload.
                            manifest.setdefault("checkpoint_rejections", []).append({
                                "path": path.relative_to(RESULT_ROOT).as_posix(), "reason": str(error)})
                    if payload is None:
                        payload = _run_method(scenario, case, method=method, seed=seed, cr=cr,
                                              streams=streams, temporary_ids=temporary_ids, identity=identity)
                        print(f"Saving trajectory checkpoint: cr={cr}% seed={seed} {method}", flush=True)
                        _save_checkpoint(path, identity, payload)
                    checkpoints.append((path, identity))
                    statuses = dict(Counter(row["status"] for row in payload["rounds"]))
                    trajectory = {"seed": seed, "cr_dynamic": cr, "method": method,
                                  "resumed": resumed, "round_status_counts": statuses,
                                  "valid_round_count": sum(row["status"] != "runtime_error" for row in payload["rounds"]),
                                  "failure": payload["failure"], "checkpoint": _identity_file(path)}
                    _record_metadata(manifest["trajectories"], trajectory, ("cr_dynamic", "seed", "method"))
                    if payload["failure"]:
                        manifest["failures"].append({"seed": seed, "cr_dynamic": cr,
                                                    "method": method, **payload["failure"]})
                    _save_manifest(manifest)
                    print(f"[{len(checkpoints)}/{config.EXPECTED_TRAJECTORIES}] cr={cr}% seed={seed} "
                          f"{method}: {'resumed' if resumed else 'saved'}; {statuses}", flush=True)
                    del payload
        rounds, events = [], []
        print(f"Reading trajectory headers/data: {len(checkpoints)} trajectories; phase=summary-input", flush=True)
        for payload in _payloads(checkpoints):
            rounds.extend(payload["rounds"])
            events.extend(payload["events"])
            del payload
        if len(rounds) != config.EXPECTED_METHOD_ROUNDS:
            raise ValueError(f"Native formal raw output must retain all {config.EXPECTED_METHOD_ROUNDS} method-round records")
        print("Writing the actual seed-2026 trajectories and event counts; no across-seed averaging", flush=True)
        summary = summarize_single_run_rounds(
            rounds, seed=2026, methods=config.METHODS,
            cr_values=config.DYNAMIC_CR_VALUES, total_rounds=config.TOTAL_ROUNDS)
        event_summary = summarize_single_run_events(
            events, rounds, seed=2026, total_rounds=config.TOTAL_ROUNDS)
        for collection in (summary, event_summary):
            for row in collection:
                row["cr_dynamic"] = int(row["cr_dynamic"])
                row.update({field: common_identity[field] for field in IDENTITY_FIELDS})
        for row in events:
            row["cr_dynamic"] = int(row["cr_dynamic"])
        outputs = {
            "long_term_rounds": (RESULT_ROOT / "raw" / "long_term_rounds.csv", rounds, ROUND_FIELDS, False),
            "corruption_events": (RESULT_ROOT / "raw" / "corruption_events.csv", events, EVENT_FIELDS, False),
            "verification_diagnostics": (RESULT_ROOT / "raw" / "verification_diagnostics.csv.gz",
                (row for payload in _payloads(checkpoints) for row in payload["diagnostics"]), DIAGNOSTIC_FIELDS, True),
            "long_term_summary": (RESULT_ROOT / "summary" / "long_term_summary.csv", summary,
                sorted({key for row in summary for key in row}), False),
            "event_detection_summary": (RESULT_ROOT / "summary" / "event_detection_summary.csv", event_summary,
                sorted({key for row in event_summary for key in row}), False),
        }
        for name, (path, records, columns, extra) in outputs.items():
            print(f"Exporting {name}: {path.name}", flush=True)
            _csv_export(path, records, columns, extra=extra)
            manifest["files"][name] = _identity_file(path)
            print(f"Exported {name}; recorded SHA256 and byte size", flush=True)
        _assert_inputs_unchanged(before, sources, manifest)
        _finish_run(manifest, method_round_records=len(rounds), trajectory_count=len(checkpoints))
    except BaseException as error:
        if manifest.get("status") != "complete":
            manifest.update(status="interrupted", interrupted_at=_now(), driver_error=traceback.format_exc())
            try:
                _save_manifest(manifest)
            except OSError as manifest_error:
                error.add_note(f"Could not save interrupted manifest: {manifest_error}")
        raise


def _write_run_report(manifest: dict) -> None:
    failures = manifest["runtime_failure_count"]
    report = ("# Group 06 native-protocol single-run execution report\n\n"
              f"Run: {manifest['run_id']}\n\n"
              f"Protocol family: {manifest['protocol_family']}\n\n"
              f"Completed at: {manifest['completed_at']} (Asia/Shanghai)\n\n"
              f"Scenarios: {len(manifest['scenarios'])}; trajectories: {manifest['trajectory_count']}; "
              f"method-round records: {manifest['method_round_records']}.\n\n"
              "Seed: 2026. Repetitions: 1. Aggregation: single_run. "
              "The five corruption ratios each use one actual 200-round trajectory per method. "
              "No across-seed average, standard deviation or confidence interval is computed.\n\n"
              f"Unexpected failed method trajectories: {failures}. Raw records retain failures; "
              "single-run summaries report the actual counts and trajectory status. "
              "GT unavailable and infeasible rounds remain valid protocol observations.\n\n"
              "No superiority is assumed. CDR uses every event, including missed/recovered ones. "
              "Detection delay averages only detected events. See the authenticated CSV summaries "
              "and baseline adaptation report before interpreting Figure L1.\n\n"
              f"Dataset fingerprint: {manifest['dataset_fingerprint']}\n\n"
              f"Source and protocol fingerprint: {manifest['run_fingerprint']}\n\n"
              "I-EDI uses Algorithm 2 random-modulus tag challenges; DVA-P uses EDI-V VMHT proofs. "
              "Algorithms, challenges and shared resource rules retain the native-protocol definitions. "
              "Earlier multi-seed checkpoint directories are preserved and their trajectories were not relabeled or reused.\n")
    if manifest.get("previous_run_manifest"):
        report += (f"\nPrevious-run manifest preserved with its original bytes at "
                   f"{manifest['previous_run_manifest']['path']} (SHA256 "
                   f"{manifest['previous_run_manifest']['sha256']}).\n")
    if manifest.get("saved_export_recovery"):
        report += (f"\nAll {config.EXPECTED_TRAJECTORIES} existing native checkpoint files and all five exported files were authenticated "
                   "against their original SHA256 and size. This invocation reused those exports; "
                   "it did not reexecute simulation or recompress verification diagnostics.\n")
    _atomic_bytes(RESULT_ROOT / "logs" / "run_report.md", report.encode("utf-8"))
