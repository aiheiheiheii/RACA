"""Atomic trial checkpoints, source identities, and paired statistics.

This module never imports a decision algorithm. Checkpoints store compact
assignment columns rather than repeating the complete G-by-target cost matrix.
"""
from __future__ import annotations

import csv
import gzip
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import time
import uuid

import numpy as np
from scipy.stats import t as student_t

from experiments.common.reproducibility import dataset_snapshot, verify_manifest, write_csv
from . import config_joint_decision as cfg

IO_RETRY_DELAYS = (0.05, 0.10, 0.20, 0.40, 0.80, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0)
_WINDOWS_SHARING_ERRORS = {5, 32, 33}


def retry_io(operation, *, description="file I/O", delays=None):
    """Retry access/sharing locks only; never replay scientific computation.

    This follows Group 02's I/O policy with a longer bounded wait (8.55 s).
    Other filesystem errors propagate immediately. The original exception is
    preserved if the lock persists; no target is deleted to bypass the lock.
    """
    waits = tuple(IO_RETRY_DELAYS if delays is None else delays)
    if any(not math.isfinite(delay) or delay < 0 for delay in waits):
        raise ValueError("I/O retry delays must be finite and nonnegative")
    for attempt in range(len(waits) + 1):
        try:
            return operation()
        except OSError as error:
            sharing_error = isinstance(error, PermissionError) or getattr(error, "winerror", None) in _WINDOWS_SHARING_ERRORS
            if not sharing_error or attempt == len(waits):
                if sharing_error:
                    error.add_note(f"{description}: access/sharing retry limit reached.")
                raise
            time.sleep(waits[attempt])
    raise AssertionError("Unreachable I/O retry state")


def replace_file(source: Path, target: Path) -> None:
    """Commit a fully closed temporary file, retaining it if all retries fail."""
    source, target = Path(source), Path(target)
    try:
        retry_io(lambda: source.replace(target), description=f"Replace output {target}")
    except OSError as error:
        error.add_note(f"Completed temporary output retained for recovery: {source}")
        raise


def _discard_incomplete(path: Path) -> None:
    try:
        path.unlink(missing_ok=True)
    except OSError:
        pass  # Cleanup must not mask the original serialization/I/O failure.


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def digest_json(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode("utf-8")).hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    path = Path(path)
    retry_io(lambda: path.parent.mkdir(parents=True, exist_ok=True), description=f"Create {path.parent}")
    temp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    completed = False
    try:
        serialized = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
        retry_io(lambda: temp.write_text(serialized, encoding="utf-8"), description=f"Write JSON {temp}")
        completed = True
        replace_file(temp, path)
    finally:
        if not completed:
            _discard_incomplete(temp)


def file_identity(path: Path, root: Path) -> dict:
    return {"path": path.resolve().relative_to(root.resolve()).as_posix(),
            "sha256": sha256_file(path), "size_bytes": path.stat().st_size}


def archive_previous_run(root: Path, manifest: dict, reason: str) -> dict:
    """Copy old exports before activating a fresh run; retain old checkpoints.

    No source file is deleted or moved. A complete archive marker is written
    only after byte identities and the active old manifest have been rechecked.
    Failure propagates before the caller replaces the active manifest.
    """
    root = Path(root).resolve()
    old_run_id = manifest.get("run_id")
    if not isinstance(old_run_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", old_run_id):
        raise ValueError("Saved run_id is not a safe archive/checkpoint path component")

    def checked(path: Path) -> Path:
        if path.is_symlink() or getattr(path, "is_junction", lambda: False)():
            raise ValueError(f"Refusing to archive a linked result path: {path}")
        if not path.resolve().is_relative_to(root):
            raise ValueError(f"Archive path escapes the Group 05 result root: {path}")
        return path

    def source_files() -> list[Path]:
        files = []

        def visit(folder: Path) -> None:
            checked(folder)
            if not folder.exists():
                return
            if not folder.is_dir():
                raise ValueError(f"Expected a result directory: {folder}")
            for child in sorted(folder.iterdir()):
                if folder == root / "logs" and child.name == "checkpoints":
                    checked(child)
                    continue
                checked(child)
                if child.is_dir():
                    visit(child)
                elif child.is_file():
                    files.append(child)
                else:
                    raise ValueError(f"Unsupported result entry: {child}")

        for name in ("raw", "summary", "figures", "logs"):
            visit(root / name)
        return files

    active_manifest = checked(root / "logs" / "run_manifest.json")
    old_manifest_bytes = active_manifest.read_bytes()
    if json.loads(old_manifest_bytes.decode("utf-8-sig")) != manifest:
        raise ValueError("The saved run changed before archiving; stop the other run and retry")
    files = source_files()
    archive_base = checked(root / "archive")
    archive_base.mkdir(parents=True, exist_ok=True)
    destination = checked(archive_base / f"{old_run_id}_{uuid.uuid4().hex}")
    destination.mkdir(exist_ok=False)
    identities = {}
    for source in files:
        relative = source.relative_to(root)
        target = checked(destination / relative)
        target.parent.mkdir(parents=True, exist_ok=True)
        before = sha256_file(source)
        shutil.copy2(source, target)
        if sha256_file(target) != before or sha256_file(source) != before:
            raise ValueError(f"Result changed or archive copy failed: {relative}")
        identities[relative.as_posix()] = {"sha256": before, "size_bytes": target.stat().st_size}
    if source_files() != files or active_manifest.read_bytes() != old_manifest_bytes:
        raise ValueError("The old result set changed while archiving; no new run was activated")
    checkpoint_dir = checked(root / "logs" / "checkpoints" / old_run_id)
    result = {
        "archive_path": destination.relative_to(root).as_posix(),
        "previous_run_id": old_run_id,
        "previous_run_fingerprint": manifest.get("run_fingerprint"),
        "reason": reason,
        "checkpoint_dir": checkpoint_dir.relative_to(root).as_posix(),
    }
    atomic_json(destination / "archive_manifest.json", {
        **result, "status": "complete", "source_files": identities,
        "checkpoint_copy_policy": "retained_in_original_location",
        "original_result_root": str(root),
    })
    return result


def source_identity() -> dict:
    files = set((cfg.PROJECT_ROOT / "experiments" / "common").glob("*.py"))
    files.update((cfg.PROJECT_ROOT / "experiments" / "05_joint_decision").glob("*.py"))
    files.update(cfg.PROJECT_ROOT / name for name in (
        "experiments/01_detection/proposed_detection.py",
        "experiments/02_overhead/overhead_resources.py", "dataset/load_dataset.py"))
    hashes = {p.relative_to(cfg.PROJECT_ROOT).as_posix(): sha256_file(p) for p in sorted(files)}
    return {"code_version": digest_json(hashes), "source_sha256": hashes}


def dataset_identity() -> dict:
    config = json.loads((cfg.DATASET_ROOT / "config.json").read_text(encoding="utf-8-sig"))
    if config.get("dataset_name") != cfg.DATASET_NAME:
        raise ValueError("The configured Dataset is not TNSE-SimSimple-v1")
    check = verify_manifest(cfg.DATASET_ROOT)
    if check["missing"] or check["mismatch"]:
        raise ValueError(f"Dataset checksum verification failed: {check}")
    snapshot = dataset_snapshot(cfg.DATASET_ROOT)
    return {"dataset_name": cfg.DATASET_NAME, "dataset_fingerprint": digest_json(snapshot),
            "dataset_sha256": snapshot, "dataset_manifest_verification": check}


def trial_key(n: int, seed: int, beta: float, sigma: float) -> str:
    return f"N{n}_seed{seed}_b{round(beta * 100):02d}_s{round(sigma * 100):03d}"


def checkpoint_path(root: Path, run_id: str, key: str) -> Path:
    return root / "logs" / "checkpoints" / run_id / f"{key}.json.gz"


def write_checkpoint(path: Path, payload: dict) -> None:
    envelope = {"payload_sha256": digest_json(payload), "payload": payload}
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    completed = False
    try:
        with gzip.open(temp, "wt", encoding="utf-8", compresslevel=6) as stream:
            json.dump(envelope, stream, ensure_ascii=False, allow_nan=False,
                      separators=(",", ":"))
        completed = True
        replace_file(temp, path)
    finally:
        if not completed:
            _discard_incomplete(temp)


def read_checkpoint(path: Path, run_fingerprint: str, key: str, identity: dict | None = None) -> dict:
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        envelope = json.load(stream)
    payload = envelope["payload"]
    if (envelope["payload_sha256"] != digest_json(payload)
            or payload["run_fingerprint"] != run_fingerprint or payload["trial_key"] != key):
        raise ValueError(f"Checkpoint integrity or source identity mismatch: {path}")
    if [r["method"] for r in payload["records"]] != list(cfg.METHODS):
        raise ValueError(f"Checkpoint does not contain exactly the five methods: {path}")
    for record in payload["records"]:
        if (record["trial_key"] != key or trial_key(record["N"], record["seed"],
                                                  record["beta_B"], record["sigma"]) != key):
            raise ValueError("Checkpoint has a foreign method record")
        if (record["N"] not in cfg.N_VALUES or record["seed"] not in cfg.SEEDS
                or (record["beta_B"], record["sigma"]) not in cfg.parameter_pairs()
                or (record["cr"], record["d"], record["H"], record["K"], record["M"],
                    record["deadline_ms"], record["lambda_value"]) != (
                    cfg.CORRUPTION_RATE_PERCENT, cfg.D, cfg.H, cfg.K, cfg.M,
                    cfg.T_ROUND_MAX_MS, cfg.LAMBDA)):
            raise ValueError("Checkpoint record has a foreign formal protocol")
        if identity is not None and any(record[field] != identity[field] for field in (
                "run_id", "code_version", "algorithm_version", "dataset_fingerprint")):
            raise ValueError("Checkpoint method record belongs to another source or run")
        if record["feasible"]:
            validate_saved_decision(record, payload["decisions"][record["method"]], payload["shared"])
    return payload


DIAGNOSTIC_COLUMNS = ("target_id", "verifier_id", "target_risk", "pair_cost_ms",
                      "verifier_load_after_ms", "makespan_after_ms")


def validate_saved_decision(record: dict, decision: dict, shared: dict) -> None:
    """Recheck saved arithmetic and constraints without loading or scheduling data."""
    columns = decision["columns"]
    if len(columns) != len(DIAGNOSTIC_COLUMNS) or len({len(c) for c in columns}) != 1:
        raise ValueError("Malformed compact assignment evidence")
    targets, verifiers, risks, costs, loads_after, spans = columns
    if len(targets) != record["selected_count"] or len(set(targets)) != len(targets):
        raise ValueError("Duplicate or missing selected targets in saved evidence")
    if record["completed_count"] != len(targets) or record["deadline_satisfied"] is not True:
        raise ValueError("Saved completion/deadline status contradicts a feasible decision")
    gt = set(shared["gt_verifier_ids"])
    loads = {g: 0.0 for g in gt}
    for target, verifier, risk, cost, load, span in zip(*columns):
        if (not isinstance(target, int) or not 0 <= target < record["N"] or target in gt
                or verifier not in gt or target == verifier
                or not math.isfinite(risk) or not 0 <= risk <= 1
                or not math.isfinite(cost) or cost < 0):
            raise ValueError("Illegal saved target/verifier/risk/cost")
        loads[verifier] += cost
        if (not math.isclose(loads[verifier], load, abs_tol=1e-9, rel_tol=1e-12)
                or not math.isclose(max(loads.values()), span, abs_tol=1e-9, rel_tol=1e-12)):
            raise ValueError("Saved per-task loads disagree with cost accumulation")
    span = max(loads.values(), default=0.0)
    reward = math.fsum(risks)
    if not 1 <= len(targets) <= record["B"] or span > record["deadline_ms"] + 1e-9:
        raise ValueError("Saved decision violates the nonempty budget/deadline constraints")
    expected = {"risk_reward": reward, "normalized_risk_reward": reward / record["B"],
                "makespan_ms": span, "verification_makespan_ms": span,
                "normalized_delay_penalty": cfg.LAMBDA * span / record["deadline_ms"],
                "joint_utility": reward / record["B"] - cfg.LAMBDA * span / record["deadline_ms"]}
    if any(not math.isclose(record[k], v, abs_tol=1e-9, rel_tol=1e-12) for k, v in expected.items()):
        raise ValueError("Saved objective is not reproducible from assignment evidence")
    actual_loads = {int(g): float(v) for g, v in decision["verifier_loads"].items()}
    if (actual_loads.keys() != loads.keys()
            or any(not math.isclose(loads[g], actual_loads[g], abs_tol=1e-9, rel_tol=1e-12) for g in loads)):
        raise ValueError("Saved verifier load vector is inconsistent")


def _statistics(values: list[float]) -> tuple[float, float, float, float]:
    if len(values) < cfg.MIN_PAIRED_TRIALS:
        return (math.nan,) * 4
    array = np.asarray(values, dtype=np.float64)
    mean, std = float(array.mean()), float(array.std(ddof=1))
    half = float(student_t.ppf(0.975, len(values) - 1)) * std / math.sqrt(len(values))
    return mean, std, mean - half, mean + half


def summarize(records: list[dict], identity: dict) -> tuple[list[dict], list[dict]]:
    """All methods use the same valid seeds at each configuration; no imputation."""
    index = {}
    for row in records:
        key = (row["N"], row["beta_B"], row["sigma"], row["seed"], row["method"])
        if key in index:
            raise ValueError(f"Duplicate raw method key: {key}")
        index[key] = row
    if len(index) != cfg.EXPECTED_METHOD_RECORDS:
        raise ValueError("Cannot summarize an incomplete formal result grid")
    summaries, feasibility = [], []
    for n in cfg.N_VALUES:
        for beta, sigma in cfg.parameter_pairs():
            paired = [s for s in cfg.SEEDS if all(index[n, beta, sigma, s, m]["feasible"]
                                                   for m in cfg.METHODS)]
            for method in cfg.METHODS:
                all_rows = [index[n, beta, sigma, s, method] for s in cfg.SEEDS]
                valid = [r for r in all_rows if r["feasible"]]
                selected = [index[n, beta, sigma, s, method] for s in paired]
                mean, std, lower, upper = _statistics([r["joint_utility"] for r in selected])
                base = {**identity, "N": n, "beta_B": beta, "sigma": sigma, "method": method,
                        "n_total": len(all_rows), "n_valid": len(valid),
                        "n_failed": len(all_rows) - len(valid), "n_paired": len(paired)}
                avg = lambda field: (float(np.mean([r[field] for r in selected]))
                                     if len(selected) >= cfg.MIN_PAIRED_TRIALS else math.nan)
                statistics = {**base, "mean_joint_utility": mean, "std_joint_utility": std,
                              "ci95_lower": lower, "ci95_upper": upper,
                              "mean_selected_count": avg("selected_count"),
                              "mean_risk_reward": avg("risk_reward"),
                              "mean_makespan_ms": avg("makespan_ms"),
                              "summary_status": "ok" if len(paired) >= cfg.MIN_PAIRED_TRIALS
                              else "insufficient_paired_trials"}
                summaries.extend({**statistics, "figure": figure} for figure in cfg.figures_for(beta, sigma))
                counts = {status: sum(r["status"] == status for r in all_rows)
                          for status in sorted({r["status"] for r in all_rows})}
                feasibility.append({**base, "valid_rate": len(valid) / len(all_rows),
                                    "failure_rate": 1 - len(valid) / len(all_rows),
                                    "paired_rate": len(paired) / len(all_rows),
                                    "status_counts_json": json.dumps(counts, sort_keys=True),
                                    "valid_seeds_json": json.dumps([r["seed"] for r in valid]),
                                    "paired_seeds_json": json.dumps(paired)})
    return summaries, feasibility


def atomic_csv(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    completed = False
    try:
        write_csv(temp, records)
        completed = True
        replace_file(temp, path)
    finally:
        if not completed:
            _discard_incomplete(temp)


def export_checkpoints(root: Path, manifest: dict) -> list[dict]:
    """Stream potentially large task diagnostics; retain only 12,150 metric rows."""
    raw = root / "raw" / "assignment_diagnostics.csv"
    raw.parent.mkdir(parents=True, exist_ok=True)
    temp = raw.with_name(f".{raw.name}.{uuid.uuid4().hex}.tmp")
    records = []
    headers = ["run_id", "trial_key", "method", "seed", "N", "beta_B", "sigma",
               *DIAGNOSTIC_COLUMNS]
    completed = False
    try:
        with temp.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=headers, lineterminator="\n")
            writer.writeheader()
            for n in cfg.N_VALUES:
                for seed in cfg.SEEDS:
                    for beta, sigma in cfg.parameter_pairs():
                        key = trial_key(n, seed, beta, sigma)
                        path = checkpoint_path(root, manifest["run_id"], key)
                        expected = manifest["checkpoints"][key]
                        if sha256_file(path) != expected["sha256"]:
                            raise ValueError(f"Checkpoint hash mismatch: {key}")
                        payload = read_checkpoint(path, manifest["run_fingerprint"], key, manifest)
                        records.extend(payload["records"])
                        for row in payload["records"]:
                            decision = payload["decisions"].get(row["method"])
                            if decision is None:
                                continue
                            for task in zip(*decision["columns"]):
                                writer.writerow({**{k: row[k] for k in headers[:7]},
                                                 **dict(zip(DIAGNOSTIC_COLUMNS, task))})
        completed = True
        replace_file(temp, raw)
    finally:
        if not completed:
            _discard_incomplete(temp)
    atomic_csv(root / "raw" / "joint_runs.csv", records)
    identity = {key: manifest[key] for key in (
        "run_id", "code_version", "algorithm_version", "dataset_fingerprint")}
    summary, feasibility = summarize(records, identity)
    atomic_csv(root / "summary" / "joint_summary.csv", summary)
    atomic_csv(root / "summary" / "feasibility_summary.csv", feasibility)
    manifest["files"] = {name: file_identity(root / relative, root) for name, relative in (
        ("joint_runs", "raw/joint_runs.csv"),
        ("assignment_diagnostics", "raw/assignment_diagnostics.csv"),
        ("joint_summary", "summary/joint_summary.csv"),
        ("feasibility_summary", "summary/feasibility_summary.csv"))}
    return records
