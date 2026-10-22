"""Small atomic I/O helpers and strict saved-result provenance checks.

This module and the plotter never import the candidate sampler or GT methods.
Incomplete runs retain .partial files and a non-complete manifest; they cannot
be used to generate the formal paper figure.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
import time
import uuid
from itertools import zip_longest

from experiments.common.reproducibility import write_csv
from .config_ground_truth import (
    PROJECT_ROOT, IMPLEMENTATION_VERSION, EXPECTED_RUNS, EXPECTED_POOLS,
    Q_VALUES, FAILURE_VALUES, SEEDS, REPETITIONS, protocol_metadata,
    READABLE_IMPLEMENTATION_VERSIONS,
    DIAGNOSTICS_VERSION,
)
from .ground_truth_metrics import SummaryAccumulator


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_identity(path: Path, result_root: Path) -> dict:
    path, root = Path(path).resolve(), Path(result_root).resolve()
    return {"path": path.relative_to(root).as_posix(),
            "sha256": sha256_file(path), "size_bytes": path.stat().st_size}


def replace_file(source: Path, target: Path) -> None:
    """Bounded retries for transient Windows sharing locks on saved files."""
    for attempt in range(6):
        try:
            Path(source).replace(target)
            return
        except PermissionError:
            if attempt == 5:
                raise
            time.sleep(0.05 * (2 ** attempt))


def atomic_json(path: Path, value: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                             encoding="utf-8")
        replace_file(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_csv(path: Path, rows: list[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        write_csv(temporary, rows)
        replace_file(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def source_identity() -> dict:
    """Content version independent of Git availability and machine paths."""
    files = set((PROJECT_ROOT / "experiments" / "common").glob("*.py"))
    files.update((PROJECT_ROOT / "experiments" / "04_ground_truth").glob("*.py"))
    files.update((PROJECT_ROOT / relative) for relative in (
        "experiments/01_detection/proposed_detection.py", "dataset/load_dataset.py"))
    hashes = {path.relative_to(PROJECT_ROOT).as_posix(): sha256_file(path) for path in sorted(files)}
    version = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode("utf-8")).hexdigest()
    return {"code_version": version, "source_sha256": hashes}


def _verified_file(root: Path, identity: dict) -> Path:
    path = (root / identity["path"]).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError(f"Missing or invalid saved source: {identity.get('path')}")
    if sha256_file(path) != identity["sha256"] or path.stat().st_size != identity["size_bytes"]:
        raise ValueError(f"Saved evidence no longer matches its run manifest: {path}")
    return path


def _read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def _verify_admission_evidence(root: Path, manifest: dict, raw_path: Path) -> None:
    """Reaggregate saved diagnostics only; never execute posthoc similarity logic."""
    from .ground_truth_diagnostics import DiagnosticAccumulator
    names = ("admission_runs", "final_k_summary", "admission_summary")
    if manifest.get("diagnostics_version") != DIAGNOSTICS_VERSION:
        raise ValueError("Revised results require their declared admission diagnostics version")
    if not all(name in manifest["files"] for name in names):
        raise ValueError("Revised results are missing admission and final-K evidence")
    paths = {name: _verified_file(root, manifest["files"][name]) for name in names}
    accumulator = DiagnosticAccumulator()
    count = 0
    with raw_path.open("r", encoding="utf-8-sig", newline="") as raw_stream, paths["admission_runs"].open("r", encoding="utf-8") as diagnostic_stream:
        for row, line in zip_longest(csv.DictReader(raw_stream), diagnostic_stream):
            if row is None or line is None:
                raise ValueError("Raw results and admission evidence have different row counts")
            record = json.loads(line)
            diagnostic = record["diagnostic"]
            digest = hashlib.sha256(json.dumps(diagnostic, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()
            if (record["run_id"] != manifest["run_id"] or record["method"] != row["method"]
                    or int(record["seed"]) != int(row["seed"])
                    or float(record["q_C"]) != float(row["q_C_requested"])
                    or float(record["p_fail"]) != float(row["p_fail_requested"])
                    or record["algorithm_version"] != manifest["algorithm_version"]
                    or record["diagnostics_version"] != DIAGNOSTICS_VERSION
                    or row["diagnostics_version"] != DIAGNOSTICS_VERSION
                    or digest != row["admission_diagnostic_sha256"]
                    or digest != record["admission_diagnostic_sha256"]):
                raise ValueError("Admission evidence is not aligned with its raw method record")
            selected = diagnostic["selected_subtype_counts"]
            if any(int(row[f"gt_{name}_count"]) != int(selected[name]) for name in (
                    "common_normal", "high_heterogeneity_normal", "severe_corrupt", "mild_corrupt")):
                raise ValueError("Selected subtype counts differ between raw and diagnostic records")
            accumulator.add(row, diagnostic)
            count += 1
    if count != EXPECTED_RUNS:
        raise ValueError("Revised admission evidence is incomplete")
    final_rows, admission_rows = accumulator.finish()
    for name, expected in (("final_k_summary", final_rows), ("admission_summary", admission_rows)):
        saved = _read_csv(paths[name])
        indexed = {(row["method"], float(row["q_C"]), float(row["p_fail"]), int(row["K"])): row for row in saved}
        if len(indexed) != len(saved) or len(saved) != len(expected):
            raise ValueError("Posthoc summary keys or row count are incomplete")
        for entry in expected:
            key = (entry["method"], entry["q_C"], entry["p_fail"], entry["K"])
            actual = indexed.get(key)
            if actual is None:
                raise ValueError(f"Missing posthoc summary: {key}")
            for field, value in entry.items():
                if value is None:
                    equal = actual[field] == ""
                elif isinstance(value, (int, float)):
                    equal = math.isclose(float(actual[field]), value, rel_tol=1e-12, abs_tol=1e-12)
                else:
                    equal = actual[field] == value
                if not equal:
                    raise ValueError(f"Saved posthoc statistics disagree with their records: {key}/{field}")
            if any(actual.get(field) != manifest[field] for field in (
                    "run_id", "dataset_fingerprint", "code_version", "algorithm_version", "diagnostics_version")):
                raise ValueError("Posthoc summary provenance disagrees with its source run")


def load_plot_sources(result_root: Path) -> tuple[list[dict], dict]:
    """Verify hashes and recalculate statistics from saved raw CSV only.

    No descriptors, dataset loading, similarity calculation or GT algorithm is
    involved. The plot uses the saved summary values after this consistency
    check, including every failure in the denominator.
    """
    root = Path(result_root).resolve()
    manifest_path = root / "logs" / "run_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"No completed Group 04 run at {root}; run run_ground_truth.py first")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    version = manifest.get("algorithm_version")
    if manifest.get("status") != "complete" or version not in READABLE_IMPLEMENTATION_VERSIONS:
        raise ValueError("Figure requires a complete, compatible Group 04 run")
    if manifest.get("protocol") != protocol_metadata(version):
        raise ValueError("Saved formal parameter grid differs from the declared Group 04 protocol")
    if manifest.get("counts", {}).get("method_runs") != EXPECTED_RUNS:
        raise ValueError("Formal run requires all 220000 method records")
    for mapping, fingerprint in (("dataset_files_sha256", "dataset_fingerprint"),
                                 ("source_sha256", "code_version")):
        saved = manifest.get(mapping)
        if not isinstance(saved, dict) or not saved:
            raise ValueError(f"Manifest lacks its saved {mapping} inventory")
        digest = hashlib.sha256(json.dumps(saved, sort_keys=True).encode("utf-8")).hexdigest()
        if digest != manifest.get(fingerprint):
            raise ValueError(f"Manifest {fingerprint} disagrees with its saved content inventory")
    required = {"raw", "summary", "diagnostics", "scenarios"}
    if not required.issubset(manifest.get("files", {})):
        raise ValueError("Run manifest lacks required saved evidence")
    paths = {name: _verified_file(root, manifest["files"][name]) for name in required}
    accumulator = SummaryAccumulator(algorithm_version=version)
    count = 0
    scene_counts = {}
    fingerprint = manifest["dataset_fingerprint"]
    run_id = manifest["run_id"]
    with paths["raw"].open("r", encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            if (row["dataset_fingerprint"] != fingerprint or row["run_id"] != run_id
                    or row["algorithm_version"] != version
                    or row["code_version"] != manifest["code_version"]):
                raise ValueError("Raw results contain mixed run/data/source identities")
            accumulator.add(row)
            key = (int(row["seed"]), float(row["q_C_requested"]), float(row["p_fail_requested"]))
            state = (row["candidate_fingerprint"], row["availability_fingerprint"],
                     row["failure_uniform_fingerprint"])
            if key in scene_counts and scene_counts[key] != state:
                raise ValueError("Methods do not share the same exogenous candidate/failure state")
            scene_counts[key] = state
            count += 1
    if count != EXPECTED_RUNS:
        raise ValueError("Raw CSV is incomplete")
    expected_summary, expected_diagnostics = accumulator.finish()
    summaries, diagnostics = _read_csv(paths["summary"]), _read_csv(paths["diagnostics"])
    for saved, expected in ((summaries, expected_summary), (diagnostics, expected_diagnostics)):
        if len(saved) != len(expected):
            raise ValueError("Saved summary row count is incomplete")
        indexed = {(row["method"], float(row["q_C"]), float(row["p_fail"])): row for row in saved}
        if len(indexed) != len(saved):
            raise ValueError("Saved statistics contain duplicate keys")
        for row in expected:
            key = (row["method"], row["q_C"], row["p_fail"])
            actual = indexed.get(key)
            if actual is None:
                raise ValueError(f"Missing formal summary: {key}")
            for field, value in row.items():
                if isinstance(value, (int, float)):
                    number = float(actual[field])
                    equal = (math.isnan(value) and math.isnan(number)) if isinstance(value, float) and math.isnan(value) else math.isclose(number, value, rel_tol=1e-12, abs_tol=1e-12)
                    if not equal:
                        raise ValueError(f"Saved statistics disagree with raw CSV at {key}/{field}")
                elif actual[field] != value:
                    raise ValueError(f"Saved statistics version mismatch at {key}/{field}")
            if (actual.get("run_id") != run_id or actual.get("dataset_fingerprint") != fingerprint
                    or actual.get("code_version") != manifest["code_version"]):
                raise ValueError("Summary provenance differs from the source run")
    # Count the independent pools without storing all diagnostic arrays.
    seen_pools = set()
    with paths["scenarios"].open("r", encoding="utf-8") as stream:
        for line in stream:
            record = json.loads(line)
            if record.get("run_id") != run_id:
                raise ValueError("Candidate-pool audit belongs to a different run")
            key = (int(record["seed"]), float(record["q_C_requested"]))
            if key in seen_pools or key[0] not in SEEDS or key[1] not in Q_VALUES:
                raise ValueError("Duplicate or unknown saved candidate pool")
            seen_pools.add(key)
            for p in FAILURE_VALUES:
                actual = scene_counts[(key[0], key[1], p)]
                expected_state = (record["candidate_fingerprint"],
                                  record["availability_fingerprints"][str(p)],
                                  record["failure_uniform_fingerprint"])
                if actual != expected_state:
                    raise ValueError("Candidate audit and method records disagree")
    if len(seen_pools) != EXPECTED_POOLS:
        raise ValueError("Candidate-pool audit is incomplete")
    if any(int(row["n_total"]) != REPETITIONS for row in summaries):
        raise ValueError("Every formal plotted point must use all 1000 seeds")
    if version == IMPLEMENTATION_VERSION:
        _verify_admission_evidence(root, manifest, paths["raw"])
    return summaries, manifest
