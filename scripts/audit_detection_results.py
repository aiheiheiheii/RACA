"""Audit saved FULL artifacts independently without running any detector.

Method availability and source identity follow the saved run manifest. The
current hash-baselines-v4 comparison has four methods and only D1/D2 figures.
Saved v3 data retain their earlier detector rules; historical Single-Reference
results retain their saved identity. Only the two audit reports are written;
all experiment artifacts are read-only.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import sys

sys.dont_write_bytecode = True
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from PIL import Image
from pypdf import PdfReader
from scipy.stats import t as student_t
from experiments.common.method_names import RACA_HIM, canonical_method_name

METHODS = (RACA_HIM, "I-EDI-Adapted", "CTCV-Adapted", "DynaEDI+-Adapted")
LEGACY_METHODS = METHODS + ("Proposed", "Single-Reference")
GT_METHODS = (RACA_HIM, "Proposed", "Single-Reference")
STATUSES = ("success", "gt_unavailable", "unsupported", "failure")
RATES = ("accuracy", "precision", "recall", "f1", "fpr", "fnr")
COUNTS = ("TP", "TN", "FP", "FN")
SUBTYPES = ("common_normal", "high_personalized_normal", "severe_corrupt", "mild_corrupt")
SEEDS = tuple(range(2026, 2056))
D1 = {(n, float(cr)) for n in range(1000, 5001, 500) for cr in range(1, 6)}
D2 = {(n, cr / 2.0) for n in range(1000, 5001, 1000) for cr in range(2, 11)}
GRID = D1 | D2


def _file_sha256(path: Path) -> str:
    checksum = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            checksum.update(chunk)
    return checksum.hexdigest()


def _snapshot(root: Path) -> dict[str, str]:
    files = sorted(path for path in root.rglob("*") if path.is_file()
                   and "quick" not in path.relative_to(root).parts)
    return {path.relative_to(root).as_posix(): _file_sha256(path) for path in files}


def _load_csv(path: Path) -> list[dict[str, str]]:
    # Full-coverage hash/digest evidence may exceed csv's small default field
    # limit; keep the saved Unicode diagnostic intact rather than truncating it.
    csv.field_size_limit(16 * 1024 * 1024)
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def _trial(row: dict[str, str]) -> tuple[int, float, int]:
    return int(row["N"]), float(row["cr"]), int(row["seed"])


def _key(row: dict[str, str]) -> tuple[int, float, int, str]:
    return *_trial(row), row["method"]


def _optional_bool(value: str | bool | None) -> bool | None:
    if value in (None, "", "None", "nan"):
        return None
    if value in (True, "True", "true", "1"):
        return True
    if value in (False, "False", "false", "0"):
        return False
    raise ValueError(f"Invalid optional boolean: {value!r}")


def _number_equal(actual: str | float, expected: float) -> bool:
    value = float(actual)
    if math.isnan(expected):
        return math.isnan(value)
    return math.isfinite(value) and math.isclose(value, expected, rel_tol=2e-12, abs_tol=2e-12)


def _divide(numerator: int | float, denominator: int | float) -> float:
    return numerator / denominator if denominator else math.nan


def _rates(counts: dict[str, int]) -> dict[str, float]:
    tp, tn, fp, fn = (counts[name] for name in COUNTS)
    return {"accuracy": _divide(tp + tn, tp + tn + fp + fn),
            "precision": _divide(tp, tp + fp), "recall": _divide(tp, tp + fn),
            "f1": _divide(2 * tp, 2 * tp + fp + fn),
            "fpr": _divide(fp, fp + tn), "fnr": _divide(fn, fn + tp)}


def _confusion(truth: np.ndarray, predictions: np.ndarray) -> dict[str, int]:
    pairs = np.bincount(2 * truth.astype(np.int64) + predictions.astype(np.int64), minlength=4)
    return {"TP": int(pairs[3]), "TN": int(pairs[0]), "FP": int(pairs[1]), "FN": int(pairs[2])}


def _target_fingerprint(values: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(values.dtype.str.encode("ascii"))
    digest.update(str(values.shape).encode("ascii"))
    digest.update(values.tobytes(order="C"))
    return digest.hexdigest()


def _statistics(values: list[float]) -> dict[str, float | int]:
    finite = [value for value in values if math.isfinite(value)]
    n = len(finite)
    mean = math.fsum(finite) / n if n else math.nan
    std = math.sqrt(math.fsum((value - mean) ** 2 for value in finite) / (n - 1)) if n > 1 else math.nan
    half = float(student_t.ppf(.975, n - 1)) * std / math.sqrt(n) if n > 1 else math.nan
    return {"n": n, "mean": mean, "std": std, "ci_low": mean - half, "ci_high": mean + half}


def audit(output: Path, report_root: Path) -> dict:
    output, report_root = output.resolve(), report_root.resolve()
    dataset_root = PROJECT_ROOT / "dataset"
    if report_root.is_relative_to(output) or report_root.is_relative_to(dataset_root.resolve()):
        raise ValueError("Audit reports must be outside saved results and dataset directories.")
    manifest = json.loads((output / "logs" / "run_manifest.json").read_text(encoding="utf-8"))
    completion = json.loads((output / "logs" / "completion.json").read_text(encoding="utf-8"))
    if manifest.get("mode", completion.get("mode")) != "FULL":
        raise RuntimeError("This audit only accepts saved FULL artifacts.")
    results_before, dataset_current_before = _snapshot(output), _snapshot(dataset_root)
    raw = _load_csv(output / "raw" / "detection_runs.csv")
    subtypes = _load_csv(output / "raw" / "subtype_metrics.csv")
    summary = _load_csv(output / "summary" / "detection_summary.csv")
    source_version = manifest.get("implementation_version", manifest.get("source_version", manifest.get("source_revision", manifest.get("code_version", "unrecorded_legacy"))))
    hash_v4 = source_version == "hash-baselines-v4"
    # Both hash generations share the artifact layout, but their prediction
    # semantics below are selected by the SAVED version, never today's code.
    hash_v3 = source_version in ("hash-baselines-v3", "hash-baselines-v4")
    # Never infer the saved comparison from today's source registry. An older
    # manifest may omit methods while its actual raw rows include Single.
    raw_names = tuple(dict.fromkeys(row["method"] for row in raw))
    raw_identity_by_canonical = {}
    for name in raw_names:
        canonical = canonical_method_name(name)
        if canonical in raw_identity_by_canonical and raw_identity_by_canonical[canonical] != name:
            raise ValueError("Saved raw artifacts mix both method aliases; ambiguous NPZ identity cannot be silently overwritten.")
        raw_identity_by_canonical[canonical] = name

    def raw_identity(name):
        return raw_identity_by_canonical.get(canonical_method_name(name), name)

    # Preserve physical raw/NPZ names. Method aliases in metadata or freshly
    # rendered labels may be canonical, without relabeling historical files.
    saved_declared_methods = tuple(manifest.get("methods", raw_names or METHODS))
    methods = tuple(raw_identity(name) for name in saved_declared_methods)
    implemented = tuple(raw_identity(name) for name in manifest.get("implemented_methods", (
        method for method in methods if any(row["method"] == method and row["status"] != "unsupported" for row in raw)
    )))
    subtypes = [{**row, "method": raw_identity(row["method"])} for row in subtypes]
    summary = [{**row, "method": raw_identity(row["method"])} for row in summary]
    availability_source = ("saved manifest implemented_methods" if "implemented_methods" in manifest
                           else "legacy raw statuses; implementation list not saved")
    schema_generation = ("four_method_raw_hash_v4" if hash_v4 else
                         "four_method_raw_hash_adaptation" if hash_v3 else
                         "five_method_descriptor_adaptation" if all(method in {canonical_method_name(name) for name in implemented}
                                                                      for method in (*METHODS, "Single-Reference"))
                         else "historical_partial_method_implementation")
    saved_source_metadata = {name: manifest[name] for name in (
        "implementation_version", "source_version", "source_revision", "code_version", "source_sha256", "source_files_sha256", "source_fingerprints", "git_revision"
    ) if name in manifest}
    config = manifest.get("config", {})
    reference_size, gt_size = int(config.get("reference_pool_size", 100)), int(config.get("M", 25))
    seeds = tuple(int(seed) for seed in manifest.get("seeds", SEEDS))
    grid = {(int(pair[0]), float(pair[1])) for pair in manifest.get("configurations", sorted(GRID))}
    expected_trials = {(n, cr, seed) for n, cr in grid for seed in seeds}
    expected_keys = {(*trial, method) for trial in expected_trials for method in methods}
    stages = defaultdict(lambda: {"checks": 0, "failures": 0})
    failures: list[dict] = []

    def check(condition: bool, stage: str, description: str) -> None:
        stages[stage]["checks"] += 1
        if not bool(condition):
            stages[stage]["failures"] += 1
            if len(failures) < 250:
                failures.append({"stage": stage, "finding": description})

    def json_dict(value: str, context: str) -> dict:
        try:
            parsed = json.loads(value or "{}")
            if not isinstance(parsed, dict):
                raise ValueError("diagnostics must be a JSON object")
            return parsed
        except (ValueError, TypeError) as error:
            check(False, "diagnostics", f"{context}: invalid diagnostic JSON: {error}")
            return {}

    check(grid == GRID and len(D1) == len(D2) == 45, "coverage", "Saved FULL grid matches 45+45-25=65")
    check(set(seeds) == set(SEEDS) and len(seeds) == len(SEEDS), "coverage", "Saved FULL seeds exactly 2026..2055")
    check(len(set(methods)) == len(methods) and set(methods) <= set(LEGACY_METHODS), "coverage", "Saved method identities unique and recognized")
    check(len(set(implemented)) == len(implemented) and set(implemented) <= set(methods), "coverage", "Implemented methods are a declared subset")
    if hash_v3:
        check(tuple(map(canonical_method_name, methods)) == METHODS and tuple(map(canonical_method_name, implemented)) == METHODS,
              "coverage", "Saved hash comparison declares four methods under current or legacy display identity")
    check(len(raw) == len(expected_keys), "coverage", "Raw count matches grid x seeds x saved methods")
    check(len(subtypes) == 4 * len(expected_keys), "coverage", "Subtype count equals all raw records x four")
    check(len(summary) == len(grid) * len(methods), "coverage", "Summary count matches grid x methods")
    raw_by_key = {_key(row): row for row in raw}
    subtype_by_key = {(*_key(row), int(row["subtype"])): row for row in subtypes}
    check(len(raw_by_key) == len(raw) and set(raw_by_key) == expected_keys, "coverage", "Unique full trial/method coverage")
    check(len(subtype_by_key) == len(subtypes) and set(subtype_by_key) == {(*key, code) for key in expected_keys for code in range(4)},
          "coverage", "Unique full trial/method/subtype coverage")
    grouped_trials, grouped_statistics, grouped_subtypes = defaultdict(list), defaultdict(list), defaultdict(list)
    for entry in subtypes:
        grouped_subtypes[(int(entry["N"]), float(entry["cr"]), entry["method"], int(entry["subtype"]))].append(entry)
    diagnostics_by_key = {}
    for row in raw:
        key = _key(row)
        n, cr, seed, method = key
        status = row["status"]
        if hash_v3:
            check(row.get("implementation_version") == source_version, "coverage", f"{key}: raw detector version matches manifest")
        grouped_trials[(n, cr, seed)].append(row)
        grouped_statistics[(n, cr, method)].append(row)
        check(status in STATUSES, "raw_status", f"{key}: recognized status")
        check(status != "success" or method in implemented, "raw_status", f"{key}: success requires saved implementation")
        check((method in implemented and status != "unsupported") or (method not in implemented and status == "unsupported"),
              "raw_status", f"{key}: availability agrees with saved source generation")
        target_count = n - reference_size
        check(int(row["target_count"]) == target_count and int(row["reference_pool_size"]) == reference_size,
              "raw_status", f"{key}: shared reference/target counts")
        check(int(row["evaluated_count"]) == (target_count if status == "success" else 0),
              "raw_status", f"{key}: complete successful predictions or unavailable")
        for field, default in (("d", 30), ("H", .5), ("tau_lower", .997), ("tau_upper", 1.)):
            check(float(row[field]) == float(config.get("dim" if field == "d" else field, default)), "raw_status", f"{key}: {field} agrees with manifest")
        diagnostics_by_key[key] = json_dict(row.get("diagnostics_json", "{}"), str(key))
        if status == "success":
            counts = {name: int(row[name]) for name in COUNTS}
            check(all(value >= 0 for value in counts.values()) and sum(counts.values()) == target_count,
                  "confusion_metrics", f"{key}: confusion counts cover all targets")
            rates = _rates(counts)
            for name, value in rates.items():
                check(_number_equal(row[name], value), "confusion_metrics", f"{key}: recomputed {name}")
            check(row["undefined_metrics"] == ";".join(name for name in RATES if math.isnan(rates[name])), "confusion_metrics", f"{key}: undefined metrics explicit")
            if method in GT_METHODS:
                check(_optional_bool(row["gt_success"]) is True and int(row["gt_size"]) == (gt_size if canonical_method_name(method) == RACA_HIM else 1),
                      "raw_status", f"{key}: successful declared GT size")
        else:
            check(all(math.isnan(float(row[name])) for name in RATES + COUNTS), "unavailable", f"{key}: no invented counts/rates")
            check(bool(row["reason"]) and row["undefined_metrics"] == "all", "unavailable", f"{key}: failure/unavailability reason retained")
    status_counts = Counter(row["status"] for row in raw)
    if "statuses" in completion:
        check(Counter(completion["statuses"]) == status_counts, "raw_status", "Completion status counts equal actual raw records")
    for trial, records in grouped_trials.items():
        check(len(records) == len(methods) and {row["method"] for row in records} == set(methods), "paired_targets", f"{trial}: all methods present")
        for field in ("target_ids_sha256", "descriptor_sha256", "evaluation_cr_percent", "realized_cr_percent"):
            check(len({row[field] for row in records}) == 1, "paired_targets", f"{trial}: shared {field}")

    prediction_folder = output / "raw" / "predictions"
    expected_names = {f"N_{n}_cr_{cr:.1f}_seed_{seed}.npz" for n, cr, seed in expected_trials}
    check({path.name for path in prediction_folder.glob("*.npz")} == expected_names, "prediction_evidence", "One NPZ per declared trial")
    prediction_count = prediction_arrays = diagnostic_arrays = 0
    cumulative_subtypes = defaultdict(lambda: {"count": 0, "errors": 0})
    for n, cr, seed in sorted(expected_trials):
        path = prediction_folder / f"N_{n}_cr_{cr:.1f}_seed_{seed}.npz"
        if not path.is_file():
            check(False, "prediction_evidence", f"Missing {path.name}")
            continue
        with np.load(path, allow_pickle=False) as evidence:
            prediction_count += 1
            for array_name in evidence.files:
                try:
                    check(not evidence[array_name].dtype.hasobject, "prediction_evidence", f"{(n, cr, seed)}/{array_name}: no pickle/object payload")
                except ValueError as error:
                    check(False, "prediction_evidence", f"{(n, cr, seed)}/{array_name}: unreadable pickle-free evidence: {error}")
            truth, codes = evidence["binary_label"], evidence["subtype_label"]
            targets, indices = evidence["target_ids"], evidence["evaluation_indices"]
            reference = evidence["reference_indices"]
            target_count, context = n - reference_size, (n, cr, seed)
            check(truth.shape == codes.shape == targets.shape == indices.shape == (target_count,), "prediction_evidence", f"{context}: aligned full evidence")
            check(np.isin(truth, (0, 1)).all() and np.isin(codes, range(4)).all() and np.array_equal(truth, (codes >= 2).astype(truth.dtype)),
                  "prediction_evidence", f"{context}: valid binary/subtype encodings")
            check(len(np.unique(targets)) == target_count, "prediction_evidence", f"{context}: unique temporary IDs")
            check(reference.shape == (reference_size,) and np.array_equal(np.sort(np.concatenate((reference, indices))), np.arange(n)),
                  "prediction_evidence", f"{context}: exhaustive disjoint partition")
            reference_ids = evidence["reference_ids"] if "reference_ids" in evidence.files else None
            reference_map = None
            if reference_ids is not None:
                valid_map = reference_ids.shape == reference.shape and len(np.unique(reference_ids)) == len(reference) and not np.isin(reference_ids, targets).any()
                check(valid_map, "reference_evidence", f"{context}: distinct pool IDs outside target IDs")
                if valid_map:
                    reference_map = dict(zip(reference.tolist(), reference_ids.tolist()))
            target_sha = _target_fingerprint(targets)
            for method in methods:
                key = (n, cr, seed, method)
                if key not in raw_by_key:
                    continue
                row, diagnostics = raw_by_key[key], diagnostics_by_key[key]
                if hash_v3 and canonical_method_name(method) != RACA_HIM and row["status"] == "success":
                    check(diagnostics.get("content_comparison") == "exact_hash", "diagnostics",
                          f"{context}/{method}: exact-hash content mode recorded")
                check(row["target_ids_sha256"] == target_sha, "paired_targets", f"{context}/{method}: NPZ fingerprint equals CSV")
                diagnostic_key = f"{method}_diagnostics_json"
                if diagnostic_key in evidence.files:
                    stored = evidence[diagnostic_key]
                    valid_json = stored.dtype.kind == "U" and stored.size == 1
                    check(valid_json, "diagnostics", f"{context}/{method}: pickle-free Unicode JSON")
                    if valid_json:
                        diagnostic_arrays += 1
                        check(json_dict(str(stored.reshape(-1)[0]), str(context)) == diagnostics, "diagnostics", f"{context}/{method}: NPZ/CSV diagnostics match")
                elif "diagnostics_json" in row:
                    check(False, "diagnostics", f"{context}/{method}: new diagnostic evidence missing")

                if method in ("I-EDI-Adapted", "CTCV-Adapted") and (row["status"] == "success" or "reference_indices" in diagnostics or "anchor_index" in diagnostics):
                    anchor_indices = diagnostics.get("reference_indices", [diagnostics["anchor_index"]] if "anchor_index" in diagnostics else [])
                    anchor_ids = diagnostics.get("reference_ids", [diagnostics["anchor_id"]] if "anchor_id" in diagnostics else [])
                    check(len(anchor_indices) == len(anchor_ids) == 1 and np.isin(anchor_indices, reference).all(), "reference_evidence", f"{context}/{method}: anchor row from pool")
                    check(reference_map is not None and all(reference_map.get(int(index)) == identifier for index, identifier in zip(anchor_indices, anchor_ids)),
                          "reference_evidence", f"{context}/{method}: anchor row/temporary ID mapping")
                if method == "DynaEDI+-Adapted" and (row["status"] == "success" or "witness_indices" in diagnostics):
                    witnesses, witness_ids = diagnostics.get("witness_indices", []), diagnostics.get("witness_ids", [])
                    check(len(witnesses) == len(witness_ids) == reference_size and len(set(witnesses)) == reference_size and set(witnesses) == set(reference.tolist()),
                          "reference_evidence", f"{context}/{method}: all pool witness rows saved")
                    check(reference_map is not None and all(reference_map.get(int(index)) == identifier for index, identifier in zip(witnesses, witness_ids)),
                          "reference_evidence", f"{context}/{method}: witness row/temporary ID mapping")

                prediction_key = f"{method}_predictions"
                if row["status"] != "success":
                    check(prediction_key not in evidence.files, "unavailable", f"{context}/{method}: unavailable prediction array absent")
                    for code in range(4):
                        subtype_row = subtype_by_key.get((*key, code))
                        if subtype_row is None:
                            continue
                        check(subtype_row["status"] == row["status"] and subtype_row["subtype_name"] == SUBTYPES[code],
                              "subtype_evidence", f"{context}/{method}/{code}: unavailable subtype identity/status")
                        check(int(subtype_row["count"]) == int(np.count_nonzero(codes == code)), "subtype_evidence", f"{context}/{method}/{code}: true stratum count retained")
                        check(all(math.isnan(float(subtype_row[name])) for name in RATES + COUNTS + ("errors", "error_rate")), "unavailable", f"{context}/{method}/{code}: undefined subtype metrics")
                    continue
                if prediction_key not in evidence.files:
                    check(False, "prediction_evidence", f"{context}/{method}: successful predictions missing")
                    continue
                predictions = evidence[prediction_key]
                valid = predictions.shape == truth.shape and np.isin(predictions, (0, 1)).all()
                check(valid, "prediction_evidence", f"{context}/{method}: one binary prediction per target")
                if not valid:
                    continue
                prediction_arrays += 1
                counts = _confusion(truth, predictions)
                for name, value in {**counts, **_rates(counts)}.items():
                    check(_number_equal(row[name], value), "prediction_evidence", f"{context}/{method}: predictions reproduce {name}")
                if method in GT_METHODS:
                    gt_key, verifier_key = f"{method}_gt_indices", f"{method}_verifier_ids"
                    available = gt_key in evidence.files and verifier_key in evidence.files
                    check(available, "reference_evidence", f"{context}/{method}: saved GT/verifier structure")
                    if available:
                        gt_indices, verifier_ids = evidence[gt_key], evidence[verifier_key]
                        gt_ids = json.loads(row["gt_ids_json"])
                        expected_size = gt_size if canonical_method_name(method) == RACA_HIM else 1
                        check(len(gt_indices) == len(gt_ids) == int(row["gt_size"]) == expected_size and np.isin(gt_indices, reference).all(), "reference_evidence", f"{context}/{method}: GT drawn only from pool")
                        check(verifier_ids.shape == truth.shape and np.isin(verifier_ids, gt_ids).all(), "reference_evidence", f"{context}/{method}: one GT verifier per target")
                        if gt_ids:
                            check(np.array_equal(verifier_ids, np.asarray(gt_ids)[np.arange(target_count) % len(gt_ids)]), "reference_evidence", f"{context}/{method}: exact cyclic or fixed verifier")
                        if reference_map is not None:
                            check(all(reference_map.get(int(index)) == identifier for index, identifier in zip(gt_indices, gt_ids)), "reference_evidence", f"{context}/{method}: GT temporary ID mapping")
                elif method in ("I-EDI-Adapted", "CTCV-Adapted"):
                    verifier_key = f"{method}_verifier_ids"
                    if verifier_key in evidence.files:
                        verifier_ids = evidence[verifier_key]
                        check(verifier_ids.shape == truth.shape and len(anchor_ids) == 1 and np.all(verifier_ids == anchor_ids[0]), "reference_evidence", f"{context}/{method}: fixed anchor assignment")
                    if method == "I-EDI-Adapted":
                        order = diagnostics.get("coverage_audit_order", [])
                        check(len(order) == target_count and set(order) == set(targets.tolist()) and diagnostics.get("full_coverage") is True, "diagnostics", f"{context}/{method}: exact full audit permutation")
                        if hash_v3:
                            equal_ids = diagnostics.get("tag_equal_target_ids", [])
                            check(len(equal_ids) == len(set(equal_ids)) and set(equal_ids) <= set(targets.tolist()),
                                  "diagnostics", f"{context}/{method}: actual tag-equal IDs are unique shared targets")
                            check(np.array_equal(predictions, (~np.isin(targets, equal_ids)).astype(np.int8)),
                                  "diagnostics", f"{context}/{method}: actual tag equality reproduces predictions")
                            if hash_v4:
                                expected_eta = int(config.get("iedi_tag_modulus", 7919))
                                check(expected_eta >= 2 and diagnostics.get("tag_modulus") == expected_eta,
                                      "diagnostics", f"{context}/{method}: eta matches saved config, source default 7919")
                                if "iedi_tag_modulus" in row:
                                    check(int(row["iedi_tag_modulus"]) == expected_eta, "diagnostics",
                                          f"{context}/{method}: raw eta agrees with saved config")
                                check(diagnostics.get("tag_equal_count") == len(equal_ids), "diagnostics",
                                      f"{context}/{method}: recorded tag-equal count matches actual equal-ID evidence")
                    elif "collector_ids" in diagnostics:
                        check(reference_ids is not None and np.isin(diagnostics["collector_ids"], reference_ids).all(), "reference_evidence", f"{context}/{method}: pool collector identities")
                        if hash_v3:
                            actual_tags = np.asarray(diagnostics.get("actual_proof_tags", []))
                            correct_tags = np.asarray(diagnostics.get("correct_proof_tags", []))
                            valid_tags = actual_tags.shape == correct_tags.shape == truth.shape
                            check(valid_tags, "diagnostics", f"{context}/{method}: aligned actual/correct modulo tags")
                            if valid_tags:
                                if hash_v4:
                                    configured_ell = int(config.get("ctcv_proof_vector_length", 0))
                                    expected_ell = configured_ell or 1000 * target_count
                                    ell = diagnostics.get("proof_vector_length")
                                    correct_bits = np.asarray(diagnostics.get("correct_combination_set_bits", []))
                                    valid_bits = (isinstance(ell, int) and ell == expected_ell and ell > 0
                                                  and np.issubdtype(actual_tags.dtype, np.integer)
                                                  and np.issubdtype(correct_tags.dtype, np.integer)
                                                  and np.all((actual_tags >= 0) & (actual_tags < ell))
                                                  and np.all((correct_tags >= 0) & (correct_tags < ell))
                                                  and np.array_equal(np.sort(correct_bits), np.unique(correct_tags)))
                                    check(valid_bits, "diagnostics",
                                          f"{context}/{method}: actual modulo tags and CR set bits agree with saved ell")
                                    block_override = int(config.get("ctcv_block_count", 0))
                                    expected_blocks = min(block_override or target_count, 4 * int(row["d"]))
                                    check(diagnostics.get("block_count") == expected_blocks, "diagnostics",
                                          f"{context}/{method}: physical nonempty byte-block count matches declared count and payload cap")
                                    expected_predictions = ((np.zeros(len(targets), dtype=np.int8)
                                        if diagnostics.get("flag") in (0, 1)
                                        else (~np.isin(actual_tags, correct_bits)).astype(np.int8))
                                        if valid_bits else None)
                                else:
                                    # V3 used own-target sigma_i inequality;
                                    # do not rewrite those historical decisions
                                    # as v4 CR bit-membership localization.
                                    expected_predictions = (np.zeros(len(targets), dtype=np.int8)
                                        if diagnostics.get("flag") in (0, 1)
                                        else (actual_tags != correct_tags).astype(np.int8))
                                if expected_predictions is not None:
                                    check(np.array_equal(predictions, expected_predictions), "diagnostics",
                                          f"{context}/{method}: saved-version Flag/tag rule reproduces predictions")
                elif method == "DynaEDI+-Adapted":
                    disagreement = np.asarray(diagnostics.get("target_disagreement_counts", []))
                    threshold = math.ceil(float(config.get("dyna_quorum_fraction", 2 / 3)) * len(witnesses))
                    valid_counts = disagreement.shape == truth.shape and np.all((disagreement >= 0) & (disagreement <= len(witnesses)))
                    check(valid_counts, "diagnostics", f"{context}/{method}: bounded per-target disagreement counts")
                    if valid_counts:
                        root_candidates = np.flatnonzero(disagreement > threshold)
                        if hash_v4:
                            # Root disagreement identifies the candidate set;
                            # a model is corrupted only after at least one
                            # expanded block has its own strict quorum.
                            localized = diagnostics.get("localized_targets", [])
                            block_count = int(diagnostics.get("block_count", config.get("baseline_block_count", 5)))
                            valid_localization = (isinstance(localized, list) and block_count > 0
                                and all(isinstance(entry, dict)
                                    and isinstance(entry.get("target_position"), int)
                                    and not isinstance(entry["target_position"], bool)
                                    and 0 <= entry["target_position"] < target_count
                                    and isinstance(entry.get("block_indices"), list)
                                    and all(isinstance(block, int) and 0 <= block < block_count
                                            and not isinstance(block, bool)
                                            for block in entry["block_indices"])
                                    and len(set(entry["block_indices"])) == len(entry["block_indices"])
                                    for entry in localized))
                            positions = [entry["target_position"] for entry in localized] if valid_localization else []
                            valid_localization = (valid_localization and len(positions) == len(set(positions))
                                                  and set(positions) == set(root_candidates.tolist()))
                            check(valid_localization, "diagnostics",
                                  f"{context}/{method}: every strict root-quorum candidate has exactly one localization record")
                            if valid_localization:
                                expected_predictions = np.zeros(target_count, dtype=np.int8)
                                block_totals = np.zeros(block_count, dtype=np.int64)
                                for entry in localized:
                                    if entry["block_indices"]:
                                        expected_predictions[entry["target_position"]] = 1
                                    for block in entry["block_indices"]:
                                        block_totals[block] += 1
                                check(np.array_equal(predictions, expected_predictions), "diagnostics",
                                      f"{context}/{method}: nonempty expanded block-quorum evidence reproduces all model decisions")
                                if "localized_target_count_by_block" in diagnostics:
                                    check(np.array_equal(np.asarray(diagnostics["localized_target_count_by_block"]), block_totals),
                                          "diagnostics", f"{context}/{method}: per-block totals match saved localization evidence")
                        else:
                            check(np.array_equal(predictions, (disagreement > threshold).astype(np.int8)), "diagnostics",
                                  f"{context}/{method}: historical root-quorum prediction rule retained")
                    # Dyna's multi-witness design needs no invented single
                    # *_verifier_ids array; None is never read as an object.
                for code, name in enumerate(SUBTYPES):
                    subtype_row = subtype_by_key.get((*key, code))
                    if subtype_row is None:
                        continue
                    mask = codes == code
                    count, errors = int(mask.sum()), int(np.count_nonzero(truth[mask] != predictions[mask]))
                    check(subtype_row["status"] == row["status"] and subtype_row["subtype_name"] == name and int(subtype_row["count"]) == count, "subtype_evidence", f"{context}/{method}/{code}: stratum identity/status/count")
                    check(int(subtype_row["errors"]) == errors and _number_equal(subtype_row["error_rate"], _divide(errors, count)), "subtype_evidence", f"{context}/{method}/{code}: recomputed subtype errors")
                    subtype_counts = _confusion(truth[mask], predictions[mask])
                    for name_, value in {**subtype_counts, **_rates(subtype_counts)}.items():
                        check(_number_equal(subtype_row[name_], value), "subtype_evidence", f"{context}/{method}/{code}: {name_}")
                    cumulative_subtypes[(method, name)]["count"] += count
                    cumulative_subtypes[(method, name)]["errors"] += errors

    summary_keys = {(int(row["N"]), float(row["cr"]), row["method"]) for row in summary}
    check(len(summary_keys) == len(summary) and summary_keys == {(n, cr, method) for n, cr in grid for method in methods}, "summary_statistics", "Unique full configuration/method summaries")
    for row in summary:
        key = int(row["N"]), float(row["cr"]), row["method"]
        if hash_v3:
            check(row.get("implementation_version") == source_version, "summary_statistics", f"{key}: summary detector version matches manifest")
        trials = grouped_statistics[key]
        statuses = Counter(trial["status"] for trial in trials)
        check(int(row["run_count"]) == len(trials) == len(seeds), "summary_statistics", f"{key}: declared actual seed count")
        for status in STATUSES:
            check(int(row[f"{status}_runs"]) == statuses[status], "summary_statistics", f"{key}: actual {status} count")
        check(int(row["prediction_total"]) == sum(int(trial["evaluated_count"]) for trial in trials), "summary_statistics", f"{key}: actual prediction total")
        attempts = [_optional_bool(trial["gt_success"]) for trial in trials if _optional_bool(trial["gt_success"]) is not None]
        check(int(row["gt_attempts"]) == len(attempts) and _number_equal(row["gt_success_rate"], _divide(sum(attempts), len(attempts))), "summary_statistics", f"{key}: actual GT/reference attempt aggregation")
        for metric in RATES + COUNTS:
            statistics = _statistics([float(trial[metric]) for trial in trials if trial["status"] == "success"])
            for statistic, value in statistics.items():
                check(_number_equal(row[f"{metric}_{statistic}"], value), "summary_statistics", f"{key}/{metric}: independent successful defined-seed {statistic}")

    subtype_summary_path = output / "summary" / "subtype_summary.csv"
    # V3 no longer uses subtype/diagnostic figures. A leftover summary or image
    # from an older run must not be audited as a newly generated v3 artifact.
    subtype_summary_checked = not hash_v3 and subtype_summary_path.is_file()
    if subtype_summary_checked:
        seen = set()
        for row in _load_csv(subtype_summary_path):
            code = int(row["subtype"]) if "subtype" in row else SUBTYPES.index(row["subtype_name"])
            key = int(row["N"]), float(row["cr"]), row["method"], code
            check(key not in seen and code in range(4), "subtype_summary", f"{key}: unique summary")
            seen.add(key)
            trials = grouped_subtypes[key]
            valid = [entry for entry in trials if entry["status"] == "success"]
            for statistic, value in _statistics([float(entry["error_rate"]) for entry in valid]).items():
                field = f"error_rate_{statistic}"
                if field in row:
                    check(_number_equal(row[field], value), "subtype_summary", f"{key}: {field}")
            if "run_count" in row:
                check(int(row["run_count"]) == len(trials), "subtype_summary", f"{key}: actual trial count")
            statuses = Counter(entry["status"] for entry in trials)
            for status in STATUSES:
                if f"{status}_runs" in row:
                    check(int(row[f"{status}_runs"]) == statuses[status], "subtype_summary", f"{key}: actual {status} count")
        check(seen == {(n, cr, method, code) for n, cr in grid for method in methods for code in range(4)},
              "subtype_summary", "Complete configuration/method/subtype summaries")

    successful_methods = tuple(method for method in implemented if any(row["method"] == method and row["status"] == "success" for row in raw))
    panel_coverage = {}
    for figure, fixed_values in (("D1", range(1, 6)), ("D2", range(1000, 5001, 1000))):
        panel_coverage[figure] = []
        for fixed in fixed_values:
            pairs = sorted((n, cr) for n, cr in grid if (cr == fixed if figure == "D1" else n == fixed))
            check(len(pairs) == 9, "figure_data", f"{figure}/{fixed}: nine actual configurations")
            for method in successful_methods:
                check(all((n, cr, method) in summary_keys for n, cr in pairs), "figure_data", f"{figure}/{fixed}/{method}: measured summaries include any failed-point gaps")
            panel_coverage[figure].append({"fixed": fixed, "points": [list(pair) for pair in pairs]})
    style_path = output / "logs" / "plot_style.json"
    plot_style = json.loads(style_path.read_text(encoding="utf-8")) if style_path.is_file() else {}
    plot_kind = next((plot_style[name] for name in ("plot_kind", "primary_plot_kind", "main_plot_kind", "main_plot_type") if name in plot_style), manifest.get("plot_kind", "unrecorded_legacy"))
    if plot_kind != "unrecorded_legacy":
        check(plot_kind in ("bar", "line"), "figure_data", "Saved main plot kind is bar or explicit line option")
    if hash_v3:
        check(isinstance(plot_style.get("broken_axis"), bool), "figure_data", "Hash results record whether actual values permit the split-axis layout")
        check(plot_style.get("break_slashes") == "double diagonal, both sides", "figure_data", "Hash results record both-side double break marks")
        if hash_v4 or "results_implementation_version" in plot_style:
            check(plot_style.get("results_implementation_version") == source_version, "figure_data",
                  "Rendered figures retain the saved detector version, including v3 data redrawn by v4 code")
        # Actual per-panel bands may expand, or a panel may fall back to a full
        # axis, to avoid hiding observations in the nominal break. Do not force
        # data to occupy the default lower/upper intervals.
        for name in ("upper_ylim", "lower_ylim"):
            if name in plot_style:
                limits = plot_style[name]
                check(isinstance(limits, (list, tuple)) and len(limits) == 2
                      and all(isinstance(value, (float, int)) and math.isfinite(value) for value in limits)
                      and limits[0] < limits[1], "figure_data", f"Hash {name} records finite ordered limits")
    if "plotted_methods" in plot_style:
        canonical_plotted = tuple(map(canonical_method_name, plot_style["plotted_methods"]))
        check(len(canonical_plotted) == len(set(canonical_plotted)) and set(canonical_plotted) == set(map(canonical_method_name, successful_methods)),
              "figure_data", "Saved plot availability equals actual successful methods, without duplicate display aliases")
    figure_checks = []
    specifications = [("Fig_D1_detection_vs_N", "abcde", bool(successful_methods)),
                      ("Fig_D2_detection_vs_cr", "abcde", bool(successful_methods))]
    if not hash_v3:
        specifications.extend((("Fig_D_diagnostic_metrics", "abcde", RACA_HIM in set(map(canonical_method_name, successful_methods))),
                               ("Fig_D3_subtype_errors", "abcd", subtype_summary_checked)))
    ignored_legacy_figures = [name for name in ("Fig_D_diagnostic_metrics", "Fig_D3_subtype_errors")
                              if hash_v3 and any((output / "figures" / f"{name}.{suffix}").is_file()
                                                 for suffix in ("pdf", "png"))]
    width_metadata = plot_style.get("figure_width_inches", manifest.get("figure_width_inches"))
    height_metadata = plot_style.get("figure_height_inches", manifest.get("figure_height_inches"))
    expected_width = float(width_metadata) if width_metadata is not None else None
    expected_height = float(height_metadata) if height_metadata is not None else None
    dimension_source = "saved plot/manifest metadata" if width_metadata is not None and height_metadata is not None else "measured legacy PDF for unrecorded dimensions"
    for name, letters, required in specifications:
        pdf, png = output / "figures" / f"{name}.pdf", output / "figures" / f"{name}.png"
        if not pdf.is_file() or not png.is_file():
            check(not required and not (pdf.is_file() or png.is_file()), "figure_files", f"{name}: required pair exists or optional legacy pair absent")
            figure_checks.append({"name": name, "status": "missing" if required else "not_present_optional", "panel_count": len(letters)})
            continue
        reader = PdfReader(pdf)
        check(pdf.stat().st_size > 1000 and len(reader.pages) == 1, "figure_files", f"{name}: readable one-page PDF")
        page = reader.pages[0]
        width, height = float(page.mediabox.width) / 72., float(page.mediabox.height) / 72.
        if expected_width is None:
            expected_width = width
        if expected_height is None:
            expected_height = height
        check(math.isclose(width, expected_width, abs_tol=1e-7) and math.isclose(height, expected_height, abs_tol=1e-7),
              "figure_files", f"{name}: dimensions match saved {expected_width:g}x{expected_height:g} inches")
        content = page.get_contents().get_data()
        xobjects = page.get("/Resources", {}).get("/XObject", {})
        if hasattr(xobjects, "get_object"):
            xobjects = xobjects.get_object()
        image_count = sum(value.get_object().get("/Subtype") == "/Image" for value in xobjects.values())
        check(image_count == 0 and (b" l" in content or b" re" in content) and (b" Tj" in content or b" TJ" in content), "figure_files", f"{name}: vector path/text, no raster image")
        check(all(f"({letter})" in (page.extract_text() or "") for letter in letters), "figure_files", f"{name}: all {len(letters)} panel labels")
        if hash_v3:
            check("accuracy" in (page.extract_text() or "").lower(), "figure_files", f"{name}: current main figure identifies Accuracy")
        with Image.open(png) as image:
            image.load()
            dpi = image.info.get("dpi", (0., 0.))
            expected_pixels = (round(width * 600), round(height * 600))
            check(png.stat().st_size > 1000 and all(abs(actual - expected) <= 1 for actual, expected in zip(image.size, expected_pixels)),
                  "figure_files", f"{name}: readable PNG dimensions match PDF at 600dpi")
            check(all(abs(value - 600.) < .05 for value in dpi), "figure_files", f"{name}: 600dpi")
            check(any(low < high for low, high in image.convert("RGB").getextrema()), "figure_files", f"{name}: nonblank pixels")
            figure_checks.append({"name": name, "status": "checked", "panel_count": len(letters), "pdf_inches": [width, height],
                                  "pdf_bytes": pdf.stat().st_size, "pdf_embedded_images": image_count,
                                  "png_pixels": list(image.size), "png_dpi": list(dpi), "png_bytes": png.stat().st_size})

    saved_before = json.loads((output / "logs" / "dataset_before.json").read_text(encoding="utf-8"))
    saved_after = json.loads((output / "logs" / "dataset_after.json").read_text(encoding="utf-8"))
    current_after = _snapshot(dataset_root)
    check(saved_before == saved_after == dataset_current_before == current_after, "immutability", "Dataset hashes unchanged across saved and independent snapshots")
    check(results_before == _snapshot(output), "immutability", "Saved FULL artifacts unchanged by audit")
    repeat_path = output / "logs" / "reproducibility_check.json"
    repeatability = {"status": "not_executed", "manifest_value": manifest.get("repeatability_check", "unrecorded")}
    if str(manifest.get("repeatability_check", "")).startswith("not_executed"):
        repeatability.update({"status": "not_executed_as_declared", "historical_log_present": repeat_path.is_file()})
        # Do not reuse a stale log when this manifest explicitly opts out.
    elif repeat_path.is_file():
        saved_repeat = json.loads(repeat_path.read_text(encoding="utf-8"))
        equivalent = bool(saved_repeat.get("raw_rows_identical") and saved_repeat.get("predictions_and_assignments_identical"))
        check(equivalent, "reproducibility_log", "Applicable saved log reports exact equivalence")
        repeatability.update({"status": "verified_from_saved_log" if equivalent else "saved_log_failure", "saved_log": saved_repeat})
    else:
        check(False, "reproducibility_log", "Repeatability log missing without explicit not_executed declaration")
        repeatability["status"] = "missing_undeclared"
    method_statistics = {}
    for method in implemented:
        records = [row for row in summary if row["method"] == method]
        default = next((row for row in records if int(row["N"]) == 3000 and float(row["cr"]) == 3.), None)
        method_statistics[method] = {
            "status_counts": dict(Counter(row["status"] for row in raw if row["method"] == method)),
            "configuration_equal_weight_mean": {metric: _statistics([float(row[f"{metric}_mean"]) for row in records])["mean"] for metric in RATES},
            "default_N3000_cr3": ({metric: {statistic: float(default[f"{metric}_{statistic}"]) for statistic in ("mean", "std", "ci_low", "ci_high", "n")} for metric in RATES} if default else None),
            "subtype_cumulative": {name: {**cumulative_subtypes[(method, name)], "error_rate": _divide(cumulative_subtypes[(method, name)]["errors"], cumulative_subtypes[(method, name)]["count"])} for name in SUBTYPES}}
    total_failures = sum(stage["failures"] for stage in stages.values())
    payload = {
        "result": "PASS" if total_failures == 0 else "FAIL", "scope": "Independent offline saved-FULL audit; no algorithms or production metric helpers executed",
        "total_checks": sum(stage["checks"] for stage in stages.values()), "failure_count": total_failures, "failures": failures, "stages": dict(stages),
        "saved_completion_flag": completion.get("completed"), "methods": list(methods), "implemented_methods": list(implemented),
        "saved_declared_methods": list(saved_declared_methods), "display_methods": list(map(canonical_method_name, methods)),
        "availability_source": availability_source, "source_version": source_version, "saved_source_metadata": saved_source_metadata,
        "artifact_schema_generation": schema_generation,
        "detector_rule_generation": "v4 eta/CR-membership/expanded-block rules" if hash_v4 else
                                    "v3 full-domain/own-tag/root rules" if hash_v3 else "saved legacy rules",
        "coverage": {"data_trials": len(expected_trials), "raw_rows": len(raw), "subtype_rows": len(subtypes), "summary_rows": len(summary),
                     "prediction_files_audited": prediction_count, "successful_prediction_arrays_audited": prediction_arrays,
                     "diagnostic_json_arrays_audited": diagnostic_arrays, "configurations": len(grid), "seeds": list(seeds), "status_counts": dict(status_counts)},
        "figure_panel_data": panel_coverage, "figure_files": figure_checks, "saved_primary_plot_kind": plot_kind,
        "figure_dimensions_inches": [expected_width, expected_height], "figure_dimension_source": dimension_source,
        "subtype_summary_audited": subtype_summary_checked,
        "ignored_legacy_figures": ignored_legacy_figures, "repeatability": repeatability,
        "dataset_files_hashed": len(current_after), "full_result_files_hashed": len(results_before), "method_statistics": method_statistics,
        "limits": ["Saved evidence/arithmetic checks do not establish scientific validity of synthetic descriptors or cryptographic guarantees.",
                   "Availability/source identity come from saved metadata; new source implementations do not upgrade historical results.",
                   "not_executed is not a passed reproducibility check; this script never reruns same-seed algorithms.",
                   "PDF/PNG structural checks do not replace visual review; v3 requires only D1/D2 Accuracy figures and ignores older extra images.",
                   "Split-axis defaults do not constrain data: actual bands can expand or fall back to a full axis to prevent hidden observations.",
                   "CIs condition on successful defined seed metrics and describe fixed-bank resampling, not independent real deployments."]}
    report_root.mkdir(parents=True, exist_ok=True)
    (report_root / "01_detection_result_audit.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# 第一组 FULL 保存结果独立审计", "", f"**结果：{payload['result']}**；{payload['total_checks']:,} 项比较，失败 {total_failures} 项。", "",
             "只读取保存CSV/NPZ/日志/Figure，没有运行检测算法或实验，没有修改数据集及结果。", "",
             f"保存来源版本：`{source_version}`；可用方法依据：{availability_source}。",
             f"本次保存实现方法：{', '.join(implemented)}；当前新增源码不改变历史结果身份。", "",
             f"- {len(grid)} 个配置×{len(seeds)} 个seed={len(expected_trials):,} 个数据试验，{len(raw):,} 个方法记录。",
             f"- 实际状态：{dict(status_counts)}。仅success要求完整预测；失败/GT不可用/unsupported保留原因、零评估计数及NaN指标。",
             f"- 核验 {prediction_count:,} 个NPZ、{prediction_arrays:,} 个成功预测数组、{diagnostic_arrays:,} 个Unicode JSON诊断数组。",
             (f"- 保存的{source_version}仅{RACA_HIM}核验GT与单verifier；I-EDI/CTCV核验pool参考映射；Dyna核验raw-hash witness映射及保存版本的root/展开block规则，不强制单verifier数组。" if hash_v3 else
              f"- 历史{RACA_HIM}/Single核验GT与单verifier；I-EDI/CTCV核验pool参考映射；Dyna核验所有witness映射及quorum，不强制单verifier数组。"),
             "- 状态数量/GT尝试数由raw重算；有效n及均值/标准差/95%CI仅由成功且定义有效seed复算。",
             f"- D1/D2五个逻辑panel各九个真实配置；保存主图类型：{plot_kind}。v3采用上下分段坐标，不把十个物理axes误算为十个子图。",
             (f"- 保存hash版本仅审计D1/D2 Accuracy图；不要求subtype_summary、D3或diagnostic图，历史遗留额外图忽略：{ignored_legacy_figures}。" if hash_v3 else
              f"- 历史subtype_summary审计：{subtype_summary_checked}；旧D3/diagnostic按保存版本检查，不改成当前图。"),
             f"- 主图尺寸来源：{dimension_source}，尺寸{expected_width}×{expected_height}英寸；PNG匹配PDF和600dpi，不固定历史图高度。",
             f"- 名称显示统一为{RACA_HIM}；原始保存身份和NPZ前缀仍按raw读取，未修改历史CSV/NPZ。",
             f"- 复现对照状态：{repeatability['status']}；未执行不等于已通过。",
             f"- dataset {len(current_after)} 文件及FULL结果 {len(results_before):,} 文件在审计前后字节未变。", "",
             "## 实际方法状态", "", "| 方法 | success | gt_unavailable | unsupported | failure |", "|---|---:|---:|---:|---:|"]
    for method in methods:
        counts = Counter(row["status"] for row in raw if row["method"] == method)
        lines.append("| " + canonical_method_name(method) + " | " + " | ".join(str(counts[status]) for status in STATUSES) + " |")
    lines.extend(["", "## 各配置均值的等权汇总", "", "单位%。只平均定义有效配置均值，不给跨配置独立总体CI。", "",
                  "| 方法 | Accuracy | Precision | Recall | F1 | FPR | FNR |", "|---|---:|---:|---:|---:|---:|---:|"])
    for method, statistics in method_statistics.items():
        lines.append("| " + canonical_method_name(method) + " | " + " | ".join(f"{100*statistics['configuration_equal_weight_mean'][metric]:.6f}" for metric in RATES) + " |")
    lines.extend(["", "## 默认配置 N=3000、cr=3%", "", "均值[95%CI]，单位%；每项有效seed数动态记录。", "",
                  "| 方法 | Accuracy | Precision | Recall | F1 | FPR | FNR |", "|---|---:|---:|---:|---:|---:|---:|"])
    for method, statistics in method_statistics.items():
        default = statistics["default_N3000_cr3"]
        if default is None:
            continue
        cells = [f"{100*default[metric]['mean']:.6f} [{100*default[metric]['ci_low']:.6f}, {100*default[metric]['ci_high']:.6f}], n={int(default[metric]['n'])}" for metric in RATES]
        lines.append("| " + canonical_method_name(method) + " | " + " | ".join(cells) + " |")
    lines.extend(["", "## 审计边界", "", *[f"- {limit}" for limit in payload["limits"]], ""])
    if failures:
        lines.extend(["## 失败项", "", *[f"- {failure['stage']}: {failure['finding']}" for failure in failures], ""])
    (report_root / "01_detection_result_audit.md").write_text("\n".join(lines), encoding="utf-8")
    return payload


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "results" / "01_detection")
    parser.add_argument("--reports", type=Path, default=PROJECT_ROOT / "reports")
    args = parser.parse_args(argv)
    report = audit(args.output, args.reports)
    print(json.dumps({name: report[name] for name in ("result", "total_checks", "failure_count", "coverage")}, ensure_ascii=False, indent=2))
    if report["failure_count"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
