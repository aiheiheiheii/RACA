"""Computation identities and verified continuation within one protocol version.

The full source inventory remains provenance. Checkpoint compatibility excludes
the runner, persistence, plots and tests, but includes every Group 02 computation
module and the actual Group 01/common dependencies. A source added or removed in
that computation set invalidates compatibility just like changed source bytes.
"""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
import hashlib
import json
from pathlib import Path, PurePosixPath

from .config_overhead import PROJECT_ROOT, OVERHEAD_IMPLEMENTATION_VERSION
from .overhead_run_setup import validate_calibration_snapshot, calibration_snapshot_sha256


_OUTPUT_MODULES = frozenset({
    "overhead_experiment.py", "overhead_storage.py", "overhead_identity.py",
    "run_overhead.py", "plot_overhead.py",
    # Independent encoded-communication replay never changes the timed v4
    # simulation; its sources are bound to its own communication fingerprint.
    "raca_wire_codec.py", "overhead_communication.py",
})
_DETECTION_DEPENDENCIES = frozenset({
    "__init__.py", "proposed_detection.py", "adaptation_utils.py",
    "iedi_adapted.py", "ctcv_adapted.py",
})
_MATCH_FIELDS = (
    "implementation_version", "config", "N_values", "d_panel_values", "seeds",
    "methods", "evaluation_modes", "main_evaluation_mode", "dataset_snapshot",
    "environment", "threadpool_limit", "kernel_calibration_policy",
    "kernel_calibrations", "kernel_calibration_sha256",
)
_SOURCE_FIELDS = (
    "source_sha256", "source_files_sha256", "computation_source_sha256",
    "computation_source_files_sha256", "orchestration_contract_sha256",
)
# The interrupted v2 release's driver was inspected before this IO repair.
# A previously unrecorded driver contract is admitted only for this exact
# known release, never for arbitrary legacy orchestration source.
_KNOWN_LEGACY_DRIVERS = {
    '85033f24ebea1e533607b27d665e71ca98c29ba0ad9e7620ef70af7f43f0a203':
    'aa42f4c7d2bfc608cfae3e2f6e86159afa83a1f3918436532cf5d7d0dec536aa',
}


def checkpoint_identity_source(computation_hash, orchestration_hash=None):
    if orchestration_hash is None:
        return computation_hash
    return hashlib.sha256((computation_hash+'\n'+orchestration_hash).encode('ascii')).hexdigest()
_SCHEMA_FIELDS = ("identity_schema", "run_fingerprint_schema", "fingerprint_schema")


def _serialized(value):
    # JSON treats tuple/list grids identically, while preserving scalar types.
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _is_computation_source(relative):
    parts = PurePosixPath(relative).parts
    if relative == "dataset/load_dataset.py":
        return True
    if len(parts) != 3 or parts[0] != "experiments" or not parts[2].endswith(".py"):
        return False
    if parts[1] == "02_overhead":
        return parts[2] not in _OUTPUT_MODULES and not parts[2].startswith("test_")
    if parts[1] == "common":
        return parts[2] != "plotting_style.py"
    return parts[1] == "01_detection" and parts[2] in _DETECTION_DEPENDENCIES


def _inventory_hash(records):
    if not isinstance(records, Mapping) or not records:
        raise ValueError("missing source file inventory")
    digest = hashlib.sha256()
    for relative, file_hash in sorted(records.items()):
        if (not isinstance(relative, str) or ":" in relative or "\\" in relative
                or PurePosixPath(relative).is_absolute()
                or ".." in PurePosixPath(relative).parts
                or PurePosixPath(relative).as_posix() != relative):
            raise ValueError("invalid relative source path")
        if (not isinstance(file_hash, str) or len(file_hash) != 64
                or any(char not in "0123456789abcdef" for char in file_hash)):
            raise ValueError("invalid source file SHA256")
        digest.update(relative.encode("utf-8") + b"\0" + file_hash.encode("ascii") + b"\n")
    return digest.hexdigest()


def computation_source_fingerprint(project_root=PROJECT_ROOT):
    """Return SHA256 and file inventory of the computation dependency set."""
    root = Path(project_root)
    sources = set((root / "experiments" / "02_overhead").glob("*.py"))
    sources.update((root / "experiments" / "common").glob("*.py"))
    sources.update(root / "experiments" / "01_detection" / name
                   for name in _DETECTION_DEPENDENCIES)
    sources.add(root / "dataset" / "load_dataset.py")
    records = {}
    for path in sorted(sources, key=lambda item: item.as_posix()):
        relative = path.relative_to(root).as_posix()
        if _is_computation_source(relative):
            records[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    return _inventory_hash(records), records


def _grid(context):
    grid = [(n, d, seed) for n in context["N_values"]
            for d in context["d_panel_values"] for seed in context["seeds"]]
    if not grid or ("grid" in context and _serialized(context["grid"]) != _serialized(grid)):
        raise ValueError("grid does not match declared N/d/seed values")
    return grid


def _identity_hash(context, source_hash):
    material = {
        "implementation_version": context["implementation_version"],
        "config": context["config"], "grid": _grid(context),
        "dataset": context["dataset_snapshot"], "source_sha256": source_hash,
        "environment": context["environment"], "threadpool_limit": context["threadpool_limit"],
    }
    if "kernel_calibration_sha256" in context:
        material["kernel_calibration_sha256"] = context["kernel_calibration_sha256"]
        material["kernel_calibration_policy"] = context["kernel_calibration_policy"]
    return hashlib.sha256(_serialized(material).encode("utf-8")).hexdigest()


def _validate_calibration_context(context):
    records, policy = context["kernel_calibrations"], context["kernel_calibration_policy"]
    validate_calibration_snapshot(records, policy, context["seeds"], context["d_panel_values"])
    if calibration_snapshot_sha256(records, policy) != context["kernel_calibration_sha256"]:
        raise ValueError("kernel_calibration_sha256 does not match its actual snapshot/policy")


def _schema(candidate):
    values = {candidate[key] for key in _SCHEMA_FIELDS if key in candidate}
    if len(values) > 1:
        raise ValueError("contradictory identity schemas")
    value = next(iter(values), "full-source-v1")
    if value not in {"full-source-v1", "legacy-full-v1", "full-source", "computation-v1"}:
        raise ValueError("unsupported identity schema")
    return value


def _verified_snapshot(candidate, current):
    if not isinstance(candidate, Mapping):
        raise ValueError("source snapshot is not an object")
    if candidate.get("implementation_version") != OVERHEAD_IMPLEMENTATION_VERSION:
        raise ValueError("incompatible implementation version; measurements from another protocol cannot be resumed")
    _validate_calibration_context(candidate)
    for key in _MATCH_FIELDS:
        if key not in candidate or key not in current or _serialized(candidate[key]) != _serialized(current[key]):
            raise ValueError(f"different or missing {key}")
    if candidate["threadpool_limit"] != 1:
        raise ValueError("unsupported threadpool limit")
    _grid(candidate)
    full_records = candidate.get("source_files_sha256")
    full_hash = _inventory_hash(full_records)
    if full_hash != candidate.get("source_sha256"):
        raise ValueError("full source inventory SHA256 mismatch")
    projected = {key: value for key, value in full_records.items() if _is_computation_source(key)}
    if projected != current["computation_source_files_sha256"]:
        raise ValueError("computation source files changed, added, removed, or absent")
    schema = _schema(candidate)
    current_contract=current.get('orchestration_contract_sha256')
    recorded_contract=candidate.get('orchestration_contract_sha256')
    if current_contract is not None:
        if recorded_contract is not None:
            if recorded_contract!=current_contract:
                raise ValueError('orchestration computation inputs changed')
        elif (schema=='computation-v1' or _KNOWN_LEGACY_DRIVERS.get(
                full_records.get('experiments/02_overhead/overhead_experiment.py'))!=current_contract):
            raise ValueError('legacy orchestration contract is not a verified release')
    elif recorded_contract is not None:
        raise ValueError('current context lacks the recorded orchestration contract')
    if schema == "computation-v1":
        computation_records = candidate.get("computation_source_files_sha256")
        computation_hash = _inventory_hash(computation_records)
        if (computation_hash != candidate.get("computation_source_sha256")
                or computation_records != projected):
            raise ValueError("computation source inventory SHA256 mismatch")
        identity_source = checkpoint_identity_source(computation_hash,recorded_contract)
    else:
        identity_source = full_hash
    if candidate.get("run_fingerprint") != _identity_hash(candidate, identity_source):
        raise ValueError("run fingerprint does not match its saved configuration/source/data/environment")
    fields = _MATCH_FIELDS + _SOURCE_FIELDS + _SCHEMA_FIELDS + ("run_fingerprint", "started_utc")
    snapshot = {key: deepcopy(candidate[key]) for key in fields if key in candidate}
    snapshot["identity_schema"] = schema
    snapshot["grid"] = _grid(candidate)
    return snapshot


def verified_checkpoint_sources(previous_manifest, current_context):
    """Return ``(accepted_snapshots, rejected_fingerprint_to_reason)``.

    Existing compatible_checkpoint_sources are independently rechecked and
    flattened. No checkpoint or manifest is edited. Status/count/timestamp
    changes cannot authorize compatibility; computation, config, data, grid,
    environment and the original fingerprint must all verify.
    """
    if previous_manifest is None:
        return [], {}
    try:
        _grid(current_context)
        _validate_calibration_context(current_context)
        if (_inventory_hash(current_context["computation_source_files_sha256"])
                != current_context["computation_source_sha256"]):
            raise ValueError("current computation source inventory SHA256 mismatch")
        if any(not _is_computation_source(key) for key in current_context["computation_source_files_sha256"]):
            raise ValueError("current computation inventory contains excluded modules")
    except (KeyError, TypeError, ValueError) as exc:
        return [], {"current_context": str(exc)}
    accepted, rejected, fingerprints, visited = [], {}, set(), set()
    pending = [previous_manifest]
    while pending:
        candidate = pending.pop(0)
        if id(candidate) in visited:
            continue
        visited.add(id(candidate))
        fingerprint = (candidate.get("run_fingerprint", "missing_run_fingerprint")
                       if isinstance(candidate, Mapping) else "invalid_snapshot")
        if isinstance(candidate, Mapping):
            inherited = candidate.get("compatible_checkpoint_sources", [])
            if isinstance(inherited, list):
                pending.extend(inherited)
        try:
            snapshot = _verified_snapshot(candidate, current_context)
        except (KeyError, TypeError, ValueError) as exc:
            rejected[str(fingerprint)] = str(exc)
            continue
        if fingerprint not in fingerprints:
            accepted.append(snapshot)
            fingerprints.add(fingerprint)
    # A verified duplicate may supersede malformed provenance with the same ID.
    for fingerprint in fingerprints:
        rejected.pop(fingerprint, None)
    return accepted, rejected
