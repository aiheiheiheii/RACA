"""Continuation accepts only independently verifiable computation identities."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import importlib
import json

import pytest

identity = importlib.import_module("experiments.02_overhead.overhead_identity")
setup = importlib.import_module("experiments.02_overhead.overhead_run_setup")
calibration_module = importlib.import_module("experiments.02_overhead.overhead_calibration")


def _inventory_digest(records):
    material = b"".join(name.encode() + b"\0" + digest.encode() + b"\n"
                        for name, digest in sorted(records.items()))
    return hashlib.sha256(material).hexdigest()


def _run_digest(manifest, source):
    grid = [(n, d, s) for n in manifest["N_values"]
            for d in manifest["d_panel_values"] for s in manifest["seeds"]]
    material = {"implementation_version": manifest["implementation_version"],
                "config": manifest["config"], "grid": grid,
                "dataset": manifest["dataset_snapshot"], "source_sha256": source,
                "environment": manifest["environment"],
                "threadpool_limit": manifest["threadpool_limit"]}
    if "kernel_calibration_sha256" in manifest:
        material["kernel_calibration_sha256"] = manifest["kernel_calibration_sha256"]
        material["kernel_calibration_policy"] = manifest["kernel_calibration_policy"]
    return hashlib.sha256(json.dumps(material, sort_keys=True,
                                    separators=(",", ":")).encode()).hexdigest()


def _calibration_record(seed,d,factor=1.):
    costs=dict(cosine_per_pair_ms=.001*factor,sha256_per_descriptor_ms=.002*factor,
               scalar_compare_ms=.0001*factor,scalar_modulo_ms=.0002*factor,
               vector_per_byte_ms=.000001*factor)
    timings=dict(cosine=costs['cosine_per_pair_ms']*calibration_module.COSINE_BATCH_PAIRS,
                 sha256=costs['sha256_per_descriptor_ms']*calibration_module.HASH_BATCH_DESCRIPTORS,
                 scalar_compare=costs['scalar_compare_ms']*calibration_module.SCALAR_BATCH_ITEMS,
                 scalar_modulo=costs['scalar_modulo_ms']*calibration_module.SCALAR_BATCH_ITEMS,
                 vector=costs['vector_per_byte_ms']*calibration_module.VECTOR_BATCH_BYTES)
    return calibration_module.KernelCalibration(
        costs,seed,d,(setup.CALIBRATION_POPULATION_N,d),
        hashlib.sha256(f'synthetic-{seed}-{d}'.encode()).hexdigest(),
        calibration_module.CALIBRATION_SAMPLE_LIMIT,sum(timings.values()),timings).to_dict()


@pytest.fixture
def source_tree(tmp_path):
    paths = [
        "dataset/load_dataset.py", "experiments/02_overhead/__init__.py",
        "experiments/02_overhead/config_overhead.py",
        "experiments/02_overhead/communication_counter.py",
        "experiments/02_overhead/overhead_baselines.py",
        "experiments/02_overhead/overhead_simulator.py",
        "experiments/02_overhead/overhead_experiment.py",
        "experiments/02_overhead/run_overhead.py",
        "experiments/02_overhead/plot_overhead.py",
        "experiments/02_overhead/test_overhead.py",
        "experiments/common/config.py", "experiments/common/reproducibility.py",
        "experiments/common/plotting_style.py",
        "experiments/01_detection/__init__.py",
        "experiments/01_detection/proposed_detection.py",
        "experiments/01_detection/adaptation_utils.py",
        "experiments/01_detection/iedi_adapted.py",
        "experiments/01_detection/ctcv_adapted.py",
        "experiments/01_detection/run_detection.py",
    ]
    for name in paths:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# " + name + "\n", encoding="utf-8")
    return tmp_path


def _full_inventory(root):
    return {path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in root.rglob("*.py")}


@pytest.fixture
def manifests(source_tree):
    computation_hash, computation_records = identity.computation_source_fingerprint(source_tree)
    full_records = _full_inventory(source_tree)
    full_hash = _inventory_digest(full_records)
    previous = {
        "implementation_version": identity.OVERHEAD_IMPLEMENTATION_VERSION, "config": {"H": .5, "dim": 30},
        "N_values": [1000, 1500], "d_panel_values": [10, 20], "seeds": [2026, 2027],
        "methods": ["RACA-HIM", "DVA-P-Adapted", "I-EDI-Adapted", "CTCV-Adapted"],
        "evaluation_modes": ["native_budget", "fixed_workload"],
        "main_evaluation_mode": "fixed_workload", "dataset_snapshot": {"bank.npz": "a" * 64},
        "environment": {"python": "3.12.4", "python_executable": "project/.venv/python.exe"},
        "threadpool_limit": 1, "source_sha256": full_hash, "source_files_sha256": full_records,
        "started_utc": "2026-10-12T09:07:00+00:00", "status": "running",
        "executed_configurations_this_invocation": 877,
    }
    records={setup.calibration_key(seed,d):_calibration_record(seed,d)
             for seed in previous['seeds'] for d in previous['d_panel_values']}
    policy=setup.calibration_policy()
    previous.update(kernel_calibrations=records,kernel_calibration_policy=policy,
                    kernel_calibration_sha256=setup.calibration_snapshot_sha256(records,policy))
    previous["run_fingerprint"] = _run_digest(previous, full_hash)
    current = {key: deepcopy(previous[key]) for key in identity._MATCH_FIELDS}
    current.update(computation_source_sha256=computation_hash,
                   computation_source_files_sha256=computation_records)
    return previous, current


def test_accepts_original_legacy_identity_and_retains_provenance(manifests):
    previous, current = manifests
    original = deepcopy(previous)
    accepted, rejected = identity.verified_checkpoint_sources(previous, current)
    assert not rejected and len(accepted) == 1
    snapshot = accepted[0]
    assert snapshot["run_fingerprint"] == previous["run_fingerprint"]
    assert snapshot["source_files_sha256"] == previous["source_files_sha256"]
    assert snapshot["identity_schema"] == "full-source-v1"
    assert "status" not in snapshot and "executed_configurations_this_invocation" not in snapshot
    snapshot["config"]["H"] = .7
    assert previous == original


@pytest.mark.parametrize("path", [
    "experiments/02_overhead/overhead_experiment.py",
    "experiments/02_overhead/run_overhead.py", "experiments/02_overhead/plot_overhead.py",
    "experiments/02_overhead/test_overhead.py", "experiments/common/plotting_style.py",
])
def test_output_or_test_change_preserves_computation_identity(path, source_tree, manifests):
    previous, current = manifests
    (source_tree / path).write_text("# revised output code\n", encoding="utf-8")
    digest, records = identity.computation_source_fingerprint(source_tree)
    assert digest == current["computation_source_sha256"]
    current.update(computation_source_sha256=digest, computation_source_files_sha256=records)
    assert identity.verified_checkpoint_sources(previous, current)[0]


@pytest.mark.parametrize("path", [
    "experiments/02_overhead/overhead_storage.py",
    "experiments/02_overhead/overhead_identity.py",
    "experiments/02_overhead/test_overhead_identity.py",
])
def test_adding_transport_modules_does_not_recompute_trials(path, source_tree, manifests):
    previous, current = manifests
    (source_tree / path).write_text("# persistence helper\n", encoding="utf-8")
    digest, records = identity.computation_source_fingerprint(source_tree)
    assert digest == current["computation_source_sha256"]
    current.update(computation_source_sha256=digest, computation_source_files_sha256=records)
    assert identity.verified_checkpoint_sources(previous, current)[0]


@pytest.mark.parametrize("path", [
    "experiments/02_overhead/overhead_simulator.py",
    "experiments/02_overhead/overhead_baselines.py",
    "experiments/02_overhead/new_algorithm.py",
    "experiments/common/reproducibility.py",
    "experiments/01_detection/proposed_detection.py",
    "experiments/01_detection/ctcv_adapted.py",
    "experiments/01_detection/iedi_adapted.py", "dataset/load_dataset.py",
])
def test_computation_change_or_addition_rejects_old_trials(path, source_tree, manifests):
    previous, current = manifests
    (source_tree / path).write_text("# changed computation\n", encoding="utf-8")
    digest, records = identity.computation_source_fingerprint(source_tree)
    current.update(computation_source_sha256=digest, computation_source_files_sha256=records)
    accepted, rejected = identity.verified_checkpoint_sources(previous, current)
    assert not accepted and "computation source" in rejected[previous["run_fingerprint"]]


@pytest.mark.parametrize(("field", "value"), [
    ("implementation_version", "overhead-events-v1"), ("config", {"H": .6, "dim": 30}),
    ("N_values", [1000, 2000]), ("d_panel_values", [10, 30]), ("seeds", [2026, 2028]),
    ("methods", ["Proposed"]), ("evaluation_modes", ["fixed_workload"]),
    ("main_evaluation_mode", "native_budget"), ("dataset_snapshot", {"bank.npz": "b" * 64}),
    ("environment", {"python": "3.13"}), ("threadpool_limit", 2),
])
def test_changed_experimental_context_rejects_legacy(field, value, manifests):
    previous, current = manifests
    current[field] = value
    accepted, rejected = identity.verified_checkpoint_sources(previous, current)
    assert not accepted and rejected


@pytest.mark.parametrize("mutation", ["source_digest", "run_fingerprint", "missing_inventory",
                                      "unknown_schema", "conflicting_schema", "grid"])
def test_unverifiable_manifest_cannot_authorize_reuse(mutation, manifests):
    previous, current = manifests
    if mutation == "source_digest":
        previous["source_sha256"] = "0" * 64
    elif mutation == "run_fingerprint":
        previous["run_fingerprint"] = "0" * 64
    elif mutation == "missing_inventory":
        del previous["source_files_sha256"]
    elif mutation == "unknown_schema":
        previous["identity_schema"] = "unrecognized-v9"
    elif mutation == "conflicting_schema":
        previous.update(identity_schema="computation-v1", fingerprint_schema="full-source-v1")
    else:
        previous["grid"] = [(1000, 10, 2026)]
    accepted, rejected = identity.verified_checkpoint_sources(previous, current)
    assert not accepted and rejected


def _as_computation_manifest(previous, current):
    candidate = deepcopy(previous)
    candidate.update(identity_schema="computation-v1",
                     computation_source_sha256=current["computation_source_sha256"],
                     computation_source_files_sha256=deepcopy(current["computation_source_files_sha256"]))
    candidate["run_fingerprint"] = _run_digest(candidate, candidate["computation_source_sha256"])
    return candidate


def test_current_schema_and_legacy_chain_verify_independently_and_flatten(manifests):
    previous, current = manifests
    candidate = _as_computation_manifest(previous, current)
    candidate["compatible_checkpoint_sources"] = [previous, deepcopy(previous)]
    before = deepcopy(candidate)
    accepted, rejected = identity.verified_checkpoint_sources(candidate, current)
    assert not rejected and len(accepted) == 2
    assert {snapshot["run_fingerprint"] for snapshot in accepted} == {
        candidate["run_fingerprint"], previous["run_fingerprint"]}
    assert all("compatible_checkpoint_sources" not in snapshot for snapshot in accepted)
    assert candidate == before


def test_forged_inherited_compatibility_is_not_trusted(manifests):
    previous, current = manifests
    candidate = _as_computation_manifest(previous, current)
    previous["environment"] = {"python": "wrong"}
    candidate["compatible_checkpoint_sources"] = [previous]
    accepted, rejected = identity.verified_checkpoint_sources(candidate, current)
    assert len(accepted) == 1 and previous["run_fingerprint"] in rejected


def test_current_schema_rejects_forged_computation_inventory(manifests):
    previous, current = manifests
    candidate = _as_computation_manifest(previous, current)
    candidate["computation_source_files_sha256"]["dataset/load_dataset.py"] = "0" * 64
    accepted, rejected = identity.verified_checkpoint_sources(candidate, current)
    assert not accepted and "inventory SHA256 mismatch" in rejected[candidate["run_fingerprint"]]


def test_invalid_current_identity_rejects_all_candidates(manifests):
    previous, current = manifests
    current["computation_source_sha256"] = "0" * 64
    accepted, rejected = identity.verified_checkpoint_sources(previous, current)
    assert not accepted and "current_context" in rejected


def test_tuple_and_json_list_grids_are_equivalent(manifests):
    previous, current = manifests
    current["grid"] = tuple((n, d, s) for n in current["N_values"]
                            for d in current["d_panel_values"] for s in current["seeds"])
    assert identity.verified_checkpoint_sources(previous, current)[0]


def test_old_v1_cannot_resume_even_when_context_also_claims_v1(manifests):
    previous, current = manifests
    previous["implementation_version"] = current["implementation_version"] = "overhead-events-v1"
    previous["run_fingerprint"] = _run_digest(previous, previous["source_sha256"])
    accepted, rejected = identity.verified_checkpoint_sources(previous, current)
    assert not accepted and "incompatible implementation version" in rejected[previous["run_fingerprint"]]


def test_no_prior_manifest_means_no_compatible_sources():
    assert identity.verified_checkpoint_sources(None, {}) == ([], {})


def _known_legacy_driver(previous, current):
    driver_hash, contract = next(iter(identity._KNOWN_LEGACY_DRIVERS.items()))
    previous['source_files_sha256']['experiments/02_overhead/overhead_experiment.py'] = driver_hash
    previous['source_sha256'] = _inventory_digest(previous['source_files_sha256'])
    previous['run_fingerprint'] = _run_digest(previous, previous['source_sha256'])
    current['orchestration_contract_sha256'] = contract
    return contract


def _with_contract(previous, current):
    contract = _known_legacy_driver(previous, current)
    candidate = _as_computation_manifest(previous, current)
    candidate['orchestration_contract_sha256'] = contract
    source = identity.checkpoint_identity_source(candidate['computation_source_sha256'], contract)
    candidate['run_fingerprint'] = _run_digest(candidate, source)
    return candidate


def test_known_interrupted_driver_is_verified_for_io_only_migration(manifests):
    previous, current = manifests
    _known_legacy_driver(previous, current)
    accepted, rejected = identity.verified_checkpoint_sources(previous, current)
    assert len(accepted) == 1 and not rejected


def test_unknown_legacy_driver_cannot_skip_orchestration_verification(manifests):
    previous, current = manifests
    current['orchestration_contract_sha256'] = next(iter(identity._KNOWN_LEGACY_DRIVERS.values()))
    accepted, rejected = identity.verified_checkpoint_sources(previous, current)
    assert not accepted and 'not a verified release' in rejected[previous['run_fingerprint']]


def test_legacy_driver_with_changed_measurement_arguments_is_rejected(manifests):
    previous, current = manifests
    _known_legacy_driver(previous, current)
    current['orchestration_contract_sha256'] = '0' * 64
    assert not identity.verified_checkpoint_sources(previous, current)[0]


def test_new_identity_binds_computation_and_driver_contract(manifests):
    previous, current = manifests
    candidate = _with_contract(previous, current)
    accepted, rejected = identity.verified_checkpoint_sources(candidate, current)
    assert len(accepted) == 1 and not rejected
    assert accepted[0]['orchestration_contract_sha256'] == current['orchestration_contract_sha256']


@pytest.mark.parametrize('mutation', ['changed', 'missing', 'fingerprint_without_contract'])
def test_new_identity_rejects_changed_or_missing_contract(mutation, manifests):
    previous, current = manifests
    candidate = _with_contract(previous, current)
    if mutation == 'changed':
        candidate['orchestration_contract_sha256'] = '0' * 64
    elif mutation == 'missing':
        del candidate['orchestration_contract_sha256']
    else:
        candidate['run_fingerprint'] = _run_digest(candidate, candidate['computation_source_sha256'])
    assert not identity.verified_checkpoint_sources(candidate, current)[0]


def test_recorded_driver_contract_requires_current_contract(manifests):
    previous, current = manifests
    candidate = _with_contract(previous, current)
    del current['orchestration_contract_sha256']
    assert not identity.verified_checkpoint_sources(candidate, current)[0]


def test_verified_contract_chain_preserves_both_origins(manifests):
    previous, current = manifests
    candidate = _with_contract(previous, current)
    candidate['compatible_checkpoint_sources'] = [previous]
    accepted, rejected = identity.verified_checkpoint_sources(candidate, current)
    assert len(accepted) == 2 and not rejected


def test_contract_changes_identity_source():
    assert identity.checkpoint_identity_source('a' * 64, 'b' * 64) != identity.checkpoint_identity_source('a' * 64, 'c' * 64)


@pytest.mark.parametrize('mutation', ['input_transform', 'seed_binding', 'threadpool', 'control_flow'])
def test_driver_contract_covers_bindings_and_control_flow_not_only_calls(monkeypatch, mutation):
    runner = importlib.import_module('experiments.02_overhead.overhead_experiment')
    original = runner.Path(runner.__file__).read_text(encoding='utf-8-sig')
    expected = runner._orchestration_contract()
    if mutation == 'input_transform':
        revised = original.replace('results=[_result_record(result,n,d,seed,point_config,fingerprint)',
                                   "group['X']=np.zeros_like(group['X'])\n                    results=[_result_record(result,n,d,seed,point_config,fingerprint)")
    elif mutation == 'seed_binding':
        revised = original.replace('tick=time.perf_counter()', 'seed+=1\n                tick=time.perf_counter()')
    elif mutation == 'threadpool':
        revised = original.replace('threadpool_limits(limits=1)', 'threadpool_limits(limits=2)')
    else:
        revised = original.replace('if fatal_reason:', 'if not fatal_reason:')
    assert revised != original
    monkeypatch.setattr(runner.Path, 'read_text', lambda *args, **kwargs: revised)
    assert runner._orchestration_contract() != expected
