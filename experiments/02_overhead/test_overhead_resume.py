"""Synthetic runner integration: continuation and output failures only.

Every output is under pytest's temporary directory. The dataset loader and
simulator are replaced; no real paper configuration, dataset or results are
read or written. Atomic JSON/gzip commits and result-directory locking remain
real, including injected Windows replacement failures.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
import gzip
import hashlib
import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

runner = importlib.import_module("experiments.02_overhead.overhead_experiment")
simulator = importlib.import_module("experiments.02_overhead.overhead_simulator")
identity = importlib.import_module("experiments.02_overhead.overhead_identity")
storage = importlib.import_module("experiments.02_overhead.overhead_storage")
plotter = importlib.import_module("experiments.02_overhead.plot_overhead")
cli = importlib.import_module("experiments.02_overhead.run_overhead")
setup = importlib.import_module("experiments.02_overhead.overhead_run_setup")
calibration_module = importlib.import_module("experiments.02_overhead.overhead_calibration")
_ORIGINAL_MATERIALIZE = runner.materialize_outputs


def _inventory_digest(records):
    material = b"".join(name.encode() + b"\0" + digest.encode() + b"\n"
                        for name, digest in sorted(records.items()))
    return hashlib.sha256(material).hexdigest()


def _synthetic_calibration(seed, d):
    """An internally consistent fixed-cost record, never a real timing claim."""
    costs = dict(cosine_per_pair_ms=.001, sha256_per_descriptor_ms=.002,
                 scalar_compare_ms=.0001, scalar_modulo_ms=.0002,
                 vector_per_byte_ms=.000001)
    timings = dict(cosine=costs['cosine_per_pair_ms']*calibration_module.COSINE_BATCH_PAIRS,
                   sha256=costs['sha256_per_descriptor_ms']*calibration_module.HASH_BATCH_DESCRIPTORS,
                   scalar_compare=costs['scalar_compare_ms']*calibration_module.SCALAR_BATCH_ITEMS,
                   scalar_modulo=costs['scalar_modulo_ms']*calibration_module.SCALAR_BATCH_ITEMS,
                   vector=costs['vector_per_byte_ms']*calibration_module.VECTOR_BATCH_BYTES)
    return calibration_module.KernelCalibration(
        costs, seed, d, (setup.CALIBRATION_POPULATION_N,d),
        hashlib.sha256(f'synthetic-{seed}-{d}'.encode()).hexdigest(),
        calibration_module.CALIBRATION_SAMPLE_LIMIT,sum(timings.values()),timings)


def _synthetic_results(config):
    """One coherent core workload per method/mode (B=1)."""
    target = 100
    workload = hashlib.sha256(np.asarray([target], dtype="<i8").tobytes()).hexdigest()
    for method in runner.METHODS:
        for mode in runner.EVALUATION_MODES:
            row = {
                "method": method, "evaluation_mode": mode, "B": config.budget(1000),
                "fixed_workload_sha256": workload, "selected_count": 1,
                "attempted_count": 1, "completed_count": 1, "deadline_violations": 0,
                "gt_success": True, "gt_applicable": method == "RACA-HIM",
                "gt_time_ms": 1., "decision_cpu_ms": .1, "verification_time_ms": 2.,
                "total_time_ms": 3.1, "gt_bytes": 0, "verification_bytes": 64,
                "control_bytes": 32, "total_bytes": 96, "message_count": 1,
                "status": "success", "reason": "",
                "within_deadline_count": 1, "deadline_met": True,
                "core_phase_counts_json": json.dumps({"verification": {
                    "message_count": 1, "payload_bytes": 64, "control_bytes": 32,
                    "total_bytes": 96, "operations": {"synthetic_compare": 1}}}),
            }
            message = {"message_id": 0, "sender": 5, "receiver": target,
                       "phase": "verification", "message_type": "synthetic_response",
                       "payload_bytes": 64, "control_bytes": 32, "total_bytes": 96,
                       "start_ms": 0., "end_ms": 1., "transfer_ms": .1, "network_ms": .9}
            task = {"task_id": 0, "target_row": target, "verifier_row": 5,
                    "start_ms": 0., "end_ms": 2., "task_ms": 2., "prediction": 0,
                    "completed": True, "deadline_violation": False}
            runtime = {"phase": "verification", "node": 5, "operation": "synthetic_compare",
                       "input_bytes": 64, "comparisons": 1, "start_ms": 0., "end_ms": .1,
                       "simulated_ms": .1, "cpu_ms": .1, "measurement_kind": "synthetic"}
            diagnostics = {"required_fixed_target_rows": [target],
                           "unscheduled_fixed_target_rows": []}
            yield simulator.RunResult(row, [message], [runtime], [task], diagnostics)


@pytest.fixture
def synthetic_runner(tmp_path, monkeypatch):
    grid = ((1000, 10, 2026), (1000, 10, 2027))
    config = runner.OverheadConfig(dim=10, beta_B=.001)
    assert config.budget(1000) == 1
    snapshot = {"synthetic-bank": "a" * 64}
    environment = {"python": "synthetic-interpreter", "packages": {"numpy": "synthetic"}}
    for name, value in (("N_VALUES", (1000,)), ("D_PANEL_VALUES", (10,)),
                        ("SEEDS", (2026, 2027))):
        monkeypatch.setattr(runner, name, value)
    monkeypatch.setattr(runner, "configurations", lambda: grid)
    monkeypatch.setattr(runner, "dataset_snapshot", lambda root: deepcopy(snapshot))
    monkeypatch.setattr(runner, "verify_manifest",
                        lambda root: {"verified": 1, "missing": [], "mismatch": []})
    monkeypatch.setattr(runner, "environment_metadata", lambda: deepcopy(environment))
    loaded, simulated, exported, plotted, reported = [], [], [], [], []
    full_loaded, calibrated, used_calibrations = [], [], []

    def load(**kwargs):
        if kwargs['n']==setup.CALIBRATION_POPULATION_N:
            full_loaded.append((kwargs['seed'],kwargs['dim']))
            return {'X':np.ones((setup.CALIBRATION_POPULATION_N,kwargs['dim']),dtype=np.float32)}
        loaded.append(kwargs["seed"])
        return {"synthetic": True}

    def calibrate(x,seed,d):
        assert x.shape==(setup.CALIBRATION_POPULATION_N,d)
        calibrated.append((seed,d))
        return _synthetic_calibration(seed,d)

    def simulate(group, n, d, seed, point_config, *, calibration):
        assert group == {"synthetic": True} and (n, d) == (1000, 10)
        assert calibration.to_dict()==_synthetic_calibration(seed,d).to_dict()
        used_calibrations.append(calibration)
        simulated.append(seed)
        return list(_synthetic_results(point_config))

    monkeypatch.setattr(runner, "load_group", load)
    monkeypatch.setattr(setup,"calibrate_kernel_costs",calibrate)
    monkeypatch.setattr(simulator, "simulate_core_configuration", simulate, raising=False)
    output = tmp_path / "synthetic-output"
    manifest_path = output / "logs" / "run_manifest.json"
    source_hash, full_records = runner.source_fingerprint()
    # A partially completed current-protocol run may resume only under the
    # exact computation/data/configuration/driver identity.
    computation_hash, computation_records = identity.computation_source_fingerprint()
    contract = runner._orchestration_contract()
    calibration_records = {setup.calibration_key(seed,10):_synthetic_calibration(seed,10).to_dict()
                           for seed in (2026,2027)}
    calibration_policy=setup.calibration_policy()
    calibration_hash=setup.calibration_snapshot_sha256(calibration_records,calibration_policy)
    legacy_fp = runner.run_identity(config, snapshot,
                                   identity.checkpoint_identity_source(computation_hash, contract),
                                   environment,kernel_calibration_sha256=calibration_hash,
                                   kernel_calibration_policy=calibration_policy)
    legacy_manifest = {
        "implementation_version": runner.OVERHEAD_IMPLEMENTATION_VERSION,
        "run_fingerprint": legacy_fp, "config": asdict(config),
        "N_values": [1000], "d_panel_values": [10], "seeds": [2026, 2027],
        "methods": list(runner.METHODS), "evaluation_modes": list(runner.EVALUATION_MODES),
        "main_evaluation_mode": runner.MAIN_EVALUATION_MODE,
        "dataset_snapshot": snapshot, "environment": environment, "threadpool_limit": 1,
        "identity_schema": "computation-v1",
        "source_sha256": source_hash, "source_files_sha256": full_records,
        "computation_source_sha256": computation_hash,
        "computation_source_files_sha256": computation_records,
        "orchestration_contract_sha256": contract,
        "kernel_calibrations":calibration_records,"kernel_calibration_policy":calibration_policy,
        "kernel_calibration_sha256":calibration_hash,
        "status": "running", "executed_configurations_this_invocation": 1,
    }
    legacy_path = output / "checkpoints" / "N1000_d10_seed2026.json.gz"
    results = [runner._result_record(result, 1000, 10, 2026, config, legacy_fp)
               for result in _synthetic_results(config)]
    legacy_payload = {"run_fingerprint": legacy_fp, "complete": True,
                      "configuration": {"N": 1000, "d": 10, "seed": 2026},
                      "wall_seconds": .01, "results": results}
    runner._atomic_json(legacy_path, legacy_payload, compressed=True)
    runner._atomic_json(manifest_path, legacy_manifest)
    legacy_bytes = legacy_path.read_bytes()

    def materialize(root, fingerprint, compatible_fingerprints=()):
        exported.append(fingerprint)
        rows = []
        for n, d, seed in grid:
            path = root / "checkpoints" / f"N{n}_d{d}_seed{seed}.json.gz"
            record = runner._checkpoint(path, fingerprint, n, d, seed, compatible_fingerprints)
            assert record is not None
            if record.get("checkpoint_origin_fingerprint") == legacy_fp:
                assert all(result["row"]["checkpoint_origin_fingerprint"] == legacy_fp
                           for result in record["results"])
            rows.extend(result["row"] for result in record["results"])
        assert all(row["run_fingerprint"] == fingerprint for row in rows)
        summaries, statistics = runner.summarize_runs(rows, fingerprint)
        return rows, summaries, statistics

    def plot(root):
        plotted.append(root)
        return {"figures_generated": []}

    monkeypatch.setattr(runner, "materialize_outputs", materialize)
    monkeypatch.setattr(plotter, "plot_results", plot)
    monkeypatch.setattr(runner, "generate_report",
                        lambda *args: reported.append(deepcopy(args[2])))
    return SimpleNamespace(output=output, config=config, legacy_fp=legacy_fp,
                           legacy_path=legacy_path, legacy_bytes=legacy_bytes,
                           manifest_path=manifest_path, loaded=loaded, simulated=simulated,
                           exported=exported, plotted=plotted, reported=reported,
                           full_loaded=full_loaded,calibrated=calibrated,used_calibrations=used_calibrations)


def test_current_protocol_resume_simulates_only_missing_seed_and_preserves_gzip(synthetic_runner):
    state = synthetic_runner
    manifest = runner.run_experiment(state.output, config=state.config)
    assert manifest["status"] == "complete"
    assert manifest["resumed_configurations"] == 1
    assert manifest["executed_configurations_this_invocation"] == 1
    assert state.loaded == state.simulated == [2027]
    assert not state.full_loaded and not state.calibrated
    assert manifest['kernel_calibration_reused']
    assert state.legacy_path.read_bytes() == state.legacy_bytes
    original = json.loads(gzip.decompress(state.legacy_bytes))
    assert original["run_fingerprint"] == state.legacy_fp
    assert manifest["run_fingerprint"] == state.legacy_fp
    assert not manifest["checkpoint_compatibility_rejections"]
    assert state.exported == [manifest["run_fingerprint"]]
    assert len(state.plotted) == len(state.reported) == 1
    saved_manifest = json.loads(state.manifest_path.read_text(encoding="utf-8"))
    assert saved_manifest["derived_outputs"] == {
        "state": "complete", "run_fingerprint": manifest["run_fingerprint"]}


def test_second_resume_retains_current_provenance_without_recomputing_either_seed(synthetic_runner):
    state = synthetic_runner
    runner.run_experiment(state.output, config=state.config)
    new_path = state.output / "checkpoints" / "N1000_d10_seed2027.json.gz"
    new_bytes = new_path.read_bytes()
    manifest = runner.run_experiment(state.output, config=state.config)
    assert manifest["resumed_configurations"] == 2
    assert manifest["executed_configurations_this_invocation"] == 0
    assert state.simulated == [2027]
    assert not state.full_loaded and not state.calibrated
    assert manifest['kernel_calibration_reused']
    assert state.legacy_path.read_bytes() == state.legacy_bytes
    assert new_path.read_bytes() == new_bytes
    assert state.legacy_fp in {source["run_fingerprint"]
                               for source in manifest["compatible_checkpoint_sources"]}


def test_v2_checkpoint_rows_and_manifest_are_not_reused_for_core_protocol(synthetic_runner):
    state = synthetic_runner
    payload = json.loads(gzip.decompress(state.legacy_path.read_bytes()))
    for result in payload["results"]:
        result["row"]["implementation_version"] = "overhead-events-v2"
    runner._atomic_json(state.legacy_path, payload, compressed=True)
    previous = json.loads(state.manifest_path.read_text(encoding="utf-8"))
    previous["implementation_version"] = "overhead-events-v2"
    runner._atomic_json(state.manifest_path, previous)
    manifest = runner.run_experiment(state.output, config=state.config)
    assert state.simulated == [2026, 2027]
    assert manifest["resumed_configurations"] == 0
    assert manifest["executed_configurations_this_invocation"] == 2
    assert "incompatible implementation version" in manifest["checkpoint_compatibility_rejections"][state.legacy_fp]
    replaced = json.loads(gzip.decompress(state.legacy_path.read_bytes()))
    assert all(result["row"]["implementation_version"] == "overhead-core-v4"
               for result in replaced["results"])


def test_complete_core_workloads_with_deadline_exceedances_remain_paired_observations(
        synthetic_runner, monkeypatch):
    state = synthetic_runner
    payload = json.loads(gzip.decompress(state.legacy_path.read_bytes()))
    for result in payload["results"]:
        result["row"].update(status="success", selected_count=1,
                             attempted_count=1, completed_count=1,
                             within_deadline_count=0, deadline_met=False,
                             deadline_violations=1, verification_time_ms=801.,
                             total_time_ms=802.1)
    runner._atomic_json(state.legacy_path, payload, compressed=True)
    # One seed in the summarizer exercises the default reporting policy.
    monkeypatch.setattr(runner, "SEEDS", (2026,))
    rows = [result["row"] for result in payload["results"]]
    summaries, policy = runner.summarize_runs(rows, state.legacy_fp)
    for row in summaries["time"]:
        assert row["n_valid"] == 1 and row["value"] == row["mean"] == 802.1
        assert row["deadline_violations_total"] == 1
        assert row["estimator"] == "single paired observation"
        assert "deadline exceedances included" in row["statistics_scope"]
        assert np.isnan(row["std"]) and np.isnan(row["ci95_lower"])
    assert policy["fixed_workload_eligible_paired_seeds"] == 1
    assert policy["confidence_intervals"].startswith("not estimated")


def test_compact_checkpoint_and_materialization_preserve_counts_without_event_exports(
        synthetic_runner, monkeypatch):
    state = synthetic_runner
    manifest = runner.run_experiment(state.output, config=state.config)
    for checkpoint in (state.output / "checkpoints").glob("*.json.gz"):
        payload = json.loads(gzip.decompress(checkpoint.read_bytes()))
        for result in payload["results"]:
            assert result["messages"] == result["runtime"] == result["tasks"] == []
            assert result["row"]["message_count"] == 1
            assert result["row"]["total_bytes"] == 96
            assert json.loads(result["row"]["core_phase_counts_json"])["verification"]["total_bytes"] == 96
    reads=[]
    original_checkpoint=runner._checkpoint

    def read_checkpoint(*args):
        reads.append(args[0])
        return original_checkpoint(*args)

    monkeypatch.setattr(runner,"_checkpoint",read_checkpoint)
    rows, summaries, policy=_ORIGINAL_MATERIALIZE(state.output,manifest["run_fingerprint"])
    assert len(reads)==2 and len(rows)==2*len(runner.METHODS)
    assert len(summaries["time"])==len(summaries["communication"])==len(runner.METHODS)
    assert policy["fixed_workload_eligible_paired_seeds"]==2
    assert {path.name for path in (state.output/"raw").iterdir()} == {
        "overhead_runs.csv","protocol_diagnostics.csv"}
    assert {path.name for path in (state.output/"summary").iterdir()} == {
        "time_overhead_summary.csv","communication_overhead_summary.csv"}


def test_default_cli_uses_requested_overhead_root_with_v4_protocol():
    args=cli.parse_args([])
    assert args.output_dir.name=="02_overhead"
    assert args.output_dir==runner.RESULT_ROOT
    config_module=importlib.import_module("experiments.02_overhead.config_overhead")
    assert config_module.SEEDS==(2026,)
    assert config_module.EVALUATION_MODES==("fixed_workload",)
    assert len(config_module.configurations())==45
    assert config_module.OVERHEAD_IMPLEMENTATION_VERSION=="overhead-core-v4"


def test_shared_execution_exception_cache_is_recomputed_on_normal_resume(synthetic_runner):
    state = synthetic_runner
    payload = json.loads(gzip.decompress(state.legacy_path.read_bytes()))
    evidence = {"exception_traceback": "ValueError: synthetic previous execution failure"}
    for result in payload["results"]:
        result["row"].update(status="failure", reason=evidence["exception_traceback"],
                             selected_count=0, attempted_count=0, completed_count=None,
                             deadline_violations=None, gt_time_ms=None, decision_cpu_ms=None,
                             verification_time_ms=None, total_time_ms=None,
                             gt_bytes=0, verification_bytes=0, control_bytes=0,
                             total_bytes=0, message_count=0,
                             diagnostics_json=json.dumps(evidence))
        result.update(messages=[], tasks=[], runtime=[], diagnostics=evidence)
    runner._atomic_json(state.legacy_path, payload, compressed=True)
    failed_bytes = state.legacy_path.read_bytes()
    manifest = runner.run_experiment(state.output, config=state.config)
    assert manifest["status"] == "complete"
    assert manifest["resumed_configurations"] == 0
    assert manifest["executed_configurations_this_invocation"] == 2
    assert state.loaded == state.simulated == [2026, 2027]
    corrected = json.loads(gzip.decompress(state.legacy_path.read_bytes()))
    assert corrected["run_fingerprint"] == manifest["run_fingerprint"]
    assert all(result["row"]["status"] == "success" for result in corrected["results"])
    assert state.legacy_path.read_bytes() != failed_bytes
    assert manifest["statistics"]["fixed_workload_eligible_paired_seeds"] == 2


def test_protocol_deadline_measurements_remain_resumable_and_keep_original_bytes(synthetic_runner):
    state = synthetic_runner
    payload = json.loads(gzip.decompress(state.legacy_path.read_bytes()))
    for result in payload["results"]:
        result["row"].update(status="deadline_failure", reason="",
                             completed_count=0, deadline_violations=1,
                             verification_time_ms=801., total_time_ms=802.1)
    runner._atomic_json(state.legacy_path, payload, compressed=True)
    deadline_bytes = state.legacy_path.read_bytes()
    manifest = runner.run_experiment(state.output, config=state.config)
    assert manifest["status"] == "complete"
    assert manifest["resumed_configurations"] == 1
    assert manifest["executed_configurations_this_invocation"] == 1
    assert state.loaded == state.simulated == [2027]
    assert state.legacy_path.read_bytes() == deadline_bytes
    count=len(runner.METHODS)*len(runner.EVALUATION_MODES)
    assert manifest["statuses"] == {"deadline_failure": count, "success": count}
    assert manifest["statistics"]["fixed_workload_eligible_paired_seeds"] == 1


def test_checkpoint_commit_failure_preserves_completed_recovery_and_skips_bulk_cleanup(
        synthetic_runner, monkeypatch):
    state = synthetic_runner
    second_path = state.output / "checkpoints" / "N1000_d10_seed2027.json.gz"
    original_replace = storage.os.replace
    denied = PermissionError("synthetic Windows reader prevents checkpoint replace")
    denied.winerror = 5

    def replace(source, target):
        if Path(target) == second_path:
            raise denied
        return original_replace(source, target)

    monkeypatch.setattr(storage, "RETRY_DELAYS", (0.,))
    monkeypatch.setattr(storage.os, "replace", replace)
    with pytest.raises(storage.AtomicWriteError) as caught:
        runner.run_experiment(state.output, config=state.config)
    assert caught.value.original_error is denied
    recovery = caught.value.recovery_path
    assert recovery.is_file() and not second_path.exists()
    retained = json.loads(gzip.decompress(recovery.read_bytes()))
    assert retained["complete"] and retained["configuration"]["seed"] == 2027
    assert len(retained["results"]) == len(runner.METHODS)*len(runner.EVALUATION_MODES)
    assert state.legacy_path.read_bytes() == state.legacy_bytes
    assert not state.exported and not state.plotted and not state.reported
    saved = json.loads(state.manifest_path.read_text(encoding="utf-8"))
    assert saved["status"] == "output_io_failed" and saved["failed_phase"] == "simulation"
    assert saved["saved_configurations"] == 1


def test_export_failure_propagates_original_exception_without_repeated_materialization(
        synthetic_runner, monkeypatch):
    state = synthetic_runner
    denied = PermissionError("synthetic locked runtime CSV")
    attempts = []

    def materialize(*args):
        attempts.append(args)
        raise denied

    monkeypatch.setattr(runner, "materialize_outputs", materialize)
    with pytest.raises(PermissionError) as caught:
        runner.run_experiment(state.output, config=state.config)
    assert caught.value is denied and len(attempts) == 1
    assert state.legacy_path.read_bytes() == state.legacy_bytes
    assert (state.output / "checkpoints" / "N1000_d10_seed2027.json.gz").is_file()
    assert not state.plotted and not state.reported
    saved = json.loads(state.manifest_path.read_text(encoding="utf-8"))
    assert saved["status"] == "output_io_failed" and saved["failed_phase"] == "materialization"
    assert saved["saved_configurations"] == 2
    assert saved["derived_outputs"]["state"] == "building"


def test_export_manifest_commit_failure_does_not_enter_bulk_export(
        synthetic_runner, monkeypatch):
    state = synthetic_runner
    atomic_json = runner._atomic_json
    denied = PermissionError("synthetic manifest publishing lock")

    def save(path, payload, compressed=False):
        if Path(path) == state.manifest_path and payload.get("status") == "exporting":
            raise denied
        return atomic_json(path, payload, compressed=compressed)

    monkeypatch.setattr(runner, "_atomic_json", save)
    with pytest.raises(PermissionError) as caught:
        runner.run_experiment(state.output, config=state.config)
    assert caught.value is denied
    assert not state.exported and not state.plotted and not state.reported
    assert state.legacy_path.read_bytes() == state.legacy_bytes
    assert (state.output / "checkpoints" / "N1000_d10_seed2027.json.gz").is_file()
    saved = json.loads(state.manifest_path.read_text(encoding="utf-8"))
    assert saved["saved_configurations"] == 2
    assert saved["failed_phase"] == "materialization"


def test_persistent_progress_manifest_lock_keeps_all_checkpoints_and_finishes(
        synthetic_runner, monkeypatch):
    state = synthetic_runner
    # Start a fresh synthetic run with enough points to hit periodic update 5.
    # The fixture's two-seed legacy files are confined to its own temp output.
    state.legacy_path.unlink()
    state.manifest_path.unlink()
    seeds = tuple(range(2026, 2032))
    grid = tuple((1000, 10, seed) for seed in seeds)
    monkeypatch.setattr(runner, "SEEDS", seeds)
    monkeypatch.setattr(runner, "configurations", lambda: grid)
    exports = []

    def materialize(root, fingerprint, compatible_fingerprints=()):
        exports.append(fingerprint)
        rows = []
        for n, d, seed in grid:
            path = root / "checkpoints" / f"N{n}_d{d}_seed{seed}.json.gz"
            record = runner._checkpoint(path, fingerprint, n, d, seed, compatible_fingerprints)
            assert record is not None
            rows.extend(result["row"] for result in record["results"])
        summaries, statistics = runner.summarize_runs(rows, fingerprint)
        return rows, summaries, statistics

    monkeypatch.setattr(runner, "materialize_outputs", materialize)
    original_replace = storage.os.replace
    denied = PermissionError("synthetic persistent progress manifest sharing lock")
    denied.winerror = 5
    attempts = []

    def replace(source, target):
        if Path(target) == state.manifest_path:
            pending = json.loads(Path(source).read_text(encoding="utf-8"))
            if (pending.get("status") == "running"
                    and pending.get("executed_configurations_this_invocation", 0) > 0):
                attempts.append(pending["executed_configurations_this_invocation"])
                raise denied
        return original_replace(source, target)

    monkeypatch.setattr(storage, "RETRY_DELAYS", (0., 0.))
    monkeypatch.setattr(storage.os, "replace", replace)
    manifest = runner.run_experiment(state.output, config=state.config)
    assert attempts == [5, 5, 5]  # The progress commit fails through all retries.
    assert manifest["status"] == "complete"
    assert manifest["executed_configurations_this_invocation"] == len(seeds)
    assert manifest["resumed_configurations"] == 0
    assert manifest["saved_configurations"] == len(seeds)
    assert state.loaded == state.simulated == list(seeds)
    assert len(list((state.output / "checkpoints").glob("*.json.gz"))) == len(seeds)
    assert exports == [manifest["run_fingerprint"]]
    assert manifest["statistics"]["fixed_workload_eligible_paired_seeds"] == len(seeds)
    assert manifest["io_warnings"]
    assert "synthetic persistent progress manifest sharing lock" in manifest["io_warnings"][0]
    saved = json.loads(state.manifest_path.read_text(encoding="utf-8"))
    assert saved["status"] == "complete"
    assert saved["derived_outputs"]["state"] == "complete"
    assert saved["io_warnings"] == manifest["io_warnings"]
    progress = (state.output / "logs" / "progress.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(progress) == len(seeds)


@pytest.mark.parametrize("derived", [
    {}, {"state": "pending", "run_fingerprint": "synthetic-id"},
    {"state": "building", "run_fingerprint": "synthetic-id"},
    {"state": "summaries_ready", "run_fingerprint": "different-generation"},
    {"state": "complete", "run_fingerprint": "different-generation"},
])
def test_plots_only_rejects_pending_or_mixed_generation_before_reading_summaries(
        tmp_path, monkeypatch, derived):
    output = tmp_path / "plot-only-output"
    manifest = {"identity_schema": "computation-v1",
                "implementation_version": runner.OVERHEAD_IMPLEMENTATION_VERSION,
                "run_fingerprint": "synthetic-id", "derived_outputs": derived}
    runner._atomic_json(output / "logs" / "run_manifest.json", manifest)
    reads = []

    def unexpected_read(*args):
        reads.append(args)
        raise AssertionError("Pending output must be rejected before reading any summary")

    monkeypatch.setattr(plotter, "_read_summary", unexpected_read)
    with pytest.raises(ValueError, match="pending or incomplete"):
        cli.main(["--plots-only", "--output-dir", str(output)])
    assert not reads
    assert not (output / "logs" / "plot_style.json").exists()
    assert not (output / "figures").exists()


def test_plots_only_rejects_legacy_running_manifest_before_reading_summaries(tmp_path, monkeypatch):
    output = tmp_path / "legacy-plot-only-output"
    manifest = {"implementation_version": runner.OVERHEAD_IMPLEMENTATION_VERSION,
                "run_fingerprint": "legacy-synthetic-id", "status": "running"}
    runner._atomic_json(output / "logs" / "run_manifest.json", manifest)
    reads = []

    def unexpected_read(*args):
        reads.append(args)
        raise AssertionError("Legacy running output must be rejected before reading summaries")

    monkeypatch.setattr(plotter, "_read_summary", unexpected_read)
    with pytest.raises(ValueError, match="pending or incomplete"):
        cli.main(["--plots-only", "--output-dir", str(output)])
    assert not reads
    assert not (output / "logs" / "plot_style.json").exists()
    assert not (output / "figures").exists()
