"""Serialized communication replay invariants; no formal overhead runs."""
from copy import deepcopy
import gzip
import hashlib
import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

m = importlib.import_module("experiments.02_overhead.overhead_communication")


@pytest.mark.parametrize("value", [0, 1, 127, 128, 255, 16384, (1 << 63) - 1])
def test_varints_roundtrip_canonical_int64(value):
    encoded = m.encode_varint(value)
    assert m.decode_varint(encoded) == (value, len(encoded))


@pytest.mark.parametrize("payload", [b"", b"\x80", b"\x80\x00", b"\xff" * 10, b"\x81" * 10 + b"\x00"])
def test_varints_reject_truncation_noncanonical_and_overflow(payload):
    with pytest.raises(ValueError):
        m.decode_varint(payload)


@pytest.mark.parametrize("value", [-1, True, 1.5, 1 << 63])
def test_varints_reject_nonidentities(value):
    with pytest.raises(ValueError):
        m.encode_varint(value)


@pytest.mark.parametrize("ids", [[1], [100, 2, 91, 17, 5000], [4999, 1, 0, 3333]])
def test_assignment_delta_packet_preserves_unsorted_primary_queue_order(ids):
    packet = m.encode_assignment(675, ids)
    assert m.decode_assignment(packet) == (675, ids)
    assert len(packet) == m.FRAME_BYTES + int.from_bytes(packet[2:4], "little")
    assert m.decode_frame(packet)[0] == m._ASSIGNMENT


@pytest.mark.parametrize("ids", [[], [1, 1], [1, -1], [True]])
def test_assignment_rejects_missing_duplicate_or_invalid_identities(ids):
    with pytest.raises(ValueError):
        m.encode_assignment(5, ids)


@pytest.mark.parametrize("count", [1, 7, 8, 9, 98, 498])
def test_result_bitmap_roundtrip_matches_actual_task_count(count):
    values = (np.arange(count) % 3 == 0).astype(np.int8)
    packet = m.encode_result(values)
    assert np.array_equal(m.decode_result(packet), values)
    assert len(packet) == 4 + len(m.encode_varint(count)) + (count + 7) // 8


def test_result_rejects_nonzero_padding_and_nonbinary_values():
    packet = m.frame(m._RESULT, m.encode_varint(1) + b"\x80")
    with pytest.raises(ValueError, match="Unused"):
        m.decode_result(packet)
    with pytest.raises(ValueError):
        m.encode_result([0, 2])


@pytest.mark.parametrize("packet", [b"", b"\x01\x01\x01\x00", b"\x02\x01\x00\x00", b"\x01\x09\x00\x00"])
def test_application_frame_validates_version_opcode_and_body_length(packet):
    with pytest.raises(ValueError):
        m.decode_frame(packet)


def test_application_frame_rejects_wrong_request_opcode():
    with pytest.raises(ValueError, match="opcode"):
        m.decode_frame(m.frame(m._QUERY, b"\x0a\x00"), m._RAW_QUERY)


@pytest.fixture
def case_and_row():
    n, d, seed = 1000, 10, 2026
    config = m.OverheadConfig(dim=d)
    ids = m.stable_model_ids(n, seed)
    references = np.argsort(ids, kind="stable")[:config.M]
    initial = np.argsort(ids, kind="stable")[:config.K]
    workload = m.nested_fixed_targets(n, ids, references, config.budget, seed)
    targets = workload[np.argsort(ids[workload], kind="stable")]
    # All candidates agree. Only two noncandidate task descriptors are rotated.
    X = np.zeros((n, d), np.float32)
    X[:, 0] = 1
    for target in [target for target in targets if target not in initial][-2:]:
        X[target, :2] = [np.cos(.25), np.sin(.25)]
    assignments = tuple((int(target), int(references[index % len(references)]))
                        for index, target in enumerate(targets))
    case = m.ReplayCase(X, ids, targets, references, assignments, (config.K,), config)
    scores = m.cached_reference_cosine(X, [target for target, _ in assignments], [verifier for _, verifier in assignments])
    predictions = (~m.within_interval(scores, config.tau_lower, config.tau_upper)).astype(np.int8)
    history = m.history_update_vectorized(np.full(n, config.h0), [target for target, _ in assignments],
        predictions, references, config.rho)
    row = dict(N=n, d=d, seed=seed, B=config.budget(n), corrupted_verdict_count=int(predictions.sum()),
        diagnostics_json=json.dumps(dict(updated_history_sha256=hashlib.sha256(history.astype("<f8").tobytes()).hexdigest())))
    return case, row


def test_encoded_gt_and_all_verification_verdicts_equal_original(case_and_row):
    case, row = case_and_row
    source = case.X.tobytes()
    totals, packets = m.replay_raca_packets(case, row)
    assert totals["verdict_count"] == 2
    assert totals["gt_membership_preserved"] and totals["verification_verdicts_preserved"]
    assert totals["history_update_preserved"]
    assert totals["encoded_total_bytes"] == sum(packet["total_bytes"] for packet in packets)
    assert totals["gt_bytes"] + totals["verification_bytes"] + totals["control_bytes"] == totals["encoded_total_bytes"]
    assert sum(packet["message_type"] == "batch_assignment" for packet in packets) == case.config.M
    assert sum(packet["message_type"] == "batch_verification_result" for packet in packets) == case.config.M
    assert all(packet["body_bytes"] + 4 == packet["total_bytes"] for packet in packets)
    assert all(packet["roundtrip_verified"] and len(packet["packet_sha256"]) == 64 for packet in packets)
    assert all(packet["sender"] != packet["receiver"] for packet in packets)
    assert case.X.tobytes() == source


def test_raw_fallback_is_transmitted_and_decoded_for_every_ambiguous_channel(case_and_row, monkeypatch):
    case, row = case_and_row
    monkeypatch.setattr(m.codec, "certify_pair", lambda *args: None)
    totals, packets = m.replay_raca_packets(case, row)
    raw = [packet for packet in packets if packet["opcode"] == m._RAW]
    queries = [packet for packet in packets if packet["opcode"] in {m._QUERY, m._REFINE_QUERY, m._RAW_QUERY}]
    assert totals["refinement_count"] == case.config.K + row["B"]
    assert totals["raw_fallback_count"] == (case.config.K + 2) // 2 + row["B"]
    assert totals["gt_raw_fallback_count"] == (case.config.K + 2) // 2
    assert len(raw) == totals["raw_fallback_count"]
    assert all(packet["total_bytes"] == 4 * row["d"] + 6 for packet in raw)
    assert all(packet["total_bytes"] == 6 for packet in queries)
    assert totals["verdict_count"] == 2


def test_uncertainty_cannot_bypass_real_low_and_raw_delivery(case_and_row):
    case, row = case_and_row
    channels = m.DescriptorChannels(case.X, m.PacketLedger(row["N"], row["d"], row["seed"]))
    target, verifier = case.assignments[0]
    with pytest.raises(ValueError, match="same channel"):
        channels.request(target, verifier, "verification", 1)
    channels.request(target, verifier, "verification", 0)
    with pytest.raises(ValueError, match="same channel"):
        channels.request(target, verifier, "verification", 2)
    with pytest.raises(ValueError, match="same channel"):
        channels.request(target, int(case.verifiers[1]), "verification", 1)
    with pytest.raises(ValueError, match="Duplicate"):
        channels.request(target, verifier, "verification", 0)


def test_coordinator_gt_cache_does_not_make_verifier_response_free(case_and_row):
    case, row = case_and_row
    channels = m.DescriptorChannels(case.X, m.PacketLedger(row["N"], row["d"], row["seed"]))
    target, verifier = case.assignments[0]
    channels.request(target, -1, "gt", 0)
    channels.request(target, verifier, "verification", 0)
    assert len(channels.ledger.packets) == 4
    assert ("gt", target, -1) in channels.decoded
    assert ("verification", target, verifier) in channels.decoded


def test_changed_original_verdict_count_cannot_be_republished(case_and_row):
    case, row = case_and_row
    changed = dict(row, corrupted_verdict_count=0)
    with pytest.raises(AssertionError, match="corruption verdict count"):
        m.replay_raca_packets(case, changed)


def test_changed_original_history_hash_cannot_be_republished(case_and_row):
    case, row = case_and_row
    changed = dict(row, diagnostics_json=json.dumps(dict(updated_history_sha256="0" * 64)))
    with pytest.raises(AssertionError, match="history-update"):
        m.replay_raca_packets(case, changed)


def test_changed_saved_gt_membership_cannot_be_republished(case_and_row):
    case, row = case_and_row
    changed = m.replace(case, verifiers=case.verifiers[::-1])
    with pytest.raises(AssertionError, match="membership/order"):
        m.replay_raca_packets(changed, row)


def test_saved_gt_cannot_expand_after_initial_majority_success(case_and_row):
    case, row = case_and_row
    changed = m.replace(case, expansion_sizes=(case.config.K, case.config.K + case.config.expansion_step))
    with pytest.raises(ValueError, match="already reaching"):
        m.replay_raca_packets(changed, row)


def _small_gt_case(angles, references, *, K, M, expansions):
    X = np.zeros((len(angles), 10), np.float32)
    X[:, :2] = np.column_stack((np.cos(angles), np.sin(angles))).astype(np.float32)
    return m.ReplayCase(X, np.arange(len(X)), np.empty(0, np.int64),
        np.asarray(references, np.int64), (), tuple(expansions),
        m.OverheadConfig(K=K, M=M, expansion_step=2, dim=10))


def test_majority_prefix_resolves_earlier_unknown_before_two_later_accepted(monkeypatch):
    case = _small_gt_case(np.deg2rad([0, 3, 3, 6]), [0, 1], K=4, M=2, expansions=[4])
    matrix = m.cosine_pairwise(case.X)
    class Channels:
        decoded, raw_rows, requests = {}, {}, []
        def request(self, row, receiver, phase, tier):
            self.requests.append((row, tier))
            value = SimpleNamespace(node=row, stage="initial" if tier == 0 else "refinement", angle_error=.01, d=10)
            self.decoded[(phase, row, receiver)] = value
            return value
    channels = Channels()
    def certificate(left, right, lower, upper):
        if (left.node == 0 and left.stage == "initial") or (right.node == 0 and right.stage == "initial"):
            return None
        return int(not bool(m.within_interval(matrix[left.node, right.node], lower, upper)))
    monkeypatch.setattr(m.codec, "certify_pair", certificate)
    audit = m._replay_gt(case, channels)
    assert channels.requests == [(0, 0), (1, 0), (2, 0), (3, 0), (0, 1)]
    assert audit["gt_certified_pair_count"] == 6


def test_irrelevant_tail_unknowns_do_not_require_extra_descriptors(monkeypatch):
    case = _small_gt_case(np.zeros(4), [0, 1], K=4, M=2, expansions=[4])
    class Channels:
        decoded, raw_rows, requests = {}, {}, []
        def request(self, row, receiver, phase, tier):
            self.requests.append((row, tier))
            value = SimpleNamespace(node=row, stage="initial", angle_error=.01, d=10)
            self.decoded[(phase, row, receiver)] = value
            return value
    channels = Channels()
    monkeypatch.setattr(m.codec, "certify_pair", lambda left, right, *args:
        None if 3 in (left.node, right.node) else 0)
    audit = m._replay_gt(case, channels)
    assert channels.requests == [(0, 0), (1, 0), (2, 0), (3, 0)]
    assert audit["gt_majority_short_circuit"]
    assert audit["gt_unresolved_pair_count"] == 3


def test_necessary_gt_expansion_recomputes_majority_threshold_and_membership():
    # The first two agree at K=3, but cannot supply M=3. At K=5 their old
    # support is below the raised threshold, and the three other nodes win.
    case = _small_gt_case([0, 0, np.pi / 2, np.pi / 2, np.pi / 2], [2, 3, 4],
                          K=3, M=3, expansions=[3, 5])
    ledger = m.PacketLedger(5, 10, 2026)
    channels = m.DescriptorChannels(case.X, ledger)
    audit = m._replay_gt(case, channels)
    assert sum(packet["opcode"] == m._INITIAL for packet in ledger.packets) == 5
    assert all(packet["phase"] == "gt" for packet in ledger.packets)
    assert audit["gt_certified_pair_count"] == 10


@pytest.fixture
def reconstruction_fixture(case_and_row):
    case, old_row = case_and_row
    n, d, seed = old_row["N"], old_row["d"], old_row["seed"]
    calibration_module = importlib.import_module("experiments.02_overhead.overhead_calibration")
    costs = dict(cosine_per_pair_ms=.001, sha256_per_descriptor_ms=.002,
        scalar_compare_ms=.0001, scalar_modulo_ms=.0002, vector_per_byte_ms=.000001)
    timings = dict(cosine=costs["cosine_per_pair_ms"] * calibration_module.COSINE_BATCH_PAIRS,
        sha256=costs["sha256_per_descriptor_ms"] * calibration_module.HASH_BATCH_DESCRIPTORS,
        scalar_compare=costs["scalar_compare_ms"] * calibration_module.SCALAR_BATCH_ITEMS,
        scalar_modulo=costs["scalar_modulo_ms"] * calibration_module.SCALAR_BATCH_ITEMS,
        vector=costs["vector_per_byte_ms"] * calibration_module.VECTOR_BATCH_BYTES)
    calibration = m.KernelCalibration(costs, seed, d, (5000, d), "a" * 64,
        calibration_module.CALIBRATION_SAMPLE_LIMIT, sum(timings.values()), timings)
    profiles = m.shared_resource_profiles(n=n, seed=seed, rounds=1)
    sim = m.EventSimulator(profiles, d, case.config, retain_events=False, calibration=calibration)
    costs_matrix = m._core_costs(m.RACA_HIM, sim, case.targets, case.verifiers)
    risk = np.full(len(case.targets), case.config.alpha + (1 - case.config.alpha) * case.config.h0)
    assignments, plan = m.joint_schedule(case.targets, case.verifiers, risk, costs_matrix, old_row["B"],
        case.config.T_round_max_ms, case.config.lambda_, fixed_workload=True, enforce_deadline=False)
    case = m.replace(case, assignments=tuple(assignments))
    workload = m.nested_fixed_targets(n, case.model_ids, case.verifiers, case.config.budget, seed)
    fixture = dict(base_size=1000, cohort_size=500, prefix_sha256=m._sha256(case.X.tobytes()),
        population_sha256="b" * 64, domain_max=5000, dimension=d)
    gt_ledger = [dict(phase="gt", message_type="gt_candidate_request", message_count=case.config.K,
        payload_bytes=0, control_bytes=case.config.K * case.config.request_bytes,
        total_bytes=case.config.K * case.config.request_bytes),
        dict(phase="gt", message_type="gt_candidate_upload", message_count=case.config.K,
        payload_bytes=case.config.K * 4 * d, control_bytes=0, total_bytes=case.config.K * 4 * d)]
    diag = dict(scaling_fixture=fixture, kernel_calibration=calibration.to_dict(),
        workload_rows=workload.tolist(), reference_rows=case.verifiers.tolist(),
        resource_fixture=m.resource_fixture_metadata(profiles), decision_plan=plan,
        selected_target_ids_sha256=m._sha256(np.sort(case.model_ids[workload]).astype("<i8").tobytes()),
        gt_expansion_sizes=[case.config.K], candidate_count=case.config.K)
    row = dict(old_row, fixed_workload_sha256=m._sha256(workload.astype("<i8").tobytes()),
        communication_phase_totals_json=json.dumps(gt_ledger))
    return dict(X=case.X.copy(), scaling_fixture=deepcopy(fixture)), dict(row=row, diagnostics=deepcopy(diag)), case.config, calibration, case


def test_reconstructs_entire_original_greedy_plan_without_gt_or_clock(reconstruction_fixture, monkeypatch):
    group, result, config, calibration, expected = reconstruction_fixture
    simulator = importlib.import_module("experiments.02_overhead.overhead_simulator")
    gt = importlib.import_module("experiments.02_overhead.raca_core")
    def forbidden(*args, **kwargs):
        raise AssertionError("Reconstruction must not execute a fresh simulation or measured GT")
    monkeypatch.setattr(simulator, "prepare_case", forbidden)
    monkeypatch.setattr(gt, "build_ground_truth_core", forbidden)
    restored = m._reconstruct_case(group, result, config, calibration)
    assert restored.assignments == expected.assignments
    assert np.array_equal(restored.targets, expected.targets)
    assert np.array_equal(restored.verifiers, expected.verifiers)


@pytest.mark.parametrize("field", ["prefix_objectives", "prefix_makespans_ms", "marginal_gains"])
def test_each_greedy_prefix_array_is_bound_to_saved_plan(reconstruction_fixture, field):
    group, result, config, calibration, _ = reconstruction_fixture
    result["diagnostics"]["decision_plan"][field][10] += 1e-5
    with pytest.raises(ValueError, match=field):
        m._reconstruct_case(group, result, config, calibration)


@pytest.mark.parametrize("mutation", ["fixture", "X", "targets", "workload_hash", "selected_hash", "resource", "calibration", "reference_order", "GT_count"])
def test_reconstruction_rejects_different_population_workload_resources_or_gt(reconstruction_fixture, mutation):
    group, result, config, calibration, _ = reconstruction_fixture
    if mutation == "fixture":
        group["scaling_fixture"]["population_sha256"] = "c" * 64
    elif mutation == "X":
        group["X"][999, 1] = .2
    elif mutation == "targets":
        result["diagnostics"]["workload_rows"][0] = result["diagnostics"]["workload_rows"][1]
    elif mutation == "workload_hash":
        result["row"]["fixed_workload_sha256"] = "0" * 64
    elif mutation == "selected_hash":
        result["diagnostics"]["selected_target_ids_sha256"] = "0" * 64
    elif mutation == "resource":
        result["diagnostics"]["resource_fixture"]["resource_fixture_sha256"] = "0" * 64
    elif mutation == "calibration":
        result["diagnostics"]["kernel_calibration"]["costs_ms"]["cosine_per_pair_ms"] *= 2
    elif mutation == "reference_order":
        result["diagnostics"]["reference_rows"] = result["diagnostics"]["reference_rows"][::-1]
    else:
        result["diagnostics"]["candidate_count"] -= 1
    with pytest.raises(ValueError):
        m._reconstruct_case(group, result, config, calibration)


def test_packet_ledger_counts_actual_bytes_in_disjoint_categories():
    ledger = m.PacketLedger(1000, 10, 2026)
    reply = m.frame(m._INITIAL, b"received descriptor")
    ledger.add(reply, 5, -1, "gt", "descriptor_response", payload=True)
    request = m.frame(m._QUERY, b"\x0a\x00")
    ledger.add(request, 7, 9, "verification", "descriptor_query")
    totals = ledger.totals()
    assert totals["gt_bytes"] == len(b"received descriptor")
    assert totals["verification_bytes"] == 0
    assert totals["control_bytes"] == 4 + 6
    assert totals["encoded_total_bytes"] == len(reply) + len(request)


def test_saved_logical_ledger_total_cannot_be_silently_reinterpreted():
    ledger = [dict(phase="gt", message_type="request", payload_bytes=0, control_bytes=64,
                   total_bytes=64, message_count=1)]
    row = dict(communication_phase_totals_json=json.dumps(ledger), total_bytes=64,
               gt_bytes=0, verification_bytes=0, control_bytes=64, message_count=1)
    m._validate_ledger(row)
    row["total_bytes"] = 63
    with pytest.raises(ValueError, match="totals"):
        m._validate_ledger(row)


@pytest.fixture
def sidecar_fixture(tmp_path, monkeypatch):
    (tmp_path / "logs").mkdir()
    (tmp_path / "raw").mkdir()
    (tmp_path / "summary").mkdir()
    (tmp_path / "figures").mkdir()
    (tmp_path / "checkpoints").mkdir()
    for relative in m._REQUIRED_BASE_FILES:
        (tmp_path / relative).write_bytes(f"frozen {relative}".encode())
    (tmp_path / "checkpoints/old_seed_file.json.gz").write_bytes(b"old preserved archive")
    protected = {relative: m._file_hash(tmp_path / relative) for relative in m._REQUIRED_BASE_FILES}
    source_hash, source_files = m.communication_source_fingerprint()
    checkpoint_stats = m.checkpoint_inventory(tmp_path)
    policy = m.communication_policy()
    metadata = dict(schema=m.COMMUNICATION_SCHEMA, status="prepared", base_run_fingerprint="base",
        communication_source_sha256=source_hash, communication_source_files_sha256=source_files,
        codec_policy=policy, frozen_files_sha256=protected, frozen_checkpoint_inventory=checkpoint_stats,
        summary_path=m.SUMMARY_PATH, encoded_runs_path=m.RUNS_PATH, packet_ledger_path=m.PACKET_LEDGER_PATH)
    metadata["communication_fingerprint"] = m._sha256(m._json_bytes(dict(schema=m.COMMUNICATION_SCHEMA,
        base_run_fingerprint="base", frozen_files_sha256=protected, frozen_checkpoint_inventory=checkpoint_stats,
        communication_source_sha256=source_hash, codec_policy=policy)))
    prepared = m.PreparedCommunicationReplay(metadata, [dict(encoded_total_bytes=10)],
        [dict(value=10)], [dict(total_bytes=10)])
    return tmp_path, prepared, protected


def test_sidecar_write_preserves_every_frozen_artifact_and_archive(sidecar_fixture):
    output, prepared, protected = sidecar_fixture
    result = m.write_communication_replay(output, prepared)
    assert result["status"] == "computed"
    assert set(result["files_sha256"]) == {m.SUMMARY_PATH, m.RUNS_PATH, m.PACKET_LEDGER_PATH}
    m.assert_frozen_files(output, protected)
    m._assert_checkpoint_inventory(output, prepared.manifest["frozen_checkpoint_inventory"])
    assert json.loads((output / m.MANIFEST_PATH).read_text())["status"] == "computed"


def test_frozen_time_mutation_blocks_sidecar_commit(sidecar_fixture):
    output, prepared, _ = sidecar_fixture
    (output / "summary/time_overhead_summary.csv").write_bytes(b"changed time")
    with pytest.raises(ValueError, match="Frozen"):
        m.write_communication_replay(output, prepared)
    assert not (output / m.MANIFEST_PATH).exists()


def test_checkpoint_archive_mutation_blocks_sidecar_commit(sidecar_fixture):
    output, prepared, _ = sidecar_fixture
    (output / "checkpoints/old_seed_file.json.gz").write_bytes(b"wrong")
    with pytest.raises(ValueError, match="checkpoint archive"):
        m.write_communication_replay(output, prepared)
    assert not (output / m.MANIFEST_PATH).exists()


def test_summary_write_failure_never_marks_sidecar_complete(sidecar_fixture, monkeypatch):
    output, prepared, protected = sidecar_fixture
    original = m.atomic_write_csv
    def fail(path, records, fieldnames=None):
        if Path(path).as_posix().endswith(m.SUMMARY_PATH):
            raise PermissionError("synthetic summary sharing denial")
        return original(path, records, fieldnames)
    monkeypatch.setattr(m, "atomic_write_csv", fail)
    with pytest.raises(PermissionError, match="synthetic summary"):
        m.write_communication_replay(output, prepared)
    assert json.loads((output / m.MANIFEST_PATH).read_text())["status"] == "failed"
    m.assert_frozen_files(output, protected)


def test_source_changes_between_prepare_and_commit_are_rejected(sidecar_fixture, monkeypatch):
    output, prepared, _ = sidecar_fixture
    monkeypatch.setattr(m, "communication_source_fingerprint", lambda: ("changed", {}))
    with pytest.raises(ValueError, match="implementation changed"):
        m.write_communication_replay(output, prepared)


def test_old_o2_files_do_not_authorize_a_completed_encoded_generation(sidecar_fixture, monkeypatch):
    output, prepared, _ = sidecar_fixture
    for extension in ("pdf", "png"):
        (output / f"figures/Fig_O2_communication_overhead.{extension}").write_bytes(b"old O2")
    metadata = dict(prepared.manifest, status="computed")
    monkeypatch.setattr(m, "validate_encoded_overlay", lambda *args, **kwargs: deepcopy(metadata))
    with pytest.raises(ValueError, match="plot metadata"):
        m.finalize_communication(output, dict(communication_fingerprint="old", base_run_fingerprint="base",
                                            figures_generated=["Fig_O2_communication_overhead"]))


def test_missing_new_o2_png_does_not_authorize_completion(sidecar_fixture, monkeypatch):
    output, prepared, _ = sidecar_fixture
    (output / "figures/Fig_O2_communication_overhead.pdf").write_bytes(b"new PDF")
    metadata = dict(prepared.manifest, status="computed")
    monkeypatch.setattr(m, "validate_encoded_overlay", lambda *args, **kwargs: deepcopy(metadata))
    with pytest.raises(ValueError, match="missing or empty"):
        m.finalize_communication(output, dict(communication_fingerprint=metadata["communication_fingerprint"],
            base_run_fingerprint="base", figures_generated=["Fig_O2_communication_overhead"]))


def test_finalization_hashes_only_new_o2_and_keeps_frozen_o1(sidecar_fixture, monkeypatch):
    output, prepared, protected = sidecar_fixture
    for extension in ("pdf", "png"):
        (output / f"figures/Fig_O2_communication_overhead.{extension}").write_bytes(f"new O2 {extension}".encode())
    metadata = dict(prepared.manifest, status="computed")
    monkeypatch.setattr(m, "validate_encoded_overlay", lambda *args, **kwargs: deepcopy(metadata))
    result = m.finalize_communication(output, dict(communication_fingerprint=metadata["communication_fingerprint"],
        base_run_fingerprint="base", figures_generated=["Fig_O2_communication_overhead"]))
    assert result["status"] == "complete"
    assert set(result["figure_files_sha256"]) == {"figures/Fig_O2_communication_overhead.pdf",
                                                 "figures/Fig_O2_communication_overhead.png"}
    m.assert_frozen_files(output, protected)


def test_final_manifest_failure_cannot_publish_complete(sidecar_fixture, monkeypatch):
    output, prepared, _ = sidecar_fixture
    for extension in ("pdf", "png"):
        (output / f"figures/Fig_O2_communication_overhead.{extension}").write_bytes(b"new O2")
    metadata = dict(prepared.manifest, status="computed")
    (output / m.MANIFEST_PATH).write_text(json.dumps(metadata))
    monkeypatch.setattr(m, "validate_encoded_overlay", lambda *args, **kwargs: deepcopy(metadata))
    monkeypatch.setattr(m, "atomic_write_json", lambda *args, **kwargs: (_ for _ in ()).throw(PermissionError("final commit denied")))
    with pytest.raises(PermissionError, match="final commit"):
        m.finalize_communication(output, dict(communication_fingerprint=metadata["communication_fingerprint"],
            base_run_fingerprint="base", figures_generated=["Fig_O2_communication_overhead"]))
    assert json.loads((output / m.MANIFEST_PATH).read_text())["status"] == "computed"


def test_completed_overlay_can_finalize_deliberately_redrawn_o2(sidecar_fixture, monkeypatch):
    output, prepared, protected = sidecar_fixture
    old_figures = {}
    for extension in ("pdf", "png"):
        relative = f"figures/Fig_O2_communication_overhead.{extension}"
        (output / relative).write_bytes(f"old O2 {extension}".encode())
        old_figures[relative] = m._file_hash(output / relative)
    complete = dict(prepared.manifest, status="complete", figure_files_sha256=old_figures)
    (output / m.MANIFEST_PATH).write_text(json.dumps(complete))
    for relative in old_figures:
        (output / relative).write_bytes(f"redrawn {relative}".encode())
    calls = []
    def validate(output, *, allow_computed=False, verify_figures=True):
        calls.append((allow_computed, verify_figures))
        if verify_figures:
            m._verify_hashes(output, old_figures, "Old O2")
        return deepcopy(complete)
    monkeypatch.setattr(m, "validate_encoded_overlay", validate)
    result = m.finalize_communication(output, dict(
        communication_fingerprint=complete["communication_fingerprint"], base_run_fingerprint="base",
        figures_generated=["Fig_O2_communication_overhead"]))
    assert calls == [(True, False)]
    assert result["status"] == "complete"
    assert result["figure_files_sha256"] != old_figures
    m.assert_frozen_files(output, protected)


@pytest.fixture
def valid_overlay_fixture(sidecar_fixture, case_and_row, monkeypatch):
    output, prepared, _ = sidecar_fixture
    case, row = case_and_row
    monkeypatch.setattr(m, "N_VALUES", (1000,))
    monkeypatch.setattr(m, "D_PANEL_VALUES", (10,))
    monkeypatch.setattr(m, "configurations", lambda: ((1000, 10, 2026),))
    monkeypatch.setattr(m, "_validate_base_manifest", lambda manifest: case.config)
    source_path = "experiments/02_overhead/config_overhead.py"
    base = dict(run_fingerprint="base", computation_source_files_sha256={
        source_path: m._file_hash(m.PROJECT_ROOT / source_path)})
    (output / "logs/run_manifest.json").write_text(json.dumps(base))
    originals = []
    for method in m.METHODS:
        originals.append(dict(N=1000, d=10, seed=2026, method=method, B=row["B"],
            fixed_workload_sha256="a" * 64, total_bytes=100000 if method == m.RACA_HIM else 300,
            gt_bytes=50, verification_bytes=150,
            control_bytes=99800 if method == m.RACA_HIM else 100, message_count=321))
    m.atomic_write_csv(output / "raw/overhead_runs.csv", originals)
    protected = {relative: m._file_hash(output / relative) for relative in m._REQUIRED_BASE_FILES}
    metadata = deepcopy(prepared.manifest)
    metadata.update(frozen_files_sha256=protected, N_values=[1000], d_panel_values=[10], seeds=[2026],
        methods=list(m.METHODS), method_records=4, saved_configurations=1)
    metadata["communication_fingerprint"] = m._sha256(m._json_bytes(dict(schema=m.COMMUNICATION_SCHEMA,
        base_run_fingerprint="base", frozen_files_sha256=protected,
        frozen_checkpoint_inventory=metadata["frozen_checkpoint_inventory"],
        communication_source_sha256=metadata["communication_source_sha256"], codec_policy=metadata["codec_policy"])))
    totals, packets = m.replay_raca_packets(case, row)
    metadata["packet_count"] = len(packets)
    runs, summaries = [], []
    for original in originals:
        values = totals if original["method"] == m.RACA_HIM else dict(encoded_total_bytes=original["total_bytes"],
            gt_bytes=original["gt_bytes"], verification_bytes=original["verification_bytes"],
            control_bytes=original["control_bytes"], message_count=original["message_count"])
        run = dict(original, **values) if original["method"] == m.RACA_HIM else dict(original, **values)
        run.update(base_total_bytes=original["total_bytes"], saved_bytes=original["total_bytes"] - values["encoded_total_bytes"],
            status="success", base_run_fingerprint="base", communication_fingerprint=metadata["communication_fingerprint"])
        runs.append(run)
        summaries.append(dict(N=1000, d=10, method=original["method"], n_valid=1, n_expected=1,
            implementation_version=m.OVERHEAD_IMPLEMENTATION_VERSION, run_fingerprint="base", base_run_fingerprint="base",
            communication_fingerprint=metadata["communication_fingerprint"], communication_schema=m.COMMUNICATION_SCHEMA,
            metric="encoded_total_bytes", value=values["encoded_total_bytes"], mean=values["encoded_total_bytes"],
            evaluation_mode=m.MAIN_EVALUATION_MODE))
    prepared = m.PreparedCommunicationReplay(metadata, runs, summaries, packets)
    metadata = m.write_communication_replay(output, prepared)
    return output, metadata, prepared


def test_full_overlay_validator_checks_serialized_packet_sums_and_paired_grid(valid_overlay_fixture):
    output, metadata, _ = valid_overlay_fixture
    checked = m.validate_encoded_overlay(output, allow_computed=True)
    assert checked["communication_fingerprint"] == metadata["communication_fingerprint"]
    with pytest.raises(ValueError, match="incomplete"):
        m.validate_encoded_overlay(output)


def test_validator_rejects_encoded_csv_replacement_with_unchanged_manifest(valid_overlay_fixture):
    output, _, _ = valid_overlay_fixture
    (output / m.SUMMARY_PATH).write_bytes(b"replaced summary")
    with pytest.raises(ValueError, match="Encoded communication output"):
        m.validate_encoded_overlay(output, allow_computed=True)


def test_validator_rejects_baseline_byte_changes_even_if_new_csv_hashes_are_recorded(valid_overlay_fixture):
    output, metadata, prepared = valid_overlay_fixture
    baseline = next(row for row in prepared.runs if row["method"] != m.RACA_HIM)
    baseline["encoded_total_bytes"] += 1
    baseline["control_bytes"] += 1
    baseline["saved_bytes"] -= 1
    summary = next(row for row in prepared.summaries if row["method"] == baseline["method"])
    summary["value"] = summary["mean"] = baseline["encoded_total_bytes"]
    for relative, rows in ((m.RUNS_PATH, prepared.runs), (m.SUMMARY_PATH, prepared.summaries)):
        m.atomic_write_csv(output / relative, rows)
        metadata["files_sha256"][relative] = m._file_hash(output / relative)
    m.atomic_write_json(output / m.MANIFEST_PATH, metadata)
    with pytest.raises(ValueError, match="baseline"):
        m.validate_encoded_overlay(output, allow_computed=True)


def test_actual_complete_to_redrawn_o2_finalization_rebinds_hashes_and_preserves_o1(valid_overlay_fixture):
    output, metadata, _ = valid_overlay_fixture
    plot = dict(communication_fingerprint=metadata["communication_fingerprint"], base_run_fingerprint="base",
                figures_generated=["Fig_O2_communication_overhead"])
    for extension in ("pdf", "png"):
        (output / f"figures/Fig_O2_communication_overhead.{extension}").write_bytes(b"first O2")
    first = m.finalize_communication(output, plot)
    m.validate_encoded_overlay(output)
    for extension in ("pdf", "png"):
        (output / f"figures/Fig_O2_communication_overhead.{extension}").write_bytes(b"redrawn O2")
    with pytest.raises(ValueError, match="O2 figure"):
        m.validate_encoded_overlay(output)
    second = m.finalize_communication(output, plot)
    assert second["figure_files_sha256"] != first["figure_files_sha256"]
    assert m.validate_encoded_overlay(output)["status"] == "complete"
    m.assert_frozen_files(output, metadata["frozen_files_sha256"])


def test_replay_wrapper_uses_independent_preparation_and_commit_without_clock_or_simulation(monkeypatch):
    calls = []
    prepared = SimpleNamespace()
    monkeypatch.setattr(m, "prepare_communication_replay", lambda output, **kwargs: calls.append(("prepare", output)) or prepared)
    monkeypatch.setattr(m, "write_communication_replay", lambda output, item: calls.append(("write", output, item)) or {"status": "computed"})
    assert m.replay_communication("output") == {"status": "computed"}
    assert calls == [("prepare", "output"), ("write", "output", prepared)]
