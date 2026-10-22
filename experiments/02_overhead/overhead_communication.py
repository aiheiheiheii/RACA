"""Replay a saved v4 workload with an independently recorded wire encoding.

This module never measures CPU or changes the v4 timing experiment.  RACA's
direction packets certify the original binary cosine verdict; an ambiguous
certificate requests the low plane and then the original float32 descriptor.
The three baseline adapters retain their saved compact protocol byte ledgers.
All packets below are application messages on the original separate links;
their four-byte frames do not claim to include TCP, TLS or authentication.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
from dataclasses import dataclass, replace
from datetime import datetime, timezone
import csv
import gzip
import hashlib
import json
from pathlib import Path
import struct

import numpy as np

from experiments.common.method_names import RACA_HIM
from experiments.common.reproducibility import dataset_snapshot
from experiments.common.similarity import cosine_aligned, cosine_pairwise, within_interval
from .config_overhead import (DATASET_ROOT, PROJECT_ROOT, N_VALUES, D_PANEL_VALUES,
    SEEDS, METHODS, MAIN_EVALUATION_MODE, OVERHEAD_IMPLEMENTATION_VERSION,
    OverheadConfig, configurations)
from .overhead_calibration import KernelCalibration
from .overhead_core_simulator import _core_costs
from .overhead_identity import verified_checkpoint_sources
from .overhead_population import load_scaling_group
from .overhead_resources import shared_resource_profiles, resource_fixture_metadata
from .overhead_run_setup import calibration_key
from .overhead_scaling import stable_model_ids, nested_fixed_targets
from .overhead_simulator import EventSimulator, joint_schedule
from .overhead_storage import atomic_write_json, atomic_write_csv, retry_io
from .raca_kernels import cached_reference_cosine, history_update_vectorized
from . import raca_wire_codec as codec


COMMUNICATION_SCHEMA = "overhead-encoded-communication-v1"
MANIFEST_PATH = "logs/communication_manifest.json"
SUMMARY_PATH = "summary/communication_encoded_summary.csv"
RUNS_PATH = "raw/communication_encoded_runs.csv"
PACKET_LEDGER_PATH = "logs/raca_encoded_packet_ledger.csv"
FRAME_VERSION = 1
FRAME_BYTES = 4
_QUERY, _INITIAL, _REFINE_QUERY, _REFINE, _RAW_QUERY, _RAW, _ASSIGNMENT, _RESULT = range(1, 9)
_REQUIRED_BASE_FILES = (
    "logs/run_manifest.json", "raw/overhead_runs.csv", "raw/protocol_diagnostics.csv",
    "summary/time_overhead_summary.csv", "summary/communication_overhead_summary.csv",
    "figures/Fig_O1_time_overhead.pdf", "figures/Fig_O1_time_overhead.png",
)


def _json_bytes(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def _sha256(value):
    return hashlib.sha256(value).hexdigest()


def _file_hash(path):
    digest = hashlib.sha256()
    with retry_io(lambda: Path(path).open("rb"), description=f"Read protected file {path}") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _relative_path(root, relative):
    path = (Path(root) / relative).resolve()
    if (not isinstance(relative, str) or Path(relative).is_absolute()
            or not path.is_relative_to(Path(root).resolve())):
        raise ValueError("Communication metadata contains an invalid relative path")
    return path


def _verify_hashes(root, records, description):
    if not isinstance(records, dict) or not records:
        raise ValueError(f"Missing {description} SHA256 inventory")
    for relative, expected in records.items():
        path = _relative_path(root, relative)
        if not path.is_file() or _file_hash(path) != expected:
            raise ValueError(f"{description} changed or disappeared: {relative}")


def assert_frozen_files(output, hashes):
    """Check every frozen numerical artifact and O1 without rewriting a file."""
    _verify_hashes(Path(output).resolve(), hashes, "Frozen v4 timing/source artifact")


def checkpoint_inventory(output):
    """Cover the old 2.4 GB checkpoint archive without rereading it as inputs."""
    root = Path(output).resolve()
    return {path.relative_to(root).as_posix(): dict(size=path.stat().st_size,
        mtime_ns=path.stat().st_mtime_ns) for path in sorted((root / "checkpoints").glob("*.json.gz"))}


def _assert_checkpoint_inventory(output, inventory):
    if not isinstance(inventory, dict) or not inventory or checkpoint_inventory(output) != inventory:
        raise ValueError("Saved checkpoint archive paths/sizes/modification times changed during communication replay")


def communication_policy():
    return dict(codec_schema=codec.WIRE_CODEC_SCHEMA, frame_version=FRAME_VERSION,
        application_frame_bytes=FRAME_BYTES, descriptor_query_body_bytes=2,
        descriptor_query="dimension byte and requested tier byte; original separate directed link",
        initial_descriptor="mode + dimension + float32 scale + high byte plane",
        refinement_descriptor="mode + dimension + low byte plane; tied to the same target/channel",
        raw_descriptor="mode + dimension + original little-endian float32 descriptor",
        assignment="unsigned varint verifier ID/count/first target ID; zigzag varint deltas preserve queue order",
        result="unsigned varint task count and little-bit-order verdict bitmap",
        ground_truth="majority support lower/upper bounds; resolve earliest unknown needed for first M accepted IDs; necessary expansion retained",
        verification="same saved assignments and binary verdicts; every ambiguous pair refines or falls back to raw",
        baseline_policy="three saved compact adapted protocol byte ledgers retained without alteration",
        bytes_scope="serialized application frames; routing/link context reused; TCP/TLS/authentication absent as in v4",
        time_scope="all v4 timing observations and O1 files frozen; no claim of encoded end-to-end runtime")


def communication_source_fingerprint():
    paths = (Path(__file__), Path(codec.__file__))
    records = {path.relative_to(PROJECT_ROOT).as_posix(): _file_hash(path) for path in paths}
    return _sha256(_json_bytes(records)), records


def _uint(value, name="integer"):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return int(value)


def encode_varint(value):
    value = _uint(value)
    if value > (1 << 63) - 1:
        raise ValueError("Varint exceeds the signed int64 identity domain")
    encoded = bytearray()
    while value >= 128:
        encoded.append((value & 127) | 128)
        value >>= 7
    encoded.append(value)
    return bytes(encoded)


def decode_varint(payload, offset=0):
    value, shift, start = 0, 0, offset
    while offset < len(payload) and offset - start < 10:
        item = payload[offset]
        offset += 1
        value |= (item & 127) << shift
        if not item & 128:
            if value > (1 << 63) - 1 or payload[start:offset] != encode_varint(value):
                raise ValueError("Noncanonical or oversized varint")
            return value, offset
        shift += 7
    raise ValueError("Truncated or oversized varint")


def frame(opcode, body):
    opcode = _uint(opcode, "opcode")
    if not 1 <= opcode <= _RESULT or not isinstance(body, bytes) or len(body) > 65535:
        raise ValueError("Invalid application frame opcode or body")
    return struct.pack("<BBH", FRAME_VERSION, opcode, len(body)) + body


def decode_frame(packet, expected_opcode=None):
    if not isinstance(packet, bytes) or len(packet) < FRAME_BYTES:
        raise ValueError("Truncated application frame")
    version, opcode, size = struct.unpack("<BBH", packet[:FRAME_BYTES])
    if version != FRAME_VERSION or not 1 <= opcode <= _RESULT or size != len(packet) - FRAME_BYTES:
        raise ValueError("Invalid application frame version/opcode/body length")
    if expected_opcode is not None and opcode != expected_opcode:
        raise ValueError("Application frame opcode does not match its request")
    return opcode, packet[FRAME_BYTES:]


def encode_assignment(verifier_id, target_ids):
    verifier_id = _uint(verifier_id, "verifier identity")
    ids = [_uint(value, "target identity") for value in target_ids]
    if not ids or len(set(ids)) != len(ids):
        raise ValueError("An assignment queue must contain unique target identities")
    body = encode_varint(verifier_id) + encode_varint(len(ids)) + encode_varint(ids[0])
    for previous, current in zip(ids, ids[1:]):
        delta = current - previous
        body += encode_varint(2 * delta if delta >= 0 else -2 * delta - 1)
    return frame(_ASSIGNMENT, body)


def decode_assignment(packet):
    _, body = decode_frame(packet, _ASSIGNMENT)
    verifier, offset = decode_varint(body)
    count, offset = decode_varint(body, offset)
    if count < 1 or count > len(body):
        raise ValueError("Invalid assignment count")
    first, offset = decode_varint(body, offset)
    ids = [first]
    for _ in range(count - 1):
        value, offset = decode_varint(body, offset)
        delta = value // 2 if value % 2 == 0 else -(value // 2) - 1
        identity = ids[-1] + delta
        if not 0 <= identity <= (1 << 63) - 1:
            raise ValueError("Decoded target identity is outside int64")
        ids.append(identity)
    if offset != len(body) or len(set(ids)) != count:
        raise ValueError("Assignment has trailing bytes or duplicate identities")
    return verifier, ids


def encode_result(predictions):
    verdicts = np.asarray(predictions)
    if verdicts.ndim != 1 or not len(verdicts) or not np.all((verdicts == 0) | (verdicts == 1)):
        raise ValueError("A result queue requires binary verdicts")
    body = encode_varint(len(verdicts)) + np.packbits(verdicts.astype(np.uint8), bitorder="little").tobytes()
    return frame(_RESULT, body)


def decode_result(packet):
    _, body = decode_frame(packet, _RESULT)
    count, offset = decode_varint(body)
    if count < 1 or len(body) - offset != (count + 7) // 8:
        raise ValueError("Invalid verdict bitmap length")
    bits = np.unpackbits(np.frombuffer(body[offset:], np.uint8), bitorder="little")
    if np.any(bits[count:]):
        raise ValueError("Unused verdict bitmap bits must be zero")
    return bits[:count].astype(np.int8)


class PacketLedger:
    """Retain actual serialization lengths and a SHA256 for each directed send."""

    def __init__(self, n, d, seed):
        self.context = dict(N=n, d=d, seed=seed, method=RACA_HIM,
            configuration_id=f"N{n}_d{d}_seed{seed}", evaluation_mode=MAIN_EVALUATION_MODE)
        self.packets = []

    def add(self, packet, sender, receiver, phase, message_type, *, payload=False,
            target_row=None, verifier_row=None, stage=""):
        opcode, body = decode_frame(packet)
        if phase not in {"gt", "verification"} or sender == receiver:
            raise ValueError("A serialized packet must use a valid separate directed link")
        payload_bytes = len(body) if payload else 0
        control_bytes = FRAME_BYTES if payload else len(packet)
        entry = dict(self.context, packet_id=len(self.packets), sender=int(sender), receiver=int(receiver),
            phase=phase, message_type=message_type, opcode=opcode, stage=stage,
            target_row=target_row, verifier_row=verifier_row, body_bytes=len(body),
            payload_bytes=payload_bytes, control_bytes=control_bytes,
            total_bytes=len(packet), packet_sha256=_sha256(packet), roundtrip_verified=True)
        self.packets.append(entry)
        return body

    def totals(self):
        gt = sum(packet["payload_bytes"] for packet in self.packets if packet["phase"] == "gt")
        verification = sum(packet["payload_bytes"] for packet in self.packets if packet["phase"] == "verification")
        control = sum(packet["control_bytes"] for packet in self.packets)
        total = sum(packet["total_bytes"] for packet in self.packets)
        if gt + verification + control != total:
            raise AssertionError("Serialized payload/control ledger is not an exact byte partition")
        return dict(gt_bytes=gt, verification_bytes=verification, control_bytes=control,
            encoded_total_bytes=total, message_count=len(self.packets))


class DescriptorChannels:
    """Each receiver owns only the packets actually delivered on its channel."""

    def __init__(self, X, ledger):
        self.X, self.ledger = X, ledger
        self.prepared, self.decoded, self.raw_rows = {}, {}, {}
        self.refinement_count = self.raw_fallback_count = 0

    def request(self, row, receiver, phase, tier):
        key = (phase, int(row), int(receiver))
        if row == receiver or tier not in (0, 1, 2):
            raise ValueError("Invalid descriptor channel or tier")
        if int(row) not in self.prepared:
            self.prepared[int(row)] = codec.encode_descriptor(self.X[row])
        prepared = self.prepared[int(row)]
        if tier == 0:
            if key in self.decoded:
                raise ValueError("Duplicate initial descriptor delivery")
            query_opcode, reply_opcode, payload = _QUERY, _INITIAL, prepared.initial
        elif tier == 1:
            if key not in self.decoded or self.decoded[key].stage != "initial":
                raise ValueError("A low plane must refine the initial packet on the same channel")
            query_opcode, reply_opcode, payload = _REFINE_QUERY, _REFINE, prepared.refinement
        else:
            if key not in self.decoded or self.decoded[key].stage != "refinement":
                raise ValueError("Raw fallback must follow the same channel's low plane")
            query_opcode, reply_opcode, payload = _RAW_QUERY, _RAW, prepared.raw
        query = frame(query_opcode, bytes((self.X.shape[1], tier)))
        body = self.ledger.add(query, receiver, row, phase, "descriptor_query", target_row=int(row),
            verifier_row=receiver if phase == "verification" else None, stage=str(tier))
        if body != bytes((self.X.shape[1], tier)) or len(query) != 6:
            raise AssertionError("Descriptor query did not roundtrip")
        body = self.ledger.add(frame(reply_opcode, payload), row, receiver, phase, "descriptor_response",
            payload=True, target_row=int(row), verifier_row=receiver if phase == "verification" else None,
            stage=str(tier))
        if tier == 0:
            decoded = codec.decode_initial(body)
        elif tier == 1:
            decoded = codec.decode_refinement(self.decoded[key], body)
            self.refinement_count += 1
        else:
            raw = codec.decode_raw(body)
            if raw.dtype != np.float32 or raw.shape != self.X[row].shape or raw.tobytes() != self.X[row].tobytes():
                raise AssertionError("Raw fallback failed the original float32 roundtrip")
            self.raw_rows[key] = raw
            decoded = codec.exact_direction(raw)
            self.raw_fallback_count += 1
        if decoded.d != self.X.shape[1]:
            raise AssertionError("Decoded descriptor does not match its queried channel dimension")
        self.decoded[key] = decoded
        return decoded


@dataclass(frozen=True)
class ReplayCase:
    X: np.ndarray
    model_ids: np.ndarray
    targets: np.ndarray
    verifiers: np.ndarray
    assignments: tuple
    expansion_sizes: tuple
    config: OverheadConfig


def _replay_gt(case, channels):
    config, X = case.config, case.X
    ordered = np.argsort(case.model_ids, kind="stable")
    previous = 0
    selected = None
    verdicts = {}
    final_unresolved = 0
    for step_id, size in enumerate(case.expansion_sizes):
        expected = config.K if step_id == 0 else min(previous + config.expansion_step, len(X))
        if size != expected or size <= previous or size > len(X):
            raise ValueError("Saved GT candidate expansion does not match its v4 policy")
        candidates = ordered[:size]
        for row in candidates[previous:]:
            channels.request(int(row), -1, "gt", 0)
        pairs = [(int(a), int(b)) for index, a in enumerate(candidates)
                 for b in candidates[index + 1:]]
        def unresolved():
            pending = []
            for left, right in pairs:
                if (left, right) in verdicts:
                    continue
                a, b = channels.decoded[("gt", left, -1)], channels.decoded[("gt", right, -1)]
                if a.stage == b.stage == "exact":
                    score = cosine_aligned(channels.raw_rows[("gt", left, -1)][None, :],
                                           channels.raw_rows[("gt", right, -1)][None, :])[0]
                    value = int(not bool(within_interval(score, config.tau_lower, config.tau_upper)))
                else:
                    value = codec.certify_pair(a, b, config.tau_lower, config.tau_upper)
                if value is None:
                    pending.append((left, right))
                else:
                    verdicts[(left, right)] = int(value)
            return pending
        threshold = (size + 2) // 2
        while True:
            ambiguous = unresolved()
            lower = dict.fromkeys(map(int, candidates), 1)  # candidate self support
            unknown = dict.fromkeys(map(int, candidates), 0)
            for left, right in pairs:
                if (left, right) not in verdicts:
                    unknown[left] += 1
                    unknown[right] += 1
                elif verdicts[(left, right)] == 0:
                    lower[left] += 1
                    lower[right] += 1
            statuses = ["accepted" if lower[int(row)] >= threshold else
                "rejected" if lower[int(row)] + unknown[int(row)] < threshold else "unknown"
                for row in candidates]
            accepted_positions = [index for index, status in enumerate(statuses) if status == "accepted"]
            unknown_positions = [index for index, status in enumerate(statuses) if status == "unknown"]
            if len(accepted_positions) >= config.M:
                cutoff = accepted_positions[config.M - 1]
                earlier_unknown = [index for index in unknown_positions if index < cutoff]
                if not earlier_unknown:
                    if step_id != len(case.expansion_sizes) - 1:
                        raise ValueError("Saved GT expanded after already reaching its majority requirement")
                    selected = candidates[accepted_positions[:config.M]]
                    break
                needed = earlier_unknown[0]
            elif len(accepted_positions) + len(unknown_positions) < config.M:
                if step_id == len(case.expansion_sizes) - 1:
                    raise AssertionError("Encoded GT did not reach the saved successful majority")
                break  # Proven insufficient accepted candidates; expansion is necessary.
            else:
                needed = unknown_positions[0]
            node = int(candidates[needed])
            neighbors = {right if left == node else left for left, right in ambiguous
                         if left == node or right == node}
            involved = neighbors | {node}
            # First refine the earliest unknown itself, then its uncertain
            # neighbors with the largest received angular error.  Raw is used
            # only once all relevant initial planes have already been refined.
            initial = [row for row in involved if channels.decoded[("gt", row, -1)].stage == "initial"]
            refined = [row for row in involved if channels.decoded[("gt", row, -1)].stage == "refinement"]
            eligible, tier = (initial, 1) if initial else (refined, 2)
            if not eligible:
                raise AssertionError("A GT support interval stayed ambiguous after receiving all relevant raw descriptors")
            chosen = node if node in eligible else min(eligible, key=lambda row: (
                -channels.decoded[("gt", row, -1)].angle_error, int(case.model_ids[row])))
            channels.request(chosen, -1, "gt", tier)
        final_unresolved = len(ambiguous)
        # Source descriptors are used only for this audit of original verdicts;
        # the receiver above computes its decisions from delivered packets.
        original = cosine_pairwise(X[candidates])
        position = {int(row): index for index, row in enumerate(candidates)}
        for left, right in pairs:
            if (left, right) not in verdicts:
                continue
            corrupted = verdicts[(left, right)]
            expected_corrupted = int(not bool(within_interval(original[position[left], position[right]],
                                                              config.tau_lower, config.tau_upper)))
            if corrupted != expected_corrupted:
                raise AssertionError("Encoded GT pair changed an original majority-support verdict")
        m_support = within_interval(original, config.tau_lower, config.tau_upper).copy()
        np.fill_diagonal(m_support, False)
        original_accepted = candidates[m_support.sum(axis=1) + 1 >= threshold]
        if selected is not None:
            if not np.array_equal(selected, original_accepted[:config.M]):
                raise AssertionError("Certified majority short circuit changed the first M original accepted identities")
        elif len(original_accepted) >= config.M:
            raise AssertionError("Certified support bounds incorrectly required a candidate expansion")
        previous = size
    if selected is None or not np.array_equal(selected, case.verifiers):
        raise AssertionError("Encoded GT changed the saved verifier membership/order")
    return dict(gt_certified_pair_count=len(verdicts), gt_unresolved_pair_count=final_unresolved,
        gt_majority_short_circuit=final_unresolved > 0)


def _replay_verification(case, channels):
    config = case.config
    queues = {}
    for target, verifier in case.assignments:
        queues.setdefault(verifier, []).append(target)
    for verifier, targets in queues.items():
        identities = case.model_ids[targets].tolist()
        packet = encode_assignment(int(case.model_ids[verifier]), identities)
        recovered_verifier, recovered_targets = decode_assignment(packet)
        if recovered_verifier != int(case.model_ids[verifier]) or recovered_targets != identities:
            raise AssertionError("Assignment queue/order failed its serialization roundtrip")
        channels.ledger.add(packet, -1, verifier, "verification", "batch_assignment",
            verifier_row=verifier)
    original_scores = cached_reference_cosine(case.X,
        [target for target, _ in case.assignments], [verifier for _, verifier in case.assignments])
    expected = (~within_interval(original_scores, config.tau_lower, config.tau_upper)).astype(np.int8)
    predictions = {}
    references = {verifier: codec.exact_direction(case.X[verifier]) for verifier in queues}
    for index, (target, verifier) in enumerate(case.assignments):
        received = channels.request(target, verifier, "verification", 0)
        value = codec.certify_pair(received, references[verifier], config.tau_lower, config.tau_upper)
        if value is None:
            received = channels.request(target, verifier, "verification", 1)
            value = codec.certify_pair(received, references[verifier], config.tau_lower, config.tau_upper)
        if value is None:
            channels.request(target, verifier, "verification", 2)
            score = cosine_aligned(channels.raw_rows[("verification", target, verifier)][None, :],
                                   case.X[verifier][None, :])[0]
            value = int(not bool(within_interval(score, config.tau_lower, config.tau_upper)))
        if int(value) != int(expected[index]):
            raise AssertionError("Encoded descriptor changed an original verification verdict")
        predictions[target] = int(value)
    for verifier, targets in queues.items():
        values = [predictions[target] for target in targets]
        packet = encode_result(values)
        if decode_result(packet).tolist() != values:
            raise AssertionError("Verdict bitmap/order failed its serialization roundtrip")
        channels.ledger.add(packet, verifier, -1, "verification", "batch_verification_result",
            verifier_row=verifier)
    return predictions


def replay_raca_packets(case, saved_row):
    """Serialize a reconstructed fixed workload and audit every original verdict."""
    n, d, seed = int(saved_row["N"]), int(saved_row["d"]), int(saved_row["seed"])
    ledger = PacketLedger(n, d, seed)
    channels = DescriptorChannels(case.X, ledger)
    gt_audit = _replay_gt(case, channels)
    predictions = _replay_verification(case, channels)
    count = sum(predictions.values())
    if count != int(saved_row["corrupted_verdict_count"]):
        raise AssertionError("Encoded verification changed the saved total corruption verdict count")
    if "diagnostics_json" in saved_row:
        diagnostics = json.loads(saved_row["diagnostics_json"])
        history = history_update_vectorized(np.full(len(case.X), case.config.h0),
            list(predictions), list(predictions.values()), case.verifiers, case.config.rho)
        if _sha256(history.astype("<f8").tobytes()) != diagnostics["updated_history_sha256"]:
            raise AssertionError("Encoded verification changed the saved history-update audit fingerprint")
    totals = ledger.totals()
    totals.update(refinement_count=channels.refinement_count, raw_fallback_count=channels.raw_fallback_count,
        gt_refinement_count=sum(packet["phase"] == "gt" and packet["opcode"] == _REFINE for packet in ledger.packets),
        gt_raw_fallback_count=sum(packet["phase"] == "gt" and packet["opcode"] == _RAW for packet in ledger.packets),
        verdict_count=count, gt_membership_preserved=True, verification_verdicts_preserved=True,
        history_update_preserved=True, **gt_audit)
    return totals, ledger.packets


def _read_csv(path):
    with retry_io(lambda: Path(path).open(encoding="utf-8-sig", newline=""),
                  description=f"Read communication source {path}") as stream:
        return list(csv.DictReader(stream))


def _validate_base_manifest(manifest):
    if (manifest.get("implementation_version") != OVERHEAD_IMPLEMENTATION_VERSION
            or manifest.get("status") != "complete" or manifest.get("missing_configurations") != 0
            or manifest.get("saved_configurations") != len(configurations())
            or manifest.get("saved_method_mode_runs") != len(configurations()) * len(METHODS)
            or manifest.get("derived_outputs", {}).get("state") != "complete"
            or manifest.get("derived_outputs", {}).get("run_fingerprint") != manifest.get("run_fingerprint")
            or manifest.get("N_values") != list(N_VALUES)
            or manifest.get("d_panel_values") != list(D_PANEL_VALUES)
            or manifest.get("seeds") != list(SEEDS)
            or manifest.get("methods") != list(METHODS)
            or manifest.get("evaluation_modes") != [MAIN_EVALUATION_MODE]):
        raise ValueError("Communication replay requires an intact, complete current v4 fixed-workload generation")
    accepted, rejected = verified_checkpoint_sources(manifest, manifest)
    if not any(record["run_fingerprint"] == manifest.get("run_fingerprint") for record in accepted):
        raise ValueError(f"Saved v4 generation identity is invalid: {rejected}")
    return OverheadConfig(**manifest["config"])


def _validate_ledger(row):
    ledger = json.loads(row["communication_phase_totals_json"])
    for packet in ledger:
        if packet["payload_bytes"] + packet["control_bytes"] != packet["total_bytes"]:
            raise ValueError("Saved logical ledger has overlapping or omitted bytes")
    if (sum(packet["total_bytes"] for packet in ledger) != int(row["total_bytes"])
            or int(row["gt_bytes"]) + int(row["verification_bytes"]) + int(row["control_bytes"]) != int(row["total_bytes"])
            or sum(packet["message_count"] for packet in ledger) != int(row["message_count"])):
        raise ValueError("Saved logical communication totals disagree with their exact phase ledger")


def _load_checkpoints(output, manifest, config):
    fingerprint, records, protected = manifest["run_fingerprint"], {}, {}
    for n, d, seed in configurations():
        relative = f"checkpoints/N{n}_d{d}_seed{seed}.json.gz"
        path = output / relative
        protected[relative] = _file_hash(path)
        with retry_io(lambda: gzip.open(path, "rt", encoding="utf-8"),
                      description=f"Read saved communication workload {path}") as stream:
            checkpoint = json.load(stream)
        results = checkpoint.get("results", [])
        if (checkpoint.get("run_fingerprint") != fingerprint or not checkpoint.get("complete")
                or checkpoint.get("configuration") != dict(N=n, d=d, seed=seed)
                or checkpoint.get("checkpoint_payload_schema") != "core-summary-v1"
                or len(results) != len(METHODS)):
            raise ValueError(f"Invalid or mixed-generation v4 checkpoint: {relative}")
        seen = set()
        for result in results:
            row = result["row"]
            method = row.get("method")
            if (method not in METHODS or method in seen or row.get("run_fingerprint") != fingerprint
                    or row.get("implementation_version") != OVERHEAD_IMPLEMENTATION_VERSION
                    or (row.get("N"), row.get("d"), row.get("seed")) != (n, d, seed)
                    or row.get("configuration_id") != f"N{n}_d{d}_seed{seed}"
                    or row.get("evaluation_mode") != MAIN_EVALUATION_MODE or row.get("status") != "success"
                    or any(row.get(field) != config.budget(n) for field in ("B", "selected_count", "attempted_count", "completed_count"))
                    or not np.isfinite(float(row["total_time_ms"])) or float(row["total_time_ms"]) < 0
                    or not isinstance(result.get("diagnostics"), dict)):
                raise ValueError(f"Incomplete/mixed paired result in {relative}")
            _validate_ledger(row)
            if json.loads(row["diagnostics_json"]) != result["diagnostics"]:
                raise ValueError("Checkpoint row and original diagnostics disagree")
            records[(n, d, seed, method)] = result
            seen.add(method)
        if len({item["row"]["fixed_workload_sha256"] for item in results}) != 1:
            raise ValueError("Saved methods do not share one common fixed-workload fingerprint")
    return records, protected


def _reconstruct_case(group, result, config, calibration):
    row, diag = result["row"], result["diagnostics"]
    n, d, seed = row["N"], row["d"], row["seed"]
    X, fixture = np.asarray(group["X"]), group["scaling_fixture"]
    if (X.dtype != np.float32 or X.shape != (n, d) or not np.isfinite(X).all()
            or fixture != diag.get("scaling_fixture")
            or fixture["prefix_sha256"] != _sha256(X.tobytes())
            or calibration.to_dict() != diag.get("kernel_calibration")):
        raise ValueError("Current descriptors/population or saved calibration differ from the v4 workload")
    ids = stable_model_ids(n, seed, base_size=fixture["base_size"], cohort_size=fixture["cohort_size"])
    original_targets, verifiers = np.asarray(diag["workload_rows"], np.int64), np.asarray(diag["reference_rows"], np.int64)
    for values, expected in ((original_targets, row["B"]), (verifiers, config.M)):
        if values.shape != (expected,) or len(np.unique(values)) != expected or np.any(values < 0) or np.any(values >= n):
            raise ValueError("Saved workload/reference rows are invalid")
    if set(original_targets) & set(verifiers) or not np.array_equal(verifiers, verifiers[np.argsort(ids[verifiers], kind="stable")]):
        raise ValueError("Saved verifier order/intersection is inconsistent")
    if _sha256(original_targets.astype("<i8").tobytes()) != row["fixed_workload_sha256"]:
        raise ValueError("Saved workload row order does not match its fixed fingerprint")
    baseline = np.argsort(ids, kind="stable")[:config.M]
    nested = nested_fixed_targets(n, ids, np.union1d(verifiers, baseline), config.budget, seed,
        base_size=fixture["base_size"], cohort_size=fixture["cohort_size"])
    if not np.array_equal(original_targets, nested):
        raise ValueError("Saved targets differ from their content-blind arrival-cohort design")
    profiles = shared_resource_profiles(seed=seed, n=n, rounds=1)
    if resource_fixture_metadata(profiles) != diag.get("resource_fixture"):
        raise ValueError("Reconstructed resource fixture differs from saved v4 resources")
    targets = original_targets[np.argsort(ids[original_targets], kind="stable")]
    sim = EventSimulator(profiles, d, config, retain_events=False, calibration=calibration)
    costs = _core_costs(RACA_HIM, sim, targets, verifiers)
    risk = config.alpha * np.ones(n) + (1 - config.alpha) * np.full(n, config.h0)
    assignments, plan = joint_schedule(targets, verifiers, risk[targets], costs, config.budget(n),
        config.T_round_max_ms, config.lambda_, fixed_workload=True, enforce_deadline=False)
    saved_plan = diag["decision_plan"]
    if set(plan) != set(saved_plan):
        raise ValueError("Reconstructed schedule has a different decision-plan schema")
    for field, value in plan.items():
        saved = saved_plan[field]
        if isinstance(value, list):
            if len(value) != len(saved) or not np.allclose(value, saved, rtol=0, atol=1e-9):
                raise ValueError(f"Reconstructed original assignment differs in {field}")
        elif value != saved:
            raise ValueError(f"Reconstructed original assignment differs in {field}")
    if len(assignments) != row["B"] or {target for target, _ in assignments} != set(targets):
        raise ValueError("Reconstructed assignment does not cover the complete original fixed workload")
    selected_hash = _sha256(np.sort(ids[[target for target, _ in assignments]]).astype("<i8").tobytes())
    if selected_hash != diag["selected_target_ids_sha256"]:
        raise ValueError("Reconstructed selected identities differ from the saved assignment fingerprint")
    expansions = tuple(diag["gt_expansion_sizes"])
    if not expansions or expansions[-1] != diag["candidate_count"]:
        raise ValueError("Saved GT candidate count and expansion sizes disagree")
    old_ledger = json.loads(row["communication_phase_totals_json"])
    requests = next(item for item in old_ledger if item["message_type"] == "gt_candidate_request")
    uploads = next(item for item in old_ledger if item["message_type"] == "gt_candidate_upload")
    if (requests["message_count"] != expansions[-1] or uploads["message_count"] != expansions[-1]
            or requests["control_bytes"] != expansions[-1] * config.request_bytes
            or uploads["payload_bytes"] != expansions[-1] * 4 * d):
        raise ValueError("Saved GT messages do not match the reconstructed candidate expansion")
    return ReplayCase(X, ids, targets, verifiers, tuple(assignments), expansions, config)


@dataclass
class PreparedCommunicationReplay:
    manifest: dict
    runs: list
    summaries: list
    packets: list


def prepare_communication_replay(output_dir, *, dataset_root=DATASET_ROOT):
    """Validate and calculate independently; do not create or modify any file."""
    output = Path(output_dir).resolve()
    manifest = json.loads((output / "logs/run_manifest.json").read_text(encoding="utf-8"))
    config = _validate_base_manifest(manifest)
    _verify_hashes(PROJECT_ROOT, manifest["computation_source_files_sha256"], "Saved v4 timing implementation")
    before = dataset_snapshot(dataset_root)
    if before != manifest["dataset_snapshot"]:
        raise ValueError("Dataset differs from the saved v4 timing generation")
    protected = {relative: _file_hash(output / relative) for relative in _REQUIRED_BASE_FILES}
    frozen_checkpoint_inventory = checkpoint_inventory(output)
    records, checkpoint_hashes = _load_checkpoints(output, manifest, config)
    protected.update(checkpoint_hashes)
    raw = _read_csv(output / "raw/overhead_runs.csv")
    keys = lambda item: (int(item["N"]), int(item["d"]), int(item["seed"]), item["method"])
    raw_by_key = {keys(item): item for item in raw}
    if len(raw) != len(records) or set(raw_by_key) != set(records):
        raise ValueError("Original raw results do not match the complete declared checkpoint grid")
    for key, result in records.items():
        for field, value in result["row"].items():
            expected = "" if value is None else str(value)
            if raw_by_key[key].get(field) != expected:
                raise ValueError(f"Original raw/checkpoint evidence disagrees for {key}/{field}")
    summary = _read_csv(output / "summary/communication_overhead_summary.csv")
    time_summary = _read_csv(output / "summary/time_overhead_summary.csv")
    summary_key = lambda item: (int(item["N"]), int(item["d"]), item["method"])
    summaries_by_key = {summary_key(item): item for item in summary}
    time_by_key = {summary_key(item): item for item in time_summary}
    expected_summary = {(n, d, method) for n in N_VALUES for d in D_PANEL_VALUES for method in METHODS}
    if (len(summary) != len(expected_summary) or set(summaries_by_key) != expected_summary
            or len(time_summary) != len(expected_summary) or {summary_key(item) for item in time_summary} != expected_summary
            or any(item.get("run_fingerprint") != manifest["run_fingerprint"] or item.get("implementation_version") != OVERHEAD_IMPLEMENTATION_VERSION
                   or int(item["n_valid"]) != 1 or int(item["n_expected"]) != 1 for item in summary + time_summary)):
        raise ValueError("Original time/communication summaries lack an intact complete paired grid")
    for n, d, seed in configurations():
        for method in METHODS:
            original = records[(n, d, seed, method)]["row"]
            if float(time_by_key[(n, d, method)]["value"]) != original["total_time_ms"]:
                raise ValueError("Original time summary differs from its frozen raw/checkpoint observation")
    source_hash, source_files = communication_source_fingerprint()
    policy = communication_policy()
    communication_fingerprint = _sha256(_json_bytes(dict(schema=COMMUNICATION_SCHEMA,
        base_run_fingerprint=manifest["run_fingerprint"], frozen_files_sha256=protected,
        frozen_checkpoint_inventory=frozen_checkpoint_inventory,
        communication_source_sha256=source_hash, codec_policy=policy)))
    runs, encoded_summaries, packets = [], [], []
    calibrations = {}
    for seed in SEEDS:
        for d in D_PANEL_VALUES:
            calibration = KernelCalibration.from_dict(manifest["kernel_calibrations"][calibration_key(seed, d)])
            full = load_scaling_group(dataset_root=dataset_root, n=max(N_VALUES),
                corruption_rate_percent=config.corruption_rate_percent, dim=d, H=config.H, seed=seed)
            X = full["X"]
            full_hash = _sha256(b"overhead-calibration-raw-float32-v1\0" + np.asarray(X.shape, dtype="<i8").tobytes()
                                + np.ascontiguousarray(X, dtype="<f4").tobytes())
            if full_hash != calibration.input_sha256:
                raise ValueError("Full raw population differs from the saved shared kernel calibration input")
            calibrations[(seed, d)] = calibration
    for n, d, seed in configurations():
        point_config = replace(config, dim=d)
        group = load_scaling_group(dataset_root=dataset_root, n=n,
            corruption_rate_percent=config.corruption_rate_percent, dim=d, H=config.H, seed=seed)
        result = records[(n, d, seed, RACA_HIM)]
        case = _reconstruct_case(group, result, point_config, calibrations[(seed, d)])
        encoded, point_packets = replay_raca_packets(case, result["row"])
        packets.extend(point_packets)
        for method in METHODS:
            original = records[(n, d, seed, method)]["row"]
            values = encoded if method == RACA_HIM else dict(
                encoded_total_bytes=original["total_bytes"], gt_bytes=original["gt_bytes"],
                verification_bytes=original["verification_bytes"], control_bytes=original["control_bytes"],
                message_count=original["message_count"], refinement_count=0, raw_fallback_count=0,
                gt_refinement_count=0, gt_raw_fallback_count=0,
                verdict_count=original["corrupted_verdict_count"], gt_membership_preserved=True,
                verification_verdicts_preserved=True, history_update_preserved=True,
                gt_certified_pair_count=0, gt_unresolved_pair_count=0, gt_majority_short_circuit=False)
            record = dict(N=n, d=d, seed=seed, method=method, evaluation_mode=MAIN_EVALUATION_MODE,
                configuration_id=original["configuration_id"], B=original["B"],
                fixed_workload_sha256=original["fixed_workload_sha256"],
                base_implementation_version=OVERHEAD_IMPLEMENTATION_VERSION,
                base_run_fingerprint=manifest["run_fingerprint"], communication_fingerprint=communication_fingerprint,
                codec_schema=codec.WIRE_CODEC_SCHEMA if method == RACA_HIM else "saved-v4-compact-baseline",
                base_total_bytes=original["total_bytes"], saved_bytes=original["total_bytes"] - values["encoded_total_bytes"],
                status="success", **values)
            runs.append(record)
            old_summary = summaries_by_key[(n, d, method)]
            if float(old_summary["value"]) != original["total_bytes"]:
                raise ValueError("Original communication summary disagrees with its paired checkpoint byte observation")
            derived = dict(old_summary, metric="encoded_total_bytes", value=values["encoded_total_bytes"],
                mean=values["encoded_total_bytes"], base_total_bytes=original["total_bytes"],
                saved_bytes=record["saved_bytes"], communication_fingerprint=communication_fingerprint,
                base_run_fingerprint=manifest["run_fingerprint"], communication_schema=COMMUNICATION_SCHEMA,
                statistics_scope="one paired original workload; serialized certified RACA direction packets; saved compact baseline ledgers; original v4 time frozen")
            encoded_summaries.append(derived)
    assert_frozen_files(output, protected)
    _assert_checkpoint_inventory(output, frozen_checkpoint_inventory)
    if dataset_snapshot(dataset_root) != before:
        raise ValueError("Dataset changed during communication replay")
    metadata = dict(schema=COMMUNICATION_SCHEMA, status="prepared",
        base_implementation_version=OVERHEAD_IMPLEMENTATION_VERSION,
        base_run_fingerprint=manifest["run_fingerprint"], communication_fingerprint=communication_fingerprint,
        communication_source_sha256=source_hash, communication_source_files_sha256=source_files,
        codec_policy=policy, frozen_files_sha256=protected,
        frozen_checkpoint_inventory=frozen_checkpoint_inventory,
        summary_path=SUMMARY_PATH, encoded_runs_path=RUNS_PATH, packet_ledger_path=PACKET_LEDGER_PATH,
        N_values=list(N_VALUES), d_panel_values=list(D_PANEL_VALUES), seeds=list(SEEDS), methods=list(METHODS),
        saved_configurations=len(configurations()), method_records=len(runs), packet_count=len(packets),
        gt_membership_preserved=True, verification_verdicts_preserved=True,
        base_time_unchanged=True, encoded_end_to_end_time_measured=False,
        created_utc=datetime.now(timezone.utc).isoformat(),
        communication_totals={method: dict(
            base_total_bytes=sum(row["base_total_bytes"] for row in runs if row["method"] == method),
            encoded_total_bytes=sum(row["encoded_total_bytes"] for row in runs if row["method"] == method),
            saved_bytes=sum(row["saved_bytes"] for row in runs if row["method"] == method),
            refinement_count=sum(row["refinement_count"] for row in runs if row["method"] == method),
            raw_fallback_count=sum(row["raw_fallback_count"] for row in runs if row["method"] == method))
            for method in METHODS})
    return PreparedCommunicationReplay(metadata, runs, encoded_summaries, packets)


def write_communication_replay(output_dir, prepared):
    """Commit only three independent CSVs and their manifest; retain base bytes."""
    output = Path(output_dir).resolve()
    metadata = deepcopy(prepared.manifest)
    assert_frozen_files(output, metadata["frozen_files_sha256"])
    _assert_checkpoint_inventory(output, metadata["frozen_checkpoint_inventory"])
    source_hash, source_files = communication_source_fingerprint()
    if source_hash != metadata["communication_source_sha256"] or source_files != metadata["communication_source_files_sha256"]:
        raise ValueError("Encoding implementation changed after replay calculation")
    metadata["status"] = "building"
    atomic_write_json(output / MANIFEST_PATH, metadata)
    try:
        for relative, rows in ((RUNS_PATH, prepared.runs), (SUMMARY_PATH, prepared.summaries),
                               (PACKET_LEDGER_PATH, prepared.packets)):
            atomic_write_csv(output / relative, rows)
        metadata["files_sha256"] = {relative: _file_hash(output / relative)
            for relative in (RUNS_PATH, SUMMARY_PATH, PACKET_LEDGER_PATH)}
        assert_frozen_files(output, metadata["frozen_files_sha256"])
        _assert_checkpoint_inventory(output, metadata["frozen_checkpoint_inventory"])
        metadata["status"] = "computed"
        atomic_write_json(output / MANIFEST_PATH, metadata)
        return metadata
    except Exception as exc:
        metadata.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        try:
            atomic_write_json(output / MANIFEST_PATH, metadata)
        except OSError as cleanup_error:
            exc.add_note(f"Could not record communication sidecar failure; its building state is retained: {cleanup_error}")
        raise


def replay_communication(output_dir, *, dataset_root=DATASET_ROOT):
    """Prepare and save an encoded communication sidecar, without touching O1."""
    return write_communication_replay(output_dir,
        prepare_communication_replay(output_dir, dataset_root=dataset_root))


def validate_encoded_overlay(output_dir, *, allow_computed=False, verify_figures=True):
    """Validate provenance and every artifact before an encoded O2 is plotted."""
    output = Path(output_dir).resolve()
    metadata = json.loads((output / MANIFEST_PATH).read_text(encoding="utf-8"))
    allowed = {"complete", "computed"} if allow_computed else {"complete"}
    if (metadata.get("schema") != COMMUNICATION_SCHEMA or metadata.get("status") not in allowed
            or metadata.get("summary_path") != SUMMARY_PATH or metadata.get("encoded_runs_path") != RUNS_PATH
            or metadata.get("packet_ledger_path") != PACKET_LEDGER_PATH
            or metadata.get("N_values") != list(N_VALUES) or metadata.get("d_panel_values") != list(D_PANEL_VALUES)
            or metadata.get("seeds") != list(SEEDS) or metadata.get("methods") != list(METHODS)
            or metadata.get("method_records") != len(configurations()) * len(METHODS)
            or metadata.get("saved_configurations") != len(configurations())
            or metadata.get("codec_policy") != communication_policy()):
        raise ValueError("Encoded communication overlay is incomplete or uses a different protocol/grid")
    source_hash, source_files = communication_source_fingerprint()
    if source_hash != metadata.get("communication_source_sha256") or source_files != metadata.get("communication_source_files_sha256"):
        raise ValueError("Encoded communication overlay uses different codec/source bytes")
    expected_fp = _sha256(_json_bytes(dict(schema=COMMUNICATION_SCHEMA,
        base_run_fingerprint=metadata["base_run_fingerprint"], frozen_files_sha256=metadata["frozen_files_sha256"],
        frozen_checkpoint_inventory=metadata["frozen_checkpoint_inventory"],
        communication_source_sha256=source_hash, codec_policy=metadata["codec_policy"])))
    if metadata.get("communication_fingerprint") != expected_fp:
        raise ValueError("Encoded communication fingerprint does not match its policy/provenance")
    assert_frozen_files(output, metadata["frozen_files_sha256"])
    _assert_checkpoint_inventory(output, metadata["frozen_checkpoint_inventory"])
    base = json.loads((output / "logs/run_manifest.json").read_text(encoding="utf-8"))
    _validate_base_manifest(base)
    if base["run_fingerprint"] != metadata["base_run_fingerprint"]:
        raise ValueError("Encoded overlay is paired with a different v4 timing generation")
    _verify_hashes(PROJECT_ROOT, base["computation_source_files_sha256"], "Saved v4 timing implementation")
    if set(metadata.get("files_sha256", {})) != {RUNS_PATH, SUMMARY_PATH, PACKET_LEDGER_PATH}:
        raise ValueError("Encoded overlay is missing an exact CSV inventory")
    _verify_hashes(output, metadata["files_sha256"], "Encoded communication output")
    summary, runs = _read_csv(output / SUMMARY_PATH), _read_csv(output / RUNS_PATH)
    expected = {(n, d, method) for n in N_VALUES for d in D_PANEL_VALUES for method in METHODS}
    key = lambda item: (int(item["N"]), int(item["d"]), item["method"])
    if (len(summary) != len(expected) or {key(item) for item in summary} != expected
            or len(runs) != len(expected) or {key(item) for item in runs} != expected):
        raise ValueError("Encoded CSVs do not contain one complete original paired observation per point/method")
    run_values = {key(item): item for item in runs}
    originals = {key(item): item for item in _read_csv(output / "raw/overhead_runs.csv")}
    if set(originals) != expected:
        raise ValueError("Frozen raw observations lack the declared original complete paired grid")
    for item in summary:
        row = run_values[key(item)]
        original = originals[key(item)]
        if (item.get("communication_fingerprint") != expected_fp or item.get("base_run_fingerprint") != base["run_fingerprint"]
                or item.get("run_fingerprint") != base["run_fingerprint"]
                or item.get("metric") != "encoded_total_bytes" or int(item["n_valid"]) != 1
                or int(item["n_expected"]) != 1 or float(item["value"]) != int(row["encoded_total_bytes"])
                or row.get("communication_fingerprint") != expected_fp
                or row.get("status") != "success" or int(row["seed"]) not in SEEDS
                or row.get("base_run_fingerprint") != base["run_fingerprint"]
                or int(row["B"]) != int(original["B"])
                or row.get("fixed_workload_sha256") != original["fixed_workload_sha256"]
                or int(row["base_total_bytes"]) != int(original["total_bytes"])
                or any(int(row[field]) < 0 for field in ("encoded_total_bytes", "gt_bytes", "verification_bytes", "control_bytes"))
                or int(row["gt_bytes"]) + int(row["verification_bytes"]) + int(row["control_bytes"]) != int(row["encoded_total_bytes"])
                or int(row["base_total_bytes"]) - int(row["encoded_total_bytes"]) != int(row["saved_bytes"])):
            raise ValueError("Encoded paired summary disagrees with its exact byte ledger/provenance")
        if row["method"] != RACA_HIM and any(int(row[field]) != int(original[source_field])
                for field, source_field in (("encoded_total_bytes", "total_bytes"), ("gt_bytes", "gt_bytes"),
                    ("verification_bytes", "verification_bytes"), ("control_bytes", "control_bytes"),
                    ("message_count", "message_count"))):
            raise ValueError("Encoded overlay changed a baseline's saved compact protocol ledger")
    packets = _read_csv(output / PACKET_LEDGER_PATH)
    if len(packets) != metadata["packet_count"]:
        raise ValueError("Encoded packet CSV has a different actual packet count")
    packet_totals = {}
    packet_ids = {}
    for item in packets:
        context = (int(item["N"]), int(item["d"]), RACA_HIM)
        expected_id = packet_ids.get(context, 0)
        payload_packet = int(item["opcode"]) in {_INITIAL, _REFINE, _RAW}
        if (int(item["packet_id"]) != expected_id or item["method"] != RACA_HIM
                or context not in expected or int(item["seed"]) not in SEEDS
                or item["phase"] not in {"gt", "verification"}
                or int(item["sender"]) == int(item["receiver"])
                or int(item["payload_bytes"]) + int(item["control_bytes"]) != int(item["total_bytes"])
                or int(item["payload_bytes"]) != (int(item["body_bytes"]) if payload_packet else 0)
                or int(item["control_bytes"]) != (FRAME_BYTES if payload_packet else int(item["total_bytes"]))
                or int(item["body_bytes"]) + FRAME_BYTES != int(item["total_bytes"])
                or len(item["packet_sha256"]) != 64 or item["roundtrip_verified"] != "True"):
            raise ValueError("Encoded packet CSV has invalid ordering/length/roundtrip evidence")
        packet_ids[context] = expected_id + 1
        aggregate = packet_totals.setdefault(context, Counter())
        aggregate["encoded_total_bytes"] += int(item["total_bytes"])
        aggregate["gt_bytes" if item["phase"] == "gt" else "verification_bytes"] += int(item["payload_bytes"])
        aggregate["control_bytes"] += int(item["control_bytes"])
        aggregate["message_count"] += 1
    for n in N_VALUES:
        for d in D_PANEL_VALUES:
            if any(packet_totals.get((n, d, RACA_HIM), {}).get(field) != int(run_values[(n, d, RACA_HIM)][field])
                    for field in ("encoded_total_bytes", "gt_bytes", "verification_bytes", "control_bytes", "message_count")):
                raise ValueError("Actual packet lengths do not sum to the encoded RACA observation")
    if metadata["status"] == "complete" and verify_figures:
        expected_figures = {"figures/Fig_O2_communication_overhead.pdf", "figures/Fig_O2_communication_overhead.png"}
        if set(metadata.get("figure_files_sha256", {})) != expected_figures:
            raise ValueError("Completed encoded overlay has no complete O2 figure inventory")
        _verify_hashes(output, metadata["figure_files_sha256"], "Encoded O2 figure")
    return metadata


def finalize_communication(output_dir, plot_metadata):
    """Mark the sidecar complete only after O2 exists and all time files match."""
    output = Path(output_dir).resolve()
    # A previously complete overlay may have just had O2 redrawn.  Revalidate
    # all numerical/source/frozen inputs, then bind the newly written figures
    # below; its old figure hash cannot identify this deliberate redraw.
    metadata = validate_encoded_overlay(output, allow_computed=True, verify_figures=False)
    if (plot_metadata.get("communication_fingerprint") != metadata["communication_fingerprint"]
            or plot_metadata.get("base_run_fingerprint", plot_metadata.get("run_fingerprint")) != metadata["base_run_fingerprint"]
            or plot_metadata.get("figures_generated") != ["Fig_O2_communication_overhead"]):
        raise ValueError("Communication plot metadata does not identify this encoded O2 generation")
    figures = ("figures/Fig_O2_communication_overhead.pdf", "figures/Fig_O2_communication_overhead.png")
    if any(not (output / relative).is_file() or (output / relative).stat().st_size == 0 for relative in figures):
        raise ValueError("Encoded O2 PDF/PNG are missing or empty")
    metadata.update(status="complete", finished_utc=datetime.now(timezone.utc).isoformat(),
        figures_generated=["Fig_O2_communication_overhead"],
        figure_files_sha256={relative: _file_hash(output / relative) for relative in figures})
    assert_frozen_files(output, metadata["frozen_files_sha256"])
    _assert_checkpoint_inventory(output, metadata["frozen_checkpoint_inventory"])
    atomic_write_json(output / MANIFEST_PATH, metadata)
    return metadata
