"""CTCV raw-content request/response hashes and actual modulo proof vectors.

Section III Algorithms 1--3 / Equations (4a)--(9) define block reordering,
r_i=SHA256(reordered data) mod ell, the ell-bit correct proof combination,
k+1 report collectors and four Flag cases. Those operations are implemented
on the stored descriptor bytes without altering their values. Full-hash
comparisons are auxiliary diagnostics and cannot override SC's vector verdict.
The unavailable vendor original is estimated by exact reference-pool hash
frequency. The raw payload byte cap and deterministic requests are explicit
adaptations; blockchain execution and original security guarantees are not
claimed. Unequal legitimate descriptors can be rejected by the replica rule.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
import struct

import numpy as np

from experiments.common.config import DetectionConfig
from experiments.common.dataset_adapter import AlgorithmInput

if __package__:
    from .adaptation_utils import canonical_descriptor_bytes, exact_hash_anchor, readonly
    from .proposed_detection import DetectionResult
else:
    from adaptation_utils import canonical_descriptor_bytes, exact_hash_anchor, readonly
    from proposed_detection import DetectionResult


@dataclass(frozen=True)
class DescriptorProof:
    """An actual response including ordered bytes, block hashes and Eq. (4b) tag."""

    target_id: int
    block_order: tuple[int, ...]
    ordered_blocks: tuple[bytes, ...]
    block_hashes: tuple[str, ...]
    response_sha256: str
    proof_tag: int
    commitment: str


def _commitment(
    target_id: int,
    block_order: tuple[int, ...],
    block_hashes: tuple[str, ...],
    response_sha256: str,
    proof_tag: int,
) -> str:
    """Commit actual message metadata; this does not certify content correctness."""
    digest = hashlib.sha256(b"CTCV-raw-response-v3\0")
    digest.update(struct.pack("<qI", int(target_id), len(block_order)))
    digest.update(np.asarray(block_order, dtype="<i8").tobytes())
    for block_hash in block_hashes:
        digest.update(bytes.fromhex(block_hash))
    digest.update(bytes.fromhex(response_sha256))
    tag_bytes = int(proof_tag).to_bytes(max(1, (int(proof_tag).bit_length() + 7) // 8), "big")
    digest.update(struct.pack("<I", len(tag_bytes)))
    digest.update(tag_bytes)
    return digest.hexdigest()


def _combination_digest(vector: np.ndarray) -> str:
    digest = hashlib.sha256(b"CTCV-modulo-proof-vector-v3\0")
    digest.update(struct.pack("<Q", len(vector)))
    digest.update(np.packbits(np.asarray(vector, dtype=np.uint8), bitorder="big").tobytes())
    return digest.hexdigest()


def _bitvector(tags, vector_length: int) -> np.ndarray:
    if isinstance(vector_length, bool) or not isinstance(vector_length, int) or vector_length < 1:
        raise ValueError("CTCV ell must be a positive integer.")
    positions = np.asarray(tags, dtype=np.int64)
    if positions.ndim != 1 or np.any((positions < 0) | (positions >= vector_length)):
        raise ValueError("CTCV proof tags must be actual integer positions in [0,ell).")
    vector = np.zeros(vector_length, dtype=np.uint8)
    vector[positions] = 1
    return vector


def assemble_proof_combination(
    messages: tuple[DescriptorProof, ...],
    target_ids: np.ndarray,
    proof_vector_length: int,
) -> np.ndarray:
    """Verify the received raw-byte proofs and construct the actual ell-bit R.

    Each collector checks actual block hashes, the hash of their concatenation,
    the modulo tag and the message commitment. It then sets every received tag
    position to one, retaining original combination collisions rather than
    replacing the vector with synthetic Normal/Corrupted slots.
    """
    if len(messages) != len(target_ids):
        raise ValueError("CTCV collector did not receive every target response.")
    if (isinstance(proof_vector_length, bool) or not isinstance(proof_vector_length, int)
            or proof_vector_length < 1):
        raise ValueError("CTCV proof-vector length must be a positive integer.")
    tags = []
    for message, expected_id in zip(messages, target_ids):
        count = len(message.block_order)
        if message.target_id != int(expected_id):
            raise ValueError("CTCV response identities changed the common target order.")
        if count == 0 or sorted(message.block_order) != list(range(count)):
            raise ValueError("CTCV must include every requested block once.")
        if len(message.ordered_blocks) != count or len(message.block_hashes) != count:
            raise ValueError("CTCV response blocks and hashes are incomplete.")
        actual_block_hashes = tuple(hashlib.sha256(block).hexdigest() for block in message.ordered_blocks)
        actual_response_hash = hashlib.sha256(b"".join(message.ordered_blocks)).hexdigest()
        if actual_block_hashes != message.block_hashes or actual_response_hash != message.response_sha256:
            raise ValueError("CTCV actual block/full-response SHA256 validation failed.")
        if int(actual_response_hash, 16) % proof_vector_length != message.proof_tag:
            raise ValueError("CTCV response did not contain its actual Eq. (4b) modulo tag.")
        if _commitment(message.target_id, message.block_order, message.block_hashes,
                       message.response_sha256, message.proof_tag) != message.commitment:
            raise ValueError("CTCV response commitment validation failed.")
        tags.append(message.proof_tag)
    return _bitvector(tags, proof_vector_length)


def co_verify_reports(
    reports: tuple[np.ndarray, ...],
    correct: np.ndarray,
    reporter_ids: np.ndarray,
    target_ids: np.ndarray,
    collusion_bound: int,
    *,
    response_tags: np.ndarray | None = None,
    fallback_reports: tuple[np.ndarray, ...] | None = None,
    fallback_reporter_ids: np.ndarray | None = None,
) -> dict:
    """Apply the four original cases to real complete proof bitvectors.

    Flags 0/1 identify a correct combination and accept all target content;
    auxiliary complete hash mismatches cannot reverse that verdict. Flags 2/3
    localize targets by testing their actual modulo-tag position in the correct
    CR bitvector, provided the response tags reconstruct the selected report.
    SC receives only compressed CR, not an individual target-to-sigma mapping;
    matching any correct occupied bit therefore retains Bloom-style collisions.
    A >k-voter fallback requires actual uniquely identified reports; no vote or
    Byzantine state is fabricated to obtain a desired result.
    """
    if collusion_bound < 0 or len(reports) != len(reporter_ids) or len(reports) < collusion_bound + 1:
        raise ValueError("CTCV requires k+1 report collectors.")
    if len(np.unique(reporter_ids)) != len(reporter_ids):
        raise ValueError("CTCV collector identities must be unique.")
    correct = np.asarray(correct)
    if correct.ndim != 1 or len(correct) < 1 or not np.isin(correct, (0, 1)).all():
        raise ValueError("CTCV correct combination must be a nonempty binary vector.")
    correct = correct.astype(np.uint8, copy=False)

    def checked(vector):
        values = np.asarray(vector)
        if values.shape != correct.shape or not np.isin(values, (0, 1)).all():
            raise ValueError("CTCV collector combinations must use the same ell-bit layout.")
        return values.astype(np.uint8, copy=False)

    reports = tuple(checked(report) for report in reports)
    matching = np.array([np.array_equal(report, correct) for report in reports])
    consistent = all(np.array_equal(report, reports[0]) for report in reports[1:])
    selected = None
    byzantine_ids = np.empty(0, dtype=np.int64)
    if consistent and matching.all():
        flag, selected = 0, correct
    elif not consistent and matching.any():
        flag, selected = 1, correct
        byzantine_ids = np.asarray(reporter_ids)[~matching]
    elif consistent:
        flag, selected = 2, reports[0]
    else:
        flag = 3
        if fallback_reports is not None:
            if fallback_reporter_ids is None or len(fallback_reports) != len(fallback_reporter_ids):
                raise ValueError("CTCV fallback requires actual reports and voter identities.")
            if len(np.unique(fallback_reporter_ids)) != len(fallback_reporter_ids):
                raise ValueError("CTCV fallback cannot count duplicate voters.")
            votes = tuple(checked(report) for report in fallback_reports)
            groups: dict[bytes, list[int]] = {}
            for position, vote in enumerate(votes):
                groups.setdefault(vote.tobytes(), []).append(position)
            qualifying = [positions for positions in groups.values() if len(positions) > collusion_bound]
            if len(qualifying) == 1:
                selected = votes[qualifying[0][0]]
                byzantine_ids = np.asarray(reporter_ids)[[
                    not np.array_equal(report, selected) for report in reports]]

    reason = ""
    predictions = None
    if selected is None:
        reason = "Conflicting CTCV combinations lack a unique actual >k-voter fallback quorum."
    elif np.array_equal(selected, correct):
        predictions = np.zeros(len(target_ids), dtype=np.int8)
    elif response_tags is None:
        reason = "CTCV has no per-target modulo-tag evidence for corruption localization."
    else:
        actual = np.asarray(response_tags, dtype=np.int64)
        if actual.shape != (len(target_ids),):
            raise ValueError("CTCV localization tags must align with all targets.")
        if not np.array_equal(_bitvector(actual, len(correct)), selected):
            reason = "The elected combination cannot be associated with the supplied actual response tags."
        else:
            predictions = (correct[actual] == 0).astype(np.int8)
    if predictions is None:
        return {"flag": flag, "resolved": False, "predictions": None,
                "corrupted_ids": [], "byzantine_collector_ids": byzantine_ids.tolist(), "VR": [], "reason": reason}
    corrupted_ids = np.asarray(target_ids)[predictions == 1].astype(np.int64)
    return {"flag": flag, "resolved": True, "predictions": readonly(predictions, np.int8),
            "corrupted_ids": corrupted_ids.tolist(), "byzantine_collector_ids": byzantine_ids.tolist(),
            "VR": np.concatenate((corrupted_ids, byzantine_ids)).astype(np.int64).tolist(), "reason": ""}


def detect_ctcv(data: AlgorithmInput, config: DetectionConfig) -> DetectionResult:
    """Execute raw-byte Request, Response, Inquiry and four-Flag co-verification."""
    targets = np.asarray(data.evaluation_indices, dtype=np.int64)
    if not len(targets):
        raise ValueError("CTCV requires a nonempty evaluation target pool.")
    target_ids = readonly(data.model_ids[targets])
    references = np.asarray(data.reference_indices, dtype=np.int64)
    k, time_limit = int(config.ctcv_collusion_bound), float(config.ctcv_time_limit)
    if k < 0 or not math.isfinite(time_limit) or time_limit < 0:
        raise ValueError("CTCV needs a nonnegative k and finite nonnegative time limit.")
    response_age = 1.0
    diagnostics = {"adaptation": "CTCV actual raw-content modulo-hash proof combinations",
                   "content_comparison": "exact_hash", "collusion_bound": k,
                   "collector_required_count": k + 1, "response_age": response_age,
                   "time_limit": time_limit, "latency_state": "uniform prior, not measured latency"}
    if len(references) < k + 1 or response_age > time_limit:
        return DetectionResult(method="CTCV-Adapted", status="failure", predictions=None,
                               target_ids=target_ids, verifier_ids=None, gt_success=None, ground_truth=None,
                               reason="CTCV lacks k+1 time-eligible collectors.", diagnostics=diagnostics)

    anchor, pool_hashes = exact_hash_anchor(data)
    anchor = int(anchor)
    anchor_bytes = canonical_descriptor_bytes(data.X[anchor])
    anchor_sha256 = hashlib.sha256(anchor_bytes).hexdigest()
    configured_length = config.ctcv_proof_vector_length
    if isinstance(configured_length, bool) or not isinstance(configured_length, int) or configured_length < 0:
        raise ValueError("ctcv_proof_vector_length must be an integer >=0 (0 selects 1000*n).")
    ell = configured_length or 1000 * max(1, len(targets))
    request_seed = int.from_bytes(hashlib.sha256(
        b"CTCV-raw-request-v3\0" + data.model_ids.astype("<i8").tobytes()).digest()[:8], "little")
    rng = np.random.default_rng(request_seed)
    block_override = getattr(config, "ctcv_block_count", 0)
    if isinstance(block_override, bool) or not isinstance(block_override, int) or block_override < 0:
        raise ValueError("ctcv_block_count must be an integer >=0 (0 selects the source count with a byte cap).")
    source_nblock_count = len(targets)
    physical_byte_cap = len(anchor_bytes)
    requested_blocks = block_override or source_nblock_count
    block_count = min(requested_blocks, physical_byte_cap)
    cuts = np.sort(rng.choice(np.arange(1, len(anchor_bytes)), block_count - 1, replace=False))
    boundaries = np.concatenate(([0], cuts, [len(anchor_bytes)])).astype(np.int64)
    shuffled = rng.permutation(block_count)
    proof_order = np.argsort(target_ids, kind="stable")
    proof_target_ids = target_ids[proof_order]
    messages, correct_tags, correct_hashes = [], [], []
    for row_index, target_id in zip(targets[proof_order], proof_target_ids):
        payload = canonical_descriptor_bytes(data.X[row_index])
        if len(payload) != len(anchor_bytes):
            raise ValueError("CTCV target and original anchor must have the same byte size.")
        order = tuple(int(value) for value in np.roll(shuffled, -(int(target_id) % block_count)))
        ordered_blocks = tuple(payload[boundaries[block]:boundaries[block + 1]] for block in order)
        block_hashes = tuple(hashlib.sha256(block).hexdigest() for block in ordered_blocks)
        response_hash = hashlib.sha256(b"".join(ordered_blocks)).hexdigest()
        proof_tag = int(response_hash, 16) % ell
        messages.append(DescriptorProof(int(target_id), order, ordered_blocks, block_hashes,
                                       response_hash, proof_tag, _commitment(int(target_id), order, block_hashes, response_hash, proof_tag)))
        # AV.Inquiry uses the SAME request transformation on the estimated
        # original. CR bits are computed from actual hashes, not assumed truth.
        correct_blocks = tuple(anchor_bytes[boundaries[block]:boundaries[block + 1]] for block in order)
        correct_hash = hashlib.sha256(b"".join(correct_blocks)).hexdigest()
        correct_hashes.append(correct_hash)
        correct_tags.append(int(correct_hash, 16) % ell)
    messages = tuple(messages)
    actual_tags = np.asarray([message.proof_tag for message in messages], dtype=np.int64)
    correct_tags = np.asarray(correct_tags, dtype=np.int64)
    correct = _bitvector(correct_tags, ell)
    collector_order = references[np.argsort(data.model_ids[references], kind="stable")]
    collectors = rng.permutation(collector_order)[:k + 1]
    collector_ids = readonly(data.model_ids[collectors])
    reports = tuple(assemble_proof_combination(messages, proof_target_ids, ell) for _ in collector_ids)
    verification = co_verify_reports(reports, correct, collector_ids, proof_target_ids, k,
                                     response_tags=actual_tags)
    full_hash_equal = np.array([message.response_sha256 == expected for message, expected in zip(messages, correct_hashes)])
    tag_equal = actual_tags == correct_tags
    membership_equal = correct[actual_tags] == 1
    inverse = np.argsort(proof_order)
    diagnostics.update({
        "anchor_index": anchor, "anchor_id": int(data.model_ids[anchor]),
        "reference_indices": [anchor], "reference_ids": [int(data.model_ids[anchor])],
        "reference_sha256": anchor_sha256,
        "reference_hash_support": sum(value == anchor_sha256 for value in pool_hashes),
        "verification_mode": "exact_hash_anchor_with_modulo_bitvector_collectors",
        "proof_tag_formula": "int(SHA256(concatenation of actual cyclically reordered raw-byte blocks),16) mod ell",
        "proof_vector_length": ell, "configured_proof_vector_length": configured_length,
        "ell_policy": "1000 * common evaluation target count when configured value is zero",
        "request_seed": request_seed, "requested_block_count": requested_blocks,
        "configured_ctcv_block_count": block_override,
        "source_nblock_count": source_nblock_count, "physical_byte_cap": physical_byte_cap,
        "block_count": block_count, "block_boundaries_bytes": boundaries.tolist(),
        "block_count_policy": "default min(source n=common target count, raw descriptor byte count); positive config is an explicit override",
        "block_count_adaptation": "source requires n blocks/node-to-block 1:1; compact byte payload caps nonempty blocks, so repeated cyclic starts replace that 1:1 mapping when capped",
        "original_node_block_mapping_preserved": block_count == source_nblock_count,
        "shuffled_block_order": shuffled.tolist(), "cyclic_start_rule": "temporary target ID modulo block count",
        "proof_message_count": len(messages),
        "proof_commitment_sha256": hashlib.sha256(b"".join(bytes.fromhex(message.commitment) for message in messages)).hexdigest(),
        "combination_layout": "actual ell-bit vector with one at every observed SHA256-mod-ell position",
        "combination_digest_encoding": "SHA256(domain || uint64 ell || packbits big-endian)",
        "correct_combination_sha256": _combination_digest(correct),
        "correct_combination_set_bits": np.flatnonzero(correct).tolist(),
        "collector_ids": collector_ids.tolist(), "collector_count": len(collector_ids),
        "collector_combination_sha256": [_combination_digest(report) for report in reports],
        "collector_combination_set_bits": np.flatnonzero(reports[0]).tolist(),
        "collector_reports_consistent": all(np.array_equal(report, reports[0]) for report in reports),
        "actual_proof_tags": actual_tags[inverse].tolist(), "correct_proof_tags": correct_tags[inverse].tolist(),
        "tag_order": "original shared evaluation target order",
        "mod_tag_equal_count": int(tag_equal.sum()), "full_response_hash_equal_count": int(full_hash_equal.sum()),
        "individual_correct_tags_auxiliary_only": True,
        "membership_match_count": int(membership_equal.sum()),
        "membership_equal_target_ids": proof_target_ids[membership_equal].tolist(),
        "cross_tag_membership_collision_count": int(np.count_nonzero(membership_equal & ~tag_equal)),
        "modulus_collision_count": int(np.count_nonzero(tag_equal & ~full_hash_equal)),
        "response_tag_duplicate_count": len(actual_tags) - len(np.unique(actual_tags)),
        "full_hash_unequal_target_ids": proof_target_ids[~full_hash_equal].tolist(),
        "full_hash_auxiliary_only": True, "flag": verification["flag"], "VR": verification["VR"],
        "byzantine_collector_ids": verification["byzantine_collector_ids"],
        "valid_target_coverage": len(targets) if verification["resolved"] else 0,
        "content_rule": "SC vector Flag first; Flag2/3 localize corrupted iff CR[actual_modulo_tag] == 0; no individual sigma comparison",
        "anchor_limitation": "hash-frequency anchor is not an authenticated vendor original; unequal legitimate model bytes are rejected",
        "guarantees_not_inherited": "original trusted vendor content; blockchain/BFT execution; randomized-request freshness",
    })
    if verification["resolved"]:
        diagnostics["combination_or_modulus_masked_hash_mismatch_count"] = int(np.count_nonzero(
            (verification["predictions"] == 0) & ~full_hash_equal))
    if messages:
        diagnostics["example_ordered_block_hashes"] = list(messages[0].block_hashes)
        diagnostics["example_actual_response_sha256"] = messages[0].response_sha256
        diagnostics["example_correct_response_sha256"] = correct_hashes[0]
    if not verification["resolved"]:
        return DetectionResult(method="CTCV-Adapted", status="failure", predictions=None,
                               target_ids=target_ids, verifier_ids=None, gt_success=None, ground_truth=None,
                               reason=verification["reason"], diagnostics=diagnostics)
    return DetectionResult(method="CTCV-Adapted", status="success",
                           predictions=readonly(verification["predictions"][inverse], np.int8), target_ids=target_ids,
                           verifier_ids=readonly(np.full(len(targets), data.model_ids[anchor], dtype=np.int64)),
                           gt_success=None, ground_truth=None, diagnostics=diagnostics)
