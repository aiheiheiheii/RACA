"""DynaEDI+ key-step adaptation to a static model-descriptor workload.

Source: Yang et al. (2025), Sections III-A--III-C and IV, Equations (5)--(7).
Actual Szudzik pairing trees, version-matched common subtrees, nonce-bound
digest commitments and inverse-tree expansion are implemented. SHA256
commitments replace the source's BLS proof only as a local digest-consistency
check: this simulation provides neither signatures nor Byzantine security.

All models are one descriptor family with five contiguous raw float32 blocks
and block version zero. SHA256 hashes the unmodified canonical block bytes
with the current nonce. Both inspection and localization compare the actual
digest integers for exact equality. A root differing from MORE THAN ceil(2R/3)
committed reference roots only identifies a candidate for inverse-tree
expansion. Final model classification follows the source block decision:
Corrupted if and only if at least one decoded leaf/block has MORE THAN
ceil(2R/3) genuine-witness digest mismatches. A root candidate with no such
block receives Normal in this binary static adaptation.

This remains an adaptation: the workload has descriptors rather than genuine
replicated content, public keys or dynamic updates. Legitimate descriptor
heterogeneity therefore can fail the exact digest check. No content transform
or tolerant comparison is introduced to conceal that consequence.
"""
from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
import hashlib
import math

import numpy as np

from experiments.common.config import DetectionConfig
from experiments.common.dataset_adapter import AlgorithmInput

if __package__:
    from .proposed_detection import DetectionResult, _readonly
else:
    from proposed_detection import DetectionResult, _readonly


def _nonnegative_integer(value: int, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return int(value)


def szudzik_pair(x: int, y: int) -> int:
    """Exact Equation (6), with unrestricted Python integer arithmetic."""
    left = _nonnegative_integer(x, "x")
    right = _nonnegative_integer(y, "y")
    return left * left + left + right if left >= right else right * right + left


def szudzik_unpair(z: int) -> tuple[int, int]:
    """Exact Equation (7); integer sqrt avoids precision loss for digest roots."""
    value = _nonnegative_integer(z, "z")
    square_root = math.isqrt(value)
    remainder = value - square_root * square_root
    if remainder < square_root:
        return remainder, square_root
    return square_root, remainder - square_root


def _integer_bytes(value: int) -> bytes:
    value = _nonnegative_integer(value, "encoded integer")
    return value.to_bytes(max(1, (value.bit_length() + 7) // 8), "big")


def _parts(*parts: bytes) -> bytes:
    """Length-prefix protocol fields; concatenation cannot merge boundaries."""
    return b"".join(len(part).to_bytes(8, "big") + part for part in parts)


@dataclass(frozen=True)
class PairingNode:
    """Known-shape subtree covering block indices [start, stop)."""

    start: int
    stop: int
    value: int
    left: PairingNode | None = None
    right: PairingNode | None = None

    @property
    def is_leaf(self) -> bool:
        return self.stop - self.start == 1


@dataclass(frozen=True)
class PairingTree:
    root: PairingNode
    versions: tuple[int, ...]

    @property
    def leaf_count(self) -> int:
        return len(self.versions)


def canonical_descriptor_bytes(row: np.ndarray) -> bytes:
    """Raw float32 bits in little-endian C order, without a feature transform.

    Other precisions are rejected rather than silently rounded to float32.
    An endian conversion only puts the same float32 bits into the declared
    wire format; signed zero and every other finite payload bit are retained.
    """
    values = np.asarray(row)
    if values.ndim != 1 or not len(values):
        raise ValueError("A descriptor block must be a nonempty 1D array")
    if values.dtype.kind != "f" or values.dtype.itemsize != 4:
        raise ValueError("Digest payload must preserve original float32 precision")
    if not np.all(np.isfinite(values)):
        raise ValueError("A descriptor block must contain only finite float32 values")
    return np.ascontiguousarray(values, dtype="<f4").tobytes(order="C")


def build_pairing_tree(
    blocks: Sequence[np.ndarray], nonce: bytes, versions: Sequence[int] | None = None
) -> PairingTree:
    """Hash actual block payloads with rt, then pair parents bottom-up.

    The balanced, ordered topology is known to both participants. Odd numbers
    of blocks are split at the integer midpoint rather than padded with a fake
    block. Equation (5) hashes raw block bytes concatenated with rt; VER is
    maintained separately and is also bound into the response commitment.
    """
    if not isinstance(nonce, bytes) or not nonce:
        raise ValueError("nonce must be nonempty bytes")
    if len(blocks) == 0:
        raise ValueError("A pairing tree needs at least one block")
    ver = tuple(0 for _ in blocks) if versions is None else tuple(
        _nonnegative_integer(version, "block version") for version in versions
    )
    if len(ver) != len(blocks):
        raise ValueError("Version count must match block count")
    leaves = []
    for block in blocks:
        payload = canonical_descriptor_bytes(block)
        leaves.append(int.from_bytes(hashlib.sha256(payload + nonce).digest(), "big"))

    def build(start: int, stop: int) -> PairingNode:
        if stop - start == 1:
            return PairingNode(start, stop, leaves[start])
        middle = (start + stop) // 2
        left = build(start, middle)
        right = build(middle, stop)
        return PairingNode(start, stop, szudzik_pair(left.value, right.value), left, right)

    return PairingTree(build(0, len(leaves)), ver)


def expand_pairing_tree(root: int, leaf_count: int) -> PairingNode:
    """Reconstruct every child digest from only the root and the known topology."""
    value = _nonnegative_integer(root, "root")
    count = _nonnegative_integer(leaf_count, "leaf_count")
    if count < 1:
        raise ValueError("leaf_count must be positive")

    def expand(value: int, start: int, stop: int) -> PairingNode:
        if stop - start == 1:
            return PairingNode(start, stop, value)
        left_value, right_value = szudzik_unpair(value)
        middle = (start + stop) // 2
        return PairingNode(start, stop, value, expand(left_value, start, middle),
                           expand(right_value, middle, stop))

    return expand(value, 0, count)


def minimum_common_subtrees(
    tree: PairingTree, requested_versions: Sequence[int]
) -> tuple[PairingNode, ...]:
    """Return the smallest covering forest of version-matched blocks (III-B 2.2).

    A whole matching interval is represented by its root; version mismatches
    cause descent and mismatched leaves are excluded. Static all-zero VER
    tables produce exactly the full root, while the implementation supports
    nonuniform version queries without inventing dynamic updates.
    """
    versions = tuple(_nonnegative_integer(version, "requested version") for version in requested_versions)
    if len(versions) != tree.leaf_count:
        raise ValueError("Requested VER table must match tree leaf count")
    selected = []

    def select(node: PairingNode) -> None:
        if all(tree.versions[index] == versions[index] for index in range(node.start, node.stop)):
            selected.append(node)
        elif not node.is_leaf:
            select(node.left)
            select(node.right)

    select(tree.root)
    return tuple(selected)


def digest_commitment(
    family_id: bytes,
    witness_id: int,
    nonce: bytes,
    versions: Sequence[int],
    subtree_roots: Sequence[PairingNode],
) -> str:
    """Local hash commitment to the digest response, NOT a BLS signature.

    It binds family, sender identity, rt, VER and common-subtree root bytes.
    A changed response or stale nonce fails verification against the earlier
    stored commitment; a malicious participant could recompute a fresh hash,
    so no cryptographic origin authentication is claimed.
    """
    encoded_versions = b"".join(_parts(_integer_bytes(version)) for version in versions)
    encoded_roots = b"".join(_parts(_integer_bytes(node.start), _integer_bytes(node.stop),
                                     _integer_bytes(node.value)) for node in subtree_roots)
    encoded = _parts(b"DynaEDI-descriptor-digest-v1", family_id, _integer_bytes(witness_id),
                     nonce, encoded_versions, encoded_roots)
    return hashlib.sha256(encoded).hexdigest()


def verify_digest_commitment(
    expected: str,
    family_id: bytes,
    witness_id: int,
    nonce: bytes,
    versions: Sequence[int],
    subtree_roots: Sequence[PairingNode],
) -> bool:
    return expected == digest_commitment(family_id, witness_id, nonce, versions, subtree_roots)


def _node_values(root: PairingNode) -> dict[tuple[int, int], int]:
    values = {(root.start, root.stop): root.value}
    if not root.is_leaf:
        values.update(_node_values(root.left))
        values.update(_node_values(root.right))
    return values


def detect_dynaedi(data: AlgorithmInput, config: DetectionConfig) -> DetectionResult:
    """Evaluate every shared target against every fixed reference witness.

    Digest frequencies retain every actual witness and its multiplicity, so
    querying an exact root equality count is equivalent to comparing it with
    all reference responses individually. Tree construction is linear in the
    raw descriptor payload size; there is no all-population square vote. The
    fixed random witness partition is independent of RACA-HIM GT and labels.
    """
    targets = np.asarray(data.evaluation_indices, dtype=np.int64)
    references = np.asarray(data.reference_indices, dtype=np.int64)
    target_ids = _readonly(data.model_ids[targets])
    witness_ids = data.model_ids[references]
    reference_count = len(references)
    block_count = int(getattr(config, "baseline_block_count", 5))
    quorum_fraction = float(getattr(config, "dyna_quorum_fraction", 2.0 / 3.0))
    if block_count < 1 or block_count > data.X.shape[1]:
        raise ValueError("baseline_block_count must be between 1 and descriptor dimension")
    if not math.isfinite(quorum_fraction) or not 0 < quorum_fraction < 1:
        raise ValueError("dyna_quorum_fraction must be in (0, 1)")
    if config.p_fail != 0.0:
        raise ValueError("Static DynaEDI+ adaptation requires p_fail=0")
    # Source Step 1.1 says at least ceil(2n/3), but Step 3.3 says MORE THAN
    # ceil(2n/3). Keep both inequalities, including their different rounding.
    ack_quorum = math.ceil(quorum_fraction * reference_count)
    inconsistent_quorum = ack_quorum + 1
    diagnostics = {
        "verification_mode": "all_reference_witnesses",
        "witness_indices": [int(value) for value in references],
        "witness_ids": [int(value) for value in witness_ids],
        "reference_count": reference_count,
        "ack_quorum": ack_quorum,
        "inconsistency_quorum": inconsistent_quorum,
        "quorum_rule": "a block is corrupted iff its exact-hash mismatch count > ceil(dyna_quorum_fraction * R); a model is corrupted iff any block is corrupted",
        "root_candidate_rule": "complete root mismatch count > ceil(dyna_quorum_fraction * R) triggers expansion only",
        "final_rule": "Corrupted iff at least one localized raw-hash block satisfies the strict inconsistency quorum; otherwise Normal",
        "algorithm_variant": "exact_hash_block_final_v4",
        "block_count": block_count,
        "all_block_versions": 0,
        "proof_mode": "SHA256 local digest commitment; no BLS, keys, or Byzantine-security claim",
        "content_comparison": "exact_hash",
        "payload_encoding": "raw float32 bits, canonical little-endian <f4, C order",
        "leaf_hash": "SHA256(raw_block_bytes || current_nonce)",
        "prediction_metric": "at least one version-matched raw-block digest inequality quorum after inverse-tree localization",
        "localization_metric": "exact inverse-decoded subtree/leaf digest inequality count",
        "localization_changes_prediction": True,
        "gt_success_semantics": "fixed witness pool and current snapshot commitments are available, not a certified GT set",
    }
    if reference_count == 0 or inconsistent_quorum > reference_count:
        return DetectionResult(
            method="DynaEDI+-Adapted", status="gt_unavailable", predictions=None,
            target_ids=target_ids, verifier_ids=None, gt_success=False, ground_truth=None,
            reason="The fixed witness pool cannot meet the source's strict inconsistency quorum.",
            diagnostics=diagnostics,
        )

    dimensions = data.X.shape[1]
    block_coordinates = np.array_split(np.arange(dimensions), block_count)
    block_ranges = [(int(indices[0]), int(indices[-1]) + 1) for indices in block_coordinates]
    versions = (0,) * block_count
    family_id = f"descriptor-family-d{dimensions}-b{block_count}-v0".encode("ascii")
    # Observable temporary ID order carries the seed's independent ID stream.
    # This is one static snapshot challenge, not a claim of a dynamic replay
    # protocol or consensus experiment. Exact same input remains reproducible.
    nonce = hashlib.sha256(_parts(b"DynaEDI-static-snapshot-rt", family_id,
                                  np.asarray(data.model_ids, dtype="<i8").tobytes())).digest()
    trees = [build_pairing_tree([row[start:stop] for start, stop in block_ranges], nonce, versions)
             for row in data.X]

    # Step 1.1: all fixed witnesses acknowledge the same deterministic rt under
    # this workload's no-failure/no-adversarial-transport setup.
    acknowledgements = [nonce for _ in references]
    ack_count = sum(value == nonce for value in acknowledgements)
    if ack_count < ack_quorum:
        return DetectionResult(
            method="DynaEDI+-Adapted", status="gt_unavailable", predictions=None,
            target_ids=target_ids, verifier_ids=None, gt_success=False, ground_truth=None,
            reason="The fixed witnesses did not acknowledge the current snapshot nonce.",
            diagnostics={**diagnostics, "ack_count": ack_count},
        )

    # Step 2.2: each verifier finds its minimum common subtree and commits its
    # root. Step 3.3 validates the later returned digest against that commitment.
    # One family/version snapshot allows these witness responses to be shared
    # across target requests, avoiding O(N*R) redundant cryptographic hashes.
    common_forests = [minimum_common_subtrees(trees[index], versions) for index in references]
    commitments = [digest_commitment(family_id, int(witness_id), nonce, versions, forest)
                   for witness_id, forest in zip(witness_ids, common_forests)]
    authentic = np.array([
        verify_digest_commitment(commitment, family_id, int(witness_id), nonce, versions, forest)
        for witness_id, forest, commitment in zip(witness_ids, common_forests, commitments)
    ], dtype=bool)
    if int(np.count_nonzero(authentic)) < inconsistent_quorum:
        return DetectionResult(
            method="DynaEDI+-Adapted", status="gt_unavailable", predictions=None,
            target_ids=target_ids, verifier_ids=None, gt_success=False, ground_truth=None,
            reason="Too few witness digest responses match the snapshot commitments.",
            diagnostics={**diagnostics, "ack_count": ack_count,
                         "commitment_valid_count": int(np.count_nonzero(authentic))},
        )

    genuine_count = int(np.count_nonzero(authentic))
    # These are actual committed tree integers, not feature buckets. A Counter
    # compresses repeated exact responses while preserving each witness vote.
    witness_root_counts = Counter(forest[0].value for forest, valid in zip(common_forests, authentic) if valid)
    disagreement_count = np.array([
        genuine_count - witness_root_counts.get(trees[int(index)].root.value, 0)
        for index in targets
    ], dtype=np.int64)
    # Root evidence is an expansion candidate gate, not a final model label.
    # Any leaf with a strict mismatch quorum necessarily has a root with that
    # quorum, because identical reversible roots have identical descendants.
    root_candidates = disagreement_count >= inconsistent_quorum
    predictions = np.zeros(len(targets), dtype=np.int8)

    # Genuine responders transmit ONLY root integers. Inverse Szudzik expansion
    # recovers their descendant digests using the known topology (IV, Eq. 7).
    # The original trees are not passed into the subtree comparison below.
    expanded_witnesses = [expand_pairing_tree(forest[0].value, block_count) for forest in common_forests]
    decoded_values = [_node_values(root) for root in expanded_witnesses]
    witness_counts_by_interval = {
        interval: Counter(values[interval] for values, valid in zip(decoded_values, authentic) if valid)
        for interval in decoded_values[0]
    }
    localized_targets = []
    local_counts = np.zeros(block_count, dtype=np.int64)
    unlocalized_candidates = 0
    for target_position in np.flatnonzero(root_candidates):
        target_position = int(target_position)
        row_index = int(targets[target_position])
        local_blocks = []
        visits = 0

        def descend(node: PairingNode) -> None:
            nonlocal visits
            interval = (node.start, node.stop)
            inconsistent_count = genuine_count - witness_counts_by_interval[interval].get(node.value, 0)
            # Equal full roots reconstruct exactly equal descendants because
            # Szudzik pairing is bijective. Consequently a differing child can
            # only belong to a full-root S_error witness. Counting all genuine
            # child mismatches is exactly counting S_error child mismatches.
            if inconsistent_count < inconsistent_quorum:
                return
            visits += 1
            if node.is_leaf:
                local_blocks.append(node.start)
            else:
                # Actual inverse pairing on the target's mismatched subtree,
                # rather than merely walking a preattached child pointer.
                left_value, right_value = szudzik_unpair(node.value)
                middle = (node.start + node.stop) // 2
                descend(PairingNode(node.start, middle, left_value))
                descend(PairingNode(middle, node.stop, right_value))

        # The comparison reconstructs target descendants from its root only;
        # it does not read the stored target tree's child pointers or payloads.
        descend(PairingNode(0, block_count, trees[row_index].root.value))
        for block in local_blocks:
            local_counts[block] += 1
        # Native DynaEDI+ finalizes corruption at the block/leaf stage. The
        # model-level OR below is the declared static descriptor aggregation.
        predictions[target_position] = int(bool(local_blocks))
        if not local_blocks:
            unlocalized_candidates += 1
        localized_targets.append({"target_position": target_position, "block_indices": local_blocks,
                                  "expanded_node_count": visits})

    diagnostics.update({
        "family_id": family_id.decode("ascii"),
        "snapshot_nonce_hex": nonce.hex(),
        "nonce_scope": "one reproducible static descriptor snapshot; no dynamic freshness experiment",
        "ack_count": ack_count,
        "commitment_valid_count": int(np.count_nonzero(authentic)),
        "witness_commitments_sha256": hashlib.sha256("".join(commitments).encode("ascii")).hexdigest(),
        "common_subtree_count_per_witness": [len(forest) for forest in common_forests],
        "example_witness_id": int(witness_ids[0]),
        "example_pairing_root_hex": format(common_forests[0][0].value, "x"),
        "example_expanded_leaf_hashes_hex": [format(decoded_values[0][(index, index + 1)], "x") for index in range(block_count)],
        "block_feature_ranges": [list(interval) for interval in block_ranges],
        "root_candidate_count": int(np.count_nonzero(root_candidates)),
        "target_root_disagreement_counts": [int(value) for value in disagreement_count],
        "target_digest_disagreement_counts": [int(value) for value in disagreement_count],
        "target_digest_agreement_counts": [int(genuine_count - value) for value in disagreement_count],
        "target_disagreement_counts": [int(value) for value in disagreement_count],
        "disagreement_count_semantics": "compatibility count fields record complete-root mismatches, not final block-based predictions",
        "witness_distinct_root_count": len(witness_root_counts),
        "localized_targets": localized_targets,
        "localized_target_count_by_block": [int(value) for value in local_counts],
        "final_corrupted_count": int(np.count_nonzero(predictions)),
        "unlocalized_root_candidate_count": unlocalized_candidates,
        "unlocalized_anomaly_count": unlocalized_candidates,
        "unlocalized_anomaly_count_semantics": "legacy alias for root candidates with no corrupted block; their final prediction is Normal",
        "localization_rule": "raw-hash subtree or leaf mismatch count > ceil(dyna_quorum_fraction * R)",
    })
    return DetectionResult(
        method="DynaEDI+-Adapted", status="success", predictions=_readonly(predictions, np.int8),
        target_ids=target_ids, verifier_ids=None, gt_success=True, ground_truth=None,
        reason="", diagnostics=diagnostics,
    )
