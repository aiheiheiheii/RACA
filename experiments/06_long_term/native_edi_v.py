"""EDI-V's variable Merkle hash tree and its four audit phases.

Primary source: Li et al., ``Auditing Cache Data Integrity in the Edge
Computing Environment'', TPDS 32(5), 2021, DOI 10.1109/TPDS.2020.3043755.
The author-uploaded accepted manuscript is publicly readable at
https://www.researchgate.net/publication/347436483_Auditing_Cache_Data_Integrity_in_the_Edge_Computing_Environment
Definition 1 and Algorithms
1--4 (manuscript pp. 4--7) specify the VMHT and request/response/proof check.
Section 5.2 uses SHA256 and a 128-bit one-audit security code.

This is the paper's VMHT, including a three-child rightmost parent when a level
has an odd number of nodes. It never duplicates leaves or pads a binary tree.
The proof is the sampled subtree's root tag, not a conventional Merkle
authentication path. The vendor recomputes its own trusted tree with fresh g.

Explicit experiment adapters:
* A descriptor is its actual little-endian float32 byte string. Four-byte
  blocks represent the existing 30 coordinates; no data are expanded.
* Distinct model contents each have their own t=0 trusted vendor original.
  Only these enrolled bytes are retained; no label or future descriptor enters
  the protocol. In the paper multiple replicas share one original.
* Algorithm 2 line 2 and Section 4.2.2 (manuscript pp. 6--7) say to choose a
  random integer 1 <= l < H according to lambda, without providing a unique
  formula, distribution or numerical default for lambda. Section 6.3, Table 3
  instead fixes subtree height ss=9 with ss=H-l+1. Thus there is no unique source
  lambda default to copy. Here its
  Theorem 2 bound determines a minimum sample size for an openly declared
  minimum of one changed block and confidence .999. A common level is drawn
  uniformly from levels whose every subtree meets that requirement. If the
  bound requires more than all available blocks, the request is for the root:
  checking all blocks is exact, rather than pretending to meet that bound by
  fractional or duplicated samples. The paper's Table 3 ss=9 and 512 KiB block
  defaults cannot apply to a 120-byte descriptor.
* Seeded pseudorandom functions make the experiment reproducible. Fresh g is
  never reused in one instance. This is a simulation of the challenge protocol,
  not a claim of production cryptographic randomness or attack validation.

No experiment, benchmark, test or demonstration runs at import time.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import hmac
import math
import random
from types import MappingProxyType
from typing import Mapping, Sequence


PROTOCOL_VERSION = "edi-v-vmht-algorithms-1-to-4-v1"
SOURCE_DOI = "10.1109/TPDS.2020.3043755"
SECURITY_CODE_BYTES = 16
HASH_BYTES = 32


def _positive_int(value: int, name: str) -> int:
    if isinstance(value, bool) or int(value) != value or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _replica_id(value: int) -> int:
    if isinstance(value, bool) or int(value) != value or value < 0:
        raise ValueError("Replica/server identity must be a nonnegative integer")
    return int(value)


def split_payload(payload: bytes, block_bytes: int = 4) -> tuple[bytes, ...]:
    """Use real fixed-size blocks, without padding, expansion or corruption data."""
    width = _positive_int(block_bytes, "block_bytes")
    if not isinstance(payload, bytes) or not payload or len(payload) % width:
        raise ValueError("Payload must be nonempty bytes, divisible by block_bytes")
    return tuple(payload[start:start + width] for start in range(0, len(payload), width))


@dataclass(frozen=True)
class VMHTNode:
    """The five-element node value in Definition 1, plus ordered child links."""

    level: int
    position: int
    leaf_count: int
    children: tuple[VMHTNode, ...]
    tag: bytes

    @property
    def child_count(self) -> int:
        return len(self.children)

    @property
    def value(self) -> tuple[int, int, int, int, bytes]:
        return self.level, self.position, self.leaf_count, self.child_count, self.tag


@dataclass(frozen=True)
class VMHT:
    root: VMHTNode
    levels: tuple[tuple[VMHTNode, ...], ...]

    @property
    def height(self) -> int:
        return len(self.levels)

    @property
    def block_count(self) -> int:
        return self.root.leaf_count

    @property
    def hash_operations(self) -> int:
        return sum(map(len, self.levels))

    def node(self, level: int, position: int) -> VMHTNode:
        l = _positive_int(level, "level")
        r = _positive_int(position, "position")
        if l > self.height or r > len(self.levels[l - 1]):
            raise ValueError("Requested VMHT node does not exist")
        return self.levels[l - 1][r - 1]


def build_vmht(ordered_blocks: Sequence[bytes], security_code: bytes) -> VMHT:
    """Algorithm 1's bottom-up construction on already ordered/shuffled blocks.

    At each level pairs are reduced except that the final three nodes have one
    three-child parent. This is the queue algorithm's same-level condition in
    Algorithm 1 lines 15--26, with no dependence on payload values.
    """
    blocks = tuple(ordered_blocks)
    if not blocks or any(not isinstance(block, bytes) or not block for block in blocks):
        raise ValueError("VMHT needs nonempty byte blocks")
    if not isinstance(security_code, bytes) or len(security_code) != SECURITY_CODE_BYTES:
        raise ValueError("EDI-V uses an explicit 128-bit security code")
    height = len(blocks).bit_length()
    current = tuple(VMHTNode(height, i + 1, 1, (), hashlib.sha256(security_code + block).digest())
                    for i, block in enumerate(blocks))
    bottom_up = [current]
    level = height
    while len(current) > 1:
        parents = []
        start = 0
        while start < len(current):
            count = 3 if len(current) - start == 3 else 2
            children = current[start:start + count]
            if len(children) != count:
                raise ValueError("Invalid VMHT topology: a parent cannot have one child")
            parents.append(VMHTNode(
                level - 1, len(parents) + 1,
                sum(child.leaf_count for child in children), children,
                hashlib.sha256(b"".join(child.tag for child in children)).digest(),
            ))
            start += count
        current = tuple(parents)
        bottom_up.append(current)
        level -= 1
    return VMHT(current[0], tuple(reversed(bottom_up)))


def _permutation(shuffle_key: int, block_count: int) -> tuple[int, ...]:
    if isinstance(shuffle_key, bool) or int(shuffle_key) != shuffle_key:
        raise ValueError("Shuffle key must be an integer")
    if not 0 <= shuffle_key < block_count:
        raise ValueError("The source shuffle key satisfies 0 <= sh < m")
    positions = list(range(block_count))
    random.Random(int(shuffle_key)).shuffle(positions)
    return tuple(positions)


def _level_leaf_counts(block_count: int) -> tuple[tuple[int, ...], ...]:
    """Content-independent source VMHT topology for validating untrusted requests."""
    current = (1,) * block_count
    bottom_up = [current]
    while len(current) > 1:
        parents = []
        start = 0
        while start < len(current):
            count = 3 if len(current) - start == 3 else 2
            parents.append(sum(current[start:start + count]))
            start += count
        current = tuple(parents)
        bottom_up.append(current)
    return tuple(reversed(bottom_up))


def vmht_gen(payload: bytes, security_code: bytes, shuffle_key: int,
             block_bytes: int = 4) -> VMHT:
    """Algorithm 1: PR(sh,m), nonce-prefixed leaf hashes, and VMHT reduction."""
    blocks = split_payload(payload, block_bytes)
    order = _permutation(shuffle_key, len(blocks))
    return build_vmht(tuple(blocks[i] for i in order), security_code)


@dataclass(frozen=True)
class EdiVRequest:
    """Definition 2's seven fields, with local round/layout audit metadata."""

    replica_id: int
    server_id: int
    shuffle_key: int
    security_code: bytes
    level: int
    position: int
    leaf_count: int
    round_id: int
    total_blocks: int
    block_bytes: int

    def as_record(self) -> dict:
        return {
            "replica_id": self.replica_id, "server_id": self.server_id,
            "shuffle_key": self.shuffle_key, "security_code_hex": self.security_code.hex(),
            "level": self.level, "position": self.position, "leaf_count": self.leaf_count,
            "round_id": self.round_id, "total_blocks": self.total_blocks,
            "block_bytes": self.block_bytes,
        }


@dataclass(frozen=True)
class EdiVProof:
    """Definition 3's actual integrity proof: (replica identity, server, tag)."""

    replica_id: int
    server_id: int
    tag: bytes

    def as_record(self) -> dict:
        return {"replica_id": self.replica_id, "server_id": self.server_id,
                "tag_hex": self.tag.hex()}


def pro_gen(current_payload: bytes, request: EdiVRequest) -> EdiVProof:
    """Algorithm 3: hash only the requested shuffled contiguous block subtree.

    Theorem 1 gives zero-based start = 2**(H-l)*(r-1). The sampled block order
    must be preserved when calling the construction part of VMHTGen: shuffling
    that sample again would not reconstruct the vendor's requested subtree.
    """
    if not isinstance(request, EdiVRequest):
        raise ValueError("Response needs an EDI-V request")
    _replica_id(request.replica_id)
    _replica_id(request.server_id)
    _positive_int(request.round_id, "round_id")
    _positive_int(request.total_blocks, "total_blocks")
    _positive_int(request.level, "level")
    _positive_int(request.position, "position")
    _positive_int(request.leaf_count, "leaf_count")
    blocks = split_payload(current_payload, request.block_bytes)
    if len(blocks) != request.total_blocks:
        raise ValueError("Response payload length differs from the registered replica")
    height = len(blocks).bit_length()
    if not 1 <= request.level < height:
        raise ValueError("Algorithm 2 requires a non-leaf sampling level")
    counts = _level_leaf_counts(len(blocks))[request.level - 1]
    if request.position > len(counts) or request.leaf_count != counts[request.position - 1]:
        raise ValueError("Request leaf_count/position differs from the source VMHT topology")
    start = (1 << (height - request.level)) * (request.position - 1)
    stop = start + request.leaf_count
    if request.position < 1 or request.leaf_count < 2 or not 0 <= start < stop <= len(blocks):
        raise ValueError("Requested sample is outside the current replica")
    order = _permutation(request.shuffle_key, len(blocks))
    sample = tuple(blocks[i] for i in order[start:stop])
    subtree = build_vmht(sample, request.security_code)
    return EdiVProof(request.replica_id, request.server_id, subtree.root.tag)


def check_pro(expected_tree: VMHT, request: EdiVRequest, proof: EdiVProof) -> bool:
    """Algorithm 4: check identities and compare the actual returned node tag."""
    expected = expected_tree.node(request.level, request.position)
    return (proof.replica_id == request.replica_id
            and proof.server_id == request.server_id
            and expected.leaf_count == request.leaf_count
            and isinstance(proof.tag, bytes) and len(proof.tag) == HASH_BYTES
            and hmac.compare_digest(expected.tag, proof.tag))


def minimum_sample_blocks(block_count: int, probability: float,
                          assumed_corrupted_blocks: int = 1) -> tuple[int, bool]:
    """Theorem 2's conservative size; full audit when its bound exceeds m.

    The assumed minimum changed-block count is declared before the run. It is
    never obtained from current corruption labels or actual changed positions.
    """
    m = _positive_int(block_count, "block_count")
    nc = _positive_int(assumed_corrupted_blocks, "assumed_corrupted_blocks")
    if nc > m or not math.isfinite(probability) or not 0 < probability < 1:
        raise ValueError("Require nc <= m and 0 < detection_probability < 1")
    required = 1 if nc == m else math.ceil(math.log1p(-probability) / math.log1p(-nc / m))
    return min(m, required), required >= m


class NativeEDIV:
    """Vendor-held t=0 originals and round-local EDI-V challenge lifecycle.

    ``prepare_round`` receives selected IDs only after the DVA-P decision. Its
    output contains no originals or expected tags. ``respond`` is the server's
    operation on its current bytes; ``verify`` performs the independent vendor
    check. Repeated/stale challenge use raises rather than accepting replay.
    """

    def __init__(self, enrolled_payloads: Mapping[int, bytes], *, seed: int,
                 block_bytes: int = 4, detection_probability: float = .999,
                 assumed_corrupted_blocks: int = 1) -> None:
        self.block_bytes = _positive_int(block_bytes, "block_bytes")
        if not enrolled_payloads:
            raise ValueError("EDI-V needs trusted t=0 vendor originals")
        if any(not isinstance(payload, bytes) for payload in enrolled_payloads.values()):
            raise ValueError("Trusted enrollment requires actual immutable payload bytes")
        payloads = {_replica_id(i): payload for i, payload in enrolled_payloads.items()}
        if len(payloads) != len(enrolled_payloads) or any(i < 0 for i in payloads):
            raise ValueError("Replica IDs must be unique nonnegative integers")
        counts = {len(split_payload(payload, self.block_bytes)) for payload in payloads.values()}
        if len(counts) != 1 or min(counts) < 2:
            raise ValueError("One audit batch requires equally sized replicas of at least two blocks")
        self.block_count = counts.pop()
        self.minimum_sample, self.full_audit_required = minimum_sample_blocks(
            self.block_count, float(detection_probability), assumed_corrupted_blocks)
        self.detection_probability = float(detection_probability)
        self.assumed_corrupted_blocks = int(assumed_corrupted_blocks)
        self._originals = MappingProxyType(payloads)
        self._rng = random.Random(int(seed))
        self._used_codes: set[bytes] = set()
        self._round = 0
        self._requests: dict[int, EdiVRequest] = {}
        self._expected: dict[int, VMHT] = {}
        self._verified: set[int] = set()

    def prepare_round(self, round_id: int, target_ids: Sequence[int]) -> Mapping[int, EdiVRequest]:
        """Setup and Algorithm 2; common fresh g/sh/l, random per-replica r."""
        t = _positive_int(round_id, "round_id")
        if t <= self._round:
            raise ValueError("EDI-V audit rounds must increase; stale rounds cannot be replayed")
        ids = tuple(_replica_id(i) for i in target_ids)
        if len(ids) != len(set(ids)) or not set(ids).issubset(self._originals):
            raise ValueError("Audit targets must be unique enrolled replica IDs")
        code = self._rng.getrandbits(128).to_bytes(SECURITY_CODE_BYTES, "big")
        while code in self._used_codes:
            code = self._rng.getrandbits(128).to_bytes(SECURITY_CODE_BYTES, "big")
        self._used_codes.add(code)
        shuffle = self._rng.randrange(self.block_count)
        self._round, self._requests, self._expected, self._verified = t, {}, {}, set()
        if not ids:
            return MappingProxyType({})
        # All batch replicas have the same topology, irrespective of contents.
        first = vmht_gen(self._originals[ids[0]], code, shuffle, self.block_bytes)
        legal_levels = [level for level in range(1, first.height)
                        if min(node.leaf_count for node in first.levels[level - 1]) >= self.minimum_sample]
        level = 1 if self.full_audit_required else self._rng.choice(legal_levels)
        for i in ids:
            tree = first if i == ids[0] else vmht_gen(self._originals[i], code, shuffle, self.block_bytes)
            position = self._rng.randrange(len(tree.levels[level - 1])) + 1
            node = tree.node(level, position)
            request = EdiVRequest(i, i, shuffle, code, level, position, node.leaf_count,
                                  t, self.block_count, self.block_bytes)
            self._requests[i], self._expected[i] = request, tree
        return MappingProxyType(dict(self._requests))

    @staticmethod
    def respond(model_id: int, current_payload: bytes, request: EdiVRequest) -> EdiVProof:
        if not isinstance(request, EdiVRequest):
            raise ValueError("Response needs an EDI-V request")
        if _replica_id(model_id) != request.replica_id or _replica_id(model_id) != request.server_id:
            raise ValueError("The receiving replica/server identity does not match the request")
        return pro_gen(current_payload, request)

    def verify(self, model_id: int, request: EdiVRequest, proof: EdiVProof) -> bool:
        i = _replica_id(model_id)
        if not isinstance(request, EdiVRequest) or not isinstance(proof, EdiVProof):
            raise ValueError("Verification needs actual EDI-V request and proof objects")
        if i not in self._requests or self._requests[i] != request or request.round_id != self._round:
            raise ValueError("Expected an outstanding current-round vendor challenge")
        if i in self._verified:
            raise ValueError("An EDI-V challenge can be checked only once")
        self._verified.add(i)
        return check_pro(self._expected[i], request, proof)

    def diagnostics(self, request: EdiVRequest) -> dict:
        i = request.replica_id
        if i not in self._requests or self._requests[i] != request:
            raise ValueError("Diagnostics require this vendor's current challenge")
        tree = self._expected[i]
        node = tree.node(request.level, request.position)
        pending = [node]
        response_hashes = 0
        while pending:
            current = pending.pop()
            response_hashes += 1
            pending.extend(current.children)
        return {
            "protocol": PROTOCOL_VERSION, "source_doi": SOURCE_DOI,
            "request": request.as_record(), "tree_height": tree.height,
            "sampling_scale": tree.height - request.level + 1,
            "minimum_sample_blocks": self.minimum_sample,
            "full_audit_required": self.full_audit_required,
            "assumed_minimum_corrupted_blocks": self.assumed_corrupted_blocks,
            "requested_detection_probability": self.detection_probability,
            "confidence_rule": "source Theorem2 bound; full coverage if its required sample exceeds m",
            "sampled_bytes": request.leaf_count * request.block_bytes,
            "vendor_tree_hash_operations": tree.hash_operations,
            "response_subtree_hash_operations": response_hashes,
            "proof_tag_bytes": HASH_BYTES,
            "vmht_three_child_parent_count": sum(
                n.child_count == 3 for nodes in tree.levels for n in nodes),
            "vendor_original_source": "explicit trusted t=0 enrollment only",
            "block_adapter": "one actual little-endian float32 coordinate per four-byte block",
            "rng_scope": "seeded reproducible challenge simulation; 128-bit g unique per audit round",
        }
