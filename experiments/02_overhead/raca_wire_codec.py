"""Progressive direction encoding with conservative cosine certificates.

This communication adapter preserves a certified binary interval verdict, not
the descriptor's individual float32 bits.  Uncertain comparisons must request
the low bit plane and, if still uncertain, the original float32 descriptor.
The existing timing implementation does not use this module.

Packets are little endian.  Their transport envelope must bind sender, target,
round and request: a two-byte mode/dimension header cannot authenticate a low
plane or associate it with a particular target by itself.  No dictionary or
undisclosed reference descriptor is required by the encoder.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
import struct

import numpy as np

from experiments.common.similarity import _unit_rows

WIRE_CODEC_SCHEMA = "raca-certified-progressive-int16-v1"
INITIAL_MODE = 1
REFINEMENT_MODE = 2
RAW_MODE = 3
MAX_DIMENSION = 255
MAX_QUANTIZED = 32767
_EPS = np.finfo(np.float64).eps


def _immutable(values, dtype) -> np.ndarray:
    """Use an immutable bytes backing rather than a reversible write flag."""
    source = np.ascontiguousarray(values, dtype=dtype)
    return np.frombuffer(source.tobytes(), dtype=dtype).reshape(source.shape)


def _row(row) -> np.ndarray:
    values = np.asarray(row)
    if (values.dtype != np.float32 or values.ndim != 1
            or not 1 <= values.size <= MAX_DIMENSION):
        raise ValueError("Expected a one-dimensional float32 descriptor of dimension 1..255")
    if not np.isfinite(values).all() or not np.any(values != 0):
        raise ValueError("Descriptor must be finite and nonzero")
    return values


@dataclass(frozen=True)
class PreparedDescriptor:
    """All actual packets, ready for length accounting and transmission."""

    initial: bytes
    refinement: bytes
    raw: bytes


@dataclass(frozen=True)
class DecodedDescriptor:
    """Unit direction and a conservative bound on its angular error.

    An initial decoder retains the received high plane and scale to decode its
    low plane.  Those fields are local state, not additional wire payload.
    """

    direction: np.ndarray
    angle_error: float
    d: int
    stage: str
    _scale: float | None = field(default=None, repr=False, compare=False)
    _high: bytes | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if isinstance(self.d, bool) or not isinstance(self.d, int) or not 1 <= self.d <= MAX_DIMENSION:
            raise ValueError("Decoded dimension must be an integer in 1..255")
        direction = np.asarray(self.direction, dtype=np.float64)
        if (direction.shape != (self.d,) or not np.isfinite(direction).all()
                or abs(float(np.linalg.norm(direction)) - 1) > 256 * (self.d + 1) * _EPS):
            raise ValueError("Decoded direction must be a finite unit vector")
        error = float(self.angle_error)
        if not math.isfinite(error) or not 0 <= error <= math.pi:
            raise ValueError("Angular error must be finite and in [0,pi]")
        if self.stage not in ("initial", "refinement", "exact"):
            raise ValueError("Unknown descriptor stage")
        if self.stage == "initial":
            if (not isinstance(self._high, bytes) or len(self._high) != self.d
                    or self._scale is None or not math.isfinite(self._scale) or self._scale <= 0):
                raise ValueError("Initial descriptor requires its received scale and high plane")
        elif self._high is not None or self._scale is not None:
            raise ValueError("Only an initial descriptor retains high-plane decoding state")
        object.__setattr__(self, "direction", _immutable(direction, np.float64))
        object.__setattr__(self, "angle_error", error)


def encode_descriptor(row) -> PreparedDescriptor:
    """Encode one unchanged source row; no reference or label is inspected."""
    source = _row(row)
    d = int(source.size)
    unit = _unit_rows(source)[0]
    scale = np.nextafter(np.float32(np.max(np.abs(unit)) / MAX_QUANTIZED), np.float32(np.inf))
    # Quantize using the exact float32 value actually carried in the packet.
    quantized = np.rint(unit / float(scale)).astype("<i2")
    if np.any(np.abs(quantized.astype(np.int32)) > MAX_QUANTIZED):
        raise AssertionError("Upward scale did not bound the signed quantizer")
    unsigned = quantized.view("<u2")
    high = (unsigned >> 8).astype(np.uint8).tobytes()
    low = (unsigned & 255).astype(np.uint8).tobytes()
    initial = bytes((INITIAL_MODE, d)) + struct.pack("<f", float(scale)) + high
    refinement = bytes((REFINEMENT_MODE, d)) + low
    raw = bytes((RAW_MODE, d)) + np.ascontiguousarray(source, dtype="<f4").tobytes()
    return PreparedDescriptor(initial, refinement, raw)


def _packet(packet, mode: int, stride: int, *, extra: int = 0) -> tuple[int, bytes]:
    if not isinstance(packet, (bytes, bytearray, memoryview)):
        raise ValueError("Packet must contain bytes")
    packet = bytes(packet)
    if len(packet) < 2 or packet[0] != mode or packet[1] == 0:
        raise ValueError("Invalid packet mode or dimension")
    d = packet[1]
    if len(packet) != 2 + extra + stride * d:
        raise ValueError("Truncated packet or unexpected trailing bytes")
    return d, packet[2:]


def _angular_error(d: int, scale: float, coefficient: float) -> float:
    # Float64 normalization/division roundoff is conservatively covered in
    # addition to the exact nearest-integer quantization error.
    roundoff = 64 * (d + 1) * _EPS
    distance = math.nextafter(math.sqrt(d) * scale * coefficient + roundoff, math.inf)
    ratio = math.nextafter(distance / (1 - roundoff), math.inf)
    if ratio >= 1:
        return math.pi
    return math.nextafter(math.asin(ratio), math.inf)


def decode_initial(packet) -> DecodedDescriptor:
    """Decode the high plane at its midpoint, retaining no hidden source."""
    d, body = _packet(packet, INITIAL_MODE, 1, extra=4)
    scale = float(struct.unpack("<f", body[:4])[0])
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("Initial scale must be positive and finite")
    maximum_scale = float(np.nextafter(np.float32(1 / MAX_QUANTIZED), np.float32(np.inf)))
    if scale > maximum_scale:
        raise ValueError("Initial scale is outside the normalized quantizer domain")
    high = body[4:]
    signed_high = np.frombuffer(high, dtype=np.int8).astype(np.float64)
    midpoint = signed_high * 256 + 127.5
    # A valid source's maximum absolute q16 is 32767, so at least one high
    # byte is 127 or -128.  This also rejects invalid all-zero bit planes.
    if not np.any((signed_high == 127) | (signed_high == -128)):
        raise ValueError("High plane has no normalized quantizer endpoint")
    # q16 differs from this midpoint by at most 127.5 integer units; source
    # rounding contributes another 0.5.  Hence 128*scale, not 128.5*scale.
    direction = _unit_rows(midpoint)[0]
    return DecodedDescriptor(direction, _angular_error(d, scale, 128), d, "initial", scale, high)


def decode_refinement(initial: DecodedDescriptor, packet) -> DecodedDescriptor:
    """Apply a transport-bound low plane to exactly its initial high plane."""
    if not isinstance(initial, DecodedDescriptor) or initial.stage != "initial":
        raise ValueError("Refinement requires an initial decoded descriptor")
    d, low = _packet(packet, REFINEMENT_MODE, 1)
    if d != initial.d:
        raise ValueError("Refinement dimension does not match its initial descriptor")
    high_values = np.frombuffer(initial._high, dtype=np.uint8).astype(np.uint16)
    low_values = np.frombuffer(low, dtype=np.uint8).astype(np.uint16)
    unsigned = (high_values << 8) | low_values
    quantized = unsigned.astype("<u2").view("<i2")
    if (np.any(quantized == -32768)
            or np.max(np.abs(quantized.astype(np.int32))) != MAX_QUANTIZED):
        raise ValueError("Refinement is outside the normalized signed quantizer domain")
    return DecodedDescriptor(_unit_rows(quantized)[0],
        _angular_error(d, float(initial._scale), 0.5), d, "refinement")


def decode_raw(packet) -> np.ndarray:
    """Return the exact original little-endian float32 row, with immutable data."""
    d, body = _packet(packet, RAW_MODE, 4)
    row = np.frombuffer(body, dtype="<f4", count=d)
    _row(row)
    return row


def exact_direction(row) -> DecodedDescriptor:
    """A verifier's already-held source requires no descriptor transmission."""
    source = _row(row)
    return DecodedDescriptor(_unit_rows(source)[0], 0.0, int(source.size), "exact")


def cosine_bounds(left: DecodedDescriptor, right: DecodedDescriptor) -> tuple[float, float]:
    """Bound the unchanged source cosine using both direction error cones."""
    if not isinstance(left, DecodedDescriptor) or not isinstance(right, DecodedDescriptor):
        raise ValueError("Cosine bounds require decoded directions")
    if left.d != right.d:
        raise ValueError("Compared descriptor dimensions differ")
    dot = float(np.einsum("i,i->", left.direction, right.direction))
    dot_margin = 32 * (left.d + 4) * _EPS
    dot_low = max(-1.0, math.nextafter(dot - dot_margin, -math.inf))
    dot_high = min(1.0, math.nextafter(dot + dot_margin, math.inf))
    theta_low = max(0.0, math.nextafter(math.acos(dot_high), -math.inf))
    theta_high = min(math.pi, math.nextafter(math.acos(dot_low), math.inf))
    total_error = math.nextafter(left.angle_error + right.angle_error + dot_margin, math.inf)
    lower_angle = max(0.0, math.nextafter(theta_low - total_error, -math.inf))
    upper_angle = min(math.pi, math.nextafter(theta_high + total_error, math.inf))
    lower = max(-1.0, math.nextafter(math.cos(upper_angle) - dot_margin, -math.inf))
    upper = min(1.0, math.nextafter(math.cos(lower_angle) + dot_margin, math.inf))
    return lower, upper


def certify_pair(left: DecodedDescriptor, right: DecodedDescriptor,
                 tau_lower: float, tau_upper: float) -> int | None:
    """Return 0 for normal, 1 for corrupted, or None when precision is insufficient."""
    if (not math.isfinite(tau_lower) or not math.isfinite(tau_upper)
            or not -1 <= tau_lower <= tau_upper <= 1):
        raise ValueError("Require finite -1 <= tau_lower <= tau_upper <= 1")
    lower, upper = cosine_bounds(left, right)
    if lower >= tau_lower and upper <= tau_upper:
        return 0
    if upper < tau_lower or lower > tau_upper:
        return 1
    return None
