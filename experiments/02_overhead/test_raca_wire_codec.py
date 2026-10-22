"""Certificates, real packets, corruption rejection and immutable decoding."""
import importlib
import math
import struct

import numpy as np
import pytest

codec = importlib.import_module("experiments.02_overhead.raca_wire_codec")
similarity = importlib.import_module("experiments.common.similarity")


@pytest.mark.parametrize("d", [1, 2, 10, 20, 30, 40, 50, 255])
def test_real_packet_lengths_and_exact_raw_roundtrip(d):
    row = np.random.default_rng(d).normal(size=d).astype(np.float32)
    prepared = codec.encode_descriptor(row)
    assert len(prepared.initial) == 6 + d
    assert len(prepared.refinement) == 2 + d
    assert len(prepared.raw) == 2 + 4 * d
    assert prepared.initial[:2] == bytes((codec.INITIAL_MODE, d))
    assert prepared.refinement[:2] == bytes((codec.REFINEMENT_MODE, d))
    recovered = codec.decode_raw(prepared.raw)
    assert recovered.dtype == np.float32
    assert recovered.tobytes() == row.astype("<f4").tobytes()
    with pytest.raises(ValueError):
        recovered.setflags(write=True)


@pytest.mark.parametrize("d", [1, 2, 10, 30, 50, 255])
@pytest.mark.parametrize("scale", [1.0, np.float32(1e-35), np.float32(1e35)])
def test_direction_angular_error_covers_stored_source(d, scale):
    rng = np.random.default_rng(d)
    for _ in range(8):
        row = (rng.uniform(-1, 1, size=d) * scale).astype(np.float32)
        packets = codec.encode_descriptor(row)
        initial = codec.decode_initial(packets.initial)
        refined = codec.decode_refinement(initial, packets.refinement)
        exact = codec.exact_direction(row)
        for decoded in (initial, refined):
            # atan2 is accurate near zero, unlike arccos(dot) cancellation.
            perpendicular = decoded.direction - np.dot(decoded.direction, exact.direction) * exact.direction
            angle = math.atan2(np.linalg.norm(perpendicular), np.dot(decoded.direction, exact.direction))
            assert angle <= decoded.angle_error
            assert decoded.d == d
            assert not decoded.direction.flags.writeable
            with pytest.raises(ValueError):
                decoded.direction.setflags(write=True)
        assert refined.angle_error < initial.angle_error


@pytest.mark.parametrize("d", [1, 2, 10, 20, 30, 40, 50, 255])
def test_cosine_certificate_contains_actual_score_and_never_changes_verdict(d):
    rng = np.random.default_rng(2000 + d)
    for _ in range(24):
        left = rng.normal(size=d).astype(np.float32)
        right = (left + rng.normal(size=d) * rng.choice([0.001, 0.08, 0.8, 4])).astype(np.float32)
        pleft, pright = codec.encode_descriptor(left), codec.encode_descriptor(right)
        ileft, iright = codec.decode_initial(pleft.initial), codec.decode_initial(pright.initial)
        lchoices = [ileft, codec.decode_refinement(ileft, pleft.refinement), codec.exact_direction(left)]
        rchoices = [iright, codec.decode_refinement(iright, pright.refinement), codec.exact_direction(right)]
        score = float(similarity.cosine_aligned(left, right)[0])
        for ld in lchoices:
            for rd in rchoices:
                lo, hi = codec.cosine_bounds(ld, rd)
                assert -1 <= lo <= score <= hi <= 1
                for lower, upper in [(-1., 1.), (0.997, 1.), (-0.8, 0.2), (-1., -0.5), (score, score)]:
                    verdict = codec.certify_pair(ld, rd, lower, upper)
                    if verdict is not None:
                        assert verdict == int(not lower <= score <= upper)


def test_signed_midpoint_and_quantizer_error_proof():
    row = np.array([-1.0, 1.0, -0.00001, 0.00001, -0.2, 0.6], dtype=np.float32)
    packets = codec.encode_descriptor(row)
    scale = struct.unpack("<f", packets.initial[2:6])[0]
    unit = similarity._unit_rows(row)[0]
    q = np.rint(unit / scale).astype("<i2")
    high = np.frombuffer(packets.initial[6:], dtype=np.int8).astype(np.int32)
    midpoint = high * 256 + 127.5
    assert np.max(np.abs(q.astype(np.float64) - midpoint)) <= 127.5
    assert np.max(np.abs(unit - q.astype(np.float64) * scale)) <= scale * 0.5 + 1e-15
    assert np.max(np.abs(unit - midpoint * scale)) <= scale * 128 + 1e-15
    decoded = codec.decode_refinement(codec.decode_initial(packets.initial), packets.refinement)
    np.testing.assert_array_equal(decoded.direction, similarity._unit_rows(q)[0])


@pytest.mark.parametrize("row", [
    np.array([np.nextafter(np.float32(0), np.float32(1)), 0], np.float32),
    np.array([np.finfo(np.float32).max, -np.finfo(np.float32).max], np.float32),
    np.array([np.finfo(np.float32).tiny, -np.nextafter(np.float32(0), np.float32(1))], np.float32),
    np.array([1, -0.0, 0.0], np.float32),
])
def test_extreme_float32_and_signed_zero(row):
    prepared = codec.encode_descriptor(row)
    initial = codec.decode_initial(prepared.initial)
    refined = codec.decode_refinement(initial, prepared.refinement)
    for decoded in (initial, refined):
        lo, hi = codec.cosine_bounds(decoded, codec.exact_direction(row))
        assert lo <= 1 <= hi
    assert codec.decode_raw(prepared.raw).tobytes() == row.tobytes()


def test_inclusive_boundary_and_negative_interval_remain_conservative():
    one = codec.exact_direction(np.array([1, 0], np.float32))
    neg = codec.exact_direction(np.array([-1, 0], np.float32))
    orthogonal = codec.exact_direction(np.array([0, 1], np.float32))
    assert codec.certify_pair(one, one, 1.0, 1.0) is None
    assert codec.certify_pair(one, neg, -1.0, -1.0) is None
    assert codec.certify_pair(one, orthogonal, 0.0, 0.0) is None
    assert codec.certify_pair(one, neg, 0.997, 1.0) == 1
    assert codec.certify_pair(one, neg, -1.0, -0.5) == 0
    assert codec.certify_pair(one, one, 0.997, 1.0) == 0
    assert codec.certify_pair(one, orthogonal, -1.0, 1.0) == 0


@pytest.mark.parametrize("bad", [
    np.zeros(10, np.float32), np.array([np.nan], np.float32), np.array([np.inf], np.float32),
    np.array([1.0], np.float64), np.ones((1, 10), np.float32), np.ones(256, np.float32),
    np.array([], np.float32),
])
def test_invalid_source_rejected(bad):
    with pytest.raises(ValueError):
        codec.encode_descriptor(bad)
    with pytest.raises(ValueError):
        codec.exact_direction(bad)


@pytest.mark.parametrize("kind", ["initial", "refinement", "raw"])
def test_truncation_trailing_bytes_bad_mode_or_dimension_rejected(kind):
    row = np.array([1, 2, 3], np.float32)
    packets = codec.encode_descriptor(row)
    packet = getattr(packets, kind)
    initial = codec.decode_initial(packets.initial)
    decode = {
        "initial": codec.decode_initial,
        "refinement": lambda p: codec.decode_refinement(initial, p),
        "raw": codec.decode_raw,
    }[kind]
    invalid = [b"", packet[:1], packet[:-1], packet + b"x",
               bytes((255, packet[1])) + packet[2:], bytes((packet[0], 0)) + packet[2:]]
    for bad in invalid:
        with pytest.raises(ValueError):
            decode(bad)


@pytest.mark.parametrize("scale", [0.0, -1.0, math.nan, math.inf, 1.0])
def test_bad_initial_scale_rejected(scale):
    prepared = codec.encode_descriptor(np.array([1, 2, 3], np.float32))
    bad = prepared.initial[:2] + struct.pack("<f", scale) + prepared.initial[6:]
    with pytest.raises(ValueError):
        codec.decode_initial(bad)


def test_low_plane_wrong_stage_dimension_and_invalid_signed_endpoint_rejected():
    prepared = codec.encode_descriptor(np.array([-1, 0.2], np.float32))
    initial = codec.decode_initial(prepared.initial)
    refined = codec.decode_refinement(initial, prepared.refinement)
    with pytest.raises(ValueError):
        codec.decode_refinement(refined, prepared.refinement)
    with pytest.raises(ValueError):
        codec.decode_refinement(codec.exact_direction(np.array([-1, .2], np.float32)), prepared.refinement)
    with pytest.raises(ValueError):
        codec.decode_refinement(initial, bytes((codec.REFINEMENT_MODE, 1, 0)))
    # A high byte of -128 followed by 0 is forbidden q16=-32768.
    bad_low = bytearray(prepared.refinement)
    bad_low[2] = 0
    with pytest.raises(ValueError):
        codec.decode_refinement(initial, bad_low)


def test_raw_invalid_float_rejected_and_high_plane_domain_checked():
    for value in [0.0, math.nan, math.inf]:
        with pytest.raises(ValueError):
            codec.decode_raw(bytes((codec.RAW_MODE, 1)) + struct.pack("<f", value))
    prepared = codec.encode_descriptor(np.array([1, 2, 3], np.float32))
    with pytest.raises(ValueError):
        codec.decode_initial(prepared.initial[:6] + b"\x00\x00\x00")


@pytest.mark.parametrize("bounds", [(1, -1), (math.nan, 1), (-1, math.inf), (-2, 1), (-1, 2)])
def test_invalid_tau_rejected(bounds):
    row = codec.exact_direction(np.array([1, 2], np.float32))
    with pytest.raises(ValueError):
        codec.certify_pair(row, row, *bounds)


def test_mismatched_dimensions_and_invalid_decoded_direction_rejected():
    with pytest.raises(ValueError):
        codec.cosine_bounds(codec.exact_direction(np.ones(2, np.float32)),
                            codec.exact_direction(np.ones(3, np.float32)))
    with pytest.raises(ValueError):
        codec.DecodedDescriptor(np.ones(2), 0, 2, "exact")
    with pytest.raises(ValueError):
        codec.DecodedDescriptor(np.array([1, 0]), -1, 2, "exact")
    with pytest.raises(ValueError):
        codec.DecodedDescriptor(np.array([1, 0]), 0, 2, "initial")
