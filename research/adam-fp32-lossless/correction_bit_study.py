"""Offline CPU arithmetic and exact correction-bit counts. No training.

All synthetic inputs are declared below. Favorable side information is
explicitly marked as FREE for an experiment, not omitted from a real budget.
"""
import json
import math
from pathlib import Path

import numpy as np

F = np.float32
N = 26_895_998_464
GIB = 2**30


def word(x):
    return int(np.asarray(x, dtype=np.float32).view(np.uint32))


def f32(u):
    return np.asarray(u, dtype=np.uint32).view(np.float32)


def ratio(m, v, t):
    mh = np.divide(m, F(1 - .9**t), dtype=np.float32)
    vh = np.divide(v, F(1 - .999**t), dtype=np.float32)
    den = np.add(np.sqrt(vh, dtype=np.float32), F(1e-8), dtype=np.float32)
    return np.divide(mh, den, dtype=np.float32)


def weight_update(w, m, v, t=1000, lr=1e-4):
    delta = np.multiply(F(lr), ratio(m, v, t), dtype=np.float32)
    return np.subtract(w, delta, dtype=np.float32)


def inverse_m_bounds(wprev, wnew, v, t=1000, lr=1e-4):
    """Exact preimage interval over nonnegative finite FP32 m bitstrings.

    wprev, wnew, v and arithmetic metadata are GIVEN to this decoder.
    Positive-m-only domain favors compression; no sign coding is charged.
    """
    shape = np.broadcast_shapes(np.shape(wprev), np.shape(wnew), np.shape(v))
    wprev, wnew, v = [np.broadcast_to(F(x), shape) for x in (wprev, wnew, v)]
    ends = []
    with np.errstate(over="ignore", invalid="raise", divide="raise"):
        for strict in (False, True):
            lo = np.zeros(shape, dtype=np.uint64)
            hi = np.full(shape, 0x7f800000, dtype=np.uint64)
            for _ in range(32):
                mid = (lo + hi) // 2
                mapped = weight_update(wprev, f32(mid), v, t, lr)
                pred = mapped < wnew if strict else mapped <= wnew
                hi = np.where(pred, mid, hi)
                lo = np.where(pred, lo, mid + 1)
            assert np.all(lo == hi)
            ends.append(lo)
    return ends[0], ends[1]


def pack_ranks(ranks, widths):
    output = bytearray()
    buffer = occupied = 0
    for r, width in zip(ranks, widths):
        r, width = int(r), int(width)
        assert 0 <= r < (1 << width) if width else r == 0
        buffer |= r << occupied
        occupied += width
        while occupied >= 8:
            output.append(buffer & 255)
            buffer >>= 8
            occupied -= 8
    if occupied:
        output.append(buffer)
    return bytes(output)


def unpack_ranks(payload, widths):
    cursor = buffer = occupied = 0
    values = []
    for width in widths:
        width = int(width)
        while occupied < width:
            buffer |= payload[cursor] << occupied
            cursor += 1
            occupied += 8
        values.append(buffer & ((1 << width) - 1))
        buffer >>= width
        occupied -= width
    return np.asarray(values, dtype=np.uint64)


records = []
for w in (.001, .01, .1, 1.):
    for lr in (1e-3, 1e-4, 1e-5):
        m, v = F(.001), F(.0001)
        new = weight_update(F(w), m, v, lr=lr)
        low, high = inverse_m_bounds(F(w), new, v, lr=lr)
        count = int(high - low)
        index = word(m) - int(low)
        width = (count - 1).bit_length()
        assert int(low) + index == word(m) and 0 <= index < count
        # Check both sides of the exact interval and its immediate neighbors.
        assert word(weight_update(F(w), f32(low), v, lr=lr)) == word(new)
        assert word(weight_update(F(w), f32(high - 1), v, lr=lr)) == word(new)
        if int(low):
            assert weight_update(F(w), f32(low - 1), v, lr=lr) > new
        if int(high) < 0x7f800000:
            assert weight_update(F(w), f32(high), v, lr=lr) < new
        records.append({
            "previous_w": float(F(w)), "learning_rate": lr,
            "next_w": float(new), "v_given": float(v), "m": float(m),
            "candidate_m_count": count, "correction_rank_bits": width,
            "log2_candidate_count": math.log2(count),
            "m_correction_gib_if_same_width_every_parameter": N * width / 8 / GIB,
            "plus_uncompressed_v_gib": N * (width + 32) / 8 / GIB,
            "side_information": "previous_w, next_w, v, step=1000 and hyperparameters GIVEN FREE for this check",
        })

zero_m, zero_v = F(1e-14), F(1e-29)
zero_new = weight_update(F(1), zero_m, zero_v)
zero_low, zero_high = inverse_m_bounds(F(1), zero_new, zero_v)
zero_count = int(zero_high - zero_low)
assert zero_new == F(1) and int(zero_low) <= word(zero_m) < int(zero_high)
zero_update_example = {
    "previous_and_next_w": 1., "m": float(zero_m), "v_given": float(zero_v),
    "candidate_nonnegative_finite_m_count": zero_count,
    "fixed_width_correction_bits": (zero_count - 1).bit_length(),
    "note": "Counts all nonnegative finite FP32 candidates under this decoder, without extra reachability constraints; not a distributional bound.",
}

# Arbitrary, non-model synthetic snapshots to test a variable-length rank codec.
rng = np.random.default_rng(20261023)
size = 4096
w = np.power(10., rng.uniform(-4, 0, size)).astype(np.float32)
v = np.power(10., rng.uniform(-12, -2, size)).astype(np.float32)
m = (np.sqrt(v) * rng.uniform(.001, .5, size)).astype(np.float32)
new = weight_update(w, m, v)
low, high = inverse_m_bounds(w, new, v)
count = high - low
assert np.all(count > 0)
widths = np.asarray([(int(c) - 1).bit_length() for c in count], dtype=np.uint8)
ranks = m.view(np.uint32).astype(np.uint64) - low
encoded = pack_ranks(ranks, widths)
decoded = (low + unpack_ranks(encoded, widths)).astype(np.uint32)
assert np.array_equal(decoded, m.view(np.uint32))
assert np.array_equal(weight_update(w, decoded.view(np.float32), v).view(np.uint32), new.view(np.uint32))

# Construct reachable states by evaluating just two scalar EMA recurrences on
# a fixed BF16 input grid. No loss, model, forward pass, or training is run.
bf16_words = np.array([
    sign | (exponent << 7) | fraction
    for sign in (0, 0x8000)
    for exponent in range(127 - 46, 127 - 42)
    for fraction in range(128)
], dtype=np.uint32)
grid = (bf16_words << 16).view(np.float32)
assert len(grid) == 1024
g1, g2 = np.repeat(grid, len(grid)), np.tile(grid, len(grid))
m1 = np.multiply(F(.1), g1, dtype=np.float32)
v1 = np.multiply(F(.001), np.multiply(g1, g1, dtype=np.float32), dtype=np.float32)
w1 = weight_update(F(1), m1, v1, t=1)
m2 = np.add(np.multiply(F(.9), m1, dtype=np.float32), np.multiply(F(.1), g2, dtype=np.float32), dtype=np.float32)
v2 = np.add(np.multiply(F(.999), v1, dtype=np.float32), np.multiply(F(.001), np.multiply(g2, g2, dtype=np.float32), dtype=np.float32), dtype=np.float32)
w2 = weight_update(w1, m2, v2, t=2)
assert np.all(w1 == F(1)) and np.all(w2 == F(1))
pairs = (m2.view(np.uint32).astype(np.uint64) << 32) | v2.view(np.uint32).astype(np.uint64)
unique_pairs = np.unique(pairs)
log2k = math.log2(len(unique_pairs))

output = {
    "scope": "Offline CPU scalar/array arithmetic only; synthetic inputs, not model training or real optimizer entropy",
    "parameter_count": N,
    "budget": {"gib": 30, "bits_per_m_v_pair": 30 * GIB * 8 / N, "raw_m_v_gib": N * 8 / GIB},
    "inverse_m_given_v_and_adjacent_weights": records,
    "zero_weight_update_example": zero_update_example,
    "synthetic_rank_codec": {
        "seed": 20261023, "elements": size,
        "w_log10_range": [-4, 0], "v_log10_range": [-12, -2], "m_over_sqrt_v_range": [.001, .5],
        "rank_payload_bytes": len(encoded),
        "rank_bits_per_parameter_including_final_byte_padding": len(encoded) * 8 / size,
        "mean_rank_width_bits": float(widths.mean()), "max_rank_width_bits": int(widths.max()),
        "rank_plus_raw_v_bits_per_parameter": len(encoded) * 8 / size + 32,
        "bitwise_roundtrip": True,
        "not_charged": ["previous weights", "v storage for the m-only rank payload", "metadata and index tables"],
    },
    "same_current_weight_reachable_state_family": {
        "gradient_source": "1024 fixed BF16-exact scalar values; all ordered pairs enumerated",
        "gradient_history_combinations": len(pairs), "distinct_m_v_pairs": len(unique_pairs),
        "w0_w1_w2": 1.0, "step": 2,
        "uniform_family_entropy_bits_per_pair": log2k,
        "uniform_family_ideal_gib_at_target_parameter_count": N * log2k / 8 / GIB,
        "minimum_fixed_width_bits_per_pair": (len(unique_pairs) - 1).bit_length(),
        "claim_scope": "A current-weight-only decoder needs extra distinguishing information. For independently selected states across parameters from this family, a universal 30GiB cap is impossible without other side information/history replay. This is NOT a bound on Qwen's actual training distribution.",
    },
    "limitations": [
        "Separate rounded FP32 operations, not Ascend fused arithmetic.",
        "No Qwen state or training trajectory used.",
        "Rank bits depend on arithmetic, state values and available side information.",
        "No proof that all side-information schemes fail; history replay has different compute/storage costs.",
    ],
}
destination = Path(__file__).with_suffix(".json")
destination.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
print(json.dumps(output, indent=2), flush=True)
