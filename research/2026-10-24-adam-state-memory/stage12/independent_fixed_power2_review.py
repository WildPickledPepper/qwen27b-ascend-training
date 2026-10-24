"""Focused exact replay of the archived fixed-power-of-two positive control."""
from fractions import Fraction as Q
from pathlib import Path
import json
import struct
from independent_algebra_checks import value, rounded_word, add, mul

root = Path(__file__).resolve().parent
case = json.loads((root / "results/fixed_power2_exception.json").read_text())
b1, b2, c1, c2 = map(value, [0x3F666666, 0x3F7FBE77, 0x3DCCCCCD, 0x3A83126F])
rows, cols = list(range(-3, 5)), list(range(-4, 4))
exponents = [i + j for i in rows for j in cols]
m, v = [Q(0)] * 64, [Q(0)] * 64
sm, sv = Q(0), Q(0)
min_exponent, max_exponent, nonzero_checked = 127, -126, 0


def check_range(x):
    global min_exponent, max_exponent, nonzero_checked
    w = rounded_word(x)
    if w & 0x7FFFFFFF:
        e = (w >> 23) & 255
        assert 0 < e < 255, "Nonzero intermediate is subnormal or nonfinite"
        min_exponent, max_exponent = min(min_exponent, e - 127), max(max_exponent, e - 127)
        nonzero_checked += 1
    return x


def step(a, b, g):
    check_range(g)
    bm = check_range(mul(b1, a))
    cg = check_range(mul(c1, g))
    bv = check_range(mul(b2, b))
    gg = check_range(mul(g, g))
    cgg = check_range(mul(c2, gg))
    return check_range(add(bm, cg)), check_range(add(bv, cgg))


for expected, record in enumerate(case["trajectory"], 1):
    assert record["step"] == expected
    c = value(record["input_scalar_word"])
    sm, sv = step(sm, sv, c)
    assert rounded_word(sm) == record["m_scalar_word"]
    assert rounded_word(sv) == record["v_scalar_word"]
    for i, exponent in enumerate(exponents):
        g = c * Q(2) ** exponent
        assert value(rounded_word(g)) == g
        m[i], v[i] = step(m[i], v[i], g)
        assert m[i] == sm * Q(2) ** exponent
        assert v[i] == sv * Q(2) ** (2 * exponent)
final = (root / "results/fixed_power2.final_m_v.f32").read_bytes()
assert list(struct.unpack("<128I", final)) == [rounded_word(x) for x in m + v]
assert case["words_compared"] == 64 * 64 * 2 == 8192
assert case["persistent_representation_and_coefficients_bytes"] == 8 + 16 + 16 + 16 == 56
assert case["raw_dense_m_v_bytes"] == 64 * 2 * 4 == 512
report = {"all_checks_passed": True, "steps": 64, "moment_words_verified": 8192,
          "archived_final_words_verified": 128,
          "scalar_state_words_verified": 128,
          "nonzero_input_intermediate_output_values_checked_normal_and_finite": nonzero_checked,
          "nonzero_values_min_unbiased_binary32_exponent": min_exponent,
          "nonzero_values_max_unbiased_binary32_exponent": max_exponent,
          "proposed_persistent_payload_bytes": 56,
          "dense_moment_payload_bytes": 512, "payload_ratio": 56 / 512,
          "starts_from_zero_moments": True,
          "scope": "Archived synthetic fixed-pattern moment recurrence only. No main grid rerun, full training, runtime peak, or arbitrary-gradient claim."}
(root / "independent_fixed_power2_review.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))
