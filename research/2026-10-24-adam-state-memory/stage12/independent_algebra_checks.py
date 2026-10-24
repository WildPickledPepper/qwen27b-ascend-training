"""Tiny exact-rational checks; no BLAS, training, dependency, or network needed.

FP32 operations use integer round-to-nearest, ties-to-even, not an FP64
approximation of a long sum. Float conversion is used only to print results.
"""
from fractions import Fraction as Q
from pathlib import Path
import json
import struct


def value(word):
    sign = -1 if word >> 31 else 1
    exponent, mantissa = (word >> 23) & 255, word & 0x7FFFFF
    assert exponent != 255
    if exponent:
        mantissa |= 1 << 23
        power = exponent - 127 - 23
    else:
        power = -149
    return sign * Q(mantissa) * Q(2) ** power


def rounded_word(x):
    """Exact finite real -> binary32, RN-even; preserve sign except exact zero."""
    x = Q(x)
    sign = 0x80000000 if x < 0 else 0
    x = abs(x)
    if not x:
        return sign
    n, d = x.numerator, x.denominator
    exponent = n.bit_length() - d.bit_length()
    if x < Q(2) ** exponent:
        exponent -= 1
    quantum = max(exponent - 23, -149)
    y = x / Q(2) ** quantum
    whole, remainder = divmod(y.numerator, y.denominator)
    if 2 * remainder > y.denominator or (
        2 * remainder == y.denominator and whole & 1
    ):
        whole += 1
    if not whole:
        return sign
    if quantum == -149 and whole < 1 << 23:
        return sign | whole
    if whole >= 1 << 24:
        whole >>= 1
        quantum += 1
    exponent = quantum + 23
    if exponent > 127:
        return sign | 0x7F800000
    return sign | ((exponent + 127) << 23) | (whole - (1 << 23))


def fp(x):
    return value(rounded_word(x))


def scalar_report(x):
    word = rounded_word(x)
    return {"word": f"0x{word:08x}", "decimal": float(value(word)),
            "exact": str(value(word))}


def add(a, b):
    return fp(a + b)


def mul(a, b):
    return fp(a * b)


def matprod(a, b):
    """a @ b.T over exact rationals."""
    return [[sum((x * y for x, y in zip(rowa, rowb)), Q(0))
             for rowb in b] for rowa in a]


def matrix_add(a, b):
    return [[x + y for x, y in zip(ra, rb)] for ra, rb in zip(a, b)]


def scale(a, c):
    return [[c * x for x in row] for row in a]


def square(a):
    return [[x * x for x in row] for row in a]


def concatenate(parts):
    return [sum((part[i] for part in parts), []) for i in range(len(parts[0]))]


def lift(a, weighted=False):
    r = len(a[0])
    return [[row[k] * row[l] * (2 if weighted and k != l else 1)
             for k in range(r) for l in range(k, r)] for row in a]


def real_factor_checks(b1, b2, o1, o2):
    aa, bb = [], []
    for t in range(3):
        aa.append([[Q(((i + 1) * (k + 2) + t) % 9 - 4, 4)
                    for k in range(2)] for i in range(3)])
        bb.append([[Q(((j + 2) * (k + 1) + 2 * t) % 7 - 3, 8)
                    for k in range(2)] for j in range(4)])
    m = [[Q(0) for _ in range(4)] for _ in range(3)]
    v = [[Q(0) for _ in range(4)] for _ in range(3)]
    for a, b in zip(aa, bb):
        g = matprod(a, b)
        assert square(g) == matprod(lift(a), lift(b, weighted=True))
        m = matrix_add(scale(m, b1), scale(g, o1))
        v = matrix_add(scale(v, b2), scale(square(g), o2))
    um = concatenate([scale(a, o1 * b1 ** (2 - t)) for t, a in enumerate(aa)])
    vm = concatenate(bb)
    uv = concatenate([scale(lift(a), o2 * b2 ** (2 - t))
                      for t, a in enumerate(aa)])
    vv = concatenate([lift(b, weighted=True) for b in bb])
    assert m == matprod(um, vm)
    assert v == matprod(uv, vv)
    return {"shape": [3, 4], "steps": 3, "token_columns_per_step": 2,
            "m_concat_width": 6, "v_symmetric_lift_width": 9,
            "exact_identity_checks_passed": True,
            "arithmetic": "Fraction, using exact stored FP32 beta and injection constants"}


def main():
    # Verify edge/tie behavior of the independent rounder before using it.
    assert rounded_word(Q(1) + Q(1, 2 ** 24)) == 0x3F800000
    assert rounded_word(Q(1) + Q(3, 2 ** 24)) == 0x3F800002
    assert rounded_word(Q(1, 2 ** 150)) == 0
    assert rounded_word(Q(3, 2 ** 150)) == 2
    for w in [0, 1, 0x007FFFFF, 0x00800000, 0x3F800000, 0xBF800000,
              0x7F7FFFFF, 0x3F666666, 0x3F7FBE77, 0x3DCCCCCD, 0x3A83126F]:
        assert rounded_word(value(w)) == w

    b1, b2 = value(0x3F666666), value(0x3F7FBE77)
    o1, o2 = value(0x3DCCCCCD), value(0x3A83126F)
    constants = {}
    for label, beta, injection in [("first_moment", b1, o1), ("second_moment", b2, o2)]:
        constants[label] = {"beta": scalar_report(beta),
                            "injection": scalar_report(injection),
                            "one_minus_stored_beta": scalar_report(1 - beta),
                            "exact_beta_plus_injection_minus_one": str(beta + injection - 1)}
        assert injection != 1 - beta

    m, v, real_m, real_v = Q(0), Q(0), Q(0), Q(0)
    trace = []
    # Integer inputs have exact FP32 squares, isolating moment-operation rounding.
    for g in map(Q, [1, 2, 3, 4]):
        m = add(mul(b1, m), mul(o1, g))
        v = add(mul(b2, v), mul(o2, mul(g, g)))
        real_m = b1 * real_m + o1 * g
        real_v = b2 * real_v + o2 * g * g
        trace.append({"gradient": int(g), "sequential_m": scalar_report(m),
                      "consolidated_m_once_rounded": scalar_report(real_m),
                      "sequential_v": scalar_report(v),
                      "consolidated_v_once_rounded": scalar_report(real_v)})
    assert rounded_word(m) == 0x3F67A786
    assert rounded_word(real_m) == 0x3F67A787
    assert rounded_word(v) == 0x3CF598A2
    assert rounded_word(real_v) == 0x3CF598A3

    # Rank-one exact-real outer product becomes exact-rank two after FP32 rounding.
    epsilon = Q(1, 2 ** 23)
    a = [Q(1), Q(1) + epsilon]
    exact = [[x * y for y in a] for x in a]
    rounded = [[fp(x) for x in row] for row in exact]
    determinant = lambda x: x[0][0] * x[1][1] - x[0][1] * x[1][0]
    assert determinant(exact) == 0
    assert determinant(rounded) == -Q(1, 2 ** 46)
    rank_example = {"factors": [scalar_report(x) for x in a],
                    "exact_product_rank": 1, "rounded_product_exact_rank": 2,
                    "rounded_product_determinant": str(determinant(rounded)),
                    "rounded_product_words": [[scalar_report(x)["word"] for x in row]
                                               for row in rounded]}
    result = {"reference": "Stage11 separate FP32 mul/add, no FMA; exact stored constants",
              "constants": constants,
              "real_algebra": real_factor_checks(b1, b2, o1, o2),
              "moment_rounding_counterexample": trace,
              "rounded_rank_counterexample": rank_example,
              "intra_step_cross_term_counterexample": {
                  "two_scalar_token_contributions": [1, 1],
                  "square_of_aggregated_gradient": 4,
                  "sum_of_token_squares_wrong": 2},
              "cross_step_counterexample": {
                  "two_steps_unit_gradient_weighted_sum_of_squares_unit_weights": 2,
                  "square_of_sum_wrong": 4},
              "all_assertions_passed": True}
    path = Path(__file__).with_name("independent_algebra_checks.json")
    path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
