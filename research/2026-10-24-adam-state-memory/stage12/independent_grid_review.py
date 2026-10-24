"""Independent read-only validation of the main Stage12 grid and rational codec.

Uses no NumPy or main-grid functions. Replays only tiny archived synthetic
gradients and moments, with the separately implemented exact RN32 checker.
"""
from fractions import Fraction as Q
from pathlib import Path
import hashlib
import json
import struct
from independent_algebra_checks import value, rounded_word, add, mul, matprod

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results"
PRIME = 2147483647


def read_matrix(name, n, m):
    raw = (OUT / name).read_bytes()
    assert len(raw) == 4 * n * m
    words = list(struct.unpack("<" + "I" * (n * m), raw))
    return [words[i * m:(i + 1) * m] for i in range(n)]


def fractions(words):
    return [[value(x) for x in row] for row in words]


def words(a):
    return [[rounded_word(x) for x in row] for row in a]


def independent_rank(a):
    a = [row.copy() for row in a]
    nr, nc, k = len(a), len(a[0]), 0
    for j in range(nc):
        pivot = next((i for i in range(k, nr) if a[i][j]), None)
        if pivot is None:
            continue
        a[k], a[pivot] = a[pivot], a[k]
        for i in range(k + 1, nr):
            if not a[i][j]:
                continue
            c = a[i][j] / a[k][j]
            for h in range(j + 1, nc):
                a[i][h] -= c * a[k][h]
            a[i][j] = Q(0)
        k += 1
        if k == nr:
            break
    return k


def rank_mod_prime(a):
    # A nonzero minor modulo p certifies a nonzero rational minor when all
    # denominators are invertible modulo p. It gives a lower bound on Q-rank.
    assert all(x.denominator % PRIME for row in a for x in row)
    a = [[(x.numerator * pow(x.denominator, -1, PRIME)) % PRIME
          for x in row] for row in a]
    nr, nc, k = len(a), len(a[0]), 0
    for j in range(nc):
        pivot = next((i for i in range(k, nr) if a[i][j]), None)
        if pivot is None:
            continue
        a[k], a[pivot] = a[pivot], a[k]
        inv = pow(a[k][j], -1, PRIME)
        for i in range(k + 1, nr):
            c = a[i][j] * inv % PRIME
            for h in range(j, nc):
                a[i][h] = (a[i][h] - c * a[k][h]) % PRIME
        k += 1
        if k == nr:
            break
    return k


def decode_codec(raw):
    n, m, km, kv = struct.unpack("<4I", raw[:16])
    pos = 16

    def uint():
        nonlocal pos
        v, shift = 0, 0
        while True:
            b = raw[pos]
            pos += 1
            v |= (b & 127) << shift
            if not b & 128:
                return v
            shift += 7

    def matrix(nr, nc):
        a = []
        for _ in range(nr):
            row = []
            for _ in range(nc):
                z, d = uint(), uint()
                num = -(z + 1) // 2 if z & 1 else z // 2
                assert d > 0
                row.append(Q(num, d))
            a.append(row)
        return a

    um, vm, uv, vv = matrix(n, km), matrix(m, km), matrix(n, kv), matrix(m, kv)
    assert pos == len(raw)
    return n, m, km, kv, matprod(um, vm), matprod(uv, vv)


def main():
    grid = json.loads((OUT / "grid.json").read_text())
    lookup = {(x["seed"], x["r"], x["steps"]): x for x in grid}
    b1, b2, c1, c2 = map(value, [0x3F666666, 0x3F7FBE77, 0x3DCCCCCD, 0x3A83126F])
    n = 12
    counts = dict(raw_factor_gradient_words_verified=0, sequential_moment_words_verified=0,
                  once_rounded_moment_words_verified=0, rational_codec_files_verified=0,
                  exact_rational_rank_values_verified=0, modular_rank_lower_bounds_matched=0)
    cases = []
    for seed, r in sorted({(x["seed"], x["r"]) for x in grid}):
        m = [[Q(0) for _ in range(n)] for _ in range(n)]
        v = [[Q(0) for _ in range(n)] for _ in range(n)]
        rm = [[Q(0) for _ in range(n)] for _ in range(n)]
        rv = [[Q(0) for _ in range(n)] for _ in range(n)]
        for t in range(1, 9):
            prefix = f"s{seed}_r{r}_t{t}"
            a = fractions(read_matrix(prefix + ".A.f32", n, r))
            b = fractions(read_matrix(prefix + ".B.f32", n, r))
            raw_g = read_matrix(prefix + ".gradient.f32", n, n)
            g = [[Q(0) for _ in range(n)] for _ in range(n)]
            for k in range(r):
                for i in range(n):
                    for j in range(n):
                        g[i][j] = add(g[i][j], mul(a[i][k], b[j][k]))
            assert words(g) == raw_g
            assert g == matprod(a, b)
            assert all(mul(x, x) == x * x for row in g for x in row)
            counts["raw_factor_gradient_words_verified"] += n * n
            for i in range(n):
                for j in range(n):
                    x = g[i][j]
                    m[i][j] = add(mul(b1, m[i][j]), mul(c1, x))
                    v[i][j] = add(mul(b2, v[i][j]), mul(c2, mul(x, x)))
                    rm[i][j] = b1 * rm[i][j] + c1 * x
                    rv[i][j] = b2 * rv[i][j] + c2 * x * x
            assert words(m) == read_matrix(prefix + ".m_seq.f32", n, n)
            assert words(v) == read_matrix(prefix + ".v_seq.f32", n, n)
            counts["sequential_moment_words_verified"] += 2 * n * n
            if (seed, r, t) not in lookup:
                continue
            q = lookup[seed, r, t]
            assert words(rm) == read_matrix(prefix + ".m_real_once.f32", n, n)
            assert words(rv) == read_matrix(prefix + ".v_real_once.f32", n, n)
            counts["once_rounded_moment_words_verified"] += 2 * n * n
            raw = (OUT / (prefix + ".real_m_v_rank_factor.rational")).read_bytes()
            nn, mm, km, kv, decoded_m, decoded_v = decode_codec(raw)
            assert (nn, mm, km, kv) == (n, n, q["m_rank_real"], q["v_rank_real"])
            assert decoded_m == rm and decoded_v == rv
            assert len(raw) == q["actual_rational_codec_bytes"]
            counts["rational_codec_files_verified"] += 1
            ranks = [(g, "gradient_rank_real"), (g, "gradient_rank_fp32_exact"),
                     (rm, "m_rank_real"), (rv, "v_rank_real"),
                     (m, "m_rank_sequential_fp32_exact"), (v, "v_rank_sequential_fp32_exact")]
            for matrix, field in ranks:
                assert independent_rank(matrix) == q[field]
                counts["exact_rational_rank_values_verified"] += 1
                assert rank_mod_prime(matrix) == q[field]
                counts["modular_rank_lower_bounds_matched"] += 1
            for matrix, real, field in [(m, rm, "m_bits"), (v, rv, "v_bits")]:
                count = sum(x != y for ra, rb in zip(words(matrix), words(real))
                            for x, y in zip(ra, rb))
                assert count == q[field]["different_words"]
            cases.append(dict(seed=seed, r=r, steps=t, codec_bytes=len(raw), passed=True))
    result = {"all_checks_passed": True, "cases": len(cases), "counts": counts,
              "main_source_sha256": hashlib.sha256((ROOT / "consolidation.py").read_bytes()).hexdigest(),
              "arithmetic": "Independent integer RN-even binary32 operations; exact Fraction recurrence and Gaussian elimination; modular rank p=2147483647",
              "codec": "Independent unsigned-varint/zigzag numerator and positive-denominator decoder; exact reconstructed moments and full byte consumption checked",
              "scope": "Verifies archived synthetic factor/gradient/moment data and rank/codec claims; not full training, NPU parity, or runtime memory peak",
              "case_results": cases}
    (ROOT / "independent_grid_review.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "case_results"}, indent=2))


if __name__ == "__main__":
    main()
