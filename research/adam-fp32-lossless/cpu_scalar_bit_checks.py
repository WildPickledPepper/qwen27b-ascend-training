"""CPU-only scalar arithmetic checks; no model, autodiff, optimizer, or NPU.

The fixed inputs below are exactly representable in BF16. This is a
counterexample to fixed zero low bits, NOT a measurement of real Adam entropy.
NumPy's explicitly rounded operations do not model a fused Ascend kernel.
"""
import json
from pathlib import Path

import numpy as np


def bits(value):
    return int(np.asarray(value, dtype=np.float32).view(np.uint32))


def describe(value):
    word = bits(value)
    return {
        "value": float(value),
        "uint32_hex": f"0x{word:08x}",
        "sign": word >> 31,
        "exponent": (word >> 23) & 255,
        "fraction": word & 0x7fffff,
        "low16_hex": f"0x{word & 0xffff:04x}",
    }


cases = []
for gradient in [0.0, 0.25, 0.5, 1.0, 2.0, -1.0]:
    g = np.float32(gradient)
    assert bits(g) & 0xffff == 0
    # m0=v0=0: single application of the defining arithmetic, no training.
    m = np.multiply(np.float32(0.1), g, dtype=np.float32)
    square = np.multiply(g, g, dtype=np.float32)
    v = np.multiply(np.float32(0.001), square, dtype=np.float32)
    cases.append({"gradient": describe(g), "m": describe(m), "v": describe(v)})

assert any(int(c["m"]["low16_hex"], 16) != 0 for c in cases)
assert any(int(c["v"]["low16_hex"], 16) != 0 for c in cases)

# Demonstrate the restricted shared-space case from the discussion.
a = np.float32(1.5)  # exactly expanded BF16 value; low 16 bits are zero
b = np.float16(-0.75)
word_a = bits(a)
word_b = int(np.asarray(b).view(np.uint16))
assert word_a & 0xffff == 0
packed = word_a | word_b
restored_a = packed & 0xffff0000
restored_b = packed & 0xffff
assert restored_a == word_a and restored_b == word_b

parameter_count = 26_895_998_464
states_gib = parameter_count * 8 / 2**30

# Integer-only BF16 round-to-nearest, ties-to-even illustration for finite FP32.
# This is a format check, not validation of any accelerator's conversion kernel.
def finite_rne_bf16(word):
    assert ((word >> 23) & 255) != 255
    return ((word + 0x7fff + ((word >> 16) & 1)) >> 16) & 0xffff


rounding_cases = []
for word in [0x3f7f8000, 0x3f808000, 0x3f800001, 0xbf7f8000]:
    high, low = word >> 16, word & 0xffff
    carry = int(low > 0x8000 or (low == 0x8000 and (high & 1)))
    bf16_word = finite_rne_bf16(word)
    restored = (((bf16_word - carry) & 0xffff) << 16) | low
    assert restored == word
    rounding_cases.append({
        "fp32_bits": f"0x{word:08x}",
        "bf16_rne_bits": f"0x{bf16_word:04x}",
        "low16_bits": f"0x{low:04x}",
        "carry": carry,
        "roundtrip_with_carry": True,
    })
assert rounding_cases[0]["bf16_rne_bits"] == rounding_cases[1]["bf16_rne_bits"]
assert rounding_cases[0]["low16_bits"] == rounding_cases[1]["low16_bits"]
bounds = []
for label, pair_bits in [
    ("raw_pair", 64),
    ("only_nonnegative_v_sign_removed_ideal", 63),
    ("two_exponents_4_bits_each_plus_unchanged_47_bits_ideal", 55),
    ("joint_exponents_4_bits_total_plus_unchanged_47_bits_ideal", 51),
    ("both_exponents_free_plus_unchanged_47_bits_ideal", 47),
]:
    bounds.append({
        "scenario": label,
        "bits_per_pair": pair_bits,
        "raw_over_encoded": 64 / pair_bits,
        "state_gib": parameter_count * pair_bits / 8 / 2**30,
        "note": "Hypothetical accounting; excludes headers, exceptions, padding, and workspace.",
    })

output = {
    "scope": "CPU-only fixed scalar arithmetic; no model or real optimizer state loaded",
    "numpy_version": np.__version__,
    "arithmetic": "FP32 multiply with separately rounded square for v; not an NPU equivalence test",
    "fixed_cases": cases,
    "restricted_bf16_plus_fp16_packing": {
        "a_bits": f"0x{word_a:08x}",
        "b_bits": f"0x{word_b:04x}",
        "packed_bits": f"0x{packed:08x}",
        "bitwise_roundtrip": True,
        "condition": "A must have zero low 16 bits before packing",
    },
    "parameter_count_from_repository": parameter_count,
    "master_bf16_rne_fixed_cases": rounding_cases,
    "master_bf16_rne_observation": "BF16 RNE plus raw low16 alone is ambiguous at ties; the finite cases roundtrip with a carry bit.",
    "weight_accounting_gib": {
        "bf16_text_weights": parameter_count * 2 / 2**30,
        "optional_fp32_text_master": parameter_count * 4 / 2**30,
        "bf16_plus_fp32_text_weights": parameter_count * 6 / 2**30,
        "ideal_packed_33bit_finite_weight_tuple": parameter_count * 33 / 8 / 2**30,
    },
    "raw_two_fp32_state_gib": states_gib,
    "state_target_60_gib": {
        "required_ratio_before_overhead": states_gib / 60,
        "required_bits_per_pair_before_overhead": 60 * 2**30 * 8 / parameter_count,
    },
    "conditional_encoding_class_bounds": bounds,
    "limitations": [
        "No actual Qwen m/v or checkpoint was available locally.",
        "No compression ratio, entropy, speed, or convergence measurement of real training.",
        "A first-step example does not characterize long-run state statistics.",
    ],
}
destination = Path(__file__).with_name("cpu_scalar_bit_checks.json")
destination.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
print(json.dumps(output, indent=2))
