"""Fixed CPU scalar checks of state reconstruction; no model or training.

Separate FP32 rounding defines the illustrative arithmetic here. This does
not establish Ascend fused-kernel behavior or real model entropy.
"""
import json
from pathlib import Path

import numpy as np


def bits(x):
    return int(np.asarray(x, dtype=np.float32).view(np.uint32))


def desc(x):
    return {"value": float(x), "bits": f"0x{bits(x):08x}"}


def first_step(g):
    # Adam t=1, m0=v0=0, beta1=.9, beta2=.999, no weight decay.
    g = np.float32(g)
    m = np.multiply(np.float32(.1), g, dtype=np.float32)
    v = np.multiply(np.float32(.001), np.multiply(g, g, dtype=np.float32), dtype=np.float32)
    mhat = np.divide(m, np.float32(.1), dtype=np.float32)
    vhat = np.divide(v, np.float32(.001), dtype=np.float32)
    denominator = np.add(np.sqrt(vhat, dtype=np.float32), np.float32(1e-8), dtype=np.float32)
    update = np.multiply(np.float32(1e-4), np.divide(mhat, denominator, dtype=np.float32), dtype=np.float32)
    w = np.subtract(np.float32(1), update, dtype=np.float32)
    return {"g": g, "m": m, "v": v, "w": w, "update": update}


def describe_case(c):
    return {k: desc(v) for k, v in c.items()}


# Same initial and final weight and same v, but different m.
pos, neg = first_step(1e-13), first_step(-1e-13)
assert bits(pos["w"]) == bits(neg["w"])
assert bits(pos["v"]) == bits(neg["v"])
assert bits(pos["m"]) != bits(neg["m"])

# Search only fixed neighboring scalar bitstrings, not gradients of a model.
# The .1 multiply can map adjacent gradients to the same m; v can differ.
base = bits(np.float32(1e-13))
seen = {}
collision = None
for word in range(base, base + 4096):
    g = np.asarray(word, dtype=np.uint32).view(np.float32)
    current = first_step(g)
    key = (bits(current["w"]), bits(current["m"]))
    if key in seen and bits(seen[key]["v"]) != bits(current["v"]):
        collision = [seen[key], current]
        break
    seen[key] = current
assert collision is not None

# Even if an update is separately known, rounded division loses information.
seen = {}
ratio_collision = None
base_v = bits(np.float32(.01))
for word in range(base_v, base_v + 4096):
    v = np.asarray(word, dtype=np.uint32).view(np.float32)
    denominator = np.add(np.sqrt(v, dtype=np.float32), np.float32(1e-8), dtype=np.float32)
    u = np.divide(np.float32(.1), denominator, dtype=np.float32)
    key = bits(u)
    if key in seen and bits(seen[key]["v"]) != bits(v):
        ratio_collision = [seen[key], {"m": np.float32(.1), "v": v, "u": u}]
        break
    seen[key] = {"m": np.float32(.1), "v": v, "u": u}
assert ratio_collision is not None

output = {
    "scope": "CPU fixed scalar format checks; no model, autograd, NPU, or optimizer training run",
    "arithmetic": "Explicit separate FP32 operations, lr=1e-4, eps=1e-8; not an Ascend kernel equivalence test",
    "same_w_and_v_different_m_after_first_step": [describe_case(pos), describe_case(neg)],
    "same_w_and_m_different_v_after_first_step": [describe_case(c) for c in collision],
    "same_m_and_normalized_ratio_different_v": [describe_case(c) for c in ratio_collision],
    "conclusions": [
        "No universally correct decoder for m from (initial w, current w, v, step metadata) under this arithmetic: a first-step collision exists.",
        "No universally correct decoder for v from (initial w, current w, m, step metadata) under this arithmetic: a first-step collision exists.",
        "Even separately retaining a normalized ratio need not determine v exactly with m known in FP32.",
        "These counterexamples rule out those specified decoders, not every possible joint encoding or recomputation scheme.",
        "No estimate of required correction bits or compression ratio on real model states is supplied.",
    ],
}
Path(__file__).with_suffix(".json").write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
print(json.dumps(output, indent=2))
