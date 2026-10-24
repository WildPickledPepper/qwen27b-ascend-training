#!/usr/bin/env python3
"""Reproduce a source-derived, *analytical* Qwen factor-history budget.

No torch import, model weights, package installation, device execution, or network
access is required. Archived official source is read as data, not executed.
This script does not prove a lossless Qwen gradient-replay implementation.
"""
from pathlib import Path
from collections import Counter
import csv
import hashlib
import json
import math

ROOT = Path(__file__).resolve().parent
SOURCES = ROOT / "qwen_audit_sources"
CONFIG = json.loads((SOURCES / "qwen3_5_27b_config.json").read_text())
C = CONFIG["text_config"]
GIB = 2**30
H, I, V, L = (C[k] for k in ("hidden_size", "intermediate_size", "vocab_size", "num_hidden_layers"))
COUNTS = Counter(C["layer_types"])
NF, ND = COUNTS["full_attention"], COUNTS["linear_attention"]
Q = C["num_attention_heads"] * C["head_dim"]
K = C["num_key_value_heads"] * C["head_dim"]
DK = C["linear_num_key_heads"] * C["linear_key_head_dim"]
DV = C["linear_num_value_heads"] * C["linear_value_head_dim"]
NV = C["linear_num_value_heads"]
assert not CONFIG["tie_word_embeddings"]
assert not C["attention_bias"]
assert C["attn_output_gate"]
assert NF + ND == L == len(C["layer_types"])

# One entry per trainable text tensor. Vision, MTP, and LoRA are not included.
inventory = []
def add(name, shape, group):
    inventory.append(dict(name=name, shape=list(shape), numel=math.prod(shape), group=group))

add("embed_tokens.weight", (V, H), "embedding_and_lm_head")
add("lm_head.weight", (V, H), "embedding_and_lm_head")
add("norm.weight", (H,), "small_final_norm")
for l, layer_type in enumerate(C["layer_types"]):
    prefix = f"layers.{l}."
    for name, shape in (("gate_proj", (I, H)), ("up_proj", (I, H)), ("down_proj", (H, I))):
        add(prefix + "mlp." + name + ".weight", shape, "mlp")
    for name in ("input_layernorm", "post_attention_layernorm"):
        add(prefix + name + ".weight", (H,), "small_block_norm")
    if layer_type == "full_attention":
        for name, shape in (("q_proj", (2 * Q, H)), ("k_proj", (K, H)), ("v_proj", (K, H)), ("o_proj", (H, Q))):
            add(prefix + "self_attn." + name + ".weight", shape, "full_attention_dense")
        for name in ("q_norm", "k_norm"):
            add(prefix + "self_attn." + name + ".weight", (C["head_dim"],), "small_qk_norm")
    else:
        for name, shape in (("in_proj_qkv", (2 * DK + DV, H)), ("in_proj_z", (DV, H)),
                            ("in_proj_b", (NV, H)), ("in_proj_a", (NV, H)), ("out_proj", (H, DV))):
            add(prefix + "linear_attn." + name + ".weight", shape, "gated_delta_dense")
        add(prefix + "linear_attn.conv1d.weight", (2 * DK + DV, 1, C["linear_conv_kernel_dim"]), "small_depthwise_conv")
        for name in ("dt_bias", "A_log"):
            add(prefix + "linear_attn." + name, (NV,), "small_delta_dt_A")
        add(prefix + "linear_attn.norm.weight", (C["linear_value_head_dim"],), "small_delta_norm")

groups = Counter()
for row in inventory:
    groups[row["group"]] += row["numel"]
P = sum(groups.values())
SMALL = sum(v for k, v in groups.items() if k.startswith("small_"))

# Store each unique same-input activation once, and each projection's dY.
# Embedding uses a sparse index-plus-row-gradient representation, NOT V inputs.
factor_elements = {
    "mlp_all_layers": L * (2 * H + 3 * I),
    "full_attention_all_layers": NF * (2 * H + 3 * Q + 2 * K),
    "gated_delta_all_layers": ND * (2 * H + 2 * DK + 3 * DV + 2 * NV),
    "lm_head": H + V,
    "embedding_row_derivative": H,
}
E = sum(factor_elements.values())
F = 2 * E + 8
EXTRA_UNSHARED = H * (L + 2 * NF + 3 * ND)
assert P == 26_895_998_464
assert SMALL == 2_645_504
assert F == 12_650_504

SCENARIOS = {
    "bf16_factor_payload_only": dict(factor_bytes_per_token=F, small_gradient_bytes_per_step=0, metadata_bytes_per_step=0, fixed_metadata_bytes=0, workspace_reserve_bytes=0),
    "bf16_plus_small_fp32_gradients": dict(factor_bytes_per_token=F, small_gradient_bytes_per_step=4 * SMALL, metadata_bytes_per_step=0, fixed_metadata_bytes=0, workspace_reserve_bytes=0),
    # Explicit hypothetical reserve, not a measured allocation or proven bound.
    "bf16_plus_small_and_illustrative_reserve": dict(factor_bytes_per_token=F, small_gradient_bytes_per_step=4 * SMALL, metadata_bytes_per_step=16 + 16 * 4, fixed_metadata_bytes=2**20, workspace_reserve_bytes=8 * 2**20),
    "fp32_factors_plus_small": dict(factor_bytes_per_token=4 * E + 8, small_gradient_bytes_per_step=4 * SMALL, metadata_bytes_per_step=0, fixed_metadata_bytes=0, workspace_reserve_bytes=0),
    "bf16_unshared_inputs_plus_small": dict(factor_bytes_per_token=F + 2 * EXTRA_UNSHARED, small_gradient_bytes_per_step=4 * SMALL, metadata_bytes_per_step=0, fixed_metadata_bytes=0, workspace_reserve_bytes=0),
    "bf16_plus_persistent_small_m_v_and_current_gradient": dict(factor_bytes_per_token=F, small_gradient_bytes_per_step=0, metadata_bytes_per_step=0, fixed_metadata_bytes=0, workspace_reserve_bytes=0, fixed_persistent_small_m_v_bytes=8 * SMALL, fixed_current_small_gradient_bytes=4 * SMALL),
}
def fixed_bytes(s):
    return (s["fixed_metadata_bytes"] + s["workspace_reserve_bytes"]
            + s.get("fixed_persistent_small_m_v_bytes", 0) + s.get("fixed_current_small_gradient_bytes", 0))

frontier = []
for n in (96, 128, 256, 512, 1024, 2048):
    for t in (1, 2, 4, 8, 16, 32):
        for name, s in SCENARIOS.items():
            history = t * (n * s["factor_bytes_per_token"] + s["small_gradient_bytes_per_step"] + s["metadata_bytes_per_step"])
            fixed = fixed_bytes(s)
            total = history + fixed
            frontier.append(dict(scenario=name, accumulated_token_slots=n, replay_steps=t,
                                 history_bytes=history, fixed_reserve_bytes=fixed,
                                 analytical_replacement_bytes=total, analytical_replacement_GiB=total / GIB,
                                 fraction_of_raw_fp32_m_v=total / (8 * P),
                                 within_30_GiB=total <= 30 * GIB,
                                 within_60_GiB=total <= 60 * GIB,
                                 smaller_than_raw_fp32_m_v=total < 8 * P))

limits = []
for name, s in SCENARIOS.items():
    for n in (128, 256, 512, 1024):
        per_step = n * s["factor_bytes_per_token"] + s["small_gradient_bytes_per_step"] + s["metadata_bytes_per_step"]
        fixed = fixed_bytes(s)
        limits.append(dict(scenario=name, accumulated_token_slots=n, bytes_per_step=per_step,
                           max_whole_steps_30_GiB=max(0, (30 * GIB - fixed) // per_step),
                           max_whole_steps_60_GiB=max(0, (60 * GIB - fixed) // per_step),
                           max_whole_steps_strictly_below_raw_m_v=max(0, (8 * P - fixed - 1) // per_step)))

source_files = [SOURCES / "qwen3_5_27b_config.json", SOURCES / "modeling_qwen3_5_v5_14_1.py"]
repo = SOURCES / "local_repo_snapshot"
snapshot = SOURCES / "local_repo_snapshot_manifest.json"
if not snapshot.exists():
    repo = ROOT.parents[2] / "references" / "qwen27b-ascend-training"
    snapshot = repo.parent / "qwen27b-ascend-training.snapshot-manifest.json"
snapshot_check = []
snapshot_commit = None
if snapshot.exists():
    snapshot_data = json.loads(snapshot.read_text())
    snapshot_commit = snapshot_data["commit_sha"]
    for entry in snapshot_data["files"]:
        data = (repo / entry["path"]).read_bytes()
        sha = hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()
        assert sha == entry["sha"] and len(data) == entry["size"]
        snapshot_check.append(dict(path=entry["path"], git_blob_sha1=sha, bytes=len(data), verified=True))
# In the lightweight Git edition, the source snapshot is intentionally omitted.
# Public-source budgets can still be reproduced; snapshot parity remains unverified.

result = dict(
    scope="Static source-derived analytical budget; no Qwen replay, full-parameter training, exact-gradient proof, memory peak, runtime, or device claim.",
    source_caveat="Official Qwen3.5 architecture matches the local repository's count, but does not authenticate the separately named Qwen3.8 checkpoint, whose tensor headers/config are absent locally.",
    dimensions=dict(hidden=H, intermediate=I, vocabulary=V, layers=L, full_attention_layers=NF, gated_delta_layers=ND,
                    full_query_width=Q, full_kv_width=K, delta_key_width=DK, delta_value_width=DV, delta_value_heads=NV),
    parameter_groups=dict(groups), parameter_tensors=len(inventory), text_parameters=P,
    non_dense_parameters=SMALL, fp32_non_dense_gradient_bytes_per_step=4 * SMALL,
    raw_fp32_m_v_bytes=8 * P, raw_fp32_m_v_GiB=8 * P / GIB,
    text_bf16_weight_bytes=2 * P, text_bf16_weight_GiB=2 * P / GIB,
    factor_elements_per_token=factor_elements, factor_float_elements_per_token=E,
    token_id_bytes_per_token=8, optimistic_bf16_factor_bytes_per_token=F,
    configured_maximum_single_device_accumulated_token_slots=1 * 4 * 128,
    actual_accumulated_token_count_verified=False,
    full_length_512_case=dict(bytes_per_step=512 * F, GiB_per_step=512 * F / GIB,
                              eight_step_bytes=8 * 512 * F, eight_step_GiB=8 * 512 * F / GIB),
    scenarios=SCENARIOS, frontier=frontier, horizon_limits=limits,
    reserve_caveat="Illustrative reserve uses 16 B per step + 16 B per microbatch for four microbatches, 1 MiB global metadata and 8 MiB workspace. These are explicit assumptions, NOT observed or sufficient-for-production bounds.",
    source_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files},
    local_repo_snapshot_commit=snapshot_commit, snapshot_blob_checks=snapshot_check,
    local_repo_snapshot_verified=bool(snapshot_check),
)
(ROOT / "qwen_audit.json").write_text(json.dumps(result, indent=2) + "\n")
with (ROOT / "qwen_audit.csv").open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(frontier[0])); w.writeheader(); w.writerows(frontier)
with (ROOT / "qwen_audit.parameters.csv").open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(inventory[0])); w.writeheader(); w.writerows(inventory)
print(json.dumps({k: result[k] for k in ("text_parameters", "non_dense_parameters", "optimistic_bf16_factor_bytes_per_token", "full_length_512_case")}, indent=2))
