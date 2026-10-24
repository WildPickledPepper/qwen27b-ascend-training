"""Read existing public optimizer tensors on CPU and test lossless codecs.

No nn.Module, forward/backward call, optimizer, or training is instantiated.
weights_only=True refuses arbitrary pickle classes. Repository code is not run.
"""
import argparse
import hashlib
import json
import struct
import sys
from pathlib import Path

import numpy as np
import torch
import zstandard as zstd

ROOT = Path(__file__).parent
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--checkpoint", type=Path, default=ROOT / "public-state" / "gatsby-nanogpt-3-ckpt.pt", help="Existing public data file; SHA256 must match the pinned checkpoint")
SOURCE = parser.parse_args().checkpoint
EXPECTED = "15d50d8d21ef28f6d7a67ee5e937abdcc9fd8e2fe295959a0680b591e82f5f7b"
N_TARGET = 26_895_998_464
GIB = 2**30
assert hashlib.sha256(SOURCE.read_bytes()).hexdigest() == EXPECTED
checkpoint = torch.load(SOURCE, map_location="cpu", weights_only=True)
states = checkpoint["optimizer"]["state"]
groups = checkpoint["optimizer"]["param_groups"]
print("loaded_inert_data", len(states), flush=True)

# The public model.py builds optimizer groups by named_parameters order,
# rank>=2 then rank<2. State dict order is module traversal order. Deduplicate
# tied tensors; require group counts and every aligned shape to match.
# No model source is imported or executed.
unique_weights = []
storage_seen = set()
for name, tensor in checkpoint["model"].items():
    if not isinstance(tensor, torch.Tensor):
        continue
    key = (tensor.untyped_storage().data_ptr(), tensor.storage_offset(), tuple(tensor.shape))
    if key in storage_seen:
        continue
    storage_seen.add(key)
    unique_weights.append((name, tensor))
by_rank = [[x for x in unique_weights if x[1].ndim >= 2], [x for x in unique_weights if x[1].ndim < 2]]
assert len(groups) == 2
mapping = {}
for group, weights in zip(groups, by_rank):
    assert len(group["params"]) == len(weights)
    for param_id, (name, w) in zip(group["params"], weights):
        state = states[param_id]
        assert state["exp_avg"].shape == w.shape == state["exp_avg_sq"].shape
        assert all(t.device.type == "cpu" for t in (w, state["exp_avg"], state["exp_avg_sq"]))
        mapping[param_id] = (name, w)

m_arrays, v_arrays, w_arrays = [], [], []
tensor_records = []
for group in groups:
    for param_id in group["params"]:
        state = states[param_id]
        name, w = mapping[param_id]
        assert all(t.dtype == torch.float32 for t in (w, state["exp_avg"], state["exp_avg_sq"]))
        m, v, w = [t.detach().numpy().reshape(-1) for t in (state["exp_avg"], state["exp_avg_sq"], w)]
        assert np.all(np.isfinite(m)) and np.all(np.isfinite(v)) and np.all(np.isfinite(w))
        assert np.all(v >= 0)
        m_arrays.append(m)
        v_arrays.append(v)
        w_arrays.append(w)
        tensor_records.append({"name": name, "elements": len(m), "step": float(state["step"]), "shape": list(state["exp_avg"].shape)})

m = np.concatenate(m_arrays)
v = np.concatenate(v_arrays)
w = np.concatenate(w_arrays)
count = len(m)
assert count == 11_015_040
print("aligned_fp32_state_elements", count, flush=True)
mb, vb, wb = [x.view(np.uint32) for x in (m, v, w)]
compressor = zstd.ZstdCompressor(level=3)
decompressor = zstd.ZstdDecompressor()


def shuffle(words, item_bytes=4):
    return words.view(np.uint8).reshape(-1, item_bytes).T.copy().tobytes()


def unshuffle(payload, item_bytes=4):
    byte_matrix = np.frombuffer(payload, dtype=np.uint8).reshape(item_bytes, -1).T.copy()
    return byte_matrix.view(np.uint32 if item_bytes == 4 else np.uint16).reshape(-1)


results = []


def record(label, payload, metadata_bytes=0, decoder_check=None, note=""):
    compressed = compressor.compress(payload)
    recovered = decompressor.decompress(compressed)
    assert recovered == payload
    if decoder_check:
        decoder_check(recovered)
    total = len(compressed) + metadata_bytes
    bpp = total * 8 / count
    entry = {
        "codec": label, "payload_bytes": len(compressed), "metadata_bytes": metadata_bytes,
        "total_bytes": total, "bits_per_m_v_pair": bpp,
        "raw_over_encoded": count * 8 / total,
        "projected_gib_if_same_bits_at_target_parameter_count": N_TARGET * bpp / 8 / GIB,
        "bitwise_roundtrip": True, "note": note,
    }
    results.append(entry)
    print(label, round(bpp, 4), "bits/pair", flush=True)


both = np.concatenate((mb, vb))
record("raw_fp32_bytes_zstd3", both.tobytes(), note="All m then all v; no side information")
record("byte_shuffled_fp32_zstd3", shuffle(both), decoder_check=lambda p: np.testing.assert_array_equal(unshuffle(p), both))

# Exact high16+low16 split. These high16 words are truncated BF16 prefixes,
# not necessarily the BF16 round-to-nearest values used by compute kernels.
high = (both >> 16).astype(np.uint16)
low = (both & 0xffff).astype(np.uint16)
high_compressed = compressor.compress(shuffle(high, 2))
low_compressed = compressor.compress(shuffle(low, 2))
high_back = unshuffle(decompressor.decompress(high_compressed), 2).astype(np.uint32)
low_back = unshuffle(decompressor.decompress(low_compressed), 2).astype(np.uint32)
assert np.array_equal((high_back << 16) | low_back, both)
split_total = len(high_compressed) + len(low_compressed) + 16
split_result = {
    "codec": "exact_high16_low16_separate_shuffle_zstd3",
    "high16_carrier_bytes": len(high_compressed),
    "low16_correction_bytes": len(low_compressed),
    "low16_correction_bits_per_m_v_pair": len(low_compressed) * 8 / count,
    "low16_correction_projected_gib": N_TARGET * len(low_compressed) / count / GIB,
    "total_bytes_including_16byte_lengths": split_total,
    "bits_per_m_v_pair": split_total * 8 / count,
    "projected_gib_if_same_bits_at_target_parameter_count": N_TARGET * split_total / count / GIB,
    "bitwise_roundtrip": True,
}

# Predictions use current weights, not unavailable previous weights. Coefficients
# are closed-form statistics, stored as FP32; there is no learned model/training.
pred_m, pred_v, coefficients = [], [], []
for ma, va, wa in zip(m_arrays, v_arrays, w_arrays):
    wd, md, vd = wa.astype(np.float64), ma.astype(np.float64), va.astype(np.float64)
    wm, mm = wd.mean(), md.mean()
    centered = wd - wm
    denom = np.dot(centered, centered)
    a = np.float32(np.dot(centered, md - mm) / denom) if denom > 0 else np.float32(0)
    b = np.float32(mm - float(a) * wm)
    pm = np.add(np.multiply(a, wa, dtype=np.float32), b, dtype=np.float32)
    square = np.multiply(ma, ma, dtype=np.float32)
    sd = square.astype(np.float64)
    denom_v = np.dot(sd, sd)
    c = np.float32(np.dot(sd, vd) / denom_v) if denom_v > 0 else np.float32(0)
    pv = np.multiply(c, square, dtype=np.float32)
    assert np.all(np.isfinite(pm)) and np.all(np.isfinite(pv))
    pred_m.append(pm)
    pred_v.append(pv)
    coefficients.append([float(a), float(b), float(c)])
pm, pv = np.concatenate(pred_m).view(np.uint32), np.concatenate(pred_v).view(np.uint32)
residual_m, residual_v = mb ^ pm, vb ^ pv
residual = np.concatenate((residual_m, residual_v))
metadata_bytes = len(coefficients) * 12 + (len(coefficients) + 1) * 8


def check_joint(payload):
    decoded = unshuffle(payload)
    # Derive m predictions again from W plus serialized-size scalar metadata;
    # do not rely on the encoder's full prediction array as free side storage.
    replay_pm = []
    for wa, coeff in zip(w_arrays, coefficients):
        predicted = np.add(np.multiply(np.float32(coeff[0]), wa, dtype=np.float32), np.float32(coeff[1]), dtype=np.float32)
        replay_pm.append(predicted.view(np.uint32))
    decoded_m = decoded[:count] ^ np.concatenate(replay_pm)
    assert np.array_equal(decoded_m, mb)
    cursor = 0
    rebuilt_v = []
    for ma, coeff in zip(m_arrays, coefficients):
        length = len(ma)
        reconstructed_m = decoded_m[cursor:cursor+length].view(np.float32)
        square = np.multiply(reconstructed_m, reconstructed_m, dtype=np.float32)
        prediction_v = np.multiply(np.float32(coeff[2]), square, dtype=np.float32).view(np.uint32)
        rebuilt_v.append(decoded[count+cursor:count+cursor+length] ^ prediction_v)
        cursor += length
    assert np.array_equal(np.concatenate(rebuilt_v), vb)


record("current_W_linear_predict_m_and_m_squared_predict_v_xor_shuffle_zstd3", shuffle(residual), metadata_bytes, check_joint,
       "Current W is given and already charged in model memory; 3 FP32 coefficients/tensor and offsets included. CPU arithmetic only.")

# An intentionally favorable reconstruction carrier: retain u and v, estimate
# m=u*(sqrt(v)+eps), then store only exact integer correction bits for m.
# Unlike a weight difference, u is retained explicitly and therefore charged.
den = np.add(np.sqrt(v, dtype=np.float32), np.float32(1e-8), dtype=np.float32)
u = np.divide(m, den, dtype=np.float32)
estimated_m = np.multiply(u, den, dtype=np.float32)
ub = u.view(np.uint32)
ratio_m_correction = mb ^ estimated_m.view(np.uint32)
ratio_carriers = np.concatenate((ub, vb))
ratio_payload = np.concatenate((ub, vb, ratio_m_correction))


def check_ratio(payload):
    decoded = unshuffle(payload)
    ru, rv, correction = decoded[:count], decoded[count:2*count], decoded[2*count:]
    rden = np.add(np.sqrt(rv.view(np.float32), dtype=np.float32), np.float32(1e-8), dtype=np.float32)
    rm_pred = np.multiply(ru.view(np.float32), rden, dtype=np.float32).view(np.uint32)
    assert np.array_equal(rm_pred ^ correction, mb)
    assert np.array_equal(rv, vb)


record("retain_u_and_v_plus_exact_m_xor_correction_shuffle_zstd3", shuffle(ratio_payload), 24, check_ratio,
       "u and v are BOTH retained and charged; correction makes m bitwise exact. u is not recoverable for free from a rounded weight update.")
ratio_carrier_compressed = compressor.compress(shuffle(ratio_carriers))
ratio_correction_compressed = compressor.compress(shuffle(ratio_m_correction))
ratio_split_total = len(ratio_carrier_compressed) + len(ratio_correction_compressed) + 24
assert np.array_equal(unshuffle(decompressor.decompress(ratio_correction_compressed)) ^ estimated_m.view(np.uint32), mb)
ratio_stats = {
    "u_plus_v_carrier_bytes": len(ratio_carrier_compressed),
    "m_correction_bytes": len(ratio_correction_compressed),
    "m_correction_bits_per_parameter": len(ratio_correction_compressed) * 8 / count,
    "m_correction_projected_gib": N_TARGET * len(ratio_correction_compressed) / count / GIB,
    "carrier_plus_correction_bits_per_m_v_pair": ratio_split_total * 8 / count,
    "carrier_plus_correction_projected_gib": N_TARGET * ratio_split_total / count / GIB,
    "m_prediction_bitwise_exact_fraction": float(np.mean(ratio_m_correction == 0)),
    "max_m_prediction_raw_uint32_distance": int(np.max(np.abs(mb.astype(np.int64) - estimated_m.view(np.uint32).astype(np.int64)))),
    "bitwise_roundtrip": True,
}

# More compact exact correction for this checkpoint: the prediction differs
# by at most one UINT32 code. Encode a changed mask and the direction only.
# A general codec would need an escape path; this snapshot has zero escapes.
ulp_delta = mb.astype(np.int64) - estimated_m.view(np.uint32).astype(np.int64)
assert np.all(np.abs(ulp_delta) <= 1)
changed = ulp_delta != 0
mask_stream = np.packbits(changed, bitorder="little").tobytes()
direction_stream = np.packbits(ulp_delta[changed] > 0, bitorder="little").tobytes()
mask_encoded = compressor.compress(mask_stream)
direction_encoded = compressor.compress(direction_stream)
mask_decoded = np.unpackbits(np.frombuffer(decompressor.decompress(mask_encoded), dtype=np.uint8), bitorder="little")[:count].astype(bool)
direction_decoded = np.unpackbits(np.frombuffer(decompressor.decompress(direction_encoded), dtype=np.uint8), bitorder="little")[:int(mask_decoded.sum())]
reconstructed_delta = np.zeros(count, dtype=np.int64)
reconstructed_delta[mask_decoded] = np.where(direction_decoded, 1, -1)
reconstructed_m = (estimated_m.view(np.uint32).astype(np.int64) + reconstructed_delta).astype(np.uint32)
assert np.array_equal(reconstructed_m, mb)
correction_bytes = len(mask_encoded) + len(direction_encoded) + 32
# Build and decode an actual three-stream container, including its real header.
container = struct.pack("<QQQQ", count, len(ratio_carrier_compressed), len(mask_encoded), len(direction_encoded)) + ratio_carrier_compressed + mask_encoded + direction_encoded
decoded_count, carrier_length, mask_length, direction_length = struct.unpack("<QQQQ", container[:32])
assert decoded_count == count and len(container) == 32 + carrier_length + mask_length + direction_length
carrier_end = 32 + carrier_length
mask_end = carrier_end + mask_length
restored_carriers = unshuffle(decompressor.decompress(container[32:carrier_end]))
restored_u, restored_v = restored_carriers[:count].view(np.float32), restored_carriers[count:].view(np.float32)
restored_den = np.add(np.sqrt(restored_v, dtype=np.float32), np.float32(1e-8), dtype=np.float32)
restored_prediction = np.multiply(restored_u, restored_den, dtype=np.float32).view(np.uint32)
restored_mask = np.unpackbits(np.frombuffer(decompressor.decompress(container[carrier_end:mask_end]), dtype=np.uint8), bitorder="little")[:count].astype(bool)
restored_directions = np.unpackbits(np.frombuffer(decompressor.decompress(container[mask_end:]), dtype=np.uint8), bitorder="little")[:int(restored_mask.sum())]
restored_delta = np.zeros(count, dtype=np.int64)
restored_delta[restored_mask] = np.where(restored_directions, 1, -1)
assert np.array_equal((restored_prediction.astype(np.int64) + restored_delta).astype(np.uint32), mb)
assert np.array_equal(restored_v.view(np.uint32), vb)
probabilities = np.array([np.mean(ulp_delta == i) for i in (-1, 0, 1)])
delta_entropy = float(-np.sum(probabilities[probabilities > 0] * np.log2(probabilities[probabilities > 0])))
ratio_stats["changed_mask_direction_codec"] = {
    "correction_bytes_including_32byte_header": correction_bytes,
    "correction_bits_per_parameter": correction_bytes * 8 / count,
    "correction_projected_gib": N_TARGET * correction_bytes / count / GIB,
    "delta_symbol_marginal_entropy_bits": delta_entropy,
    "u_v_carrier_plus_this_correction_bits_per_pair": (len(ratio_carrier_compressed) + correction_bytes) * 8 / count,
    "u_v_carrier_plus_this_correction_projected_gib": N_TARGET * (len(ratio_carrier_compressed) + correction_bytes) / count / GIB,
    "escape_count_in_this_checkpoint": 0,
    "actual_container_bytes": len(container),
    "container_header_bytes": 32,
    "bitwise_roundtrip": True,
    "limitation": "At-most-one-code correction verified for this checkpoint only; general inputs need escape handling, and u plus v still consume memory.",
}

# Coarse BF16 prefix for v and exact residual from a local m^2 predictor is
# not assumed sufficient: the entire residual stream above is always retained.
fractions = {
    "m_zero_fraction": float(np.mean(mb & 0x7fffffff == 0)),
    "v_zero_fraction": float(np.mean(vb & 0x7fffffff == 0)),
    "m_low16_zero_fraction": float(np.mean((mb & 0xffff) == 0)),
    "v_low16_zero_fraction": float(np.mean((vb & 0xffff) == 0)),
}
output = {
    "scope": "Existing public 11M model's saved FP32 AdamW tensors, read only on CPU; no model or training run",
    "source_repository": "https://huggingface.co/sup-computer/gatsby-nanogpt-3",
    "source_revision": "fcefc5fbc149f9763bd79475fa9ee793abb62ee3",
    "source_sha256": EXPECTED, "checkpoint_iteration": checkpoint.get("iter_num"),
    "torch_version": torch.__version__, "zstd_version": zstd.__version__,
    "state_elements": count, "state_tensors": len(m_arrays), "raw_m_v_bytes": count * 8,
    "target_model_parameter_count": N_TARGET, "target_30gib_bits_per_m_v_pair": 30 * GIB * 8 / N_TARGET,
    "codec_results": results, "exact_high16_low16_correction_test": split_result,
    "retain_u_v_and_correct_m_test": ratio_stats,
    "fractions": fractions, "tensor_records": tensor_records,
    "coefficients_m_from_current_w_and_v_from_m_squared": coefficients,
    "limitations": [
        "A single public 11M checkpoint, not Qwen27B or Ascend; results do not certify the target model.",
        "All projected GiB values scale measured bytes/element only; they are not a model measurement.",
        "These codecs/predictors are examples, not an exhaustive search or a universal entropy lower bound.",
        "CPU encode/decode equality is verified; no online NPU codec or speed/peak workspace test.",
        "Floating predictor arithmetic requires identical decoder evaluation for exact reconstruction.",
        "Full tensors in CPU analysis are not a proposal to retain uncompressed full states in HBM.",
    ],
}
destination = ROOT / "public_state_audit.json"
destination.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
print(json.dumps({k:v for k,v in output.items() if k not in ("tensor_records", "coefficients_m_from_current_w_and_v_from_m_squared")}, indent=2), flush=True)
