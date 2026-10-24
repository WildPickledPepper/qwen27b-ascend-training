"""FP32 ENEC device roundtrip and offline checkpoint conversion, not a training pager.

CPU tensors here are checkpoint input/verification only; no running training state
is offloaded. Device codec scratch and raw+compressed copies coexist during tests.
"""
import argparse, ctypes, hashlib, json, os, time, traceback
from pathlib import Path
os.environ['TORCH_DEVICE_BACKEND_AUTOLOAD'] = '0'
os.environ.pop('ASCEND_RT_VISIBLE_DEVICES', None)  # ACL API uses authorized physical IDs.
import numpy as np
import torch
from safetensors import safe_open

p = argparse.ArgumentParser()
p.add_argument('--model', required=True)
p.add_argument('--device', type=int, choices=[2, 3], required=True)
p.add_argument('--mode', choices=['probe', 'samples', 'all'], default='probe')
p.add_argument('--output', required=True)
p.add_argument('--chunk-mib', type=int, default=64)
a = p.parse_args()
torch.set_num_threads(4)
root = Path(__file__).resolve().parent
out = Path(a.output); out.mkdir(parents=True, exist_ok=False)
lib = ctypes.CDLL(str(root / 'build/libenec_fp32_bridge.so'))
lib.enec_error.restype = ctypes.c_char_p
lib.enec_init.argtypes = [ctypes.c_int]
lib.enec_roundtrip.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_void_p,
                             ctypes.c_void_p, ctypes.c_uint32,
                             ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(ctypes.c_double)]
report = {'device': a.device, 'mode': a.mode, 'codec': 'ENEC-FP32-120-6-3-16',
          'dtype': 'float32', 'scope': 'Offline checkpoint conversion and NPU codec validation; no model training or online memory saving claim.',
          'rows': [], 'complete': False, 'model_modified': False}
started = time.time()
if (root / 'build_manifest.json').exists():
    report['build'] = json.loads((root / 'build_manifest.json').read_text())
    report['codec'] = report['build']['variant']

def check(code):
    if code:
        raise RuntimeError(lib.enec_error().decode())

def save():
    (out / 'report.json').write_text(json.dumps(report, indent=2))

def roundtrip(t, name, offset, stream=None, diagnostic=False):
    t = t.contiguous().view(-1)
    assert t.dtype == torch.float32 and t.device.type == 'cpu'
    data = t.numpy().view(np.uint8)
    raw_bytes = data.nbytes
    if raw_bytes < 65536:
        row = {'name': name, 'element_offset': offset, 'elements': t.numel(),
               'raw_fp32_bytes': raw_bytes, 'stored_bytes': raw_bytes,
               'stored_fraction': 1., 'storage_kind': 'raw_small_tensor',
               'bit_exact': True, 'stored_lossless': True, 'diagnostic': diagnostic,
               'input_sha256': hashlib.sha256(data).hexdigest(),
               'decoded_sha256': hashlib.sha256(data).hexdigest()}
        if stream is not None:
            row['archive_offset'] = stream.tell()
            stream.write(memoryview(data))
        report['rows'].append(row); save()
        return
    restored = np.empty(raw_bytes, dtype=np.uint8)
    capacity = raw_bytes * 2 + (16 << 20)
    encoded = np.empty(capacity, dtype=np.uint8)
    size = ctypes.c_uint32(); metrics = (ctypes.c_double * 6)()
    start = time.monotonic()
    check(lib.enec_roundtrip(data.ctypes.data, raw_bytes, restored.ctypes.data,
                            encoded.ctypes.data, capacity, ctypes.byref(size), metrics))
    exact = bool(np.array_equal(data, restored))
    row = {'name': name, 'element_offset': offset, 'elements': t.numel(),
           'raw_fp32_bytes': raw_bytes, 'author_encoded_bytes': size.value,
           'bit_exact': exact, 'input_sha256': hashlib.sha256(data).hexdigest(),
           'decoded_sha256': hashlib.sha256(restored).hexdigest(),
           'compress_event_ms': metrics[0], 'merge_event_ms': metrics[1],
           'decompress_event_ms': metrics[2], 'explicit_live_device_allocation_bytes': metrics[3],
           'hbm_free_during_operation': metrics[4], 'hbm_total': metrics[5],
           'roundtrip_wall_seconds': time.monotonic() - start,
           'nonzero_low16_fraction': float(np.count_nonzero(t.numpy().view(np.uint32) & 65535) / t.numel()),
           'diagnostic': diagnostic}
    # Unsafe/expanding author output is never accepted as a replacement.
    use_raw = not exact or size.value >= raw_bytes
    row['storage_kind'] = 'raw_fallback' if use_raw else 'enec'
    row['stored_bytes'] = raw_bytes if use_raw else size.value
    row['stored_fraction'] = row['stored_bytes'] / raw_bytes
    stored = data if use_raw else encoded[:size.value]
    row['stored_sha256'] = hashlib.sha256(stored).hexdigest()
    row['stored_lossless'] = True  # exact original bytes on any codec mismatch.
    if stream is not None:
        row['archive_offset'] = stream.tell()
        stream.write(memoryview(stored))
    report['rows'].append(row)
    save()
    print(json.dumps(row), flush=True)

check(lib.enec_init(a.device))
print('DEVICE_READY', a.device, flush=True)
try:
    if a.mode == 'probe':
        rng = np.random.default_rng(20261024)
        for n in [1 << 20, (1 << 20) + 17]:
            w = rng.normal(0, .02, n).astype(np.float32)
            roundtrip(torch.from_numpy(w), 'finite_fp32_probe', 0)
        # The paper does not establish arbitrary IEEE special-word behavior.
        words = np.array([0, 0x80000000, 1, 0x007fffff, 0x00800000,
                          0x7f7fffff, 0xff7fffff, 0x7f800000, 0xff800000,
                          0x7fc12345, 0x7fa54321], dtype=np.uint32)
        words = np.resize(words, 1 << 20).copy()
        roundtrip(torch.from_numpy(words.view(np.float32)), 'ieee_special_words', 0, diagnostic=True)
    else:
        model = Path(a.model)
        mapping = json.loads((model / 'model.safetensors.index.json').read_text())['weight_map']
        keys = sorted(k for k in mapping if k == 'lm_head.weight' or k.startswith('model.language_model.'))
        shapes = {}
        for filename in sorted(set(mapping[k] for k in keys)):
            with safe_open(str(model / filename), framework='pt', device='cpu') as f:
                for k in keys:
                    if mapping[k] == filename:
                        shape = list(f.get_slice(k).get_shape())
                        shapes[k] = shape
        bins = [[], []]; loads = [0, 0]
        for k in sorted(keys, key=lambda k: -int(np.prod(shapes[k]))):
            rank = min(range(2), key=lambda i: loads[i])
            bins[rank].append(k); loads[rank] += int(np.prod(shapes[k]))
        assigned = bins[a.device - 2]
        report['text_parameters'] = sum(int(np.prod(shapes[k])) for k in keys)
        report['assigned_parameters'] = loads[a.device - 2]
        report['checkpoint_dtypes'] = {}
        report['scope_excludes'] = ['vision', 'MTP', 'gradients', 'optimizer states', 'training activations']
        report['tensor_index'] = {k: shapes[k] for k in assigned}
        if a.mode == 'samples':
            assigned = [k for k in assigned if any(s in k for s in
                        ['layers.0.mlp.', 'layers.31.mlp.', 'layers.63.mlp.',
                         'layers.0.linear_attn.in_proj_qkv', 'layers.31.self_attn.q_proj'])]
        count = a.chunk_mib * (1 << 20) // 4
        with (out / 'weights.enec.parts').open('wb') as archive:
            for key in sorted(assigned):
                with safe_open(str(model / mapping[key]), framework='pt', device='cpu') as f:
                    t = f.get_tensor(key)
                report['checkpoint_dtypes'][key] = str(t.dtype)
                # Exact lift from checkpoint BF16; no BF16 compute copy is kept on NPU.
                flat = t.flatten()
                for offset in range(0, flat.numel(), count):
                    chunk = flat[offset:offset + count].float().contiguous()
                    roundtrip(chunk, key, offset, archive)
                    if a.mode == 'samples' and offset == 0:
                        # Controlled FP32 AdamW update: codec stress, NOT model-training evidence.
                        g = torch.from_numpy(np.random.default_rng(20261024).normal(0,.001,chunk.numel()).astype(np.float32))
                        m = torch.zeros_like(chunk); v = torch.zeros_like(chunk)
                        for step in range(1, 9):
                            m.mul_(.9).add_(g, alpha=.1)
                            v.mul_(.999).addcmul_(g,g,value=.001)
                            chunk.mul_(1-1e-4*.01).addcdiv_(m/(1-.9**step), (v/(1-.999**step)).sqrt().add_(1e-8), value=-1e-4)
                        roundtrip(chunk, key + ':controlled_fp32_update8', offset, diagnostic=False)
                        del g, m, v
                    del chunk
                del flat, t
    ordinary = [r for r in report['rows'] if not r['diagnostic'] and ':controlled_' not in r['name']]
    report['aggregate'] = {'raw_fp32_bytes': sum(r['raw_fp32_bytes'] for r in ordinary),
                           'stored_bytes': sum(r['stored_bytes'] for r in ordinary),
                           'all_bit_exact': all(r['bit_exact'] for r in ordinary),
                           'all_stored_lossless': all(r['stored_lossless'] for r in ordinary),
                           'author_mismatch_chunks': sum(not r['bit_exact'] for r in ordinary)}
    report['aggregate']['stored_fraction'] = report['aggregate']['stored_bytes'] / report['aggregate']['raw_fp32_bytes']
    report['complete'] = True
except Exception:
    report['error'] = traceback.format_exc()
    raise
finally:
    report['close_status'] = lib.enec_close()
    report['elapsed_seconds'] = time.time() - started
    save()
