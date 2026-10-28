"""Interleaved A/B/C harness: load noise cancels because variants alternate.

Measures on the real slice sequence:
  torch  : torch bf16 add_ per micro (current production path)
  k1     : hand-written fused kernel, 1 micro per accumulate (bitwise-same semantics)
  k2     : kernel fusing 2 microbatches per accumulator touch
"""
import ctypes
import json
import os
import time

import torch

CHUNK = 64 * 1024 * 1024
MEASURE_PASSES = 4          # 4 of the 16 real passes per measurement window
LOG = ('/home/yaopengju/mindstudio/projects/qwen-sql-pipeline-overlap-20261026/'
       'cache4-full.private/rank0.jsonl')


def load():
    return open('/proc/loadavg').read().split()[0]


rows = [json.loads(l) for l in open(LOG)]
manifest = [r for r in rows if r.get('event') == 'model_ready'][0]['manifest']
numel = [m['numel'] for m in manifest if m['numel'] > 0]
TOTAL = sum(numel)
slices = []
for n in numel:
    for start in range(0, n, CHUNK):
        slices.append((start, min(CHUNK, n - start)))

print(f'params={len(numel)} elements={TOTAL/1e9:.3f}e9 slices={len(slices)} '
      f'load={load()}', flush=True)

# pin feasibility first (cheap, no NPU needed when torch reports it)
for gib in (8, 27, 54):
    n = gib * 2**30 // 2
    t = time.monotonic()
    try:
        b = torch.empty(n, dtype=torch.bfloat16, pin_memory=True)
        b.fill_(0.5)
        print(f'  pin {gib:2d} GiB OK {time.monotonic()-t:5.1f}s', flush=True)
        del b
    except Exception as exc:
        print(f'  pin {gib:2d} GiB FAIL {type(exc).__name__}: {str(exc)[:90]}', flush=True)

lib = ctypes.CDLL('./accum2.so')
lib.accum1.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t]
lib.accumk.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p), ctypes.c_int,
                       ctypes.c_size_t]
gomp = ctypes.CDLL('libgomp.so.1')
gomp.omp_set_num_threads(ctypes.c_int(24))

acc = torch.empty(TOTAL, dtype=torch.bfloat16)
grads = [torch.empty(TOTAL, dtype=torch.bfloat16) for _ in range(2)]
acc.fill_(0.25)
for g in grads:
    g.fill_(0.125)
print(f'allocated, load={load()}', flush=True)

aptr = acc.data_ptr()
gptrs = [g.data_ptr() for g in grads]


def run_torch(passes):
    torch.set_num_threads(24)
    g = grads[0]
    t = time.monotonic()
    for _ in range(passes):
        for start, n in slices:
            acc[start:start + n].add_(g[start:start + n])
    return time.monotonic() - t


def run_k1(passes):
    t = time.monotonic()
    for _ in range(passes):
        for start, n in slices:
            lib.accum1(ctypes.c_void_p(aptr + start * 2),
                       ctypes.c_void_p(gptrs[0] + start * 2), ctypes.c_size_t(n))
    return time.monotonic() - t


def run_k2(passes):
    shifted = (ctypes.c_void_p * 2)()
    t = time.monotonic()
    for _ in range(passes // 2):
        for start, n in slices:
            shifted[0] = ctypes.c_void_p(gptrs[0] + start * 2)
            shifted[1] = ctypes.c_void_p(gptrs[1] + start * 2)
            lib.accumk(ctypes.c_void_p(aptr + start * 2), shifted, 2, ctypes.c_size_t(n))
    return time.monotonic() - t


VARIANTS = {'torch': (run_torch, 1), 'k1': (run_k1, 1), 'k2': (run_k2, 2)}
traffic = {'torch': TOTAL * 6, 'k1': TOTAL * 6, 'k2': TOTAL * 8}   # per pass / per pair

for v in VARIANTS:          # warm caches / first touch
    VARIANTS[v][0](VARIANTS[v][1])

order = ['torch', 'k1', 'k2', 'k2', 'k1', 'torch', 'k1', 'k2', 'torch']
res = {v: [] for v in VARIANTS}
for name in order:
    fn, unit = VARIANTS[name]
    passes = MEASURE_PASSES if unit == 1 else MEASURE_PASSES
    dt = fn(passes)
    bytes_moved = traffic[name] * passes / unit
    gbps = bytes_moved / dt / 1e9
    res[name].append(dt)
    print(f'  {name:6s} passes={passes:2d} {dt:6.2f}s {gbps:5.2f} GB/s  load={load()}',
          flush=True)


def median(xs):
    xs = sorted(xs)
    return xs[len(xs) // 2]


print('\n=== medians over interleaved runs ===')
base = median(res['torch'])
for name in ('torch', 'k1', 'k2'):
    m = median(res[name])
    print(f'  {name:6s} median {m:6.2f}s  speedup x{base/m:4.2f}  '
          f'samples {[round(x,1) for x in res[name]]}')
