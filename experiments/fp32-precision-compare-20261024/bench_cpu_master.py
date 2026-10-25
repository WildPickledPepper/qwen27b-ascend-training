"""CPU-resident FP32 master/m/v with bounded pinned gradient/weight staging.

Loads the installed DeepSpeed CPUAdam extension without modifying any environment.
FP32 AdamW math, including its floating-point rounding, is the library's own.
This is not claimed bitwise identical to the explicit NPU Adam reference.
"""
import importlib.util
import os
import time
import torch
import bench_resident as bench
from bench_resident import probe_before, probe_after, TIMES

POOLS = {}
DETAIL = {'gradient_d2h_seconds': 0., 'cpu_adam_seconds': 0.,
          'weight_stage_seconds': 0., 'weight_h2d_seconds': 0.,
          'initial_master_seconds': 0., 'windows': 0}
_OP = None


def cpu_op():
    global _OP
    if _OP is None:
        path = os.environ['CPU_ADAM_EXTENSION']
        spec = importlib.util.spec_from_file_location('cpu_adam', path)
        _OP = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(_OP)
        _OP.create_adam(7201, 1e-5, .9, .999, 1e-8, .01, True, True)
    return _OP


def pool(device, window):
    key = (str(device), window)
    if key not in POOLS:
        POOLS[key] = torch.empty((2, window), dtype=torch.float32, pin_memory=True)
        assert POOLS[key].is_pinned()
    return POOLS[key]


class CpuMasterMoments(bench.CpuMoments):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.master = torch.empty(self.n, dtype=torch.float32)
        self.initialized = False
        cpu_op()

    def bytes(self):
        return self.n * 12

    @torch.no_grad()
    def initialize_frames(self, frames):
        t = time.monotonic()
        offset = 0
        for frame in frames:
            assert frame['kind'] == 'raw'
            n = frame['n']
            self.master[offset:offset+n].copy_(frame['data'][:n*4].view(torch.float32))
            offset += n
        assert offset == self.n
        self.initialized = True
        DETAIL['initial_master_seconds'] += time.monotonic() - t

    @torch.no_grad()
    def update(self, start, w, g, step):
        assert self.initialized and w.dtype == g.dtype == torch.float32
        token = probe_before(self.name, self.n, start, w, g, step)
        if not torch.isfinite(g).all().item():
            raise FloatingPointError('Nonfinite CPUAdam gradient')
        host = pool(w.device, self.window)
        for off in range(0, w.numel(), self.window):
            n = min(self.window, w.numel()-off)
            begin = start+off
            t = time.monotonic()
            host[0, :n].copy_(g[off:off+n], non_blocking=True)
            torch.npu.current_stream(w.device).synchronize()
            elapsed = time.monotonic()-t
            DETAIL['gradient_d2h_seconds'] += elapsed
            TIMES['host_copy_seconds'] += elapsed
            t = time.monotonic()
            cpu_op().adam_update(7201, step, self.lr, .9, .999, 1e-8, self.wd, True,
                                 self.master[begin:begin+n], host[0, :n],
                                 self.m[begin:begin+n], self.v[begin:begin+n])
            elapsed = time.monotonic()-t
            DETAIL['cpu_adam_seconds'] += elapsed
            TIMES['cpu_adam_seconds'] += elapsed
            t = time.monotonic()
            host[1, :n].copy_(self.master[begin:begin+n])
            DETAIL['weight_stage_seconds'] += time.monotonic()-t
            t = time.monotonic()
            w[off:off+n].copy_(host[1, :n], non_blocking=True)
            torch.npu.current_stream(w.device).synchronize()
            elapsed = time.monotonic()-t
            DETAIL['weight_h2d_seconds'] += elapsed
            TIMES['device_copy_seconds'] += elapsed
            DETAIL['windows'] += 1
            self.updated += n
            if self.progress and (self.updated % (8 << 20) == 0 or off+n == w.numel()):
                self.progress()
        probe_after(self.name, step, w, token)


def install():
    bench.base.QuantMoments = CpuMasterMoments
