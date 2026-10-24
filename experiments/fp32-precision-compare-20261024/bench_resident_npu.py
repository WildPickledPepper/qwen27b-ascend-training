"""Full FP32 states on CPU, windowed FP32 update on NPU; no state quantization.

Keeps the previously executed CPU-update adapter unchanged as an auxiliary control.
The compressed group uses exactly its existing NPU quantized update implementation.
"""
import time
import torch
import bench_resident as cpu
from bench_resident import (Codec, CompressedLinear, CompressedEmbedding, load_frames,
                            WeightStore, precision_context, evaluation_context,
                            CpuMoments, RawCodec, PROBES, TIMES, probe_before, probe_after)

TIMES.setdefault('npu_adam_seconds', 0.)


class NpuOffloadedMoments(CpuMoments):
    @torch.no_grad()
    def update(self, start, w, g, step):
        token = probe_before(self.name, self.n, start, w, g, step)
        for off in range(0, w.numel(), self.window):
            length = min(self.window, w.numel()-off)
            begin = start+off
            chunk = w[off:off+length]
            grad = g[off:off+length].float()
            if not torch.isfinite(grad).all().item():
                raise FloatingPointError('Nonfinite offloaded-state gradient')
            t = time.monotonic()
            m = self.m[begin:begin+length].to(w.device)
            v = self.v[begin:begin+length].to(w.device)
            torch.npu.current_stream(w.device).synchronize()
            TIMES['device_copy_seconds'] += time.monotonic()-t
            t = time.monotonic()
            m.mul_(.9).add_(grad, alpha=.1)
            v.mul_(.999).addcmul_(grad, grad, value=.001)
            chunk.mul_(1-self.lr*self.wd).addcdiv_(
                m/(1-.9**step), (v/(1-.999**step)).sqrt().add_(1e-8), value=-self.lr)
            torch.npu.current_stream(w.device).synchronize()
            TIMES['npu_adam_seconds'] += time.monotonic()-t
            t = time.monotonic()
            self.m[begin:begin+length].copy_(m)
            self.v[begin:begin+length].copy_(v)
            TIMES['host_copy_seconds'] += time.monotonic()-t
            del m, v
            self.updated += length
            if self.progress and (self.updated % (8 << 20) == 0 or off+length == w.numel()):
                self.progress()
        probe_after(self.name, step, w, token)


def configure(mode):
    cpu.configure(mode)
    if mode != 'compressed':
        cpu.base.QuantMoments = NpuOffloadedMoments
