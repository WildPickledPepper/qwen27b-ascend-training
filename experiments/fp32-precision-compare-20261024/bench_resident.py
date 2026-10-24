"""Controlled comparison adapters for the previously executed resident trainer.

Reference groups keep uncompressed FP32 device weights and CPU FP32 Adam states.
BF16 means on-demand BF16 linear compute and autocast, not a BF16 master model.
This is a common custom harness, not a tuned DeepSpeed performance baseline.
"""
import contextlib
import json
import time
import torch
import torch.nn.functional as F
import resident as base

Codec = base.Codec
CompressedLinear = base.CompressedLinear
CompressedEmbedding = base.CompressedEmbedding
MODE = None
TIMES = {'cpu_adam_seconds': 0., 'host_copy_seconds': 0., 'device_copy_seconds': 0.}
PROBES = []
OriginalMoments = base.QuantMoments
FP32LinearFunction = base.LinearFunction


def probe_before(name, total, start, w, g, step):
    # Deterministic coordinates spanning every real tensor, not proxy parameters.
    indices = sorted(set((i * (total - 1)) // 63 for i in range(64)))
    chosen = [i for i in indices if start <= i < start + w.numel()]
    if not chosen:
        return None
    ix = torch.tensor([i-start for i in chosen], device=w.device)
    return chosen, ix, w[ix].detach().float().cpu().tolist(), g[ix].detach().float().cpu().tolist()


def probe_after(name, step, w, token):
    if token is None:
        return
    chosen, ix, before, grad = token
    after = w[ix].detach().float().cpu().tolist()
    PROBES.append({'name': name, 'step': step, 'indices': chosen,
                   'weight_before': before, 'gradient': grad, 'weight_after': after})


class TrackedQuantMoments(OriginalMoments):
    def update(self, start, w, g, step):
        token = probe_before(self.name, self.n, start, w, g, step)
        super().update(start, w, g, step)
        probe_after(self.name, step, w, token)


class CpuMoments:
    def __init__(self, n, device, block=128, lr=1e-5, wd=.01, window=1048576):
        self.n = n
        self.lr = lr
        self.wd = wd
        self.window = window
        self.m = torch.zeros(n, dtype=torch.float32)
        self.v = torch.zeros(n, dtype=torch.float32)
        self.updated = 0
        self.progress = None

    def bytes(self):
        return self.n * 8

    @torch.no_grad()
    def update(self, start, w, g, step):
        token = probe_before(self.name, self.n, start, w, g, step)
        for off in range(0, w.numel(), self.window):
            length = min(self.window, w.numel()-off)
            begin = start + off
            t = time.monotonic()
            host_w = w[off:off+length].float().cpu().clone()
            host_g = g[off:off+length].float().cpu()
            TIMES['host_copy_seconds'] += time.monotonic()-t
            if not torch.isfinite(host_g).all():
                raise FloatingPointError('Nonfinite reference gradient')
            t = time.monotonic()
            m = self.m[begin:begin+length]
            v = self.v[begin:begin+length]
            m.mul_(.9).add_(host_g, alpha=.1)
            v.mul_(.999).addcmul_(host_g, host_g, value=.001)
            host_w.mul_(1-self.lr*self.wd).addcdiv_(
                m/(1-.9**step), (v/(1-.999**step)).sqrt().add_(1e-8), value=-self.lr)
            TIMES['cpu_adam_seconds'] += time.monotonic()-t
            t = time.monotonic()
            w[off:off+length].copy_(host_w)
            torch.npu.current_stream(w.device).synchronize()
            TIMES['device_copy_seconds'] += time.monotonic()-t
            self.updated += length
            if self.progress and (self.updated % (8 << 20) == 0 or off+length == w.numel()):
                self.progress()
        probe_after(self.name, step, w, token)


class RawCodec:
    """Uncompressed FP32 device storage; copies avoid retaining parameter gradients."""
    fallbacks = 0
    encodes = 0
    decodes = 0

    def encode(self, w):
        w = w.contiguous().view(-1)
        return {'kind': 'raw', 'n': w.numel(), 'bytes': w.numel()*4,
                'data': w.view(torch.uint8).clone(), 'header': None}

    def decode(self, frame, output=None):
        raw = frame['data'][:frame['n']*4].view(torch.float32)
        if output is None:
            return raw.clone()
        output.copy_(raw)
        return output


def configure(mode):
    global MODE
    MODE = mode
    base.QuantMoments = TrackedQuantMoments if mode == 'compressed' else CpuMoments
    base.LinearFunction = BF16LinearFunction if mode == 'bf16' else FP32LinearFunction


def precision_context():
    return torch.autocast('npu', dtype=torch.bfloat16) if MODE == 'bf16' else contextlib.nullcontext()


@contextlib.contextmanager
def evaluation_context(canonical=True):
    previous = base.LinearFunction
    if canonical:
        base.LinearFunction = FP32LinearFunction
    try:
        with (torch.autocast('npu', enabled=False) if canonical else precision_context()):
            yield
    finally:
        base.LinearFunction = previous


def load_frames(folder, report, name, device):
    frames = base.load_frames(folder, report, name, device)
    if MODE == 'compressed':
        return frames
    decoder = Codec(__import__('pathlib').Path(__file__).resolve().parent/'libonline_codec.so')
    raw = RawCodec()
    for i, frame in enumerate(frames):
        if frame['kind'] != 'raw':
            value = decoder.decode(frame)
            frames[i] = raw.encode(value)
            del value, frame
    return frames


def WeightStore(name, shape, frames, codec, device, block, lr, wd, window):
    s = base.WeightStore(name, shape, frames, codec if MODE == 'compressed' else RawCodec(),
                         device, block, lr, wd, window)
    s.moments.name = name
    return s


class BF16LinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, proxy, s):
        ctx.s = s
        ctx.input_device, ctx.input_dtype, ctx.shape = x.device, x.dtype, x.shape
        with torch.npu.device(s.device):
            z = x.detach().to(device=s.device, dtype=torch.bfloat16).reshape(-1, s.shape[1])
            ctx.save_for_backward(z)
            if s.n <= 128_000_000:
                w = s.read_all().to(torch.bfloat16)
                y = F.linear(z, w)
            else:
                outputs = []
                for row in range(0, s.shape[0], 4096):
                    count = min(4096, s.shape[0]-row)
                    w = s.read_range(row*s.shape[1], count*s.shape[1]).view(count, -1).to(torch.bfloat16)
                    outputs.append(F.linear(z, w))
                y = torch.cat(outputs, dim=-1)
            return y.reshape(*x.shape[:-1], s.shape[0]).to(x.device)

    @staticmethod
    def backward(ctx, dy):
        s = ctx.s
        (x,) = ctx.saved_tensors
        with torch.no_grad(), torch.npu.device(s.device):
            gy = dy.to(device=s.device, dtype=torch.bfloat16).reshape(-1, s.shape[0])
            if s.n <= 128_000_000:
                w = s.read_all()
                gx = (gy @ w.to(torch.bfloat16)).float()
                g = (gy.T @ x).float()
                if s.enabled:
                    s.update_all(w, g)
            else:
                gx = torch.zeros(x.shape, device=s.device, dtype=torch.float32)
                for row in range(0, s.shape[0], 4096):
                    count = min(4096, s.shape[0]-row)
                    start = row*s.shape[1]
                    w = s.read_range(start, count*s.shape[1]).view(count, -1)
                    sy = gy[:, row:row+count]
                    gx.add_((sy @ w.to(torch.bfloat16)).float())
                    g = (sy.T @ x).float()
                    if s.enabled:
                        s.moments.update(start, w.view(-1), g.view(-1), s.step)
                        s.write_range(start, w.view(-1))
                    del w, g
                if s.enabled:
                    s.changed_steps += 1
            return gx.reshape(ctx.shape).to(device=ctx.input_device, dtype=ctx.input_dtype), None, None
