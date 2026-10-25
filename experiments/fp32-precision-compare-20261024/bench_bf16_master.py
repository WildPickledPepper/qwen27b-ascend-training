"""Persistent BF16 linear weights, CPU FP32 master/m/v, explicit mixed compute.

No global autocast: recurrent/normalization/attention math retains the reference
FP32 path. Only linear matmuls use BF16, with FP32 outputs and chunk reductions.
Embedding and tiny native parameters retain FP32. No shared library edits.
"""
import contextlib
import types
import threading
import torch
import torch.nn.functional as F
import bench_resident as bench
import bench_cpu_pipeline as pipeline

CANONICAL = False
COUNTS = {'forward_bf16_matmuls': 0, 'backward_bf16_matmuls': 0}
SUBMIT_LOCK = threading.RLock()


class MixedMoments(pipeline.PipelinedMoments):
    def targets(self, start, n):
        result = []; offset = 0
        for frame in self.frames:
            end = offset + frame['n']; lo = max(offset, start); hi = min(end, start+n)
            if hi > lo:
                result.append((lo, frame['data'].view(torch.bfloat16)[lo-offset:hi-offset]))
            offset = end
            if offset >= start+n: break
        assert sum(x.numel() for _, x in result) == n
        return result

    def update(self, start, w, g, step):
        # Master update always uses the CPU FP32 master, never this rounded view.
        with SUBMIT_LOCK, torch.npu.device(w.device), torch.no_grad():
            w=w.float();g=g.float()
            if getattr(self,'synchronous',False):return super().update(start,w,g,step)
            assert self.initialized
            if not torch.isfinite(g).all().item():raise FloatingPointError('Nonfinite mixed gradient')
            buffers=pipeline.pool(w.device,self.window)
            for off in range(0,w.numel(),self.window):
                n=min(self.window,w.numel()-off);begin=start+off
                token=pipeline.probe_before(self.name,self.n,begin,w[off:off+n],g[off:off+n],step)
                slot=buffers.acquire();targets=self.targets(begin,n)
                # Keep the producer event alive until the slot is retired. Native
                # autograd hooks may execute under a different current device.
                ready=torch.npu.Event();ready.record(torch.npu.current_stream(w.device))
                slot['producer_ready']=ready
                buffers.d2h.wait_event(ready)
                with torch.npu.stream(buffers.d2h):
                    slot['events'][0].record()
                    slot['host'][0,:n].copy_(g[off:off+n],non_blocking=True)
                    g.record_stream(buffers.d2h)
                    slot['events'][1].record()
                pipeline.DETAIL['submitted_coordinates']+=n
                slot['future']=pipeline.executor().submit(self.work,buffers,slot,begin,n,step,token,targets)

    def work(self, *args):
        super().work(*args)
        slot = args[1]
        if slot['audit']:
            token, state, begin, step, expected, targets = slot['audit']
            rounded = torch.tensor(expected).to(torch.bfloat16).float().tolist()
            slot['audit'] = token, state, begin, step, rounded, targets


def install_store(store, is_embedding=False):
    if is_embedding:
        pipeline.install_store(store, is_embedding=True)
        return
    pipeline.install_store(store)
    # initialize_frames above captured the original exact FP32 master first.
    for frame in store.frames:
        frame['data'] = frame['data'][:frame['n']*4].view(torch.float32).to(torch.bfloat16).view(torch.uint8)
        frame['kind'] = 'bf16'; frame['bytes'] = frame['n']*2

    def read_range(s, start, n):
        with torch.npu.device(s.device):
            if CANONICAL:
                return s.moments.master[start:start+n].to(s.device)
            out = torch.empty(n, dtype=torch.bfloat16, device=s.device)
            offset = 0
            for frame in s.frames:
                end = offset+frame['n']; lo = max(start, offset); hi = min(start+n, end)
                if hi > lo:
                    out[lo-start:hi-start].copy_(frame['data'].view(torch.bfloat16)[lo-offset:hi-offset])
                offset = end
                if offset >= start+n: break
            return out

    def read_all(s):
        return s.read_range(0, s.n).reshape(s.shape)

    store.read_range = types.MethodType(read_range, store)
    store.read_all = types.MethodType(read_all, store)


class MixedLinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, proxy, s):
        ctx.s = s; ctx.shape = x.shape; ctx.input_device = x.device; ctx.input_dtype = x.dtype
        with torch.npu.device(s.device), torch.autocast('npu', enabled=False):
            z = x.detach().to(device=s.device, dtype=torch.bfloat16).reshape(-1, s.shape[1])
            ctx.save_for_backward(z)
            outputs = []
            for row in range(0, s.shape[0], 4096):
                count = min(4096, s.shape[0]-row)
                w = s.read_range(row*s.shape[1], count*s.shape[1]).view(count, -1)
                assert w.dtype == z.dtype == torch.bfloat16
                outputs.append(F.linear(z, w).float()); COUNTS['forward_bf16_matmuls'] += 1
            y = torch.cat(outputs, dim=-1)
            return y.reshape(*x.shape[:-1], s.shape[0]).to(ctx.input_device)

    @staticmethod
    def backward(ctx, dy):
        s = ctx.s; (x,) = ctx.saved_tensors
        with torch.no_grad(), torch.npu.device(s.device), torch.autocast('npu', enabled=False):
            gy = dy.to(device=s.device, dtype=torch.bfloat16).reshape(-1, s.shape[0])
            gx = torch.zeros(x.shape, dtype=torch.float32, device=s.device)
            for row in range(0, s.shape[0], 4096):
                count = min(4096, s.shape[0]-row); start = row*s.shape[1]
                w = s.read_range(start, count*s.shape[1]).view(count, -1)
                sy = gy[:, row:row+count]
                gx.add_((sy @ w).float())
                g = (sy.T @ x).float(); COUNTS['backward_bf16_matmuls'] += 2
                if s.enabled:
                    s.moments.update(start, w.view(-1), g.view(-1), s.step)
                del w, g
            if s.enabled: s.changed_steps += 1
            return gx.reshape(ctx.shape).to(device=ctx.input_device, dtype=ctx.input_dtype), None, None


def precision_context():
    return torch.autocast('npu', enabled=False)


@contextlib.contextmanager
def evaluation_context(canonical=True):
    global CANONICAL
    previous = bench.base.LinearFunction; old = CANONICAL
    CANONICAL = canonical
    bench.base.LinearFunction = bench.FP32LinearFunction if canonical else MixedLinear
    try:
        with torch.autocast('npu', enabled=False): yield
    finally:
        CANONICAL = old; bench.base.LinearFunction = previous


def install():
    bench.base.QuantMoments = MixedMoments
    bench.base.LinearFunction = MixedLinear
