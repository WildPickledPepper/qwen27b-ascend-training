"""Bounded FP32 CPUAdam pipeline across independent parameter frames.

Raw FP32 device frames are persistent destinations. Input-gradient matmul finishes
before each submission. Each layer's old weight read buffers are separate clones.
All jobs drain before the next forward/evaluation; no stale-gradient updates.
Embedding and tiny native tensors retain the synchronous controlled adapters.
"""
import concurrent.futures
import importlib.util
import os
import threading
import time
import types
import torch
import bench_cpu_master as sync
import bench_resident as bench
from bench_resident import PROBES, TIMES, probe_before

DETAIL = {'d2h_event_seconds': 0., 'h2d_event_seconds': 0.,
          'cpu_adam_seconds': 0., 'weight_stage_seconds': 0.,
          'slot_wait_seconds': 0., 'submitted_coordinates': 0,
          'completed_coordinates': 0, 'windows': 0, 'audited_coordinates': 0}
POOLS = {}
RUNTIME = {}
_OP = None
_EXECUTOR = None


def op():
    global _OP
    if _OP is None:
        spec = importlib.util.spec_from_file_location('cpu_adam_async', os.environ['CPU_ADAM_ASYNC_EXTENSION'])
        _OP = importlib.util.module_from_spec(spec); spec.loader.exec_module(_OP)
        _OP.create_adam(7202, 1e-5, .9, .999, 1e-8, .01, True, True)
    return _OP


def executor():
    global _EXECUTOR
    if _EXECUTOR is None:
        op()
        def initialize_worker():
            torch.set_num_threads(24)
            RUNTIME.update(worker_native_id=threading.get_native_id(),
                           worker_torch_threads=torch.get_num_threads(),
                           worker_affinity=sorted(os.sched_getaffinity(0)))
        _EXECUTOR = concurrent.futures.ThreadPoolExecutor(max_workers=1, initializer=initialize_worker)
    return _EXECUTOR


class Pool:
    def __init__(self, device, window):
        self.device = torch.device(device)
        with torch.npu.device(device):
            self.d2h = torch.npu.Stream(); self.h2d = torch.npu.Stream()
            self.slots = []
            for _ in range(4):
                self.slots.append({'host': torch.empty((2, window), dtype=torch.float32, pin_memory=True),
                                   'events': [torch.npu.Event(enable_timing=True) for _ in range(4)],
                                   'future': None, 'audit': None})
        self.next = 0

    def retire(self, slot):
        if slot['future'] is None: return
        t = time.monotonic()
        slot['future'].result()
        slot['events'][3].synchronize()
        DETAIL['slot_wait_seconds'] += time.monotonic()-t
        DETAIL['d2h_event_seconds'] += slot['events'][0].elapsed_time(slot['events'][1])/1000
        DETAIL['h2d_event_seconds'] += slot['events'][2].elapsed_time(slot['events'][3])/1000
        audit = slot['audit']
        if audit:
            token, state, start, step, expected, targets = audit
            chosen, _, before, gradient = token
            got = {}
            with torch.npu.device(self.device):
                for begin, dest in targets:
                    selected = [i for i in chosen if begin <= i < begin+dest.numel()]
                    if selected:
                        ix = torch.tensor([i-begin for i in selected], device=self.device)
                        vals = dest[ix].cpu().tolist()
                        got.update(zip(selected, vals))
            after = [got[i] for i in chosen]
            assert torch.equal(torch.tensor(after).view(torch.int32), torch.tensor(expected).view(torch.int32)), 'CPU master/device copy audit failed'
            DETAIL['audited_coordinates'] += len(chosen)
            PROBES.append({'name': state.name, 'step': step, 'indices': chosen,
                           'weight_before': before, 'gradient': gradient, 'weight_after': after})
        slot['future'] = None; slot['audit'] = None

    def acquire(self):
        slot = self.slots[self.next % len(self.slots)]; self.next += 1
        self.retire(slot)
        return slot


def pool(device, window):
    key = (str(device), window)
    if key not in POOLS: POOLS[key] = Pool(device, window)
    return POOLS[key]


def flush():
    for buffers in POOLS.values():
        for slot in buffers.slots: buffers.retire(slot)
    assert DETAIL['completed_coordinates'] == DETAIL['submitted_coordinates']


class PipelinedMoments(sync.CpuMasterMoments):
    def bind_frames(self, frames):
        self.frames = frames
        self.initialize_frames(frames)

    def targets(self, start, n):
        result = []; offset = 0
        for frame in self.frames:
            end = offset+frame['n']; lo = max(offset, start); hi = min(end, start+n)
            if hi > lo:
                result.append((lo, frame['data'][(lo-offset)*4:(hi-offset)*4].view(torch.float32)))
            offset = end
            if offset >= start+n: break
        assert sum(x.numel() for _, x in result) == n
        return result

    @torch.no_grad()
    def update(self, start, w, g, step):
        if getattr(self, 'synchronous', False):
            return super().update(start, w, g, step)
        assert self.initialized and w.dtype == g.dtype == torch.float32
        if not torch.isfinite(g).all().item(): raise FloatingPointError('Nonfinite pipeline gradient')
        buffers = pool(w.device, self.window)
        for off in range(0, w.numel(), self.window):
            n = min(self.window, w.numel()-off); begin = start+off
            token = probe_before(self.name, self.n, begin, w[off:off+n], g[off:off+n], step)
            slot = buffers.acquire(); targets = self.targets(begin, n)
            ready = torch.npu.Event(); ready.record(torch.npu.current_stream(w.device))
            buffers.d2h.wait_event(ready)
            with torch.npu.stream(buffers.d2h):
                slot['events'][0].record()
                slot['host'][0, :n].copy_(g[off:off+n], non_blocking=True)
                g.record_stream(buffers.d2h)
                slot['events'][1].record()
            DETAIL['submitted_coordinates'] += n
            slot['future'] = executor().submit(self.work, buffers, slot, begin, n, step, token, targets)

    @torch.no_grad()
    def work(self, buffers, slot, begin, n, step, token, targets):
        # The private binding releases the GIL during unmodified CPUAdam math.
        with torch.npu.device(buffers.device):
            slot['events'][1].synchronize()
            t = time.monotonic()
            op().adam_update(7202, step, self.lr, .9, .999, 1e-8, self.wd, True,
                             self.master[begin:begin+n], slot['host'][0, :n],
                             self.m[begin:begin+n], self.v[begin:begin+n])
            elapsed = time.monotonic()-t
            DETAIL['cpu_adam_seconds'] += elapsed; TIMES['cpu_adam_seconds'] += elapsed
            t = time.monotonic(); slot['host'][1, :n].copy_(self.master[begin:begin+n])
            DETAIL['weight_stage_seconds'] += time.monotonic()-t
            expected = None
            if token:
                ix = torch.tensor([i-begin for i in token[0]])
                expected = self.master[begin:begin+n][ix].tolist()
            with torch.npu.stream(buffers.h2d):
                slot['events'][2].record()
                for offset, dest in targets:
                    rel = offset-begin
                    dest.copy_(slot['host'][1, rel:rel+dest.numel()], non_blocking=True)
                slot['events'][3].record()
            slot['audit'] = (token, self, begin, step, expected, targets) if token else None
            self.updated += n
            DETAIL['completed_coordinates'] += n; DETAIL['windows'] += 1


def install_store(store, is_embedding=False):
    if is_embedding:
        store.moments.synchronous = True
        store.moments.initialize_frames(store.frames)
        return
    store.moments.bind_frames(store.frames)

    def update_all(s, w, g):
        offset = 0
        for frame in s.frames:
            n = frame['n']
            s.moments.update(offset, w.view(-1)[offset:offset+n], g.view(-1)[offset:offset+n], s.step)
            offset += n
        s.changed_steps += 1

    def write_range(s, start, w):
        # update() already targets the persistent raw device frames.
        assert 0 <= start and start+w.numel() <= s.n

    store.update_all = types.MethodType(update_all, store)
    store.write_range = types.MethodType(write_range, store)


def install():
    bench.base.QuantMoments = PipelinedMoments
