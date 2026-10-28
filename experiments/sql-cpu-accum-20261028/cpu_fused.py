"""Fused BF16 gradient accumulation for the CPU-offload training path.

Modes
-----
k1 : one microbatch per accumulator touch. Same addition order and same bytes as
     ``acc.add_(grad)``, but the elementwise loop is tighter (bitwise equal
     results, verified against torch on the real slice sequence).
k2 : two microbatches are staged into a ping-pong pinned stash and the
     accumulator is read/modified/written once per pair, halving accumulator
     traffic. The summation order changes, so results are NOT bitwise equal to
     sequential accumulation.

The module needs only CPU tensors plus an event object with ``synchronize()``;
``event_factory`` is injectable so the whole staging/fusing flow can be unit
tested without a device.
"""
import ctypes
import queue
import subprocess
import threading
import time
from pathlib import Path

import torch

_HERE = Path(__file__).resolve().parent
_SO = _HERE / 'cpu_fused_accum.so'
_SRC = _HERE / 'cpu_fused_accum.c'


def build_kernel(force=False):
    if force or not _SO.exists() or _SO.stat().st_mtime < _SRC.stat().st_mtime:
        subprocess.run(['gcc', '-O3', '-fopenmp', '-march=armv8.2-a', '-fPIC', '-shared',
                        '-o', str(_SO), str(_SRC)], check=True)
    lib = ctypes.CDLL(str(_SO))
    lib.accum1.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t]
    lib.accumk.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p), ctypes.c_int,
                           ctypes.c_size_t]
    return lib


class FusedOffload:
    """Drop-in replacement for the single-worker gradient offload class."""

    def __init__(self, params, mode='k1', chunk=64 * 1024 * 1024, threads=24,
                 pinned=True, lib=None, event_factory=None, name='offload'):
        assert mode in ('k1', 'k2'), mode
        self.mode = mode
        self.k = 2 if mode == 'k2' else 1
        self.chunk = chunk
        self.name = name
        self.params = list(params)
        self.slices = [[(s, min(chunk, p.numel() - s))
                        for s in range(0, p.numel(), chunk)] for p in self.params]
        self.total = sum(p.numel() for p in self.params)
        self.accum = [torch.zeros(p.numel(), dtype=torch.bfloat16) for p in self.params]
        self.stash = [torch.empty(self.total, dtype=torch.bfloat16, pin_memory=pinned)
                      for _ in range(self.k)]
        self.offset, off = [], 0
        for p in self.params:
            self.offset.append(off)
            off += p.numel()
        assert off == self.total
        self.lib = lib or build_kernel()
        ctypes.CDLL('libgomp.so.1').omp_set_num_threads(ctypes.c_int(threads))

        self._event_factory = event_factory or (lambda: torch.npu.Event())
        self._cv = threading.Condition()
        # staged slice count per (parameter, parity); a value of 0 means the stash
        # region is free for the next microbatch of that parity
        self._stage_count = [[0] * self.k for _ in self.params]
        self._micro = [0] * len(self.params)
        self.micro = 0
        self.error = None
        self.bytes = 0
        self.fuse_calls = 0
        self.copy_wait = 0.0
        self.fuse_seconds = 0.0
        self.slot_wait = 0.0
        self.jobs = queue.Queue()
        self._stop = False
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()

    # ------------------------------------------------------------------ public
    def begin_micro(self, micro=None):
        """Call once before each backward microbatch; sets the stash parity."""
        self.micro = self.micro + 1 if micro is None else micro

    def hook(self, idx):
        def take(param, idx=idx):
            grad = param.grad.detach().view(-1)
            parity = self.micro % self.k
            with self._cv:
                if self._stage_count[idx][parity]:
                    t0 = time.monotonic()
                    self._cv.wait_for(lambda: self._stage_count[idx][parity] == 0)
                    self.slot_wait += time.monotonic() - t0
                self._micro[idx] = self.micro
            base = self.offset[idx]
            for i, (start, n) in enumerate(self.slices[idx]):
                dest = self.stash[parity][base + start:base + start + n]
                dest.copy_(grad[start:start + n], non_blocking=True)
                ev = self._event_factory()
                ev.record()
                self.bytes += n * 2
                self.jobs.put((idx, parity, i, n, ev))
            param.grad = None
        return take

    def drain(self):
        self.jobs.join()
        assert self.error is None, self.error
        with self._cv:
            leftover = [(idx, p) for idx, row in enumerate(self._stage_count)
                        for p in range(self.k) if row[p]]
        for idx, parity in leftover:
            self._fuse(idx)

    def close(self):
        self.jobs.join()
        self.jobs.put(None)

    def zero(self):
        for buf in self.accum:
            buf.zero_()
        with self._cv:
            self._stage_count = [[0] * self.k for _ in self.params]
        self.bytes = 0
        self.fuse_calls = 0
        self.copy_wait = self.fuse_seconds = self.slot_wait = 0.0

    # ------------------------------------------------------------------ worker
    def _worker(self):
        while True:
            job = self.jobs.get()
            if job is None:
                self.jobs.task_done()
                return
            idx, parity, i, n, ev = job
            try:
                t0 = time.monotonic()
                ev.synchronize()
                self.copy_wait += time.monotonic() - t0
                with self._cv:
                    self._stage_count[idx][parity] += 1
                    staged = self._stage_count[idx][parity]
                    complete = staged == len(self.slices[idx])
                    if complete and self.k == 2:
                        complete = all(self._stage_count[idx][p] == len(self.slices[idx])
                                       for p in range(self.k))
                if complete:
                    self._fuse(idx)
            except BaseException:
                import traceback
                self.error = traceback.format_exc()
            finally:
                self.jobs.task_done()

    def _fuse(self, idx):
        a = self.accum[idx]
        with self._cv:
            filled = [p for p in range(self.k) if self._stage_count[idx][p]]
        t0 = time.monotonic()
        if len(filled) == 1:
            # odd microbatch count or a one-off resident flush: single accumulate
            src = ctypes.c_void_p(self.stash[filled[0]].data_ptr() + self.offset[idx] * 2)
            self.lib.accum1(ctypes.c_void_p(a.data_ptr()), src, ctypes.c_size_t(a.numel()))
        else:
            ptrs = (ctypes.c_void_p * 2)(
                ctypes.c_void_p(self.stash[0].data_ptr() + self.offset[idx] * 2),
                ctypes.c_void_p(self.stash[1].data_ptr() + self.offset[idx] * 2))
            self.lib.accumk(ctypes.c_void_p(a.data_ptr()), ptrs, 2,
                            ctypes.c_size_t(a.numel()))
        self.fuse_calls += 1
        with self._cv:
            for p in range(self.k):
                self._stage_count[idx][p] = 0
            self._cv.notify_all()
        self.fuse_seconds += time.monotonic() - t0
