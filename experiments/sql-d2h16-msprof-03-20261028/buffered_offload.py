"""Bounded device backlog -> D2H stream -> ready-only CPU FIFO.

Keeps the exact per-parameter BF16 addition order. No optimizer math changes.
The CPU slots themselves are the receiving buffers; no extra host memcpy.
"""
import queue
import threading
import time
import traceback
from diagnostic_events import Diagnostics


def create_base_offload(torch, params, rank, device, args, scope):
    class BufferedOffload:
        def __init__(self):
            self.chunk = 65536 if args.tiny else 64 * 1024 * 1024
            self.budget = 2**20 if args.tiny else 512 * 2**20
            self.accum = [torch.zeros(p.numel(), dtype=torch.bfloat16) for p in params]
            self.slots = [torch.empty(self.chunk, dtype=torch.bfloat16, pin_memory=True) for _ in range(args.slots)]
            self.free = queue.Queue()
            for i in range(args.slots):
                self.free.put(i)
            self.pending = queue.Queue()
            self.completions = queue.Queue()
            self.jobs = queue.Queue()  # Contains only completed D2H jobs.
            self.condition = threading.Condition()
            self.pending_bytes = 0
            self.pending_peak = 0
            self.error = None
            self.copy_stream = torch.npu.Stream(device=device)
            self.bytes = self.wait = self.copy_wait = self.cpu_add = 0
            self.copy_slot_wait = 0
            self.cpu_spans = []
            self.host_trace = []
            self.active_start = time.monotonic()
            self.diag = Diagnostics(torch,args.slots,rank)
            self.thread = threading.Thread(target=self.worker, daemon=True)
            self.completer = threading.Thread(target=self.complete, daemon=True)
            self.copier = threading.Thread(target=self.copy, daemon=True)
            for t in [self.thread, self.completer, self.copier]:
                t.start()
            self.hooks = [p.register_post_accumulate_grad_hook(self.hook(i)) for i, p in enumerate(params)]

        def fail(self):
            with self.condition:
                self.error = traceback.format_exc()
                self.condition.notify_all()

        def check(self):
            assert self.error is None, self.error

        def span(self, name, lane, start, end):
            self.host_trace.append(dict(name=name, lane=lane, start=start, end=end))

        def free_slot(self):
            t = time.monotonic()
            while True:
                self.check()
                try:
                    slot = self.free.get(timeout=.2)
                    break
                except queue.Empty:
                    pass
            end = time.monotonic()
            self.copy_slot_wait += end-t
            self.span('copy_dispatch/wait_cpu_slot', 'copy_dispatch', t, end)
            return slot

        def hook(self, idx):
            def take(param):
                hook_enter = time.monotonic()
                grad = param.grad.detach().view(-1)
                size = 0 if idx in getattr(self, 'resident', {}) else grad.numel() * grad.element_size()
                # Exact producer-stream dependency, not a device-wide sync.
                produced = torch.npu.Event(enable_timing=self.diag.active)
                produced.record(torch.npu.current_stream())
                t = time.monotonic()
                with self.condition:
                    while self.pending_bytes and self.pending_bytes+size > self.budget:
                        self.check()
                        self.condition.wait(timeout=.2)
                    self.check()
                    self.pending_bytes += size
                    self.pending_peak = max(self.pending_peak, self.pending_bytes)
                end = time.monotonic()
                self.wait += end-t
                self.span('main/wait_device_backlog_budget', 'main', t, end)
                owner = self.diag.producer(idx,grad,produced,hook_enter,end)
                self.pending.put((idx,grad,produced,size,owner))
                param.grad = None
            def marked(param):
                with scope(f"diag/gradient_hook/p{idx}/n{params[idx].numel()}"):
                    return take(param)
            return marked

        def copy(self):
            torch.npu.set_device(rank)
            try:
                while True:
                    job = self.pending.get()
                    if job is None:
                        self.pending.task_done()
                        return
                    idx, grad, produced, size, owner = job
                    # Keep source ownership until the final copy completes.
                    with torch.npu.stream(self.copy_stream):
                        self.copy_stream.wait_event(produced)
                        grad.record_stream(self.copy_stream)
                        for start in range(0, grad.numel(), self.chunk):
                            slot = self.free_slot()
                            n = min(self.chunk, grad.numel()-start)
                            t = time.monotonic()
                            with scope(f"diag/D2H/p{idx}/offset{start}/bytes{n*2}"):
                                diagnostic = self.diag.begin(owner,slot,start,n)
                                self.slots[slot][:n].copy_(grad[start:start+n],non_blocking=True)
                                event = torch.npu.Event(enable_timing=owner is not None)
                                event.record(self.copy_stream)
                                self.diag.submitted(diagnostic,event)
                            self.span('D2H/enqueue_only', 'copy_dispatch', t, time.monotonic())
                            last = start+n == grad.numel()
                            self.completions.put((slot,idx,start,n,event,grad,size if last else 0,diagnostic))
                            self.bytes += n*2
                    self.pending.task_done()
                    del grad, job
            except BaseException:
                self.fail()

        def complete(self):
            torch.npu.set_device(rank)
            try:
                while True:
                    job = self.completions.get()
                    if job is None:
                        self.completions.task_done()
                        return
                    slot,idx,start,n,event,owner,release,diagnostic = job
                    t = time.monotonic()
                    event.synchronize()
                    end = time.monotonic()
                    self.copy_wait += end-t
                    self.span('D2H/wait_completion_includes_dependencies', 'completion_monitor', t, end)
                    self.diag.completed(diagnostic)
                    self.jobs.put((slot,idx,start,n,diagnostic))
                    if release:
                        with self.condition:
                            self.pending_bytes -= release
                            self.condition.notify_all()
                    self.completions.task_done()
                    del owner, job
            except BaseException:
                self.fail()

        def worker(self):
            try:
                while True:
                    t = time.monotonic()
                    snapshot = self.diag.snapshot(self.pending_bytes,self.jobs.qsize()) if self.jobs.empty() else None
                    job = self.jobs.get()
                    end = time.monotonic()
                    if job is None:
                        self.jobs.task_done()
                        return
                    self.span('CPU/wait_ready_data', 'cpu_accumulator', max(t, self.active_start), end)
                    slot,idx,start,n,diagnostic = job
                    self.diag.consuming(diagnostic,max(t,self.active_start),end,snapshot)
                    t = time.monotonic()
                    self.accum[idx][start:start+n].add_(self.slots[slot][:n])
                    end = time.monotonic()
                    self.cpu_add += end-t
                    self.cpu_spans.append((t, end))
                    self.span('CPU/BF16_gradient_add', 'cpu_accumulator', t, end)
                    self.diag.consumed(diagnostic,t,end)
                    self.free.put(slot)
                    self.jobs.task_done()
            except BaseException:
                self.fail()

        def join_queue(self, q):
            # Poll only on the bounded drain path so errors cannot silently hang.
            while True:
                self.check()
                with q.all_tasks_done:
                    if not q.unfinished_tasks:
                        return
                    q.all_tasks_done.wait(timeout=.2)

        def drain(self):
            for q in [self.pending, self.completions, self.jobs]:
                self.join_queue(q)
            self.check()
            assert self.pending_bytes == 0

        def reset(self):
            self.drain()
            for buf in self.accum:
                buf.zero_()
            self.bytes = self.wait = self.copy_wait = self.cpu_add = 0
            self.copy_slot_wait = self.pending_peak = 0
            self.cpu_spans = []
            self.host_trace = []
            self.active_start = time.monotonic()
            self.diag.reset(bool(getattr(args,"profile_active",False)))

        def metrics(self):
            spans = list(self.cpu_spans)
            active = spans[-1][1]-spans[0][0] if spans else 0
            busy = sum(b-a for a, b in spans)
            gaps = [max(0, b[0]-a[1]) for a, b in zip(spans, spans[1:])]
            return dict(cpu_add_count=len(spans), cpu_first_to_last_span=active,
                        cpu_add_busy_seconds=busy, cpu_between_add_gap_seconds=sum(gaps),
                        cpu_max_between_add_gap=max(gaps, default=0),
                        cpu_busy_fraction=busy/active if active else None,
                        copy_thread_wait_free_slot_seconds=self.copy_slot_wait,
                        main_thread_wait_device_budget_seconds=self.wait,
                        device_backlog_peak_bytes=self.pending_peak,
                        device_backlog_budget_bytes=self.budget,
                        cpu_receiving_slot_bytes=len(self.slots)*self.chunk*2,
                        scope='CPU busy over first-to-last add; excludes initial fill/final tail. Gaps include scheduling overhead; not all pure starvation.')

        def close(self):
            self.drain()
            for q, thread in [(self.pending, self.copier), (self.completions, self.completer), (self.jobs, self.thread)]:
                q.put(None)
                self.join_queue(q)
                thread.join(timeout=10)
                assert not thread.is_alive()
    return BufferedOffload
