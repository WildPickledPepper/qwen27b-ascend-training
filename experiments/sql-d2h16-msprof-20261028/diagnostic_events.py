"""Fixed16-slot snapshots and cross-stream device events, profiling step only."""
import json
import threading
import time


class Diagnostics:
    def __init__(self, torch, slots, rank):
        assert slots == 16
        self.torch = torch
        self.rank = rank
        self.lock = threading.Lock()
        self.active = False
        self.states = [dict(state='free') for _ in range(slots)]
        self.cases = []
        self.sequence = 0
        self.anchor = None

    def reset(self, active):
        self.active = active
        self.cases = []
        self.sequence = 0
        with self.lock:
            self.states = [dict(state='free') for _ in self.states]
        if active:
            self.anchor_host_before = time.monotonic()
            self.anchor = self.torch.npu.Event(enable_timing=True)
            self.anchor.record(self.torch.npu.current_stream())
            self.anchor_host_after = time.monotonic()

    def producer(self, idx, grad, event, entered, admitted):
        if not self.active:
            return None
        self.sequence += 1
        return dict(sequence=self.sequence, param=idx, elements=grad.numel(),
                    producer_hook_host=entered, producer_admit_host=admitted,
                    _producer_event=event)

    def begin(self, owner, slot, start, count):
        if owner is None:
            return None
        d = {k:v for k,v in owner.items()}
        d.update(slot=slot, element_offset=start, bytes=count*2,
                 copy_dispatch_host=time.monotonic())
        with self.lock:
            self.states[slot] = dict(state='reserved_or_copying', sequence=owner['sequence'], param=owner['param'])
        e = self.torch.npu.Event(enable_timing=True)
        e.record(self.torch.npu.current_stream())
        d['_copy_start'] = e
        return d

    def submitted(self, d, end):
        if d is not None:
            d['_copy_end'] = end
            d['copy_submitted_host'] = time.monotonic()

    def completed(self, d):
        if d is not None:
            d['completion_notice_host'] = time.monotonic()
            with self.lock:
                self.states[d['slot']]['state'] = 'ready'
            self.cases.append(d)

    def snapshot(self, pending_bytes, ready_count):
        if not self.active:
            return None
        with self.lock:
            counts = {}
            for x in self.states:
                counts[x['state']] = counts.get(x['state'], 0)+1
            return dict(host=time.monotonic(), counts=counts, ready_queue_size=ready_count,
                        device_backlog_bytes=pending_bytes,
                        slots=[dict(x) for x in self.states])

    def consuming(self, d, wait_start, got, snapshot):
        if d is not None:
            d.update(cpu_get_wait_start_host=wait_start, cpu_get_return_host=got,
                     snapshot_before_get=snapshot)
            with self.lock:
                self.states[d['slot']]['state'] = 'cpu_processing'

    def consumed(self, d, start, end):
        if d is not None:
            d.update(cpu_add_start_host=start, cpu_add_end_host=end)
            with self.lock:
                self.states[d['slot']] = dict(state='free')

    def export(self, out, step, origin):
        if not self.active:
            return
        result = []
        for c in self.cases:
            d = {k:v for k,v in c.items() if not k.startswith('_')}
            # All completion events have finished; no timing query in supply path.
            p = self.anchor.elapsed_time(c['_producer_event'])
            s = self.anchor.elapsed_time(c['_copy_start'])
            e = self.anchor.elapsed_time(c['_copy_end'])
            assert p >= -.01 and s >= p-.01 and e >= s-.01, (p,s,e)
            d.update(device_source_ready_ms=p, device_copy_start_ms=s, device_copy_end_ms=e,
                     device_ready_to_copy_ms=s-p, device_copy_service_ms=e-s)
            for k in list(d):
                if k.endswith('_host'):
                    d[k] -= origin
            snap = d.get('snapshot_before_get')
            if snap is not None:
                snap = dict(snap); snap['host'] -= origin; d['snapshot_before_get'] = snap
            result.append(d)
        payload = dict(rank=self.rank, step=step, slots=16, bytes_per_slot=128*2**20,
                       anchor_host_before=self.anchor_host_before-origin,
                       anchor_host_after=self.anchor_host_after-origin,
                       chunks=result,
                       limits=['Host monotonic and device-anchor elapsed clocks are separate axes.',
                               'Device copy interval is bounded by timing events after source dependency; includes event overhead.',
                               'Slot state snapshots have small host bookkeeping race; qsize is advisory.',
                               'Profiling instrumentation changes timing; not normal throughput data.'])
        (out/f'device-diagnostics-rank{self.rank}-step{step}.private.json').write_text(json.dumps(payload))
