"""Controlled CPU BF16 add replay; no model weights, forward, backward or Adam.

One idle physical3, fixed CPU96:119/NUMA preferred4, fixed allocation addresses.
Use aten::add_ CPU profiler duration vs outer Python wall to separate op time.
This does not reconstruct historical NUMA placement or real gradient production.
"""
import os, ctypes, json, queue, threading, time, traceback, hashlib, subprocess, resource
from pathlib import Path

root = Path(__file__).resolve().parent
assert os.environ['ASCEND_RT_VISIBLE_DEVICES'] == '3'
smi = subprocess.check_output(['npu-smi', 'info'], text=True)
assert 'No running processes found in NPU 3' in smi
os.sched_setaffinity(0, list(range(96, 120)))
lib = ctypes.CDLL('libnuma.so.1', use_errno=True)
lib.set_mempolicy.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_ulong), ctypes.c_ulong]
mask = ctypes.c_ulong(1 << 4)
assert lib.set_mempolicy(1, ctypes.byref(mask), 8) == 0
import torch, torch_npu
torch.set_num_threads(24)
torch.npu.set_device(0)
chunk = 64 * 1024 * 1024
jobs = 128
destination = torch.zeros(13447996672, dtype=torch.bfloat16)
slots = [torch.ones(chunk, dtype=torch.bfloat16, pin_memory=True) for _ in range(32)]
source = torch.ones(chunk, dtype=torch.bfloat16, device='npu:0')
stream = torch.npu.Stream(device=0)
torch.npu.synchronize()
state = dict(state='running', scope='Controlled isolated replay; no training or model checkpoint', results=[])

def save():
    (root/'results.private.json').write_text(json.dumps(state, indent=2))

def run(name, nslots, copying, tracing, rep):
    destination.zero_()
    trace = []
    free = queue.Queue()
    for i in range(nslots):
        free.put(i)
    completed = queue.Queue()
    ready = queue.Queue()
    errors = []
    result = dict(name=name, repetition=rep, slots=nslots, copying=copying, host_tracing=tracing)

    def span(name, a, b):
        if tracing:
            trace.append((name, a, b))

    def copier():
        try:
            torch.npu.set_device(0)
            with torch.npu.stream(stream):
                for i in range(jobs):
                    t = time.monotonic(); slot = free.get(); span('free_slot', t, time.monotonic())
                    t = time.monotonic()
                    if name != 'event32':
                        slots[slot].copy_(source, non_blocking=True)
                    event = torch.npu.Event(); event.record(stream)
                    span('enqueue', t, time.monotonic())
                    completed.put((i, slot, event))
        except BaseException:
            errors.append(traceback.format_exc())

    def monitor():
        try:
            torch.npu.set_device(0)
            for _ in range(jobs):
                i, slot, event = completed.get()
                t = time.monotonic(); event.synchronize(); span('completion', t, time.monotonic())
                ready.put((i, slot))
        except BaseException:
            errors.append(traceback.format_exc())

    def consumer():
        try:
            durations = []; thread_cpu = []; waits = []
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU], record_shapes=False, with_stack=False, profile_memory=False) as prof:
                started = time.monotonic()
                for i in range(jobs):
                    t = time.monotonic()
                    if copying:
                        index, slot = ready.get()
                        assert index == i
                    else:
                        slot = i % nslots
                    end = time.monotonic(); waits.append(end-t); span('ready', t, end)
                    target = destination[i*chunk:(i+1)*chunk]
                    tc = time.thread_time(); t = time.monotonic()
                    target.add_(slots[slot])
                    end = time.monotonic(); thread_cpu.append(time.thread_time()-tc); durations.append(end-t)
                    span('add', t, end)
                    if copying:
                        free.put(slot)
                elapsed = time.monotonic()-started
            ops = [e for e in prof.key_averages() if e.key == 'aten::add_']
            assert len(ops) == 1 and ops[0].count == jobs
            result.update(cpu_add_outer_wall_seconds=sum(durations), cpu_add_aten_wall_seconds=ops[0].cpu_time_total/1e6,
                          cpu_add_calling_thread_cpu_seconds=sum(thread_cpu),
                          consumer_elapsed_seconds=elapsed, ready_wait_seconds=sum(waits), trace_events=len(trace),
                          individual_add_wall_seconds=durations)
        except BaseException:
            errors.append(traceback.format_exc())

    threads = []
    if copying:
        threads += [threading.Thread(target=monitor), threading.Thread(target=copier)]
    t = time.monotonic()
    usage0 = resource.getrusage(resource.RUSAGE_SELF)
    for thread in threads:
        thread.daemon = True; thread.start()
    consumer()  # Native profiler and consumer remain on main thread.
    for thread in threads:
        thread.join(timeout=120)
        assert not thread.is_alive(), 'Trial timeout'
    assert not errors, errors
    torch.npu.synchronize()
    assert float(destination[:jobs*chunk:chunk].sum()) == jobs
    result.update(total_wall_seconds=time.monotonic()-t, processed_bytes=jobs*chunk*2,
                  profiled=True, dtype='BF16', chunk_bytes=chunk*2)
    usage1 = resource.getrusage(resource.RUSAGE_SELF)
    result['process_cpu_seconds']=(usage1.ru_utime+usage1.ru_stime)-(usage0.ru_utime+usage0.ru_stime)
    result['actual_d2h']=copying and name != 'event32'
    result['outer_minus_aten_seconds'] = result['cpu_add_outer_wall_seconds']-result['cpu_add_aten_wall_seconds']
    state['results'].append(result); save()
    print(json.dumps({k:v for k,v in result.items() if k != 'individual_add_wall_seconds'}), flush=True)

try:
    run('warmup32',32,False,False,0)
    variants = [('cpu32',32,False,False), ('event32',32,True,False),
                ('copy32',32,True,False), ('copy32trace',32,True,True)]
    for rep, cases in enumerate([variants, list(reversed(variants))], 1):
        for args in cases:
            run(*args, rep)
    state.update(state='complete', source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                 destination_bytes=destination.numel()*2, cpu_slots_bytes=32*chunk*2, npu_source_bytes=chunk*2,
                 limits=['Not a full training benchmark; gradients are pre-produced and all blocks are128MiB.',
                         'Warmup excluded; CPU profiler/consumer on main thread. Event-only controls preserve queues/events but omit memcpy.',
                         'CPU profiler used in every variant; absolute speeds need not equal unprofiled training.',
                         'Does not reproduce prior NUMA placement, compiler/CPU load, or gradient arrival gaps.'])
    save()
finally:
    del source
    torch.npu.synchronize(); torch.npu.empty_cache()
