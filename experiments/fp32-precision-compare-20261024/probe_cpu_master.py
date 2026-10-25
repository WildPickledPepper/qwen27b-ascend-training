"""CPUAdam transfer+math and numerical check versus existing NPU math, no model."""
import argparse, ctypes, gc, json, os, statistics, time
from pathlib import Path
assert ctypes.CDLL(None).prctl(41, 1, 0, 0, 0) == 0
os.sched_setaffinity(0, set(range(96, 120)) & os.sched_getaffinity(0))
import torch, torch_npu
from bench_resident_npu import NpuOffloadedMoments
from bench_cpu_master import CpuMasterMoments, DETAIL, cpu_op

p = argparse.ArgumentParser(); p.add_argument('--output', required=True); a = p.parse_args()
torch.set_num_threads(4); torch.manual_seed(20261024)
torch.npu.config.allow_internal_format = False
assert os.environ['ASCEND_RT_VISIBLE_DEVICES'] == '2,3'
cpu_op()
rows = []
for dev in [0, 1]:
 with torch.npu.device(dev):
  device = 'npu:'+str(dev); n = 32*1048576+4096
  seed = torch.randn(n, device=device)*.01
  grad = torch.randn(n, device=device)*.001
  ref = NpuOffloadedMoments(n, device, window=16*1048576); ref.name = 'cpu_master.probe'
  w = seed.clone(); ref_times = []
  # Re-running only this small optimizer test, not any model baseline.
  for step in [1, 2, 3]:
   t = time.monotonic(); ref.update(0, w, grad, step); torch.npu.synchronize(dev)
   ref_times.append(time.monotonic()-t)
  reference = [w.cpu(), ref.m.clone(), ref.v.clone()]
  del ref, w
  for threads in [4, 12, 24]:
   torch.set_num_threads(threads)
   state = CpuMasterMoments(n, device, window=16*1048576); state.name = 'cpu_master.probe'
   state.master.copy_(seed); state.initialized = True
   w = seed.clone(); times = []; before = dict(DETAIL)
   for step in [1, 2, 3]:
    t = time.monotonic(); state.update(0, w, grad, step); torch.npu.synchronize(dev)
    times.append(time.monotonic()-t)
   actual = [w.cpu(), state.m, state.v]
   checks = {}
   for name, got, expected in zip(['weight', 'm', 'v'], actual, reference):
    diff = (got-expected).double()
    checks[name] = {'finite': bool(torch.isfinite(got).all()),
                    'max_abs': float(diff.abs().max()),
                    'relative_l2': float(diff.norm()/expected.double().norm()),
                    'exact_bits': bool(torch.equal(got.view(torch.int32), expected.view(torch.int32)))}
    assert checks[name]['finite'] and checks[name]['relative_l2'] < 1e-4, checks
   row = {'logical_device': dev, 'coordinates': n, 'threads': threads,
          'reference_seconds': ref_times, 'candidate_seconds': times,
          'warm_speedup': statistics.mean(ref_times[1:])/statistics.mean(times[1:]),
          'phases': {k: DETAIL[k]-before[k] for k in DETAIL}, 'checks': checks}
   rows.append(row); print(json.dumps(row), flush=True)
   Path(a.output).write_text(json.dumps({'complete': False, 'results': rows}, indent=2))
   del state, w, actual; gc.collect(); torch.npu.empty_cache()
  del reference, seed, grad; gc.collect(); torch.npu.empty_cache()
Path(a.output).write_text(json.dumps({'complete': True, 'results': rows}, indent=2))
