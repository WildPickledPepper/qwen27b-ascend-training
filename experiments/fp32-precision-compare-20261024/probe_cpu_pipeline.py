"""Exercise slot reuse, tails, stream lifetimes and full FP32 state equality."""
import argparse, ctypes, gc, json, os, time
from pathlib import Path
assert ctypes.CDLL(None).prctl(41, 1, 0, 0, 0) == 0
os.sched_setaffinity(0, set(range(96, 120)) & os.sched_getaffinity(0))
import torch, torch_npu
from bench_cpu_master import CpuMasterMoments
from bench_cpu_pipeline import PipelinedMoments, flush, DETAIL
p = argparse.ArgumentParser(); p.add_argument('--output', required=True); a = p.parse_args()
assert os.environ['ASCEND_RT_VISIBLE_DEVICES'] == '2,3'
torch.set_num_threads(24); torch.manual_seed(20261024)
torch.npu.config.allow_internal_format = False
rows = []
for dev in [0, 1]:
 with torch.npu.device(dev):
  device = 'npu:'+str(dev); window = 16*1048576; n = 6*window+4096
  initial = torch.randn(n, device=device)*.01
  grad = torch.randn(n, device=device)*.001
  ref = CpuMasterMoments(n, device, window=window); ref.name = 'cpu.pipeline.probe'
  ref.master.copy_(initial); ref.initialized = True; w = initial.clone()
  frames = []
  # Frame boundaries differ from optimizer windows to exercise intersecting views.
  frame_size = 8*1048576
  for begin in range(0, n, frame_size):
   part = initial[begin:begin+frame_size].clone()
   frames.append({'kind': 'raw', 'n': part.numel(), 'data': part.view(torch.uint8)})
  candidate = PipelinedMoments(n, device, window=window); candidate.name = ref.name
  candidate.bind_frames(frames)
  for step in [1, 2, 3]:
   t = time.monotonic(); ref.update(0, w, grad, step); torch.npu.synchronize(dev)
   ref_seconds = time.monotonic()-t
   old = torch.cat([f['data'].view(torch.float32) for f in frames])
   t = time.monotonic(); candidate.update(0, old, grad, step); flush(); torch.npu.synchronize(dev)
   pipeline_seconds = time.monotonic()-t
   actual = torch.cat([f['data'].view(torch.float32) for f in frames])
   checks = {'device_weight_bits': torch.equal(actual.view(torch.int32), w.view(torch.int32)),
             'cpu_master_bits': torch.equal(candidate.master.view(torch.int32), ref.master.view(torch.int32)),
             'm_bits': torch.equal(candidate.m.view(torch.int32), ref.m.view(torch.int32)),
             'v_bits': torch.equal(candidate.v.view(torch.int32), ref.v.view(torch.int32)),
             'complete_coordinates': candidate.updated == n*step}
   assert all(checks.values()), checks
   row = {'logical_device': dev, 'step': step, 'coordinates': n,
          'reference_seconds': ref_seconds, 'pipeline_seconds': pipeline_seconds, 'checks': checks}
   rows.append(row); print(json.dumps(row), flush=True)
   Path(a.output).write_text(json.dumps({'complete': False, 'rows': rows, 'detail': DETAIL}, indent=2))
   grad.mul_(.7).add_(.0001)
   del old, actual
  del ref, w, frames, candidate, initial, grad; gc.collect(); torch.npu.empty_cache()
Path(a.output).write_text(json.dumps({'complete': True, 'rows': rows, 'detail': DETAIL}, indent=2))
