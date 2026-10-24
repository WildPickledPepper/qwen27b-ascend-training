"""Compare original/staged FP32 offload at two windows; same state trajectory."""
import ctypes,os,time,json,argparse,gc,statistics
from pathlib import Path
assert ctypes.CDLL(None).prctl(41,1,0,0,0)==0
os.sched_setaffinity(0,set(range(96,100))&os.sched_getaffinity(0))
import torch,torch_npu
from bench_resident_npu import NpuOffloadedMoments
from bench_staged_offload import StagedNpuMoments,DETAIL,POOLS
p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args()
torch.set_num_threads(4);torch.manual_seed(20261024);torch.npu.config.allow_internal_format=False
results=[]
for dev in [0,1]:
 with torch.npu.device(dev):
  device='npu:'+str(dev);n=32*1048576+4096
  seed=torch.randn(n,device=device)*.01;g=torch.randn(n,device=device)*.001
  reference=None
  for cls,window in [(NpuOffloadedMoments,1048576),(NpuOffloadedMoments,16*1048576),(StagedNpuMoments,1048576),(StagedNpuMoments,16*1048576)]:
   state=cls(n,device,window=window);state.name='staging.window.probe';w=seed.clone();rows=[]
   for step in [1,2,3]:
    t=time.monotonic();state.update(0,w,g,step);torch.npu.synchronize(dev);rows.append(time.monotonic()-t)
   values=(w,state.m,state.v)
   if reference is None:reference=tuple(x.clone() for x in values)
   checks={name:torch.equal(x.view(torch.int32),y.view(torch.int32)) for name,x,y in zip(['weight_bits','m_bits','v_bits'],reference,values)}
   assert all(checks.values()),checks
   result={'logical_device':dev,'coordinates':n,'updates':3,'method':cls.__name__,'window_coordinates':window,'seconds':rows,'warm_mean':statistics.mean(rows[1:]),'exact_checks':checks}
   results.append(result);Path(a.output).write_text(json.dumps({'passed':True,'affinity':sorted(os.sched_getaffinity(0)),'results':results,'staged_counters':DETAIL},indent=2))
   print(json.dumps(result),flush=True)
   del state,w,values;gc.collect();torch.npu.empty_cache()
  del reference,seed,g;gc.collect();torch.npu.empty_cache()
