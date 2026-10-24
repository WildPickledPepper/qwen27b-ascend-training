import ctypes,time,json
assert ctypes.CDLL(None).prctl(41,1,0,0,0)==0
import torch,torch_npu
from bench_resident import CpuMoments,TIMES
from pathlib import Path
torch.set_num_threads(4);torch.manual_seed(20261024);torch.npu.set_device(0)
n=16777216;device='npu:0';seed=torch.randn(n,device=device)*.01;g=torch.randn(n,device=device)*.001
# Warm transfer path without a large benchmark state.
z=seed[:1048576].cpu();seed[:1048576].copy_(z);del z
results=[];previous=None
for window in [1048576,16777216]:
 s=CpuMoments(n,device,window=window);s.name='window.probe';w=seed.clone()
 for k in TIMES:TIMES[k]=0.
 start=time.monotonic()
 for step in [1,2]:s.update(0,w,g,step)
 torch.npu.synchronize();elapsed=time.monotonic()-start
 final=w.cpu();exact=True if previous is None else torch.equal(previous.view(torch.int32),final.view(torch.int32))
 results.append({'window':window,'steps':2,'parameters':n,'seconds':elapsed,'timing':dict(TIMES),'same_final_fp32_bits':exact})
 previous=final;del s,w
 torch.npu.empty_cache()
Path(__file__).with_suffix('.json').write_text(json.dumps({'results':results},indent=2));print(json.dumps(results),flush=True)
