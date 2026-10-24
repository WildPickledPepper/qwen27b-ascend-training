import ctypes,time,json
assert ctypes.CDLL(None).prctl(41,1,0,0,0)==0
import torch,torch_npu
from bench_resident_npu import NpuOffloadedMoments,CpuMoments,TIMES
from pathlib import Path
torch.set_num_threads(4);torch.manual_seed(20261024);torch.npu.set_device(0)
n=16777216;device='npu:0';seed=torch.randn(n,device=device)*.01;g=torch.randn(n,device=device)*.001
rows=[]
for cls in [CpuMoments,NpuOffloadedMoments]:
 s=cls(n,device,window=1048576);s.name='offload.probe';w=seed.clone()
 for k in TIMES:TIMES[k]=0.
 start=time.monotonic()
 for step in [1,2]:s.update(0,w,g,step)
 torch.npu.synchronize();elapsed=time.monotonic()-start
 rows.append({'class':cls.__name__,'steps':2,'parameters':n,'seconds':elapsed,'timing':dict(TIMES)})
 if cls==NpuOffloadedMoments:
  reference=seed.clone();m=torch.zeros_like(seed);v=torch.zeros_like(seed)
  for step in [1,2]:
   m.mul_(.9).add_(g,alpha=.1);v.mul_(.999).addcmul_(g,g,value=.001)
   reference.mul_(1-1e-5*.01).addcdiv_(m/(1-.9**step),(v/(1-.999**step)).sqrt().add_(1e-8),value=-1e-5)
  rows[-1]['same_weight_bits_as_resident_fp32_npu']=torch.equal(w.view(torch.int32),reference.view(torch.int32))
  rows[-1]['same_m_bits']=torch.equal(s.m.view(torch.int32),m.cpu().view(torch.int32))
  rows[-1]['same_v_bits']=torch.equal(s.v.view(torch.int32),v.cpu().view(torch.int32))
  assert all(rows[-1][k] for k in ['same_weight_bits_as_resident_fp32_npu','same_m_bits','same_v_bits'])
 del s,w
 torch.npu.empty_cache()
Path(__file__).with_suffix('.json').write_text(json.dumps({'passed':True,'results':rows},indent=2));print(json.dumps(rows),flush=True)
