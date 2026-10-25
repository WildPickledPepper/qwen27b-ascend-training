"""Audit full branched/chunked FFN backward against direct BF16 autograd."""
import argparse, ctypes, json, os, types
from pathlib import Path
assert ctypes.CDLL(None).prctl(41,1,0,0,0)==0
import torch, torch_npu
import torch.nn.functional as F
import bench_bf16_master as mixed
import bench_cpu_pipeline as pipeline
from bench_resident import WeightStore,configure
p=argparse.ArgumentParser();p.add_argument('--output',required=True)
p.add_argument('--hidden',type=int,default=513);p.add_argument('--intermediate',type=int,default=5120)
p.add_argument('--tokens',type=int,default=3);p.add_argument('--native',action='store_true');p.add_argument('--cross',action='store_true');a=p.parse_args()
assert os.environ['ASCEND_RT_VISIBLE_DEVICES']=='2,3'
torch.npu.config.allow_internal_format=False
torch.set_num_threads(4);configure('fp32');mixed.install()
if a.native:
 from bench_bf16_native import NativeLinear
else:NativeLinear=mixed.MixedLinear
rows=[]
for update in [False,True]:
 torch.manual_seed(42)
 device='npu:0';h=a.hidden;i=a.intermediate;stores={};captured={};refs={}
 for name,shape in [('gate',(i,h)),('up',(i,h)),('down',(h,i))]:
  weight_device='npu:1' if a.cross and name=='gate' else device
  w=(torch.randn(shape,device=weight_device)*.01).to(torch.bfloat16).float().view(-1)
  frames=[]
  for begin in range(0,w.numel(),32768):
   part=w[begin:begin+32768].clone()
   frames.append({'kind':'raw','n':part.numel(),'bytes':part.numel()*4,'data':part.view(torch.uint8),'header':None})
  s=WeightStore(name,shape,frames,None,weight_device,128,1e-5,.01,65536)
  mixed.install_store(s);s.enabled=True;s.step=1;stores[name]=s
  refs[name]=w.reshape(shape).to(torch.bfloat16).detach().requires_grad_(True)
  original=s.moments.update
  def make_capture(name,original):
   def capture(start,w,g,step):
    captured.setdefault(name,[]).append((start,g.detach().float().cpu().clone()))
    if update:return original(start,w,g,step)
   return capture
  s.moments.update=make_capture(name,original)
 x=torch.randn((a.tokens,h),device=device,requires_grad=True);xr=x.detach().clone().requires_grad_(True)
 def linear(z,name):return NativeLinear.apply(z,torch.zeros((),device=device,requires_grad=True),stores[name])
 yc=linear(F.silu(linear(x,'gate'))*linear(x,'up'),'down')
 def reflinear(z,name):return F.linear(z.to(refs[name].device).to(torch.bfloat16),refs[name]).float().to(z.device)
 yr=reflinear(F.silu(reflinear(xr,'gate'))*reflinear(xr,'up'),'down')
 dy=torch.randn_like(yc)
 yr.backward(dy);yc.backward(dy);pipeline.flush()
 def diff(a,b):
  a=a.detach().float().cpu();b=b.detach().float().cpu()
  return {'relative_l2':float((a-b).norm()/a.norm().clamp_min(1e-30)),
          'max_abs':float((a-b).abs().max()),'reference_l2':float(a.norm()),'candidate_l2':float(b.norm())}
 report={'updates_enabled':update,'output':diff(yr,yc),'input_gradient':diff(xr.grad,x.grad),'weight_gradients':{}}
 for name,s in stores.items():
  got=torch.empty(s.n)
  for start,g in captured[name]:got[start:start+g.numel()].copy_(g.reshape(-1))
  report['weight_gradients'][name]=diff(refs[name].grad.reshape(-1),got)
 rows.append(report);print(json.dumps(report),flush=True)
Path(a.output).write_text(json.dumps({'rows':rows},indent=2))
