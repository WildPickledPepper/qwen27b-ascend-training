"""Exercise native mixed-device autograd with reentrant checkpoint and hooks."""
import argparse,json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--output',required=True)
p.add_argument('--single-thread-backward',action='store_true')
p.add_argument('--non-reentrant',action='store_true')
a=p.parse_args()
import torch,torch_npu
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from bench_resident import WeightStore,configure
import bench_bf16_native as mixed
import bench_cpu_pipeline as pipeline
torch.set_num_threads(24);torch.manual_seed(42)
torch.npu.config.allow_internal_format=False
configure('fp32');mixed.install()
if a.single_thread_backward:torch.autograd.set_multithreading_enabled(False)
stores={}; captured={}
for name,shape,device in [('gate',(2048,1024),'npu:1'),('up',(2048,1024),'npu:0'),('down',(1024,2048),'npu:1')]:
 w=(torch.randn(shape,device=device)*.01).to(torch.bfloat16).float().view(-1)
 frames=[]
 for begin in range(0,w.numel(),32768):
  part=w[begin:begin+32768].clone()
  frames.append({'kind':'raw','n':part.numel(),'bytes':part.numel()*4,'data':part.view(torch.uint8),'header':None})
 s=WeightStore(name,shape,frames,None,device,128,1e-5,.01,65536)
 mixed.install_store(s);s.enabled=False;s.step=1;stores[name]=s
scale=torch.nn.Parameter(torch.ones(1024,device='npu:0'))
def hook(p):
 with torch.npu.device(p.device):
  captured['tiny_gradient_finite']=bool(torch.isfinite(p.grad).all().item())
  p.grad=None
scale.register_post_accumulate_grad_hook(hook)
def linear(z,name):return mixed.NativeLinear.apply(z,None,stores[name])
def block(x):return x+linear(F.silu(linear(x*scale,'gate'))*linear(x,'up'),'down')
with torch.no_grad():
 for _ in range(2):
  x=torch.randn(28,1024,device='npu:0');y=block(block(x));float(y.norm())
for s in stores.values():s.enabled=True
x=torch.randn(28,1024,device='npu:0',requires_grad=True)
y=checkpoint(block,x,use_reentrant=not a.non_reentrant)
y.square().mean().backward();pipeline.flush()
assert captured['tiny_gradient_finite']
assert bool(torch.isfinite(x.grad).all().item())
result={'completed':True,'single_thread_backward':a.single_thread_backward,'non_reentrant':a.non_reentrant,'tiny_gradient_finite':True,
        'input_gradient_l2':float(x.grad.norm()),'changed_steps':{k:s.changed_steps for k,s in stores.items()}}
Path(a.output).write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
