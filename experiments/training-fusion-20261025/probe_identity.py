"""Bit-level audit of finite FP32 gradient multiply-by-one and CPUAdam state."""
import importlib.util,json,os,time
from pathlib import Path
import torch
torch.set_num_threads(24);os.sched_setaffinity(0,range(96,120))
torch.manual_seed(20261025)
edge=torch.tensor([0,-2147483648,1,2,8388607,8388608,2139095039,-1+(-2147483648)+2**23],dtype=torch.int32).view(torch.float32)
assert torch.isfinite(edge).all()
sample=torch.cat([torch.randn(16777216),edge]);copied=sample.clone();t=time.monotonic();copied.mul_(1.)
elapsed=time.monotonic()-t
result={'coordinates':sample.numel(),'identity_bit_differences':int((sample.view(torch.int32)!=copied.view(torch.int32)).sum()),'identity_pass_seconds':elapsed}
assert result['identity_bit_differences']==0
spec=importlib.util.spec_from_file_location('cpu_adam',os.environ['CPU_ADAM_EXTENSION']);ext=importlib.util.module_from_spec(spec);spec.loader.exec_module(ext)
ext.create_adam(31001,1e-5,.9,.999,1e-8,.01,True,False)
ext.create_adam(31002,1e-5,.9,.999,1e-8,.01,True,False)
params=[torch.randn(1048576) for _ in [0]];p=params[0];q=p.clone()
m=torch.zeros_like(p);v=m.clone();m2=m.clone();v2=v.clone()
for step in range(1,4):
 grad=torch.randn_like(p);scaled=grad.clone().mul_(1.)
 for ident,weight,first,second,g in [(31001,p,m,v,scaled),(31002,q,m2,v2,grad)]:
  ext.adam_update(ident,step,1e-5,.9,.999,1e-8,.01,True,weight,g,first,second)
result['after_three_updates_bit_differences']={key:int((a.view(torch.int32)!=b.view(torch.int32)).sum()) for key,a,b in [('master',p,q),('m',m,m2),('v',v,v2)]}
assert not any(result['after_three_updates_bit_differences'].values())
result['completed']=True;Path('identity-probe.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
