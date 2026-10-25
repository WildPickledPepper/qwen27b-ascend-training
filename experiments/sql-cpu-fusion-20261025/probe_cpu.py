"""CPU-only bit-pattern checks and paired operator timing before model integration."""
import ctypes,importlib.util,json,os,statistics,time
from pathlib import Path
os.sched_setaffinity(0,range(96,120));numa=ctypes.CDLL('libnuma.so.1',use_errno=True)
mask=ctypes.c_ulong(1<<4);numa.set_mempolicy.argtypes=[ctypes.c_int,ctypes.POINTER(ctypes.c_ulong),ctypes.c_ulong]
assert numa.set_mempolicy(1,ctypes.byref(mask),8)==0
import torch,torch_npu
assert os.environ['ASCEND_RT_VISIBLE_DEVICES']=='2,3'
torch.npu.set_device(0)  # Register/use our allowed NPU's host pinned allocator; no NPU math.
torch.set_num_threads(24)
r=Path(__file__).resolve().parent
def load(name,path):
 s=importlib.util.spec_from_file_location(name,str(path));m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
native=load('cpu_adam',os.environ['CPU_ADAM_EXTENSION']);fused=load('cpu_adam_fused',r/'build.private/cpu_adam_fused.so')
torch.manual_seed(42);checks=[]
def equal(a,b):return torch.equal(a.contiguous().view(torch.uint8),b.contiguous().view(torch.uint8))
for case,n in enumerate([1,3,4,15,16,31,32,33,127,1025,1000003]):
 reference=[torch.randn(n),torch.randn(n)*.1,torch.rand(n)*.1]
 candidate=[x.clone() for x in reference];output=torch.empty(n,dtype=torch.bfloat16)
 native.create_adam(case,2e-5,.9,.999,1e-8,.01,True,False);fused.create_adam(case,2e-5,.9,.999,1e-8,.01,True,False)
 for step,scale in enumerate([1.,.5,.3333333432674408,1e-5,1.],1):
  grad=torch.randn(n);scaled=grad.clone().mul_(scale)
  native.adam_update(case,step,2e-5,.9,.999,1e-8,.01,True,reference[0],scaled,reference[1],reference[2])
  fused.adam_update_fused(case,step,2e-5,.9,.999,1e-8,.01,True,candidate[0],grad,candidate[1],candidate[2],scale,output)
  states=all(equal(a,b) for a,b in zip(reference,candidate));compute=equal(output,reference[0].to(torch.bfloat16))
  assert states and compute,(n,step,states,compute)
  checks.append({'elements':n,'step':step,'fp32_master_m_v_bits_equal':states,'bf16_output_bits_equal':compute})
 native.destroy_adam(case);fused.destroy_adam(case)
# High/low exponent and BF16 halfway-rounding cases without NaN optimizer states.
bits=torch.randint(-(2**31),2**31-1,(1000003,),dtype=torch.int32);weights=bits.view(torch.float32)
weights=weights[torch.isfinite(weights)].clone();n=weights.numel();a=[weights.clone(),torch.zeros(n),torch.ones(n)];b=[x.clone() for x in a]
g=torch.zeros(n);out=torch.empty(n,dtype=torch.bfloat16)
native.create_adam(900,0.,.9,.999,1e-8,0.,True,False);fused.create_adam(900,0.,.9,.999,1e-8,0.,True,False)
native.adam_update(900,1,0.,.9,.999,1e-8,0.,True,a[0],g,a[1],a[2]);fused.adam_update_fused(900,1,0.,.9,.999,1e-8,0.,True,b[0],g,b[1],b[2],1.,out)
assert all(equal(x,y) for x,y in zip(a,b)) and equal(out,a[0].to(torch.bfloat16))
probe={'passed':True,'checks':checks,'finite_raw_fp32_cast_coordinates':n,'bit_comparison':'torch.uint8 view; distinguishes signed zero','scope':'CPU operator, not whole-model training'}
(r/'cpu-probe.json').write_text(json.dumps(probe,indent=2));print(json.dumps({'cpu_numerical_probe':'passed','cases':len(checks),'cast_coordinates':n}),flush=True)
# One representative 128M-element shard. Preparation is outside timed regions.
n=128*2**20;initial=[torch.randn(n)*.1,torch.zeros(n),torch.ones(n)*.01];gradient=torch.randn(n)*.01
timings=[]
for repeat in range(4):
 for mode in (['native','fused'] if repeat%2==0 else ['fused','native']):
  state=[x.clone() for x in initial];grad=gradient.clone();output=torch.empty(n,dtype=torch.bfloat16,pin_memory=True)
  ident=1000+repeat*2+(mode=='fused');op=native if mode=='native' else fused
  op.create_adam(ident,2e-5,.9,.999,1e-8,.01,True,False)
  start=time.monotonic();detail={}
  if mode=='native':
   t=time.monotonic();grad.mul_(.3333333432674408);detail['scale_seconds']=time.monotonic()-t
   t=time.monotonic();native.adam_update(ident,1,2e-5,.9,.999,1e-8,.01,True,state[0],grad,state[1],state[2]);detail['adam_seconds']=time.monotonic()-t
   t=time.monotonic();output.copy_(state[0]);detail['bf16_cast_seconds']=time.monotonic()-t
  else:fused.adam_update_fused(ident,1,2e-5,.9,.999,1e-8,.01,True,state[0],grad,state[1],state[2],.3333333432674408,output)
  elapsed=time.monotonic()-start;op.destroy_adam(ident)
  row={'repeat':repeat,'mode':mode,'elements':n,'seconds':elapsed,**detail};timings.append(row);print(json.dumps(row),flush=True)
  del state,grad,output
summary={mode:statistics.mean(x['seconds'] for x in timings if x['mode']==mode and x['repeat']>0) for mode in ['native','fused']}
(r/'cpu-benchmark.json').write_text(json.dumps({'timings':timings,'steady_mean_seconds':summary,'time_reduction_percent':100*(1-summary['fused']/summary['native']),'limits':'Single-node CPU operator; not full-model wall-time speedup. First pair excluded.'},indent=2));print(json.dumps(summary),flush=True)
