"""Load pinned VeOmni raw training kernels without importing its trainer stack.

No third-party source is modified. The namespace registry shim only supplies the
SavedState value type; forward/backward math remains upstream code.
"""
import dataclasses,importlib,os,sys,types
from pathlib import Path
import torch

@dataclasses.dataclass(frozen=True)
class SavedState:
 tensors:tuple
 metadata:object=None

def load_raw():
 root=Path(os.environ['VEOMNI_SOURCE']).resolve()/'veomni'
 packages=['veomni','veomni.ops','veomni.ops.kernels','veomni.ops.kernels.gated_delta_rule',
 'veomni.ops.kernels.gated_delta_rule.chunk_gated_delta_rule',
 'veomni.ops.kernels.gated_delta_rule.chunk_gated_delta_rule.standard',
 'veomni.ops.kernels.gated_delta_rule.vendor','veomni.ops.kernels.gated_delta_rule.vendor.triton',
 'veomni.ops.kernels.gated_delta_rule.vendor.triton_core']
 for name in packages:
  assert name not in sys.modules,'Use an isolated process, not an imported VeOmni trainer'
  module=types.ModuleType(name);module.__path__=[str(root.joinpath(*name.split('.')[1:]))]
  sys.modules[name]=module
 registry=types.ModuleType('veomni.ops.registry');registry.SavedState=SavedState
 sys.modules['veomni.ops.registry']=registry
 return importlib.import_module('veomni.ops.kernels.gated_delta_rule.chunk_gated_delta_rule.standard.triton')

_raw=None
class GDN(torch.autograd.Function):
 @staticmethod
 def forward(ctx,q,k,v,g,beta,initial,cu,use_norm,chunk,final):
  global _raw
  if _raw is None:_raw=load_raw()
  outputs,saved=_raw.forward(q,k,v,g,beta,initial,cu,use_qk_l2norm_in_kernel=use_norm,chunk_size=chunk,output_final_state=final)
  ctx.save_for_backward(*saved.tensors);ctx.meta=saved.metadata
  return outputs
 @staticmethod
 def backward(ctx,dout,dfinal):
  grads=_raw.backward((dout,dfinal),SavedState(ctx.saved_tensors,ctx.meta))
  return (*grads,None,None,None)

def chunk_gated_delta_rule(q,k,v,g,beta,initial_state=None,output_final_state=False,
                          use_qk_l2norm_in_kernel=False,cu_seqlens=None,chunk_size=64,**kwargs):
 assert not kwargs,kwargs
 # Restrict this adapter to the bounded length-512, no-cache SQL training experiment. Padding
 # future positions keeps valid causal outputs unchanged and fixes JIT shapes.
 assert initial_state is None and cu_seqlens is None and not output_final_state
 length=q.shape[1];assert 0<length<=512
 pad=max(128,((length+63)//64)*64)-length
 if pad:
  q,k,v=[torch.nn.functional.pad(x,(0,0,0,0,0,pad)) for x in [q,k,v]]
  g,beta=[torch.nn.functional.pad(x,(0,0,0,pad)) for x in [g,beta]]
 out,final=GDN.apply(q,k,v,g,beta,initial_state,cu_seqlens,use_qk_l2norm_in_kernel,chunk_size,output_final_state)
 return out[:,:length],None
