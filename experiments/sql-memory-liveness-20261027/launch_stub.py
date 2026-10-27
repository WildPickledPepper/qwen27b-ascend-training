"""Run Python tensor allocation wrappers without executing device kernels."""
import contextlib,functools,importlib,sys,types
import torch

def install():
 import triton
 from triton.runtime.jit import JITFunction
 from triton.runtime.autotuner import Autotuner,Heuristics
 launches=[]
 def subscript(self,grid):
  def skip(*args,**kwargs):
   name=getattr(self,'__name__',getattr(getattr(self,'fn',None),'__name__',type(self).__name__))
   launches.append(name)
   return None
  return skip
 for cls in [JITFunction,Autotuner,Heuristics]:cls.__getitem__=subscript
 # Only the device-selection utility is replaced. Allocation wrapper functions
 # and explicit contiguous operations come from the pinned upstream sources.
 name='veomni.ops.kernels.gated_delta_rule.vendor.triton.utils'
 u=types.ModuleType(name)
 def guard(fn):
  @functools.wraps(fn)
  def wrapper(*args,**kwargs):
   args=tuple(v.contiguous() if isinstance(v,torch.Tensor) else v for v in args)
   kwargs={k:v.contiguous() if isinstance(v,torch.Tensor) else v for k,v in kwargs.items()}
   return fn(*args,**kwargs)
  return wrapper
 def varlen(*a,**kw):raise RuntimeError('Variable-length GDN metadata rule is not implemented')
 u.input_guard=guard;u.prepare_chunk_indices=varlen;u.prepare_chunk_offsets=varlen
 u.is_amd=False;u.make_tensor_descriptor=varlen
 u.get_npu_properties=lambda:dict(num_aicore=24,num_vectorcore=48)
 u.get_autotune_config=lambda **kw:[triton.Config({},num_warps=4,num_stages=1)]
 # Only used inside skipped JIT functions.
 u.exp=lambda x:None
 sys.modules[name]=u
 return launches
