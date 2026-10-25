"""Process-local observation only: preserve original ZeRO methods and tensor operations."""
import collections,contextlib,inspect,threading,time,types
import torch
from torch.utils._python_dispatch import TorchDispatchMode

class TransferObserver(TorchDispatchMode):
 def __init__(self,stats):super().__init__();self.stats=stats
 def __torch_dispatch__(self,func,types,args=(),kwargs=None):
  kwargs=kwargs or {};source=None;destination=None;dtype=None
  if func==torch.ops.aten._to_copy.default and args and isinstance(args[0],torch.Tensor):
   source=args[0];destination=torch.device(kwargs.get('device',source.device));dtype=kwargs.get('dtype',source.dtype)
  elif func==torch.ops.aten.copy_.default and len(args)>=2:
   destination=args[0].device;source=args[1];dtype=args[0].dtype
  direction=None
  if source is not None:
   if source.device.type=='cpu' and destination.type in ['npu','privateuseone']:direction='h2d'
   elif destination.type=='cpu' and source.device.type in ['npu','privateuseone']:direction='d2h'
  if direction:
   self.stats[direction+'_calls']+=1
   self.stats[direction+'_source_bytes']+=source.numel()*source.element_size()
   self.stats[direction+'_destination_bytes']+=source.numel()*torch.empty((),dtype=dtype,device='meta').element_size()
   start=time.monotonic()
   with torch.profiler.record_function('gradient_transfer/'+direction):result=func(*args,**kwargs)
   self.stats[direction+'_host_call_seconds']+=time.monotonic()-start
   return result
  return func(*args,**kwargs)

def install(optimizer):
 stats=collections.defaultdict(float);local=threading.local()
 methods={'_finish_deferred_boundary':'deferred_boundary','partition_grads':'partition_gradients',
 '_DeepSpeedZeroOptimizer_Stage3__reduce_and_partition_ipg_grads':'reduce_and_partition',
 '_DeepSpeedZeroOptimizer_Stage3__avg_scatter_grads':'average_scatter',
 '_offload_grad_partition_at_boundary':'boundary_gradient_offload',
 'complete_grad_norm_calculation_for_cpu_offload':'global_gradient_norm'}
 originals={}
 for name,label in methods.items():
  method=getattr(optimizer,name);originals[name]=method
  def wrapped(self,*args,_original=method,_label=label,**kwargs):
   observe=_label in ['partition_gradients','boundary_gradient_offload','deferred_boundary'] and not getattr(local,'observing',False)
   if observe:local.observing=True
   started=time.monotonic();stats[_label+'_calls']+=1
   try:
    with torch.profiler.record_function('zero/'+_label):
     with TransferObserver(stats) if observe else contextlib.nullcontext():
      return _original(*args,**kwargs)
   finally:
    stats[_label+'_host_seconds']+=time.monotonic()-started
    if observe:local.observing=False
  setattr(optimizer,name,types.MethodType(wrapped,optimizer))
 return stats
