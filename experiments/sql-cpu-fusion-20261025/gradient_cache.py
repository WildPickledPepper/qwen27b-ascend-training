"""Bounded resident gradient shards. Keep native accumulation/communication/Adam math."""
import hashlib,inspect,json
import torch

def unwrap(fn):
 fn=getattr(fn,'__func__',fn)
 for _ in range(8):
  if fn.__module__!='deepspeed.utils.nvtx':break
  fn=inspect.getclosurevars(fn).nonlocals['func']
 return fn

def install(optimizer,budget_gib):
 assert optimizer.offload_optimizer and not optimizer.swap_optimizer
 assert optimizer.micro_step_id<=0
 source=inspect.getsource(unwrap(optimizer.partition_grads))
 assert 'elif get_accelerator().on_accelerator(grad_buffer):' in source
 assert 'grad_buffer.add_(grad_partition.to(self.gradient_accumulation_dtype)' in source
 assert 'self._offload_grad_partition_at_boundary(' in source
 mapping=optimizer._DeepSpeedZeroOptimizer_Stage3__param_id_to_grad_partition
 assert all(v.device.type=='cpu' and v.dtype==optimizer.gradient_accumulation_dtype for v in mapping.values())
 budget=int(budget_gib*2**30);used=0;selected=[]
 for key,value in sorted(mapping.items(),key=lambda x:(-x[1].numel(),x[0])):
  size=value.numel()*value.element_size()
  if size and used+size<=budget:
   selected.append((key,value.numel(),str(value.dtype)));used+=size
 if selected:
  torch.npu.empty_cache()
  for key,_,_ in selected:
   # Native micro_step_id==0 overwrites before use; no stale CPU contents needed.
   mapping[key]=torch.empty_like(mapping[key],device=f'npu:{torch.npu.current_device()}')
  torch.npu.synchronize()
 plan={'budget_bytes':budget,'resident_bytes':used,'resident_parameters':len(selected),
       'total_parameters':len(mapping),'accumulation_dtype':str(optimizer.gradient_accumulation_dtype),
       'plan_sha256':hashlib.sha256(json.dumps(selected).encode()).hexdigest(),
       'native_partition_source_sha256':hashlib.sha256(source.encode()).hexdigest(),
       'native_math_unmodified':True}
 return plan
