"""Inspect installed ZeRO gradient/offload paths before adding profiling scopes."""
import inspect,json
from deepspeed.runtime.zero.stage3 import DeepSpeedZeroOptimizer_Stage3 as Z
for name in dir(Z):
 if any(k in name for k in ['partition_grads','copy_grad_to_fp32','avg_scatter','reduce_and_partition','reduce_ready','complete_grad_norm','offload_gradient','offload_grad_partition','managed_offload']):
  method=getattr(Z,name)
  if not callable(method):continue
  print('METHOD',name,flush=True)
  for _ in range(8):
   if getattr(method,'__module__','')!='deepspeed.utils.nvtx':break
   nested=inspect.getclosurevars(method).nonlocals.get('func')
   assert inspect.isfunction(nested)
   method=nested
  print(inspect.getsource(method),flush=True)
