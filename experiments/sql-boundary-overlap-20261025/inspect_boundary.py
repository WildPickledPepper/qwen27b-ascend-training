"""Read installed ZeRO epilogue/finalization contracts without allocating NPU tensors."""
import inspect,textwrap
from deepspeed.runtime.zero.stage3 import DeepSpeedZeroOptimizer_Stage3 as Z
def unwrap(fn):
 for _ in range(8):
  if getattr(fn,'__module__','')!='deepspeed.utils.nvtx':break
  fn=inspect.getclosurevars(fn).nonlocals['func']
 return fn
for name in dir(Z):
 fn=getattr(Z,name)
 if not callable(fn):continue
 try:s=inspect.getsource(unwrap(fn))
 except (TypeError,OSError):continue
 if any(w in s for w in ['_finalize_offload_gradient_accumulation','_offload_boundary_param_ids.clear','_coalesce_grad_reduction','def independent_gradient_partition_epilogue','def overlapping_partition_gradients_reduce_epilogue']):
  print('METHOD',name,flush=True);print(s,flush=True)
