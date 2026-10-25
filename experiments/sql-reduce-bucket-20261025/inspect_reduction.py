"""Read native reduction packing and supported coalescing contracts."""
import inspect
from deepspeed.runtime.zero.stage3 import DeepSpeedZeroOptimizer_Stage3 as Z
from deepspeed.runtime.comm.coalesced_collectives import reduce_scatter_coalesced
for name in ['reduce_independent_p_g_buckets_and_remove_grads','_DeepSpeedZeroOptimizer_Stage3__add_grad_to_ipg_bucket','_DeepSpeedZeroOptimizer_Stage3__reduce_and_partition_ipg_grads']:
 fn=getattr(Z,name)
 for _ in range(8):
  if fn.__module__!='deepspeed.utils.nvtx':break
  fn=inspect.getclosurevars(fn).nonlocals['func']
 print('METHOD',name);print(inspect.getsource(fn))
print('COALESCED');print(inspect.getsource(reduce_scatter_coalesced))
