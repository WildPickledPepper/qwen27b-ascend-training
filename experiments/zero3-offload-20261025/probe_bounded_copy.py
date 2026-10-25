"""Compare the original and bounded offload cast bit for bit, including tail."""
import json,types
from pathlib import Path
import torch,torch_npu
import deepspeed.runtime.zero.stage3 as stage3
from bounded_grad_copy import install
torch.npu.set_device(0);torch.set_num_threads(4)
original=stage3.DeepSpeedZeroOptimizer_Stage3._offload_grad_partition_at_boundary
install();bounded=stage3.DeepSpeedZeroOptimizer_Stage3._offload_grad_partition_at_boundary
rows=[]
for dtype in [torch.bfloat16,torch.float32]:
 n=16777216+257
 g=torch.randn(n,device='npu:0',dtype=dtype)
 def state():
  p=torch.nn.Parameter(torch.zeros(n));p.grad=torch.empty(n)
  return types.SimpleNamespace(grad_position={0:(0,0,n)},get_param_id=lambda p:0,
   norm_for_param_grads={},_constant_buffered_norm2=lambda x:0,
   _swappable_optimizer_subgroup=lambda i:False,fp32_partitioned_groups_flat=[p],
   master_weights_and_grads_dtype=torch.float32)
 old=state();new=state()
 original(old,None,g,{},{});torch.npu.synchronize()
 bounded(new,None,g,{},{});torch.npu.synchronize()
 assert torch.equal(old.fp32_partitioned_groups_flat[0].grad.view(torch.int32),new.fp32_partitioned_groups_flat[0].grad.view(torch.int32))
 rows.append({'dtype':str(dtype),'coordinates':n,'fp32_output_bits_equal':True})
 del old,new,g
result={'completed':True,'rows':rows}
Path('bounded-copy-probe.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
