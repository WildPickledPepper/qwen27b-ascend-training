"""Stress actual staged copy callback with two slots and repeated buffer reuse."""
import json,os
from pathlib import Path
os.environ['ASCEND_RT_VISIBLE_DEVICES']='3'
import torch,torch_npu
from cpu_update import install
torch.npu.set_device(0);torch.set_num_threads(12);torch.manual_seed(20261025)
class Harness:
 offload_optimizer=True;swap_optimizer=False;clip_grad=0;loss_scale=1.
 def unscale_and_clip_grads(self,group,norm):
  combined_scale=self.loss_scale
  self.fp32_partitioned_groups_flat[group].grad.mul_(1. / combined_scale)
 def _optimizer_step(self,*args):pass
 def _reassign_or_swap_out_partitioned_parameters(self,*args):pass
 def _prepare_sub_group(self,*args):pass
 def _release_sub_group(self,*args):pass
 def _post_step(self,*args):pass
 def _unflatten_partitioned_parameters(self,*args):pass
obj=Harness()
obj.fp32_partitioned_groups_flat=[torch.randn(1048576+i*17) for i in range(8)]
obj.fp16_partitioned_groups_flat=[torch.empty(p.shape,dtype=torch.bfloat16,device='npu') for p in obj.fp32_partitioned_groups_flat]
stats=install(obj,skip_identity=True,pipeline=True)
differences=0
for repeat in range(3):
 for i,source in enumerate(obj.fp32_partitioned_groups_flat):
  source.add_(.001);obj._reassign_or_swap_out_partitioned_parameters(i)
 obj._post_step(set())
 for src,dst in zip(obj.fp32_partitioned_groups_flat,obj.fp16_partitioned_groups_flat):
  differences+=int((dst.cpu().view(torch.int16)!=src.to(torch.bfloat16).view(torch.int16)).sum())
result={'completed':True,'rounds':3,'groups_per_round':8,'bf16_bit_differences':differences,'stats':dict(stats)}
assert differences==0
Path('pipeline-probe.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
