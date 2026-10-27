"""NPU gradient/loss check, no CPU optimizer or model weights."""
import json,os
from pathlib import Path
import torch,torch_npu
import torch.nn.functional as F
from chunked_loss import chunked_linear_ce,shifted_targets
torch.set_num_threads(4);torch.npu.set_device(0);torch.manual_seed(42)
rows=[]
for b,s,h,v,chunk in [(2,17,64,256,8),(1,32,5120,248320,16)]:
 x=(torch.randn(b,s,h,device='npu:0',dtype=torch.bfloat16)*.1).requires_grad_()
 w=(torch.randn(v,h,device='npu:0',dtype=torch.bfloat16)*.02).requires_grad_()
 labels=torch.randint(0,v,(b,s),device='npu:0');labels[:,:5]=-100
 ref=F.cross_entropy(F.linear(x,w).float().view(-1,v),shifted_targets(labels),ignore_index=-100)
 gr=torch.autograd.grad(ref,(x,w));torch.npu.synchronize()
 loss=chunked_linear_ce(x,w,labels,chunk);gc=torch.autograd.grad(loss,(x,w));torch.npu.synchronize()
 differences=[float((a.float()-b.float()).norm()/b.float().norm().clamp_min(1e-20)) for a,b in zip(gc,gr)]
 item=dict(shape=[b,s,h,v],chunk_tokens=chunk,reference_loss=float(ref),chunk_loss=float(loss),loss_abs_error=float((ref-loss).abs()),gradient_relative_l2=differences,gradients_finite=all(bool(g.isfinite().all()) for g in gc))
 assert item['gradients_finite'] and item['loss_abs_error']<1e-4 and max(differences)<.02,item
 rows.append(item);del x,w,labels,ref,loss,gr,gc;torch.npu.empty_cache()
Path('loss-probe.private.json').write_text(json.dumps(dict(device='npu',physical_card=3,cases=rows,scope='Short-token output-head shapes, not full-model convergence'),indent=2))
print(json.dumps(rows))
