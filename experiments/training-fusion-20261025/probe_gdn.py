"""Output and all input gradients against the installed HF training fallback."""
import json,os,time,traceback
from pathlib import Path
os.environ.setdefault('ASCEND_RT_VISIBLE_DEVICES','2,3')
import torch,torch_npu
from transformers.models.qwen3_5.modeling_qwen3_5 import torch_chunk_gated_delta_rule
from community_gdn_unpadded import chunk_gated_delta_rule
torch.set_num_threads(12);torch.npu.set_device(0)
torch.npu.matmul.allow_hf32=False;torch.npu.config.allow_internal_format=False
torch.manual_seed(20261025)
result={'cases':[],'device':torch.npu.get_device_name(0)}
try:
 for batch,length,heads in [(1,32,4),(1,128,4),(2,64,4)]:
  q=torch.randn(batch,length,heads,128,device='npu',dtype=torch.bfloat16)*.1
  k=torch.randn_like(q)*.1;v=torch.randn_like(q)
  g=-torch.rand(batch,length,heads,device='npu',dtype=torch.float32)
  beta=torch.sigmoid(torch.randn(batch,length,heads,device='npu',dtype=torch.bfloat16))
  originals=[q,k,v,g,beta];upstream=torch.randn_like(v)
  variants=[]
  for name,fn in [('hf',torch_chunk_gated_delta_rule),('community',chunk_gated_delta_rule)]:
   inputs=[x.detach().clone().requires_grad_(True) for x in originals]
   torch.npu.synchronize();started=time.monotonic()
   out,_=fn(*inputs,use_qk_l2norm_in_kernel=True)
   out.backward(upstream);torch.npu.synchronize()
   variants.append((out.detach().cpu(),[x.grad.detach().cpu() for x in inputs]))
   print(json.dumps({'case':[batch,length,heads],'variant':name,'first_forward_backward_seconds':time.monotonic()-started}),flush=True)
  def metrics(a,b):
   a=a.float();b=b.float();diff=(a-b).abs()
   return {'max_abs':float(diff.max()),'relative_l2':float(torch.linalg.vector_norm(a-b)/torch.linalg.vector_norm(a).clamp_min(1e-12)),
           'cosine':float(torch.nn.functional.cosine_similarity(a.flatten(),b.flatten(),dim=0)),
           'finite':bool(torch.isfinite(b).all())}
  row={'shape':[batch,length,heads,128],'output':metrics(variants[0][0],variants[1][0]),
       'gradients':{key:metrics(a,b) for key,a,b in zip(['q','k','v','g','beta'],variants[0][1],variants[1][1])}}
  result['cases'].append(row);print(json.dumps(row),flush=True)
 result['completed']=True
except Exception:
 result.update(completed=False,error=traceback.format_exc());print(result['error'],flush=True)
Path('gdn-probe.json').write_text(json.dumps(result,indent=2))
