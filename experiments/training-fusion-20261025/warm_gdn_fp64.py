"""Compile/verify actual model GDN shapes before starting collective workers."""
import json,os,time
from pathlib import Path
os.environ['ASCEND_RT_VISIBLE_DEVICES']='2,3'
import torch,torch_npu
from transformers.models.qwen3_5.modeling_qwen3_5 import torch_chunk_gated_delta_rule
from community_gdn import chunk_gated_delta_rule
torch.npu.set_device(0);torch.set_num_threads(12)
torch.npu.matmul.allow_hf32=False;torch.npu.config.allow_internal_format=False
torch.manual_seed(20261025);result={'cases':[]}
for batch,length in [(1,31),(1,128),(16,125)]:
 q=torch.randn(batch,length,48,128,device='npu',dtype=torch.bfloat16)*.1
 k=torch.randn_like(q)*.1;v=torch.randn_like(q)
 g=-torch.rand(batch,length,48,device='npu',dtype=torch.float32)
 beta=torch.sigmoid(torch.randn(batch,length,48,device='npu',dtype=torch.bfloat16))
 originals=[q,k,v,g,beta];upstream=torch.randn_like(v);variants=[]
 for name,fn in [('hf',torch_chunk_gated_delta_rule),('community',chunk_gated_delta_rule)]:
  inputs=[x.detach().clone().requires_grad_(True) for x in originals]
  torch.npu.synchronize();started=time.monotonic()
  out,_=fn(*inputs,use_qk_l2norm_in_kernel=True);out.backward(upstream);torch.npu.synchronize()
  variants.append((out.detach().cpu(),[x.grad.detach().cpu() for x in inputs]))
  print(json.dumps({'shape':[batch,length,48,128],'variant':name,'first_seconds':time.monotonic()-started}),flush=True)
 def metrics(a,b):
  a=a.double();b=b.double()
  return {'relative_l2':float(torch.linalg.vector_norm(a-b)/torch.linalg.vector_norm(a).clamp_min(1e-12)),
   'cosine':float(torch.nn.functional.cosine_similarity(a.flatten(),b.flatten(),dim=0)), 'finite':bool(torch.isfinite(b).all())}
 row={'shape':[batch,length,48,128],'output':metrics(variants[0][0],variants[1][0]),
  'gradients':{key:metrics(a,b) for key,a,b in zip(['q','k','v','g','beta'],variants[0][1],variants[1][1])}}
 assert all(m['finite'] and m['relative_l2']<.03 and m['cosine']>.995 for m in [row['output'],*row['gradients'].values()])
 result['cases'].append(row);print(json.dumps(row),flush=True)
 # Warm timing uses a new graph each time and synchronization at both ends.
 timings={}
 for name,fn in [('hf',torch_chunk_gated_delta_rule),('community',chunk_gated_delta_rule)]:
  elapsed=[]
  for _ in range(3):
   inputs=[x.detach().clone().requires_grad_(True) for x in originals]
   torch.npu.synchronize();start=time.monotonic()
   out,_=fn(*inputs,use_qk_l2norm_in_kernel=True);out.backward(upstream);torch.npu.synchronize()
   elapsed.append(time.monotonic()-start)
  timings[name]=elapsed
 row['warm_forward_backward_seconds']=timings
 result['completed']=False;Path('gdn-actual-fp64-probe.json').write_text(json.dumps(result,indent=2))
result['completed']=True;Path('gdn-actual-fp64-probe.json').write_text(json.dumps(result,indent=2))
