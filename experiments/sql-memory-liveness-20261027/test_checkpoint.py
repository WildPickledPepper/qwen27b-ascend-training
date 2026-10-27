import gc,json
import torch
from torch.utils.checkpoint import checkpoint
import torch.utils.checkpoint as cp
from storage_tracker import StorageTracker
cp._infer_device_type=lambda *args:'cpu'

def execute(use_checkpoint):
 with torch.device('meta'):
  block=torch.nn.Sequential(torch.nn.Linear(128,256,bias=False),torch.nn.SiLU(),torch.nn.Linear(256,128,bias=False))
 t=StorageTracker();t.track_events=True
 for name,p in block.named_parameters():
  t.register(p,'weight',name,True)
  p.register_post_accumulate_grad_hook(lambda v:setattr(v,'grad',None))
 with t:
  x=torch.empty(16,256,128,device='meta',requires_grad=True)
  with t.scope('forward'):
   y=checkpoint(block,x,use_reentrant=False) if use_checkpoint else block(x)
  t.sweep();forward_bytes=t.total
  with t.scope('backward'):y.sum().backward()
  del x,y;gc.collect();t.sweep()
 r=dict(checkpoint=use_checkpoint,forward_live_bytes=forward_bytes,peak_bytes=t.peak,
        recompute_allocations=sum(e.get('category')=='checkpoint_recompute_intermediates' for e in t.events),
        live_end_bytes=t.total,weight_bytes=sum(p.numel()*p.element_size() for p in block.parameters()))
 assert r['live_end_bytes']==r['weight_bytes'],r
 t.close();return r
plain=execute(False);checked=execute(True)
assert checked['forward_live_bytes']<plain['forward_live_bytes']
assert checked['recompute_allocations']>0
print(json.dumps(dict(checks=dict(checkpoint_reduces_forward_retention=True,real_nonreentrant_recompute_traced=True,graph_release_returns_to_weights=True),plain=plain,checkpoint=checked),indent=2))
