import json
from pathlib import Path
import torch
r=Path(__file__).resolve().parent
def tensors(v):
 if isinstance(v,torch.Tensor):yield v
 elif isinstance(v,dict):
  for x in v.values():yield from tensors(x)
 elif isinstance(v,(list,tuple)):
  for x in v:yield from tensors(x)
comparisons=[]
for rank in [0,1]:
 for step in range(3):
  for kind in ['gradient','state']:
   name=f'probe-{kind}-rank{rank}-step{step}.private.pt'
   a=list(tensors(torch.load(r/'tiny-native'/name,weights_only=True)))
   b=list(tensors(torch.load(r/'tiny-cache'/name,weights_only=True)))
   assert len(a)==len(b) and a
   equal=all(torch.equal(x,y) for x,y in zip(a,b))
   diff=max(float((x.float()-y.float()).abs().max()) for x,y in zip(a,b))
   comparisons.append({'rank':rank,'step':step+1,'kind':kind,'bitwise_equal':equal,'max_abs_difference':diff,'coordinates':sum(x.numel() for x in a)})
assert all(x['bitwise_equal'] for x in comparisons),comparisons
result={'passed':True,'comparisons':comparisons,'scope':'All tiny-model FP32 gradients/master/m/v; 3 updates, 8 microbatches each; not a full-model bitwise proof.'}
(r/'probe-comparison.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
