"""Compare every tiny gradient/master/m/v tensor, requiring bitwise equality."""
import argparse,json
from pathlib import Path
import torch
p=argparse.ArgumentParser();p.add_argument('--reference',type=Path,required=True);p.add_argument('--candidate',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
rows=[]
for rank in [0,1]:
 for step in [1,2,3]:
  b=torch.load(a.reference/f'probe-rank{rank}-step{step}.private.pt',map_location='cpu',weights_only=True)
  c=torch.load(a.candidate/f'probe-rank{rank}-step{step}.private.pt',map_location='cpu',weights_only=True)
  checks={}
  for group in ['master','grad','m','v']:
   def items(s):
    if group in ['master','grad']:return s[group]
    key='exp_avg' if group=='m' else 'exp_avg_sq'
    return [s['states']['state'][k][key] for k in sorted(s['states']['state'])]
   bb,cc=items(b),items(c);assert len(bb)==len(cc)
   exact=all(x.dtype==y.dtype and torch.equal(x,y) for x,y in zip(bb,cc))
   maximum=max(float((x.double()-y.double()).abs().max()) for x,y in zip(bb,cc))
   checks[group]=dict(exact=exact,tensors=len(bb),maximum_absolute_error=maximum)
   assert exact,(rank,step,group,maximum)
  rows.append(dict(rank=rank,step=step,checks=checks))
a.output.write_text(json.dumps(dict(all_exact=True,scope='All tiny tensors at three steps; not a full-model bitwise proof',rows=rows),indent=2))
print(json.dumps(dict(all_exact=True,steps=3,ranks=2)))
