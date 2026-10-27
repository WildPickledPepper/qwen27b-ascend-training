"""Compare complete CPU snapshots from tiny matched pipeline runs."""
import argparse,json,math
from pathlib import Path
import torch
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--candidate',default='tiny-chunk-v2.private');a=p.parse_args()
rows=[]
for rank in [0,1]:
 for step in [1,2,3]:
  b=torch.load(a.root/'tiny-baseline.private'/f'probe-rank{rank}-step{step}.private.pt',map_location='cpu',weights_only=True)
  c=torch.load(a.root/a.candidate/f'probe-rank{rank}-step{step}.private.pt',map_location='cpu',weights_only=True)
  metrics={}
  for group in ['master','grad','m','v']:
   def values(s):
    if group in ['master','grad']:return s[group]
    key='exp_avg' if group=='m' else 'exp_avg_sq'
    return [s['states']['state'][k][key] for k in sorted(s['states']['state'])]
   bb,cc=values(b),values(c);assert len(bb)==len(cc)
   ds=rs=maximum=0.;finite=True
   for x,y in zip(bb,cc):
    assert x.shape==y.shape
    x=x.double();y=y.double();delta=x-y
    ds+=float(delta.square().sum());rs+=float(x.square().sum())
    maximum=max(maximum,float(delta.abs().max()));finite &= bool(y.isfinite().all())
   metrics[group]=dict(relative_l2=math.sqrt(ds/max(rs,1e-40)),maximum_absolute_error=maximum,finite=finite)
  assert all(v['finite'] for v in metrics.values())
  assert metrics['grad']['relative_l2']<.02 and metrics['master']['relative_l2']<1e-3,metrics
  rows.append(dict(rank=rank,step=step,metrics=metrics))
a.output.write_text(json.dumps(dict(scope='All tensors of tiny three-step pipeline; BF16 chunking not bitwise equivalent',checks_passed=True,rows=rows),indent=2))
print(json.dumps(dict(checks_passed=True,max_gradient_relative_l2=max(r['metrics']['grad']['relative_l2'] for r in rows),max_master_relative_l2=max(r['metrics']['master']['relative_l2'] for r in rows))))
