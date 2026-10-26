"""CPU-only aggregate diagnosis of tiny replicated norm gradients."""
import argparse,json
from pathlib import Path
import torch
p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--output',required=True);a=p.parse_args()
root=Path(a.root)
data=[torch.load(root/f'norm-diagnostic-rank{r}.private.pt',weights_only=True) for r in [0,1]]
def rel(x,y):return float((x.float()-y.float()).norm()/y.float().norm().clamp_min(1e-20))
rows=[]
for name in data[0]:
    values=[x[name] for x in data];expected=values[0]['expected']
    row=dict(name=name,hook_calls=[len(v['history']) for v in values],
             rank_difference_relative=rel(values[0]['actual'],values[1]['actual']),
             rank_sum_vs_reference_relative=rel(values[0]['actual']+values[1]['actual'],expected),ranks=[])
    for rank,v in enumerate(values):
        r=dict(rank=rank,actual_vs_reference_relative=rel(v['actual'],expected))
        if v['history']:
            before=torch.zeros_like(v['history'][0][0]);after=torch.zeros_like(before)
            for b,e in v['history']:before.add_(b);after.add_(e)
            r.update(actual_vs_before_hook_relative=rel(v['actual'],before),
                     actual_vs_after_hook_relative=rel(v['actual'],after),after_hook_vs_reference_relative=rel(after,expected))
        row['ranks'].append(r)
    if values[0]['history'] and len(values[0]['history'])==len(values[1]['history']):
        row['all_post_hook_micro_gradients_equal_across_ranks']=all(torch.equal(l[1],r[1]) for l,r in zip(values[0]['history'],values[1]['history']))
    rows.append(row)
result=dict(rows=rows,scope='tiny diagnostic only; no raw tensors or paths exported')
Path(a.output).write_text(json.dumps(result,indent=2));print(json.dumps(result))
