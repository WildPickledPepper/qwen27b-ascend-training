"""Compare all tiny parameters, accumulated gradients and FP32 Adam states, 3 steps."""
import argparse,json
from pathlib import Path
import torch
p=argparse.ArgumentParser();p.add_argument('--reference',required=True);p.add_argument('--candidate',required=True);p.add_argument('--output',required=True);a=p.parse_args()
rows=[]
for rank in [0,1]:
    lines=[json.loads(x) for x in (Path(a.candidate)/f'rank{rank}.jsonl').read_text().splitlines()]
    plan=next(x for x in lines if x['event']=='resident_plan')
    assert 0<plan['selected_parameters']<plan['total_parameters'],'Tiny must exercise resident and CPU paths'
    for step in [1,2,3]:
        ref=torch.load(Path(a.reference)/f'probe-rank{rank}-step{step}.private.pt',weights_only=False)
        got=torch.load(Path(a.candidate)/f'probe-rank{rank}-step{step}.private.pt',weights_only=False)
        counts={}
        for key in ['master','grad']:
            assert len(ref[key])==len(got[key])
            for i,(left,right) in enumerate(zip(ref[key],got[key])):
                assert torch.equal(left,right),(rank,step,key,i,float((left.float()-right.float()).abs().max()))
            counts[key]=len(ref[key])
        assert set(ref['states']['state'])==set(got['states']['state'])
        for i in ref['states']['state']:
            for key in ['exp_avg','exp_avg_sq']:
                assert torch.equal(ref['states']['state'][i][key],got['states']['state'][i][key]),(rank,step,key,i)
        rows.append(dict(rank=rank,step=step,all_gradient_master_m_v_exact=True,tensors=counts))
result=dict(passed=True,rows=rows,scope='All tiny gradients/master/m/v for three updates, resident+CPU paths; not whole-model bitwise proof')
Path(a.output).write_text(json.dumps(result,indent=2));print(json.dumps(result))
