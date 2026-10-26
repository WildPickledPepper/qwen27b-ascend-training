"""Compact read-only job status without model manifests or raw sample lists."""
import argparse,collections,json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--root',required=True);a=p.parse_args()
root=Path(a.root);result={}
for rank in [0,1]:
    file=root/f'rank{rank}.jsonl'
    if not file.exists():continue
    rows=[json.loads(x) for x in file.read_text().splitlines()]
    item=dict(events=dict(collections.Counter(x['event'] for x in rows)))
    ready=next((x for x in rows if x['event']=='model_ready'),None)
    if ready:item['model']={k:ready[k] for k in ['local_parameters','checkpoint_policy','checkpointed_layers','allocated_after_init','reserved_after_init'] if k in ready}
    for event in ['validation','micro','step','complete']:
        found=[x for x in rows if x['event']==event]
        if found:
            last=found[-1]
            if event=='step':last={k:v for k,v in last.items() if k!='norm_hook_calls'}
            item['last_'+event]=last
    result[rank]=item
print(json.dumps(result))
