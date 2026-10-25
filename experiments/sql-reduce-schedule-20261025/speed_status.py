import json,statistics
from pathlib import Path
r=Path(__file__).resolve().parent;values=[]
for rank in [0,1]:
 p=r/'cpu-full'/f'rank{rank}.jsonl'
 events=[json.loads(line) for line in p.read_text().splitlines()] if p.exists() else []
 steps=[x for x in events if x['event']=='step_complete'];values.append(steps)
 print(json.dumps({'rank':rank,'steps':[{'step':x['step'],'seconds':x['seconds'],'profiled':x['profiled']} for x in steps],
 'validation_at_768':[x['metrics'] for x in events if x['event']=='sample_checkpoint' and x['samples']==768]}))
if all(len(x)>=3 for x in values):
 sec=statistics.mean(max(v[i]['seconds'] for v in values) for i in [1,2])
 print(json.dumps({'seconds_per_256':sec,'baseline':107.97092516999692,'time_reduction_percent':100*(1-sec/107.97092516999692)}))
