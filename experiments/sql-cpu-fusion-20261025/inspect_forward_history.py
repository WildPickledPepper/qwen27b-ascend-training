"""Read existing runs; export bounded timing/placement evidence without inputs."""
import json
from pathlib import Path
r=Path(__file__).resolve().parent
cases={'reference':r.parent/'qwen-sql-boundary-overlap-20261025','eager_reduce':r.parent/'qwen-sql-reduce-bucket-20261025','cpu_fused':r}
out={}
for name,base in cases.items():
 ranks=[]
 for rank in [0,1]:
  rows=[json.loads(s) for s in (base/'cpu-full'/f'rank{rank}.jsonl').read_text().splitlines()]
  step=[]
  lengths=next(x['training_lengths'] for x in rows if x['event']=='data')
  shape_history=[];seen=set()
  for update in range(3):
   shapes=[]
   for micro in range(8):
    group=[lengths[update*256+(micro*16+j)*2+rank] for j in range(16)]
    padded=max(128,((max(group)+63)//64)*64)
    shapes.append({'micro':micro+1,'padded_length':padded,'first_training_occurrence':padded not in seen});seen.add(padded)
   shape_history.append({'step':update+1,'shapes':shapes})
  for index in [2,3]:
   progress=[x for x in rows if x['event']=='micro_complete' and x['step']==index]
   lr=next(x for x in rows if x['event']=='learning_rate' and x['step']==index)
   prior=lr['unix'];times=[]
   for x in progress:times.append(x['unix']-prior);prior=x['unix']
   full=next(x for x in rows if x['event']=='step_complete' and x['step']==index)
   step.append({'step':index,'micro_wall_seconds':times,'phase_wall_seconds':full['phase_wall_seconds'],'peak_reserved':full['peak_reserved'],'peak_allocated':full['peak_allocated']})
  placement=[{k:v for k,v in x.items() if k not in ['unix','event']} for x in rows if x['event']=='numa_placement']
  ranks.append({'rank':rank,'steps':step,'shape_history':shape_history,'numa_placement':placement})
 out[name]=ranks
(r/'forward-history.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
