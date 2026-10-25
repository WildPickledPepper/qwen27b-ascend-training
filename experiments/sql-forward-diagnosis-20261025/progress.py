import json,sys
from pathlib import Path
r=Path(__file__).resolve().parent;light='--light' in sys.argv
v={'job':json.loads((r/('light-job.json' if light else 'job.json')).read_text())}
p=r/('light-full' if light else 'cpu-full')/'rank0.jsonl'
if p.exists():
 rows=[json.loads(x) for x in p.read_text().splitlines()]
 v['last_event']=rows[-1]['event'] if rows else None
 v['micro']=[{k:x[k] for k in ['step','micro']} for x in rows if x['event']=='micro_complete'][-1:]
 v['steps']=[{k:x[k] for k in ['step','seconds','phase_wall_seconds','profiled']} for x in rows if x['event']=='step_complete']
 v['forward']=[{k:x[k] for k in ['step','micro','parts','before_memory','after_memory']} for x in rows if x['event']=='forward_parts'][-1:]
 if light:v['forward']=[{k:x[k] for k in ['step','micro','seconds','parts','allocator_delta','gc_seconds_by_generation']} for x in rows if x['event']=='forward_light'][-1:]
 if any(x['event']=='forward_cpu_profile' for x in rows):v['cpu_top']=next(x for x in rows if x['event']=='forward_cpu_profile')['top_self'][:8]
print(json.dumps(v,indent=2))
