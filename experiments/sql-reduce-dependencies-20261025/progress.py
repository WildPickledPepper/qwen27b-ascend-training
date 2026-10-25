import json
from pathlib import Path
r=Path(__file__).resolve().parent
v={'job':json.loads((r/'capture-job.json').read_text())}
name=next((n for n in ['capture-full','tiny-smoke'] if (r/n/'rank0.jsonl').exists()),None)
if name:
 rows=[json.loads(x) for x in (r/name/'rank0.jsonl').read_text().splitlines()]
 v['run']=name;v['last_event']=rows[-1]['event'] if rows else None
 v['steps']=[{k:x[k] for k in ['step','seconds','profiled']} for x in rows if x['event']=='step_complete']
 v['micro']=[{k:x[k] for k in ['step','micro']} for x in rows if x['event']=='micro_complete'][-1:]
if v['job']['stage']=='failed':
 entry=v['job']['runs'][-1]['name'];v['error']=(r/(entry+'.private.log')).read_text(errors='replace')[-2200:]
print(json.dumps(v,indent=2))
