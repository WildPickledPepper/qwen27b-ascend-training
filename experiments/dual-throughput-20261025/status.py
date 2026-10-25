"""Compact read-only progress for the sequential tuning jobs."""
import json
from pathlib import Path
r=Path(__file__).resolve().parent
for filename in ['advanced-job.json','matched-job.json']:
 if not (r/filename).exists():continue
 state=json.loads((r/filename).read_text())
 output={'job':filename,'stage':state['stage'],'best':state.get('best',state.get('best_throughput_configuration')),'runs':[]}
 names=[x['name'] for x in state.get('runs',[])]
 if filename=='matched-job.json' and state.get('pid'):names.append('matched-g2-optimized')
 for name in names:
  row={'name':name}
  path=r/name/'rank0.jsonl'
  if path.exists():
   events=[]
   for line in path.read_text().splitlines():
    try:events.append(json.loads(line))
    except json.JSONDecodeError:pass
   row['last_event']=events[-1]['event'] if events else None
   row['steps']=[{'step':e['step'],'seconds':e['seconds'],'phase_wall_seconds':e['phase_wall_seconds']} for e in events if e['event']=='step_complete']
  output['runs'].append(row)
 print(json.dumps(output))
