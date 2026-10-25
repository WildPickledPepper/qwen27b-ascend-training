import json,sys
from pathlib import Path
r=Path(__file__).resolve().parent
s=json.loads((r/(sys.argv[1] if len(sys.argv)>1 else 'research2-job.json')).read_text())
result={'stage':s['stage'],'cpu_best':s.get('cpu_best'),'runs':[]}
for run in s['runs'][-2:]:
 row={k:run[k] for k in ['name','returncode','failure','result'] if k in run}
 path=r/run['name']/'rank0.jsonl'
 if path.exists():
  events=[]
  for line in path.read_text().splitlines():
   try:events.append(json.loads(line))
   except json.JSONDecodeError:pass
  row['last_event']=events[-1]['event'] if events else None
  row['steps']=[{k:e[k] for k in ['step','seconds','cpu_update_stats']} for e in events if e['event']=='step_complete']
 result['runs'].append(row)
print(json.dumps(result))
