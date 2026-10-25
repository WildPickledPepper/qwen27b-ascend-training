import json,time
from pathlib import Path
r=Path(__file__).resolve().parent;j=json.loads((r/'job.json').read_text())
out={'stage':j['stage'],'elapsed_seconds':round(time.time()-j['started_unix']),'completed_runs':[{'name':x['name'],'returncode':x['returncode']} for x in j['runs'] if 'returncode' in x],'ranks':[]}
for rank in [0,1]:
 p=r/j['stage']/f'rank{rank}.jsonl'
 if not p.exists():continue
 events=[]
 for line in p.read_text().splitlines():
  try:events.append(json.loads(line))
  except json.JSONDecodeError:pass
 if not events:continue
 e=events[-1];row={'rank':rank,'last_event':e['event'],'age_seconds':round(time.time()-e['unix'])}
 row.update({k:e[k] for k in ['step','micro','total_micros','tensor_count'] if k in e})
 row['completed_steps']=[{'step':v['step'],'seconds':v['seconds']} for v in events if v['event']=='step_complete']
 row['validation_checkpoints']=[{'samples':v['samples'],'loss':v['metrics']['loss']} for v in events if v['event']=='sample_checkpoint']
 out['ranks'].append(row)
print(json.dumps(out,indent=2))
