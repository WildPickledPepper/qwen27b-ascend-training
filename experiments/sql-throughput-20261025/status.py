"""Print compact benchmark progress without dataset text or private paths."""
import json,time
from pathlib import Path
r=Path(__file__).resolve().parent
matched=(r/'matched-job.json').exists()
j=json.loads((r/('matched-job.json' if matched else 'job.json')).read_text())
out={'stage':j['stage'],'elapsed_seconds':round(time.time()-j['started_unix']),'ranks':[]}
for rank in [0,1]:
 path=r/('train-g256-m8' if matched else 'train-g256-m4')/f'rank{rank}.jsonl'
 if not path.exists():continue
 events=[]
 for line in path.read_text().splitlines():
  try:events.append(json.loads(line))
  except json.JSONDecodeError:pass
 if not events:continue
 e=events[-1];entry={'rank':rank,'last_event':e['event'],'age_seconds':round(time.time()-e['unix'])}
 for key in ['step','micro','total_micros','tensor_count','seconds']:
  if key in e:entry[key]=e[key]
 entry['completed_steps']=[{'step':v['step'],'seconds':v['seconds']} for v in events if v['event']=='step_complete']
 out['ranks'].append(entry)
print(json.dumps(out,indent=2))
