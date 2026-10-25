import json
from pathlib import Path
r=Path(__file__).resolve().parent
s=json.loads((r/'job.json').read_text());result={'stage':s['stage'],'runs':[{'name':x['name'],'returncode':x.get('returncode')} for x in s['runs']]}
for name in (['cpu-full'] if (r/'cpu-full'/'rank0.jsonl').exists() else ['tiny-cache']):
 p=r/name/'rank0.jsonl'
 if not p.exists():continue
 rows=[json.loads(x) for x in p.read_text().splitlines() if x.strip()]
 result[name]={'last_event':rows[-1]['event'] if rows else None,'steps':[{'step':x['step'],'seconds':x['seconds'],'profiled':x['profiled'],'cpu_accumulation':x['cpu_gradient_accumulation_stats']} for x in rows if x['event']=='step_complete']}
 if rows:result[name]['last_progress']={k:rows[-1][k] for k in ['step','micro','tensor_count'] if k in rows[-1]}
if s['stage']=='failed':
 result['failure']=(r/(s['runs'][-1]['name']+'.log')).read_text(errors='replace')[-2800:]
print(json.dumps(result,indent=2))
