import json
from pathlib import Path
r=Path(__file__).resolve().parent
state=json.loads((r/'job.json').read_text());print(json.dumps(state))
for name in ['tiny-cache','cpu-full']:
 p=r/name/'rank0.jsonl'
 if not p.exists():continue
 rows=[]
 for line in p.read_text().splitlines():
  try:v=json.loads(line)
  except ValueError:continue
  if v.get('event') in ['cpu_gradient_accumulation','step_complete','sample_checkpoint','validation_after','complete','micro_complete']:rows.append(v)
 print(name,json.dumps(rows[-2:]))
if state['stage']=='failed':
 print((r/'supervisor.log').read_text(errors='replace')[-800:])
