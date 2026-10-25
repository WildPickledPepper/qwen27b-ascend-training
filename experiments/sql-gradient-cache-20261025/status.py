import json
from pathlib import Path
r=Path(__file__).resolve().parent
print((r/'job.json').read_text())
for name in ['tiny-native','tiny-cache','cache12-full']:
 p=r/name/'rank0.jsonl'
 if not p.exists():continue
 rows=[]
 for line in p.read_text().splitlines():
  try:v=json.loads(line)
  except ValueError:continue
  if v.get('event') in ['step_complete','gradient_cache','sample_checkpoint','validation_after','complete','micro_complete']:rows.append(v)
 print(name,json.dumps(rows[-2:]))
log=r/'supervisor.log'
print(log.read_text(errors='replace')[-700:] if log.exists() else '')
