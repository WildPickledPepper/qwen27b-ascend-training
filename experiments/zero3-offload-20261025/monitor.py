"""Compact progress without training text or private paths."""
import json
from pathlib import Path
r=Path(__file__).resolve().parent
out={}
for name in ['single-run2','dual-run2']:
 f=r/name/'rank0.jsonl'
 if not f.exists():continue
 rows=[]
 for line in f.read_text().splitlines():
  try:rows.append(json.loads(line))
  except json.JSONDecodeError:pass
 steps=[x for x in rows if x['event']=='step_complete']
 out[name]={'last_event':rows[-1]['event'],'completed':(r/name/'rank0-result.json').exists(),
            'step_seconds':[x['seconds'] for x in steps]}
 for event in ['validation_before','validation_after']:
  metrics=[x['metrics'] for x in rows if x['event']==event]
  if metrics:out[name][event]=metrics[-1]
print(json.dumps(out),flush=True)
