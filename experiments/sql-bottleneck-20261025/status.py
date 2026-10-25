import json
from pathlib import Path
r=Path(__file__).resolve().parent
print((r/'job.json').read_text())
for rank in [0,1]:
 p=r/'profile-g256-m16'/f'rank{rank}.jsonl'
 if p.exists():
  for line in p.read_text().splitlines():
   try:v=json.loads(line)
   except ValueError:continue
   if v.get('event') in ['step_complete','sample_checkpoint','validation_after','learning_rate']:
    print(json.dumps(v))
log=r/'profile-g256-m16.log'
if log.exists():
 lines=[x for x in log.read_text(errors='replace').splitlines() if not x.startswith('{')]
 print('\n'.join(lines[-6:]))
