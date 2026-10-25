import json,time
from pathlib import Path
r=Path(__file__).resolve().parent
assert json.loads((r/'job.json').read_text())['stage']=='failed'
dest=r/f'guard-failure-{time.time_ns()}';dest.mkdir(exist_ok=False)
for name in ['job.json','tiny-cache.log','tiny-cache','supervisor.log']:
 p=r/name
 if p.exists():
  assert p.resolve().is_relative_to(r.resolve())
  p.rename(dest/name)
