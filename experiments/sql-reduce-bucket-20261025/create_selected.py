"""Save a replay entry only after measured improvement and matched validation."""
import hashlib,json
from pathlib import Path
r=Path(__file__).resolve().parent
e=json.loads((r.parents[1]/'evidence/2026-10-25/sql-reduce-bucket-results.json').read_text())
raw=json.loads((r/'rank0-result.private.json').read_text())
previous=r.parent/'sql-boundary-overlap-20261025'
assert raw['completed'] and raw['config']==json.loads((previous/'zero3-selected.json').read_text())
assert e['time_reduction_percent']>=1 and e['validation_loss_difference']==0
assert e['validation_at_768_per_rank'][0]['correct']==e['reference_validation_at_768']['correct']
(r/'zero3-selected.json').write_text(json.dumps(raw['config'],indent=2),encoding='utf-8',newline='\n')
v=json.loads((previous/'selected-profile.json').read_text())
v.update(name='sql-fixed256-ready-singleton',seconds_per_256=e['candidate_seconds_per_256'],reference_seconds_per_256=e['reference_seconds_per_256'],time_reduction_percent=e['time_reduction_percent'],boundary_policy=raw['cpu_gradient_accumulation']['boundary_policy'])
names=['train_ready.py' if n=='train_boundary.py' else n for n in v['files_sha256']]+['ready_reduce.py']
v['ready_reduce']=raw['ready_reduce']
v['files_sha256']={n:hashlib.sha256((r/n).read_bytes().replace(b'\r\n',b'\n')).hexdigest() for n in names}
(r/'selected-profile.json').write_text(json.dumps(v,indent=2),encoding='utf-8',newline='\n')
(r/'run_selected.py').write_text((previous/'run_selected.py').read_text().replace('train_boundary.py','train_ready.py').replace("'--cpu-gradient-accum']","'--cpu-gradient-accum','--ready-reduce']"),encoding='utf-8',newline='\n')
print(json.dumps({'selected':v['name'],'seconds_per_256':v['seconds_per_256'],'time_reduction_percent':v['time_reduction_percent']}))
