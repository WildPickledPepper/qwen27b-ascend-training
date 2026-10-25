import json,subprocess,time,re,sys
from pathlib import Path
r=Path(__file__).resolve().parent
assert json.loads((r/'job.json').read_text())['stage']=='complete'
for rank in [0,1]:
 rows=[]
 for line in (r/'cpu-full'/f'rank{rank}.jsonl').read_text().splitlines():
  x=json.loads(line)
  if x['event'] in ['sample_checkpoint','gradient_cache','step_complete','validation_after','complete']:rows.append(x)
 (r/f'events-rank{rank}.json').write_text(json.dumps(rows,indent=2))
if '--events-only' in sys.argv:
 print('Filtered events exported; resource check deferred until all authorized runs finish.');raise SystemExit(0)
s=subprocess.check_output(['npu-smi','info'],text=True)
devices={}
for i in [2,3]:
 m=re.search(r'\| '+str(i)+r'\s+910B2.*?\n\| 0\s+.*?\s(\d+)\s*/\s*65536',s)
 devices[str(i)]={'no_running_processes':f'No running processes found in NPU {i}' in s,'hbm_mib':int(m.group(1)) if m else None}
assert all(x['no_running_processes'] for x in devices.values())
(r/'resource-release.json').write_text(json.dumps({'unix':time.time(),'devices':devices,'other_known_service_still_present':'1424210' in s},indent=2))
print('Filtered events exported; devices 2/3 released.')
