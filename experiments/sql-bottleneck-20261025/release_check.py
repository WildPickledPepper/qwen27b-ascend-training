import json,re,subprocess,time
from pathlib import Path
r=Path(__file__).resolve().parent
s=subprocess.check_output(['npu-smi','info'],text=True)
devices={}
for i in [2,3]:
 m=re.search(r'\| '+str(i)+r'\s+910B2.*?\n\| 0\s+.*?\s(\d+)\s*/\s*65536',s)
 devices[str(i)]={'no_running_processes':f'No running processes found in NPU {i}' in s,'hbm_mib':int(m.group(1)) if m else None}
v={'unix':time.time(),'job_complete':json.loads((r/'job.json').read_text())['stage']=='complete','devices':devices,'other_known_service_still_present':'1424210' in s}
assert v['job_complete'] and all(x['no_running_processes'] for x in devices.values())
(r/'resource-release.json').write_text(json.dumps(v,indent=2));print(json.dumps(v))
