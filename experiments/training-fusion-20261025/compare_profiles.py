"""Compare rank0 profiles with the same global batch, samples and checkpointing."""
import json
from pathlib import Path
r=Path(__file__).resolve().parents[2]
old=json.loads((r/'evidence/2026-10-25/dual-throughput-cpu24-profile.json').read_text())['results']
new=json.loads((r/'evidence/2026-10-25/training-fusion-profile.json').read_text())['results']
result={'same_protocol':'Global batch32, three updates, same data, rank0 final-update profile, 48/64 checkpointed layers',
 'metrics':{},'limits':['Summed API/kernel/communication durations overlap; not a critical-path partition',
 'Unprofiled 34.8777 vs 27.5756 seconds are used for speed, not profile-window ratios',
 'aten::copy_ includes multiple purposes and is not pure PCIe time']}
for key in ['compute_total','compute_interval_union','cpu_adam_ranges','communication_top20','device_window']:
 result['metrics'][key]={'before':old[key],'after':new[key]}
result['compute_call_reduction_fraction']=1-new['compute_total'][0]['calls']/old['compute_total'][0]['calls']
result['matmul_v3']={'before':next(x for x in old['compute_top20'] if x['op_type']=='MatMulV3'),
 'after':next(x for x in new['compute_top20'] if x['op_type']=='MatMulV3')}
(r/'evidence/2026-10-25/training-fusion-profile-comparison.json').write_text(json.dumps(result,indent=2))
print(json.dumps({'compute_call_reduction_fraction':result['compute_call_reduction_fraction'],'matmul_calls':result['matmul_v3']['after']['calls']}))
