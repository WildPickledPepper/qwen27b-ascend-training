"""Derive explicitly conditional scheduling limits from one matched profile."""
import json
from pathlib import Path
r=Path(__file__).resolve().parents[1];out=r/'evidence/2026-10-25'
p=json.loads((out/'sql-boundary-overlap-profile.json').read_text())['results']
o=json.loads((out/'sql-boundary-overlap-communication-overlap.json').read_text())['summary']
w=p['device_window'][0]['window_seconds'];c=p['compute_interval_union'][0]['nonoverlapping_compute_seconds']
adam=sum(x['seconds'] for x in p['exclusive_timeline'] if x['mask']&8 and not x['mask']&1)
rows=json.loads((out/'sql-boundary-overlap-communication-overlap.json').read_text())['rows']
assert not any(x['mask']&2 and x['mask']&4 and x['seconds']>0 for x in rows)
rs=o['reduce_scatter']['without_compute_seconds'];ag=o['allgather']['without_compute_seconds']
scenarios={name:{'seconds':t,'time_reduction_percent':100*(1-t/w),'speedup':w/t} for name,t in [
 ('hide_reduce_only',w-rs),('hide_all_collectives',w-rs-ag),('fixed_compute_and_serial_adam',c+adam),('fixed_compute_with_perfect_cross_step_update_overlap',c)]}
result={'basis':'Same rank0 profile only, not unprofiled seconds/256 conversion','window_seconds':w,'compute_union_seconds':c,'cpu_adam_without_compute_seconds':adam,'scenarios':scenarios,
 'assumptions':['Observed compute execution time treated as fixed.','Perfect hiding of chosen ranges without new contention is hypothetical.','Serial-Adam scenario retains current optimizer boundary; cross-step overlap is a separate unproven scenario.'],
 'limits':'Conditional scheduling estimates, not physical hardware bounds or achieved speedups. No new training.'}
(out/'sql-scheduling-ceiling.json').write_text(json.dumps(result,indent=2));print(json.dumps(scenarios,indent=2))
