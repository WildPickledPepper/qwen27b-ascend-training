"""Separate an observed historical stall from non-reproducing diagnostics."""
import json
from pathlib import Path
r=Path(__file__).resolve().parent;e=r.parents[1]/'evidence/2026-10-25'
load=lambda n:json.loads((e/n).read_text())
history=load('sql-forward-history.json');detailed=load('sql-forward-diagnosis.json');light=load('sql-forward-light.json');db=load('sql-forward-scopes.json');profile=load('sql-forward-profile.json')
assert detailed['completed'] and light['completed'] and profile['valid']
assert detailed['same_validation_at_768']
def first(rows,event):return next(x for x in rows if x['event']==event and x['step']==3 and x['micro']==1)
light_ranks=[]
for rank in light['ranks']:
 event=first(rank['events'],'forward_light')
 all_forward=[x for x in rank['events'] if x['event']=='forward_light']
 light_ranks.append({'rank':rank['rank'],'third_first':event,'steps':[{k:x[k] for k in ['step','seconds','phase_wall_seconds','profiled']} for x in rank['events'] if x['event']=='step_complete'],
 'total_forward_alloc_retries':sum(x['allocator_delta'].get('num_alloc_retries',0) for x in all_forward),
 'total_forward_ooms':sum(x['allocator_delta'].get('num_ooms',0) for x in all_forward),
 'long_gc_seen':all_forward[-1]['long_gc_seen']})
v={'historical_rank0_third_update':{n:{'first_micro_wall_seconds':history[n][0]['steps'][1]['micro_wall_seconds'][0],'all_forward_wall_seconds':history[n][0]['steps'][1]['phase_wall_seconds']['forward_wall']} for n in history},
 'detailed_third_first_per_rank':[first(x['events'],'forward_parts') for x in detailed['ranks']],
 'light_per_rank':light_ranks,'detailed_first_forward_cann_top':db['results']['first_engine_cann'][:5],
 'detailed_third_allocator_calls':db['results']['allocator_calls'],
 'detailed_validation_exact':detailed['same_validation_at_768'],'light_validation_exact':light['same_validation_at_768'],
 'light_validation_comparisons':light['validation_comparisons'],
 'both_runs_devices_released':detailed['devices_2_3_released'] and light['devices_2_3_released'],
 'conclusion':'Historical extra delay is concentrated in first microbatch of update3. Neither detailed nor lightweight reproduction reproduced the extra ~4 seconds. A persistent forward regression from CPU fusion is not established; the historical root cause remains unobserved.',
 'limits':['Detailed Level1+cProfile and light counters perturb timing and allocation patterns.',
 'No old anomalous third-update DB exists, so healthy repeat traces cannot retroactively exclude an allocator, compiler, GC, or scheduling stall in the historical run.',
 'Do not replace the selected throughput configuration using these diagnostic repetitions.']}
(e/'sql-forward-conclusion.json').write_text(json.dumps(v,indent=2));print(json.dumps(v,indent=2))
