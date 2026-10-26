"""Conditional scheduling ceilings from non-overlapping measured CPU phases."""
import argparse,json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--reference',required=True);p.add_argument('--candidate',required=True)
p.add_argument('--output',required=True);a=p.parse_args()
def summary(path):
    data=json.loads(Path(path).read_text());ranks=data['rank_results'];rows=[]
    for rank,r in enumerate(ranks):
        steps=r['steps'][1:3]
        add=sum(x['worker_cpu_add_seconds'] for x in steps)/2
        update=sum(x['cpu_update_return_seconds'] for x in steps)/2
        assert all(x['worker_cpu_add_seconds']+x['cpu_update_return_seconds']<=x['seconds'] for x in steps)
        rows.append(dict(rank=rank,cpu_bf16_add_seconds=add,update_return_tail_seconds=update,
            conditional_floor_seconds=add+update,normal_mean_seconds=sum(x['seconds'] for x in steps)/2))
    wall=sum(max(r['steps'][s]['seconds'] for r in ranks) for s in [1,2])/2
    floor=max(x['conditional_floor_seconds'] for x in rows)
    return dict(seconds_per_256=wall,ranks=rows,conditional_scheduling_floor_seconds=floor,
        maximum_remaining_seconds_if_only_other_waits_removed=wall-floor,
        maximum_remaining_time_reduction_percent=(1-floor/wall)*100)
out=dict(reference=summary(a.reference),candidate=summary(a.candidate),
    interpretation='CPU gradient additions finish before norm/update; these measured add and update durations cannot overlap in the current algorithm.',
    conditions=['Hold the measured CPU-add and update/return durations fixed; hypothetical removal of all other work or waits.',
        'This is an optimistic scheduling-only envelope, not a hardware bound or promised speed.',
        'CPU timings include contention/host scheduling and may change after a new optimization.',
        'Reducing CPU work itself, e.g. more residency or exact fusion, changes this conditional floor.',
        'Normal updates only; profiled-window seconds are never mixed into the arithmetic.'])
Path(a.output).write_text(json.dumps(out,indent=2));print(json.dumps(out))
