"""Plan bounded gradient residency from recorded real peaks, no NPU work."""
import argparse,json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--reference',required=True);p.add_argument('--output',required=True);a=p.parse_args()
root=Path(a.reference);rows=[]
for rank in [0,1]:
    logs=[json.loads(x) for x in (root/f'rank{rank}.jsonl').read_text().splitlines()]
    manifest=next(x['manifest'] for x in logs if x['event']=='model_ready')
    result=json.loads((root/f'rank{rank}-result.json').read_text())
    oldplan=next(x for x in logs if x['event']=='resident_plan')
    baseline=oldplan['allocated_bytes'];plans=[]
    for gib in [2,4,6]:
        budget=gib*2**30;used=0;chosen=[]
        for idx,info in sorted(enumerate(manifest),key=lambda x:(-x[1]['numel'],x[0])):
            size=info['numel']*2
            if size<=budget-used:chosen.append(idx);used+=size
        extra=used-baseline
        # Observed idle physical-card overhead was 3415MiB; not free application HBM.
        projected_reserved=max(x['reserved_peak'] for x in result['steps'])+extra
        projected_total=projected_reserved+3415*2**20
        plans.append(dict(gib=gib,allocated_bytes=used,selected_parameters=len(chosen),
            extra_bytes=extra,saved_logical_d2h_per_update=extra*15,
            projected_peak_allocated=max(x['allocated_peak'] for x in result['steps'])+extra,
            projected_reserved_plus_observed_idle_overhead=projected_total,
            remaining_physical_bytes=64*2**30-projected_total))
    rows.append(dict(rank=rank,plans=plans))
out=dict(ranks=rows,assumptions=['Same input lengths/checkpointing and allocator behavior as recorded four steps.',
    'Projection is a capacity estimate, not a guarantee; actual full-run peaks must be checked.',
    '6GiB exceeds the current cache hard bound and the projected rank0 physical budget; not tested.'],
    choice='If further caching is tested, choose 4GiB; keep source hard bound and avoid 6GiB.')
Path(a.output).write_text(json.dumps(out,indent=2));print(json.dumps(out))
