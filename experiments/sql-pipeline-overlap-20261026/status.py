"""Compact progress without raw data, model paths or huge parameter counters."""
import argparse,json
from pathlib import Path
root=Path(__file__).resolve().parent
p=argparse.ArgumentParser();p.add_argument('--cache4',action='store_true');a=p.parse_args()
state=root/('cache4-job.private.json' if a.cache4 else 'job.private.json');print(state.read_text() if state.exists() else 'not started')
for rank in [0,1]:
    log=root/('cache4-full.private' if a.cache4 else 'async2-full.private')/f'rank{rank}.jsonl'
    if not log.exists():continue
    rows=[json.loads(x) for x in log.read_text().splitlines()]
    steps=[{k:v for k,v in x.items() if k in ['step','seconds','allocated_peak','worker_cpu_add_seconds',
        'offload_slot_wait','worker_copy_wait_seconds','cpu_update_return_seconds','forward_backward_and_drain_seconds','grad_norm']} for x in rows if x['event']=='step']
    micros=[x for x in rows if x['event']=='micro']
    print(json.dumps(dict(rank=rank,last_event=rows[-1]['event'],steps=steps,last_micro=micros[-1] if micros else None)))
