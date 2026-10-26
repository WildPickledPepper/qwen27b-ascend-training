"""Compact pipeline/codec progress; never dump sample data or full manifest."""
import json
from pathlib import Path
root=Path(__file__).resolve().parent
state=root/'job.private.json';print(state.read_text() if state.exists() else 'not started')
for name in ['tiny','full']:
    for rank in [0,1]:
        file=root/(name+'.private')/f'rank{rank}.jsonl'
        if not file.exists():continue
        rows=[json.loads(x) for x in file.read_text().splitlines()]
        steps=[{k:v for k,v in x.items() if k in ['step','seconds','allocated_peak','reserved_peak','worker_cpu_add_seconds',
            'offload_slot_wait','cpu_update_return_seconds','forward_backward_and_drain_seconds','grad_norm','compressed_weight_stats']} for x in rows if x['event']=='step']
        last=rows[-1];last={k:v for k,v in last.items() if k not in ['manifest','training_hashes','training_lengths','validation_hashes','checks','frames']}
        print(json.dumps(dict(name=name,rank=rank,last=last,steps=steps)))
