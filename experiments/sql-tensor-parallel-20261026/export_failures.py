"""Filter capacity failures; retain no server paths or raw sample data."""
import hashlib,importlib.metadata,json,re
from pathlib import Path
root=Path(__file__).resolve().parent;rows=[]
for name,checkpoint in [('full','selective48'),('full-all','all64')]:
    folder=root/(name+'.private');logfile=root/(name+'.private.log')
    ranks=[]
    for rank in [0,1]:
        data=[json.loads(x) for x in (folder/f'rank{rank}.jsonl').read_text().splitlines()]
        ready=next(x for x in data if x['event']=='model_ready')
        micros=[x for x in data if x['event']=='micro'];steps=[x for x in data if x['event']=='step']
        ranks.append(dict(rank=rank,completed_micros=len(micros),completed_updates=len(steps),
                          local_parameters=ready['local_parameters'],
                          allocated_after_init=ready.get('allocated_after_init'),reserved_after_init=ready.get('reserved_after_init')))
    messages=[]
    for line in logfile.read_text(errors='replace').splitlines():
        if 'torch.OutOfMemoryError:' in line:
            messages.append(line.split('torch.OutOfMemoryError:',1)[1].split('If reserved memory',1)[0].strip())
    assert messages and all(x['completed_updates']==0 for x in ranks)
    rows.append(dict(checkpoint=checkpoint,zero_stage=1,ranks=ranks,oom_messages=messages))
source=Path(importlib.metadata.distribution('deepspeed').locate_file('deepspeed/runtime/zero/stage_1_and_2.py'))
text=source.read_text();gate='if self.partition_gradients or self.is_gradient_accumulation_boundary() or self.zenflow:'
assert gate in text
result=dict(capacity_failures=rows,native_source=dict(lf_sha256=hashlib.sha256(text.encode()).hexdigest(),
            function='reduce_ready_partitions_and_remove_grads',guard=gate),
            conclusion='Stage1 skips per-micro reduction/cleanup outside the accumulation boundary; Stage2 enables it.',
            limits=['OOM trials have no completed updates and no speed result.',
                    'Stage2 success and actual per-micro storage checks are reported separately.'])
(root/'capacity-failures.json').write_text(json.dumps(result,indent=2));print('Exported two Stage1 capacity failures, zero successful updates')
