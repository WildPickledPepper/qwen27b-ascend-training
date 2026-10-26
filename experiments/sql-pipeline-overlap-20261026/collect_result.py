"""Compare async D2H against the sealed same-cache, same-workload pipeline."""
import argparse,hashlib,json,subprocess
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--reference',required=True)
p.add_argument('--zero-reference',required=True);p.add_argument('--output',required=True);p.add_argument('--cache-change',action='store_true');a=p.parse_args()
root=Path(a.root);reference=Path(a.reference);zero=Path(a.zero_reference)
new=[json.loads((root/f'rank{r}-result.json').read_text()) for r in [0,1]]
old=[json.loads((reference/f'rank{r}-result.json').read_text()) for r in [0,1]]
checks=[]
for rank in [0,1]:
    logs=[json.loads(x) for x in (root/f'rank{rank}.jsonl').read_text().splitlines()]
    refs=[json.loads(x) for x in (reference/f'rank{rank}.jsonl').read_text().splitlines()]
    for event,keys in [('data',['training_hashes','training_lengths','validation_hashes','global_batch','micro_batch','pipeline_micros','padding']),
                       ('model_ready',['parameters','manifest','checkpoint_layers'])]:
        n=next(x for x in logs if x['event']==event);o=next(x for x in refs if x['event']==event)
        for key in keys:assert n[key]==o[key],(rank,event,key)
    plan=next(x for x in logs if x['event']=='resident_plan');op=next(x for x in refs if x['event']=='resident_plan')
    if not a.cache_change:
        assert {k:v for k,v in plan.items() if k not in ['unix']}=={k:v for k,v in op.items() if k not in ['unix']}
    else:assert plan['allocated_bytes']==4*2**30 and op['allocated_bytes']==2*2**30
    assert new[rank]['validation_before']==old[rank]['validation_before']
    assert new[rank]['validation_after_768']==old[rank]['validation_after_768']
    assert new[rank]['state_bytes']==old[rank]['state_bytes']
    assert len(new[rank]['steps'])==len(old[rank]['steps'])==4
    for s,(n,o) in enumerate(zip(new[rank]['steps'],old[rank]['steps'])):
        for key in ['input_tokens','padded_tokens','grad_norm']:
            assert n[key]==o[key],(rank,s,key,n[key],o[key])
        saved=(plan['allocated_bytes']-op['allocated_bytes'])*15
        assert o['gradient_d2h_bytes']-n['gradient_d2h_bytes']==saved
        assert n['resident_add_calls']==plan['selected_parameters']*16
        assert n['resident_add_bytes']==plan['allocated_bytes']*16
        assert n['profiled']==(s==3)
        if 'offload_parameter_stats' in n:assert sum(t['bytes'] for t in n['offload_parameter_stats'].values())==n['gradient_d2h_bytes']
    checks.append(dict(rank=rank,same_data_shapes_padding_and_precision=True,resident_bytes=plan['allocated_bytes'],
        saved_d2h_per_update=(plan['allocated_bytes']-op['allocated_bytes'])*15,all_four_gradient_norms_exact=True,
        validation_before_and_after_768_exact=True))
normal=[max(r['steps'][s]['seconds'] for r in new) for s in [1,2]]
previous=[max(r['steps'][s]['seconds'] for r in old) for s in [1,2]]
mean=sum(normal)/2;baseline=sum(previous)/2
metadata=json.loads((zero/'selected-profile.json').read_text())
integrity={name:hashlib.sha256((zero/name).read_bytes().replace(b'\r\n',b'\n')).hexdigest()==expected for name,expected in metadata['files_sha256'].items()}
assert all(integrity.values())
smi=subprocess.check_output(['npu-smi','info'],text=True)
assert all(f'No running processes found in NPU {i}' in smi for i in [2,3])
result=dict(normal_seconds=normal,reference_normal_seconds=previous,seconds_per_256=mean,reference_seconds_per_256=baseline,
    time_reduction_percent=(1-mean/baseline)*100,checks=checks,rank_results=new,
    validation_after_768=new[0]['validation_after_768'],original_zero3_integrity=integrity,
    resource_release=dict(cards_2_3_no_processes=True,other_service_pid_present='1424210' in smi),
    limits=['Two normal timed updates after warmup; fourth profiled update excluded.',
            'Tiny three-step full-state exactness and eight validation rows are not long-term convergence proof.',
            'Host worker/slot waits overlap backward; do not add them to step time.'])
Path(a.output).write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k!='rank_results'}))
