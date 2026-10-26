"""Verify workload against the original matched pipeline and summarize residency."""
import argparse,hashlib,json,subprocess
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--reference',required=True);p.add_argument('--output',required=True);a=p.parse_args()
root=Path(a.root);ref=Path(a.reference)
new=[json.loads((root/f'rank{r}-result.json').read_text()) for r in [0,1]]
old=[json.loads((ref/f'rank{r}-result.json').read_text()) for r in [0,1]]
plans=[];rows=[]
for r in [0,1]:
    logs=[json.loads(x) for x in (root/f'rank{r}.jsonl').read_text().splitlines()]
    oldlogs=[json.loads(x) for x in (ref/f'rank{r}.jsonl').read_text().splitlines()]
    plan=dict(next(x for x in logs if x['event']=='resident_plan'));plan.pop('unix');plan.pop('event');plans.append(plan)
    data=next(x for x in logs if x['event']=='data');od=next(x for x in oldlogs if x['event']=='data')
    for key in ['training_hashes','training_lengths','validation_hashes']:assert data[key]==od[key],key
    assert new[r]['validation_before']==old[r]['validation_before']
    assert new[r]['state_bytes']==old[r]['state_bytes']
    assert new[r]['parameters']==old[r]['parameters']
    for s,(n,o) in enumerate(zip(new[r]['steps'],old[r]['steps'])):
        assert n['input_tokens']==o['input_tokens'] and n['padded_tokens']==o['padded_tokens']
        assert o['gradient_d2h_bytes']-n['gradient_d2h_bytes']==plan['allocated_bytes']*15
        rows.append(dict(rank=r,step=s+1,expected_and_actual_saved_d2h_bytes=plan['allocated_bytes']*15,
                         grad_norm_exact=n['grad_norm']==o['grad_norm']))
normal=[max(x['steps'][s]['seconds'] for x in new) for s in [1,2]]
oldnormal=[max(x['steps'][s]['seconds'] for x in old) for s in [1,2]]
mean=sum(normal)/2;previous=sum(oldnormal)/2
def metrics(rows):
    return {key:sum(x[key] for x in rows)/len(rows) for key in ['forward_backward_and_drain_seconds','cpu_update_return_seconds','worker_cpu_add_seconds','offload_slot_wait','worker_copy_wait_seconds','gradient_d2h_bytes']}
smi=subprocess.check_output(['npu-smi','info'],text=True)
assert all(f'No running processes found in NPU {i}' in smi for i in [2,3])
result=dict(passed_workload=True,pipeline_reference_seconds_per_256=previous,cache_seconds_per_256=mean,
            normal_seconds=normal,reference_normal_seconds=oldnormal,time_reduction_percent=(1-mean/previous)*100,
            versus_zero3_time_reduction_percent=(1-mean/95.74024082010146)*100,
            validation_reference=old[0]['validation_after_768'],validation_candidate=new[0]['validation_after_768'],
            validation_exact=new[0]['validation_after_768']==old[0]['validation_after_768'],
            resident_plans=plans,byte_and_norm_checks=rows,rank0_reference_normal_metrics=metrics(old[0]['steps'][1:3]),
            rank0_candidate_normal_metrics=metrics(new[0]['steps'][1:3]),rank_results=new,
            resource_release=dict(cards_2_3_no_processes=True,other_service_pid_present='1424210' in smi),
            limits=['Two normal timed updates after warmup; profiling excluded.',
                    'Nested worker and slot-wait times overlap backward; do not add them.',
                    'Tiny three-update bitwise checks and eight validation rows do not prove long-term quality.'])
Path(a.output).write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k!='rank_results'}))
