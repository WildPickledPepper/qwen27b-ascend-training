"""Fixed-workload TP/optimized-pipeline comparison; no raw data or host paths."""
import argparse,hashlib,json,math,subprocess
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--pipeline',required=True)
p.add_argument('--zero-reference',required=True);p.add_argument('--output',required=True);a=p.parse_args()
root=Path(a.root);reference=Path(a.pipeline);zero=Path(a.zero_reference)
new=[json.loads((root/f'rank{r}-result.json').read_text()) for r in [0,1]]
old=[json.loads((reference/f'rank{r}-result.json').read_text()) for r in [0,1]]
checks=[]
for rank in [0,1]:
    logs=[json.loads(x) for x in (root/f'rank{rank}.jsonl').read_text().splitlines()]
    refs=[json.loads(x) for x in (reference/f'rank{rank}.jsonl').read_text().splitlines()]
    data=next(x for x in logs if x['event']=='data');ref=next(x for x in refs if x['event']=='data')
    for key in ['training_hashes','training_lengths','validation_hashes']:assert data[key]==ref[key],key
    ready=next(x for x in logs if x['event']=='model_ready')
    assert len(ready['original_shapes'])==851
    assert sum(math.prod(s) for s in ready['original_shapes'].values())==26895998464
    assert ready['checkpointed_layers']==48 and ready['rebound_norm_parameters']==32
    cfg=new[rank]['config']
    assert (cfg['train_batch_size'],cfg['train_micro_batch_size_per_gpu'],cfg['gradient_accumulation_steps'])==(256,16,16)
    assert cfg['tensor_parallel']['autotp_size']==2 and cfg['zero_optimization']['stage']==2
    assert cfg['zero_optimization']['offload_optimizer']['device']=='cpu' and new[rank]['fp32_cpu_master_m_v']
    assert len(new[rank]['steps'])==4
    for step,(n,o) in enumerate(zip(new[rank]['steps'],old[rank]['steps'])):
        assert n['input_tokens']==o['input_tokens'] and n['padded_tokens']==o['padded_tokens']
        assert n['profiled']==(step==3)
        assert len(n['norm_hook_calls'])==32 and all(v==16 for v in n['norm_hook_calls'].values())
        assert math.isfinite(n['grad_norm'])
    checks.append(dict(rank=rank,same_data_order_lengths_validation_split=True,same_input_and_padded_tokens=True,
                       original_parameters=26895998464,original_parameter_tensors=851,checkpointed_layers=48,
                       tp=2,dp=1,norm_hook_calls_per_micro=32))
normal=[max(r['steps'][s]['seconds'] for r in new) for s in [1,2]]
pipeline_normal=[max(r['steps'][s]['seconds'] for r in old) for s in [1,2]]
mean=sum(normal)/2;previous=sum(pipeline_normal)/2
assert new[0]['validation_after_768']==new[1]['validation_after_768']
metadata=json.loads((zero/'selected-profile.json').read_text())
integrity={name:hashlib.sha256((zero/name).read_bytes().replace(b'\r\n',b'\n')).hexdigest()==expected for name,expected in metadata['files_sha256'].items()}
assert all(integrity.values())
smi=subprocess.check_output(['npu-smi','info'],text=True)
released=all(f'No running processes found in NPU {i}' in smi for i in [2,3]);assert released
result=dict(tp_seconds_per_256=mean,normal_seconds=normal,pipeline_seconds_per_256=previous,
    tp_time_over_pipeline=mean/previous,tp_extra_time_percent=(mean/previous-1)*100,
    validation_initial=dict(tp=new[0]['validation_before'],pipeline=old[0]['validation_before']),
    validation_after_768=dict(tp=new[0]['validation_after_768'],pipeline=old[0]['validation_after_768']),
    workload_checks=checks,rank_results=new,fp32_m_v_bytes_total=sum(r['state_bytes'] for r in new),
    original_zero3_integrity=integrity,
    resource_release=dict(cards_2_3_no_processes=released,other_service_pid_present='1424210' in smi),
    conclusion='Current optimized pipeline is faster; retain it as selected candidate, not native TP CPU-offload.',
    limits=['Same data/batch/micro/checkpoint count, but BF16 TP reduction tree and distributed CE differ.',
            'Native TP ZeRO-2 CPU offload does not include the optimized pipeline FIFO or resident cache.',
            'This implementation comparison does not establish an intrinsic upper bound for tensor parallelism.',
            'Two timed normal updates and eight validation examples; no long-term convergence claim.'])
Path(a.output).write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k!='rank_results'}))
