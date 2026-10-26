"""Same-workload, storage and precision audit against the sealed 4GiB pipeline."""
import argparse,hashlib,json,subprocess
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--reference',required=True)
p.add_argument('--baseline-source',required=True);p.add_argument('--zero-reference',required=True);p.add_argument('--output',required=True);a=p.parse_args()
root=Path(a.root);ref=Path(a.reference);baseline=Path(a.baseline_source);zero=Path(a.zero_reference)
new=[json.loads((root/f'rank{r}-result.json').read_text()) for r in [0,1]]
old=[json.loads((ref/f'rank{r}-result.json').read_text()) for r in [0,1]]
checks=[];audits=[]
for rank in [0,1]:
    logs=[json.loads(x) for x in (root/f'rank{rank}.jsonl').read_text().splitlines()]
    refs=[json.loads(x) for x in (ref/f'rank{rank}.jsonl').read_text().splitlines()]
    for event,keys in [('model_ready',['parameters','manifest','checkpoint_layers']),
                       ('data',['training_hashes','training_lengths','validation_hashes','global_batch','micro_batch','pipeline_micros','padding'])]:
        n=next(x for x in logs if x['event']==event);o=next(x for x in refs if x['event']==event)
        for key in keys:assert n[key]==o[key],(rank,event,key)
    plan=next(x for x in logs if x['event']=='resident_plan');previous=next(x for x in refs if x['event']=='resident_plan')
    assert plan['allocated_bytes']==7*2**30 and previous['allocated_bytes']==4*2**30
    audit=next(x for x in logs if x['event']=='compressed_store');audit={k:v for k,v in audit.items() if k not in ['unix','event']}
    assert audit['selected_parameters']==96 and audit['original_ffn_weight_bytes']==17112760320
    expected=audit['original_ffn_weight_bytes']-audit['resident_payload_bytes']
    assert expected-2**20<=audit['allocated_before']-audit['allocated_after']<=expected+2**20
    assert audit['all_initial_weights_bit_exact'];audits.append(audit)
    assert new[rank]['parameters']==old[rank]['parameters'] and new[rank]['state_bytes']==old[rank]['state_bytes']
    assert new[rank]['validation_before']==old[rank]['validation_before']
    assert new[rank]['validation_after_768']==old[rank]['validation_after_768']
    assert len(new[rank]['steps'])==len(old[rank]['steps'])==4
    for step,(n,o) in enumerate(zip(new[rank]['steps'],old[rank]['steps'])):
        for key in ['input_tokens','padded_tokens','grad_norm']:assert n[key]==o[key],(rank,step,key,n[key],o[key])
        assert o['gradient_d2h_bytes']-n['gradient_d2h_bytes']==3*2**30*15
        assert n['resident_add_bytes']==7*2**30*16 and n['resident_add_calls']==plan['selected_parameters']*16
        stats=n['compressed_weight_stats']
        assert stats['encode_calls']==288 and stats['saved_weight_packs']==stats['saved_weight_unpacks']==384
        assert stats['payload_bytes']<audit['original_ffn_weight_bytes']
    checks.append(dict(rank=rank,same_data_shapes_padding_checkpoint_and_precision=True,
        all_four_gradient_norms_exact=True,validation_before_and_after_768_exact=True,
        saved_logical_d2h_per_update=3*2**30*15,resident_bytes=plan['allocated_bytes'],
        all_288_updated_ffn_frames_encoded_and_bit_checked_or_raw_each_step=True))
normal=[max(r['steps'][s]['seconds'] for r in new) for s in [1,2]]
previous=[max(r['steps'][s]['seconds'] for r in old) for s in [1,2]]
mean=sum(normal)/2;previous_mean=sum(previous)/2
cfg=json.loads((baseline/'selected-profile.json').read_text())
integrity={name:hashlib.sha256((baseline/name).read_bytes().replace(b'\r\n',b'\n')).hexdigest()==value for name,value in cfg['source_lf_sha256'].items()}
assert all(integrity.values())
cfg=json.loads((zero/'selected-profile.json').read_text())
zero_checks={name:hashlib.sha256((zero/name).read_bytes().replace(b'\r\n',b'\n')).hexdigest()==value for name,value in cfg['files_sha256'].items()}
assert all(zero_checks.values())
smi=subprocess.check_output(['npu-smi','info'],text=True)
assert all(f'No running processes found in NPU {d}' in smi for d in [2,3])
result=dict(seconds_per_256=mean,reference_seconds_per_256=previous_mean,normal_seconds=normal,
    reference_normal_seconds=previous,time_change_percent=(mean/previous_mean-1)*100,
    selected=mean<previous_mean,checks=checks,initial_storage_audits=audits,
    validation_after_768=new[0]['validation_after_768'],rank_results=new,
    original_4gib_source_integrity=integrity,original_zero3_integrity=zero_checks,
    resource_release=dict(cards_2_3_no_processes=True),
    limits=['Two steady normal updates after warmup; fourth profile excluded.',
        'First warmup has a large CPU-update tail; its cause is not isolated and it is not a steady timing.',
        'Same short validation and gradient norms do not establish long-term full-model bitwise identity.',
        'Only FFN weight paging tested; not a universal conclusion about all compression implementations.'])
Path(a.output).write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k!='rank_results'}))
