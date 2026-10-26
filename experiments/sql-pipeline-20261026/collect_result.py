"""Export aggregate results and fixed-workload checks; no raw samples or paths."""
import argparse,hashlib,json,math
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--reference',required=True);p.add_argument('--output',required=True)
p.add_argument('--diagnostic-initial-mismatch',action='store_true');a=p.parse_args()
root=Path(a.root);refroot=Path(a.reference)
logs=[[json.loads(x) for x in (root/f'rank{r}.jsonl').read_text().splitlines()] for r in [0,1]]
reference_logs=[[json.loads(x) for x in (refroot/f'rank{r}.jsonl').read_text().splitlines()] for r in [0,1]]
new=[json.loads((root/f'rank{r}-result.json').read_text()) for r in [0,1]]
old=[json.loads((refroot/f'rank{r}-result.json').read_text()) for r in [0,1]]
data=[next(x for x in lines if x['event']=='data') for lines in logs]
refdata=next(x for x in reference_logs[0] if x['event']=='data')
assert all(x['training_hashes']==refdata['training_hashes'] for x in data)
assert all(x['training_lengths']==refdata['training_lengths'] for x in data)
assert all(x['validation_hashes']==refdata['validation_hashes'] for x in data)
manifests=[next(x for x in lines if x['event']=='model_ready') for lines in logs]
assert sum(x['parameters'] for x in manifests)==26895998464
names=[v['name'] for m in manifests for v in m['manifest']]
assert len(names)==len(set(names))==851
assert sum(x['checkpoint_layers'] for x in manifests)==48
assert sum(x['state_bytes'] for x in new)==26895998464*8
normal=[max(r['steps'][s]['seconds'] for r in new) for s in [1,2]]
mean=sum(normal)/2;reference=95.74024082010146
for r in [0,1]:
    assert len(new[r]['steps'])==4
    assert all(not s['profiled'] for s in new[r]['steps'][:3]) and new[r]['steps'][3]['profiled']
    original=old[r]['steps']
    for s in range(4):
        # Old rank-local padding must sum to the PP global padding.
        padded=sum(old[k]['steps'][s]['padded_input_tokens'] for k in [0,1])
        tokens=sum(old[k]['steps'][s]['input_tokens'] for k in [0,1])
        assert new[r]['steps'][s]['padded_tokens']==padded
        assert new[r]['steps'][s]['input_tokens']==tokens
    initial=next(x for x in logs[r] if x['event']=='validation')['metrics']
    reference_initial=next(x for x in reference_logs[r] if x['event']=='validation_before')['metrics']
    if not a.diagnostic_initial_mismatch:
        assert initial['assistant_tokens']==reference_initial['assistant_tokens'] and initial['correct']==reference_initial['correct'],(initial,reference_initial)
        assert abs(initial['loss']-reference_initial['loss'])<=1e-6,(initial,reference_initial)
val=new[0]['validation_after_768']; assert new[1]['validation_after_768']==val
reference_val=next(x for x in reference_logs[0] if x['event']=='sample_checkpoint' and x['samples']==768)['metrics']
assert math.isfinite(val['loss'])
result=dict(valid_for_speed_comparison=not a.diagnostic_initial_mismatch,
            initial_validation=dict(pipeline=initial,reference=reference_initial),
            reference_seconds_per_256=reference,pipeline_seconds_per_256=mean,
            normal_slow_rank_seconds=normal,speedup=reference/mean,time_reduction_percent=(1-mean/reference)*100,
            samples_per_second=256/mean,rank_results=new,
            verification=dict(same_training_order=True,same_lengths=True,same_validation_split=True,
                              same_input_and_padded_tokens=True,unique_parameters=851,all_text_parameters=26895998464,
                              checkpointed_layers=48,cpu_fp32_m_v_bytes=sum(x['state_bytes'] for x in new)),
            validation=dict(reference=reference_val,pipeline=val,loss_difference=val['loss']-reference_val['loss'],
                            correct_difference=val['correct']-reference_val['correct']),
            limits=['BF16 accumulation tree changes; no full-model bitwise equivalence claim.',
                    'Eight validation examples are a short numerical check, not convergence evidence.',
                    'Fourth update profiling excluded from speed ranking.'])
Path(a.output).write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k not in ['rank_results']}))
