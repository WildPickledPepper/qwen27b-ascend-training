"""Summarize measured phase tradeoffs and verify own device cleanup."""
import argparse,hashlib,json,subprocess
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--reference',required=True);p.add_argument('--output',required=True);a=p.parse_args()
root=Path(a.root);ref=Path(a.reference)
result=json.loads((root/'results.json').read_text());profile=json.loads((root/'profile.json').read_text())
assert result['valid_for_speed_comparison']
assert json.loads((root/'sequence.private.json').read_text())['state']=='complete'
old=json.loads((ref/'rank0-result.json').read_text())
def avg(rows,key):return sum(x[key] for x in rows)/len(rows)
normal=result['rank_results'][0]['steps'][1:3]
oldnormal=old['steps'][1:3]
before_old=sum(x['phase_wall_seconds']['forward_wall']+x['phase_wall_seconds']['backward_wall'] for x in oldnormal)/2
before_new=sum(x['forward_backward_and_drain_seconds']+x['global_norm_seconds'] for x in normal)/2
phase=dict(reference_before_update_seconds=before_old,pipeline_before_update_seconds=before_new,
           reference_update_seconds=sum(x['phase_wall_seconds']['step_wall'] for x in oldnormal)/2,
           pipeline_update_seconds=avg(normal,'cpu_update_return_seconds'),
           reference_cpu_bf16_add_seconds=sum(x['cpu_gradient_accumulation_stats']['cpu_add_seconds'] for x in oldnormal)/2,
           pipeline_cpu_bf16_add_seconds=avg(normal,'worker_cpu_add_seconds'),
           pipeline_main_gradient_slot_wait_seconds=avg(normal,'offload_slot_wait'),
           pipeline_worker_copy_wait_seconds=avg(normal,'worker_copy_wait_seconds'))
for entry in profile['ranks']:
    assert all(x['op_type'] not in ['hcom_allGather_','hcom_reduceScatter_'] for x in entry['communication'])
    assert any(x['op_type']=='hcom_batchSendRecv_' for x in entry['communication'])
smi=subprocess.check_output(['npu-smi','info'],text=True)
release=dict(physical_cards=[2,3],no_running_processes=all(f'No running processes found in NPU {i}' in smi for i in [2,3]),
             existing_other_service_pid_present='1424210' in smi)
assert release['no_running_processes'],'Own cards have not been released'
sources={f.name:hashlib.sha256(f.read_bytes().replace(b'\r\n',b'\n')).hexdigest() for f in root.glob('*.py') if 'private' not in f.name}
selected=json.loads((ref.parent/'selected-profile.json').read_text())
default_integrity={name:hashlib.sha256((ref.parent/name).read_bytes().replace(b'\r\n',b'\n')).hexdigest()==expected for name,expected in selected['files_sha256'].items()}
assert all(default_integrity.values()),default_integrity
summary=dict(decision='Keep original selected ZeRO-3; measured 0.45% is below run-to-run spread, not a robust speedup',
             performance={k:result[k] for k in ['reference_seconds_per_256','pipeline_seconds_per_256','normal_slow_rank_seconds','speedup','time_reduction_percent']},
             phase=phase,validation=result['validation'],workload_verified=result['verification'],
             profile=[{k:r[k] for k in ['rank','coverage','communication']} for r in profile['ranks']],
             resource_release=release,original_selected_source_integrity=default_integrity,source_lf_sha256=sources,
             notes=['CPU BF16 add, worker-copy-wait and main slot wait overlap with backward; do not add them.',
                    'Before-update and update are main-thread scopes; update includes FP32 gradient preparation/CPUAdam/BF16 return.',
                    'Only two normal timed updates after warmup; no statistical speedup claim.',
                    'This isolated implementation is not an upper bound for pipeline parallel training.'])
Path(a.output).write_text(json.dumps(summary,indent=2));print(json.dumps({k:v for k,v in summary.items() if k not in ['source_lf_sha256','profile']}))
