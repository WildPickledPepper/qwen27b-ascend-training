"""Validate stopped profiling, export bounded aggregate reports, verify release."""
import json,subprocess,sys
from pathlib import Path
r=Path(__file__).resolve().parent
assert json.loads((r/'job.json').read_text())['stage']=='complete'
for script,args in [('export_evidence.py',[]),('analyze_boundary.py',['--root',str(r/'cpu-full'/'profile'),'--output',str(r/'profile-analysis.json')]),('communication_overlap.py',['--root',str(r/'cpu-full'/'profile'),'--output',str(r/'communication-overlap.json')])]:
 with (r/(script+'.private.log')).open('w') as log:
  subprocess.run([sys.executable,str(r/script),*args],cwd=r,check=True,stdout=log,stderr=subprocess.STDOUT)
 print(json.dumps({'completed':script}),flush=True)
probe=json.loads((r/'probe-comparison.json').read_text());assert probe['passed']
reference=r.parent/'qwen-sql-boundary-overlap-20261025'/'cpu-full'
for rank in [0,1]:
 result=json.loads((r/'cpu-full'/f'rank{rank}-result.json').read_text())
 old=json.loads((reference/f'rank{rank}-result.json').read_text())
 assert result['config']==old['config']
 current_events=[json.loads(x) for x in (r/'cpu-full'/f'rank{rank}.jsonl').read_text().splitlines()]
 old_events=[json.loads(x) for x in (reference/f'rank{rank}.jsonl').read_text().splitlines()]
 data=next(x for x in current_events if x['event']=='data')
 old_data=next(x for x in old_events if x['event']=='data')
 for key in ['training_hashes','training_lengths','validation_hashes']:assert data[key]==old_data[key],key
 current_validation=next(x['metrics'] for x in current_events if x['event']=='sample_checkpoint' and x['samples']==768)
 old_validation=next(x['metrics'] for x in old_events if x['event']=='sample_checkpoint' and x['samples']==768)
 assert current_validation==old_validation
 assert result['cpu_fusion']['native_adam_simd_math_reused']
 assert result['completed'] and len(result['steps'])==4
 assert result['reduce_schedule']['event_limit']==4 and result['cpu_gradient_accumulation']['slots']==32
 assert 'boundary_policy' in result['cpu_gradient_accumulation']
 for row in result['steps']:
  assert row['cpu_gradient_accumulation_stats']['deferred_boundary_parameters']==851
  assert row['cpu_gradient_accumulation_stats'].get('boundary_wait_seconds',0)==0
  assert row['fp32_m_v_bytes']==107583993856
  assert row['cpu_update_stats']['fused_scale_adam_bf16_calls']==44
  assert row['cpu_update_stats']['async_copy_bytes']==26895998464
  assert sum(row['gradient_observer_stats'].get(k,0) for k in ['h2d_source_bytes','d2h_source_bytes'])==295855983104
v={'passed':True,'both_ranks_complete':True,'tiny_full_gradient_master_m_v_bitwise':True,
 'same_boundary_parameter_count_all_steps':True,'native_reduce_event_limit_4_preserved':True,
 'cpu_receive_slots_32_preserved':True,'cpu_fp32_master_m_v_preserved':True,
 'same_data_hashes_order_lengths_validation_split':True,'same_deepspeed_configuration':True,
 'same_768_sample_validation_metrics':True,'same_logical_gradient_transfer_bytes':True,
 'profile_valid':json.loads((r/'profile-analysis.json').read_text())['valid'],
 'resource_release':json.loads((r/'resource-release.json').read_text()),
 'original_cpu_probe_passed':json.loads((r/'cpu-probe.json').read_text())['passed'],
 'fused_optimizer_calls_44_all_full_steps':True,
 'limits':'Tiny bitwise checks and eight heldout samples do not establish full-model long-term identity.'}
(r/'verification.json').write_text(json.dumps(v,indent=2));print(json.dumps(v),flush=True)
