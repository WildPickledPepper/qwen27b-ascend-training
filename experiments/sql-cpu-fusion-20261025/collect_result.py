"""Audit the fixed workload against the selected 95.74-second reference."""
import json,statistics
from pathlib import Path
r=Path(__file__).resolve().parent;out=r.parents[1]/'evidence/2026-10-25'
prior=json.loads((out/'sql-boundary-overlap-results.json').read_text())
ranks=[json.loads((r/f'rank{i}-result.private.json').read_text()) for i in [0,1]]
events=[json.loads((r/f'events-rank{i}.private.json').read_text()) for i in [0,1]]
for i,result in enumerate(ranks):
 assert result['cpu_fusion']['native_adam_simd_math_reused']
 assert result['completed'] and result['global_batch']==256 and result['micro_batch']==16
 assert result['fp32_master'] and result['fp32_m_v'] and result['reduce_schedule']['event_limit']==4
 assert result['config']==json.loads((r.parent/'sql-reduce-schedule-20261025'/'zero3-selected.json').read_text())
 for j in range(3):
  a=result['steps'][j];b=prior['ranks'][i]['steps'][j]
  assert not a['profiled'] and a['fp32_m_v_bytes']==b['fp32_m_v_bytes']==107583993856
  assert a['input_tokens']==b['input_tokens'] and a['padded_input_tokens']==b['padded_input_tokens']
  assert a['cpu_gradient_accumulation_stats']['deferred_boundary_parameters']==851
  assert a['cpu_update_stats']['fused_scale_adam_bf16_calls']==44
  assert a['cpu_update_stats']['async_copy_bytes']==26895998464
  assert a['cpu_gradient_accumulation_stats'].get('boundary_wait_seconds',0)==0
val=[next(x['metrics'] for x in e if x['event']=='sample_checkpoint' and x['samples']==768) for e in events]
new=statistics.mean(max(x['steps'][j]['seconds'] for x in ranks) for j in [1,2]);base=prior['candidate_seconds_per_256']
result={'completed':True,'global_batch':256,'micro_batch_per_rank':16,'accumulation':8,
 'reference_seconds_per_256':base,'candidate_seconds_per_256':new,'time_reduction_percent':100*(1-new/base),
 'same_token_and_padding_workload':True,'fp32_master_m_v_unchanged':True,
 'reference_validation_at_768':prior['validation_at_768_per_rank'][0],'validation_at_768_per_rank':val,
 'validation_loss_difference':val[0]['loss']-prior['validation_at_768_per_rank'][0]['loss'],
 'ranks':[{'rank':i,'config':x['config'],'steps':x['steps'],'cpu_accumulation_plan':x['cpu_gradient_accumulation'],'cpu_fusion':x['cpu_fusion']} for i,x in enumerate(ranks)],
 'limits':['Three training updates and eight validation samples do not establish long-run convergence or full-model bitwise identity.',
 'Fourth update repeats the first 256 examples for profiling only; excluded from speed.',
 'Logical transfer bytes are submission accounting, not hardware link counters.']}
(out/'sql-cpu-fusion-results.json').write_text(json.dumps(result,indent=2))
print(json.dumps({k:v for k,v in result.items() if k not in ['ranks','limits']}))
