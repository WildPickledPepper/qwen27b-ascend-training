import json,statistics
from pathlib import Path
r=Path(__file__).resolve().parent;repo=r.parents[1];out=repo/'evidence/2026-10-25'
old=json.loads((out/'sql-batch-tuning-results.json').read_text())['runs'][0]
ranks=[json.loads((r/f'rank{i}-result.private.json').read_text()) for i in [0,1]]
logs=[json.loads((r/f'events-rank{i}.private.json').read_text()) for i in [0,1]]
for rank,result in enumerate(ranks):
 assert result['completed'] and result['global_batch']==256 and result['micro_batch']==16
 assert result['fp32_master'] and result['fp32_m_v'] and result['cpu_gradient_accumulation']
 assert result['gradient_cache']['resident_bytes']==0
 for i in range(3):
  row=result['steps'][i];prior=old['steps'][i]
  assert not row['profiled'] and row['fp32_m_v_bytes']==107583993856
  assert sum(x['steps'][i]['input_tokens'] for x in ranks)==prior['input_tokens']
  assert sum(x['steps'][i]['padded_input_tokens'] for x in ranks)==prior['padded_tokens']
  transfer=row['gradient_observer_stats']
  assert transfer.get('h2d_source_bytes',0)+transfer.get('d2h_source_bytes',0)==295855983104
base=old['steady_mean_step_seconds'];new=statistics.mean(max(x['steps'][i]['seconds'] for x in ranks) for i in [1,2])
checkpoint=[next(x for x in log if x['event']=='sample_checkpoint' and x['samples']==768)['metrics'] for log in logs]
v={'completed':True,'fixed_global_batch':256,'micro_batch_per_rank':16,'accumulation':8,
 'baseline_seconds_per_256':base,'cpu_accumulation_seconds_per_256':new,'time_reduction_percent':100*(1-new/base),'throughput_multiplier':base/new,
 'cpu_fp32_master_m_v':True,'gradient_accumulation_dtype':'BF16, unchanged from reference',
 'same_768_sample_workload_verified':True,'same_token_and_padding_counts_verified':True,
 'baseline_validation_at_768':old['validation_checkpoints'][-1]['metrics'],'validation_at_768_per_rank':checkpoint,
 'validation_loss_difference':checkpoint[0]['loss']-old['validation_checkpoints'][-1]['metrics']['loss'],
 'baseline_gradient_transfer_bytes_per_rank':484127972352,'cpu_accumulation_gradient_transfer_bytes_per_rank':295855983104,
 'gradient_transfer_reduction_percent':100*(1-295855983104/484127972352),
 'peak_allocated_gib_per_rank':[max(s['peak_allocated'] for s in x['steps'])/2**30 for x in ranks],
 'ranks':[{'rank':i,'cpu_accumulation_plan':x['cpu_gradient_accumulation'],'steps':x['steps'],'validation_after_1024':x['validation_after']} for i,x in enumerate(ranks)],
 'limits':['Small-model full gradient/master/m/v bitwise comparison passed; full-model validation is not a full-model bitwise proof.',
 'First update is cold and excluded; second/third updates provide same-workload speed comparison.',
 'Fourth update is profiler-only and repeats the first 256 examples; its shape distribution differs from old third-update profile.',
 'Logical copy submissions are not hardware link-byte counters.']}
(out/'sql-gradient-pipeline-results.json').write_text(json.dumps(v,indent=2))
print(json.dumps({k:v[k] for k in ['baseline_seconds_per_256','cpu_accumulation_seconds_per_256','time_reduction_percent','throughput_multiplier','validation_loss_difference','peak_allocated_gib_per_rank']}))
