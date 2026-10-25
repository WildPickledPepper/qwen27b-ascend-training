"""Export aggregate measurements only; source text and private runtime paths stay remote."""
import argparse,json,statistics
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--root',default='.');a=p.parse_args();r=Path(a.root)
job=json.loads((r/'matched-job.json').read_text());assert job['stage']=='complete'
results=[json.loads((r/'train-g256-m8'/f'rank{i}-result.json').read_text()) for i in [0,1]]
data=json.loads((r/'data'/'data-summary.json').read_text())
assert all(x['completed'] and len(x['steps'])==3 and x['parameters']==26895998464 and x['fp32_master'] and x['fp32_m_v'] for x in results)
assert not set(data['training_hashes'])&set(data['validation_hashes'])
assert len(set(data['training_hashes']))==768
rows=[]
for i in range(3):
 steps=[x['steps'][i] for x in results]
 assert all(s['global_batch_size']==256 and s['accumulation']==16 for s in steps)
 assert sum(s['fp32_m_v_bytes'] for s in steps)==26895998464*8
 seconds=max(s['seconds'] for s in steps)
 input_tokens=sum(s['input_tokens'] for s in steps);padded=sum(s['padded_input_tokens'] for s in steps)
 rows.append({'step':i+1,'global_samples':256,'seconds':seconds,'samples_per_second':256/seconds,
  'input_tokens':input_tokens,'input_tokens_per_second':input_tokens/seconds,'padded_tokens':padded,
  'padded_tokens_per_second':padded/seconds,'peak_allocated_gib_per_rank':[s['peak_allocated']/2**30 for s in steps],
  'weighted_loss':sum(s['weighted_local_loss'] for s in steps)/2,'fp32_m_v_total_bytes':sum(s['fp32_m_v_bytes'] for s in steps),
  'cpu_update_stats':[s['cpu_update_stats'] for s in steps],'phase_wall_seconds':[s['phase_wall_seconds'] for s in steps]})
mean=statistics.mean(x['seconds'] for x in rows[1:]);ours=256/mean
ref=data['source_manifest']['reference'];author=ref['author_reported_samples']/ref['author_reported_total_seconds']
export={'completed':True,'cards':2,'hardware':'Ascend 910B2 64 GiB','model':'Qwen3.5-27B text backbone',
 'parameters_updated':26895998464,'compute':'BF16','master_and_adam_states':'CPU FP32','global_batch':256,'micro_batch_per_rank':8,
 'gradient_accumulation':16,'max_length':512,'padding':'per microbatch rounded up to 64, minimum 128','activation_checkpointing':'all 64 layers',
 'steps':rows,'steady_mean_seconds_excluding_first':mean,'steady_samples_per_second':ours,
 'author_samples_per_second_from_reported_total':author,'author_equivalent_256_sample_seconds':256/author,
 'author_whole_machine_throughput_divided_by_ours':author/ours,
 'author_per_card_throughput_divided_by_ours':(author/16)/(ours/2),
 'estimated_seconds_for_178000_samples_training_only':178000/ours,
 'validation_before':results[0]['validation_before'],'validation_after':results[0]['validation_after'],
 'data':{k:v for k,v in data.items() if not k.endswith('hashes')},'source_sha256':job['source_sha256'],
 'capacity_check_micro4_first_step':job['capacity_step'],'capacity_run_disposition':job['capacity_run_disposition'],
 'limitations':['Author total wall time versus our short steady-state training time; validation, checkpoints, startup are not equal.',
 'Same original Qwen3.5 checkpoint family, data sources, global batch, length cap, and reported optimizer hyperparameters; exact author preprocessing and examples unavailable.',
 'Our padding, sharding, fused implementation, microbatch, activation recomputation, and CPU offload differ.',
 'Eight held-out SQL examples and three updates do not measure full-epoch quality. No complete trained checkpoint saved.']}
(r/'comparison.json').write_text(json.dumps(export,indent=2),encoding='utf-8')
print(json.dumps({k:v for k,v in export.items() if k not in {'steps','data','source_sha256','limitations'}},indent=2))
