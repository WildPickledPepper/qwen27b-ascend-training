"""Compare fixed examples, shared warmup window and a preregistered short-validation target."""
import argparse,hashlib,json,math,statistics
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--root',default='.');a=p.parse_args();r=Path(a.root)
j=json.loads((r/'job.json').read_text());assert j['stage']=='complete'
target=.86;rows=[];expected_hashes=None;initial_losses=[]
for case in j['runs']:
 if case['returncode']:continue
 root=r/case['name'];batch=case['batch'];micro=case['micro_batch']
 for name,digest in case['source_sha256'].items():assert hashlib.sha256((r/name).read_bytes()).hexdigest()==digest,name
 results=[json.loads((root/f'rank{i}-result.json').read_text()) for i in [0,1]]
 events=[[json.loads(s) for s in (root/f'rank{i}.jsonl').read_text().splitlines()] for i in [0,1]]
 for result,ev in zip(results,events):
  assert result['completed'] and result['parameters']==26895998464 and result['fp32_master'] and result['fp32_m_v']
  data=next(e for e in ev if e['event']=='data')
  if expected_hashes is None:expected_hashes=data['training_hashes']
  assert data['training_hashes']==expected_hashes and len(set(expected_hashes))==len(expected_hashes)==768
  assert not set(expected_hashes)&set(data['validation_hashes']) and not data['repeats']
  initial_losses.append(result['validation_before']['loss'])
  assert len(result['steps'])*batch==768
 steps=[]
 for index in range(len(results[0]['steps'])):
  pair=[x['steps'][index] for x in results]
  assert all(x['global_batch_size']==batch and x['micro_batch_size']==micro and not x['profiled'] for x in pair)
  assert sum(x['fp32_m_v_bytes'] for x in pair)==215167987712
  steps.append({'step':index+1,'samples_complete':(index+1)*batch,'seconds':max(x['seconds'] for x in pair),
   'input_tokens':sum(x['input_tokens'] for x in pair),'padded_tokens':sum(x['padded_input_tokens'] for x in pair),
   'peak_allocated_gib_per_rank':[x['peak_allocated']/2**30 for x in pair],
   'cpu_adam_seconds_per_rank':[x['cpu_update_stats']['cpu_adam_step_seconds'] for x in pair],
   'phase_wall_seconds_per_rank':[x['phase_wall_seconds'] for x in pair]})
 points=[]
 for n in [256,512]:
  metrics=[next(e for e in ev if e['event']=='sample_checkpoint' and e['samples']==n)['metrics'] for ev in events]
  assert abs(metrics[0]['loss']-metrics[1]['loss'])<1e-5
  points.append({'samples':n,'training_seconds':sum(s['seconds'] for s in steps if s['samples_complete']<=n),'metrics':metrics[0]})
 points.append({'samples':768,'training_seconds':sum(s['seconds'] for s in steps),'metrics':results[0]['validation_after']})
 assert all(math.isfinite(x['metrics']['loss']) for x in points)
 warm=[s for s in steps if s['samples_complete']>256];seconds=sum(x['seconds'] for x in warm)
 hit=next((x for x in points if x['metrics']['loss']<=target),None)
 rows.append({'name':case['name'],'global_batch':batch,'micro_batch_per_rank':micro,'accumulation':batch//(2*micro),
  'steps':steps,'validation_checkpoints':points,'initial_validation':results[0]['validation_before'],
  'steady_512_sample_seconds':seconds,'steady_samples_per_second':512/seconds,
  'steady_mean_step_seconds':statistics.mean(s['seconds'] for s in warm),
  'short_validation_target':target,'first_target_checkpoint':hit,
  'peak_allocated_gib':max(max(s['peak_allocated_gib_per_rank']) for s in steps),
  'source_sha256':case['source_sha256']})
assert len(rows)==3 and {x['global_batch'] for x in rows}=={64,128,256}
assert max(initial_losses)-min(initial_losses)<1e-5
assert len({sum(s['input_tokens'] for s in x['steps'] if s['samples_complete']>256) for x in rows})==1
assert len({sum(s['padded_tokens'] for s in x['steps'] if s['samples_complete']>256) for x in rows})==1
qualified=[x for x in rows if x['first_target_checkpoint'] is not None and x['peak_allocated_gib']<57]
if qualified:
 fastest=min(x['first_target_checkpoint']['training_seconds'] for x in qualified)
 near=[x for x in qualified if x['first_target_checkpoint']['training_seconds']<=fastest*1.05]
 winner=max(near,key=lambda x:x['steady_samples_per_second'])
 reason='Shortest observed training time to validation loss <=0.86; within 5% prefer higher steady throughput.'
else:
 winner=min(rows,key=lambda x:x['validation_checkpoints'][-1]['metrics']['loss'])
 reason='No run met the short-validation target; provisional lowest final validation loss, not a validated quality winner.'
profile={'model_family':'Qwen3.5-27B text-only SQL','physical_cards':[2,3],'global_batch':winner['global_batch'],
 'micro_batch_per_rank':winner['micro_batch_per_rank'],'gradient_accumulation_steps':winner['accumulation'],
 'max_sequence_length':512,'checkpoint':'on','cpu_threads_per_rank':24,'numa_spread':True,'cpu_mode':'pipeline','fuse_gdn':True,
 'pad_mode':'dynamic64','config':'zero3-selected.json','compute_dtype':'BF16','master_and_adam_states':'CPU FP32',
 'lr_schedule_basis':'processed samples, reference batch256, warmup8960 samples, total178688 samples',
 'short_validation_target':target,'selection_reason':reason,'selection_scope':'Fixed 768 SQL samples and eight validation examples; not long-run optimality.'}
export={'completed':True,'same_training_examples':768,'warmup_samples':256,'measured_samples_per_case':512,
 'validation_target_preregistered':target,'runs':rows,'selected':profile,
 'failures':[{k:v for k,v in x.items() if k in ['name','returncode','failure']} for x in j['runs'] if x['returncode']],
 'limitations':['Eight validation examples provide an operational short-run target, not general model quality.',
 'Validation checkpoints every256 examples bound the target crossing; intermediate crossing times were not measured.',
 'Time-to-target includes first-update cold execution and optimizer initialization, excludes model loading and validation wall time.',
 'Global batch changes optimization trajectories and update count; the sample-based LR progress is held constant.',
 'Only the recorded batch/microbatch choices were measured; no hardware or long-run global optimum is claimed.']}
(r/'batch-results.json').write_text(json.dumps(export,indent=2)+'\n',encoding='utf-8')
(r/'selected-profile.json').write_text(json.dumps(profile,indent=2)+'\n',encoding='utf-8')
cfg=json.loads((r/'zero3-sql.json').read_text())
cfg.update(train_micro_batch_size_per_gpu=winner['micro_batch_per_rank'],train_batch_size=winner['global_batch'],gradient_accumulation_steps=winner['accumulation'])
(r/'zero3-selected.json').write_text(json.dumps(cfg,indent=2)+'\n',encoding='utf-8')
print(json.dumps({'runs':[{k:x[k] for k in ['name','steady_mean_step_seconds','steady_samples_per_second','peak_allocated_gib','first_target_checkpoint']} for x in rows],'selected':profile},indent=2))
