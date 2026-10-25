"""Audit completed runtime config, source hashes and independent benchmark examples."""
import hashlib,json
from pathlib import Path
r=Path(__file__).resolve().parent
j=json.loads((r/'matched-job.json').read_text());assert j['stage']=='complete'
for name,digest in j['source_sha256'].items():assert hashlib.sha256((r/name).read_bytes()).hexdigest()==digest,name
data=json.loads((r/'data'/'data-summary.json').read_text())
assert len(data['training_hashes'])==len(set(data['training_hashes']))==768
assert not set(data['training_hashes'])&set(data['validation_hashes'])
for rank in [0,1]:
 x=json.loads((r/'train-g256-m8'/f'rank{rank}-result.json').read_text())
 assert x['completed'] and x['parameters']==26895998464
 assert x['global_batch']==256 and x['micro_batch']==8 and x['max_sequence_length']==512
 cfg=x['config'];assert cfg['train_batch_size']==256 and cfg['gradient_accumulation_steps']==16
 assert cfg['gradient_clipping']==1.0 and cfg['bf16']['enabled']
 assert cfg['optimizer']['params']=={'lr':2e-5,'betas':[.9,.999],'eps':1e-8,'weight_decay':.01}
 assert cfg['zero_optimization']['offload_optimizer']['device']=='cpu'
 assert x['fp32_master'] and x['fp32_m_v'] and x['checkpoint']=='on'
 events=[json.loads(line) for line in (r/'train-g256-m8'/f'rank{rank}.jsonl').read_text().splitlines()]
 e=next(v for v in events if v['event']=='data')
 assert e['training_hashes']==data['training_hashes'] and not e['repeats']
 lrs=[v['learning_rate'] for v in events if v['event']=='learning_rate']
 assert len(lrs)==3 and lrs[0]==0 and lrs[1]>0 and lrs[2]>lrs[1]
 assert len(x['steps'])==3 and all(s['fp32_m_v_bytes']==26895998464*4 for s in x['steps'])
result={'passed':True,'source_hashes_match':True,'distinct_training_examples':768,'validation_disjoint':True,
 'global_batch':256,'max_length':512,'nonzero_learning_rate_updates':2,'fp32_m_v_total_bytes_per_update':215167987712,
 'author_optimizer_scalar_settings_matched':True,'author_preprocessing_exactly_reproduced':False,
 'parameter_groups_exactly_reproduced':False,'full_epoch_completed':False}
(r/'verification.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
print(json.dumps(result))
