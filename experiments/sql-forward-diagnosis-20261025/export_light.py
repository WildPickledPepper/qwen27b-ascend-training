"""Verify identical data/state settings, export scoped data and released cards."""
import hashlib,json,re,subprocess
from pathlib import Path
r=Path(__file__).resolve().parent
assert json.loads((r/'light-job.json').read_text())['stage']=='complete'
reference=r.parent/'qwen-sql-cpu-fusion-20261025'/'cpu-full'
out=[];validation=[]
for rank in [0,1]:
 rows=[json.loads(x) for x in (r/'light-full'/f'rank{rank}.jsonl').read_text().splitlines()]
 old=[json.loads(x) for x in (reference/f'rank{rank}.jsonl').read_text().splitlines()]
 data=next(x for x in rows if x['event']=='data');previous=next(x for x in old if x['event']=='data')
 for key in ['training_hashes','training_lengths']:assert data[key]==previous[key][:768]
 assert data['validation_hashes']==previous['validation_hashes']
 expected=next(x['metrics'] for x in old if x['event']=='sample_checkpoint' and x['samples']==768)
 actual=next(x['metrics'] for x in rows if x['event']=='sample_checkpoint' and x['samples']==768)
 assert actual['assistant_tokens']==expected['assistant_tokens'] and actual['correct']==expected['correct']
 validation.append({'rank':rank,'actual':actual,'reference':expected,'exactly_equal':actual==expected,'loss_difference':actual['loss']-expected['loss']})
 result=json.loads((r/'light-full'/f'rank{rank}-result.json').read_text())
 prior=json.loads((reference/f'rank{rank}-result.json').read_text())
 assert result['completed'] and result['config']==prior['config'] and result['fp32_master'] and result['fp32_m_v']
 assert result['cpu_fusion']==prior['cpu_fusion']
 allowed=['forward_light','step_complete','sample_checkpoint']
 out.append({'rank':rank,'events':[{k:v for k,v in x.items() if k!='unix'} for x in rows if x['event'] in allowed]})
info=subprocess.check_output(['npu-smi','info'],text=True)
assert all(f'No running processes found in NPU {i}' in info for i in [2,3])
v={'completed':True,'same_768_data_order_lengths_and_validation_split':True,'same_validation_at_768':all(x['exactly_equal'] for x in validation),'validation_comparisons':validation,'same_config_and_fp32_state':True,'same_cpu_fusion_extension_and_plan':True,'devices_2_3_released':True,'other_known_service_present':'1424210' in info,'ranks':out,'source_sha256':{p.name:hashlib.sha256(p.read_bytes().replace(b'\r\n',b'\n')).hexdigest() for p in r.glob('*.py') if 'private' not in p.name},'limits':'Light counters perturb Python allocation/GC timing; cannot retroactively attribute older untraced stalls.'}
(r/'light-diagnosis.json').write_text(json.dumps(v,indent=2));print(json.dumps({k:v for k,v in v.items() if k not in ['ranks','source_sha256']}))
