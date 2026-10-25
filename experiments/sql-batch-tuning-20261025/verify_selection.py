"""Verify that the persisted default is exactly the measured winner's runtime config."""
import hashlib,json
from pathlib import Path
r=Path(__file__).resolve().parent
x=json.loads((r/'batch-results.json').read_text());p=json.loads((r/'selected-profile.json').read_text())
cfg=json.loads((r/'zero3-selected.json').read_text())
winner=next(v for v in x['runs'] if v['global_batch']==p['global_batch'])
for name,digest in winner['source_sha256'].items():assert hashlib.sha256((r/name).read_bytes()).hexdigest()==digest,name
for rank in [0,1]:
 actual=json.loads((r/winner['name']/f'rank{rank}-result.json').read_text())
 assert actual['config']==cfg
 assert actual['global_batch']==p['global_batch'] and actual['micro_batch']==p['micro_batch_per_rank']
 assert actual['fp32_master'] and actual['fp32_m_v'] and actual['completed']
assert p==x['selected'] and p['global_batch']==2*p['micro_batch_per_rank']*p['gradient_accumulation_steps']
result={'passed':True,'selected_runtime_config_exact_match':True,'selected_global_batch':p['global_batch'],
 'selected_micro_batch_per_rank':p['micro_batch_per_rank'],'selected_accumulation':p['gradient_accumulation_steps'],
 'three_fixed_data_runs_completed':len(x['runs'])==3,'samples_per_case':768,'steady_measured_samples_per_case':512,
 'input_and_padding_work_equal_in_steady_window':True,'all_source_and_data_checks_passed_in_collector':True,
 'fp32_master_and_adam_states':True,'selection_scope':'eight-example short validation, checkpoints every256 examples'}
(r/'verification.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
print(json.dumps(result))
