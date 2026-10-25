"""Audit a completed annotated capture against the unchanged selected setup."""
import hashlib,json,subprocess,sqlite3
from pathlib import Path
r=Path(__file__).resolve().parent
assert json.loads((r/'capture-job.json').read_text())['stage']=='complete'
reference=r.parent/'qwen-sql-boundary-overlap-20261025'/'cpu-full';ranks=[]
for rank in [0,1]:
 data=[json.loads(x) for x in (r/'capture-full'/f'rank{rank}.jsonl').read_text().splitlines()]
 old=[json.loads(x) for x in (reference/f'rank{rank}.jsonl').read_text().splitlines()]
 current_input=next(x for x in data if x['event']=='data');prior_input=next(x for x in old if x['event']=='data')
 for key in ['training_hashes','training_lengths','validation_hashes']:assert current_input[key]==prior_input[key]
 result=json.loads((r/'capture-full'/f'rank{rank}-result.json').read_text());prior=json.loads((reference/f'rank{rank}-result.json').read_text())
 db=next((r/'capture-full/profile').rglob(f'ascend_pytorch_profiler_{rank}.db'))
 infofile=next((r/'capture-full/profile').rglob(f'profiler_info_{rank}.json'))
 metadata=json.loads(infofile.read_text());assert metadata.get('end_info'),list(metadata)
 connection=sqlite3.connect(f'file:{db}?mode=ro',uri=True)
 window=connection.execute('SELECT (MAX(endNs)-MIN(startNs))/1e9 FROM TASK').fetchone()[0]
 wall=result['steps'][-1]['seconds'];assert abs(window-wall)<max(1,wall*.1)
 assert result['completed'] and len(result['steps'])==4 and result['config']==prior['config']
 assert result['fp32_master'] and result['fp32_m_v']
 for row in result['steps']:
  assert row['fp32_m_v_bytes']==107583993856
  assert row['cpu_gradient_accumulation_stats']['deferred_boundary_parameters']==851
  assert sum(row['gradient_observer_stats'].get(k,0) for k in ['h2d_source_bytes','d2h_source_bytes'])==295855983104
 actual=next(x['metrics'] for x in data if x['event']=='sample_checkpoint' and x['samples']==768)
 expected=next(x['metrics'] for x in old if x['event']=='sample_checkpoint' and x['samples']==768)
 assert actual['correct']==expected['correct'] and actual['assistant_tokens']==expected['assistant_tokens']
 manifest=next(x['parameters'] for x in data if x['event']=='parameter_manifest');assert len(manifest)==851
 ranks.append({'rank':rank,'validation_at_768':actual,'validation_exact':actual==expected,'loss_difference':actual['loss']-expected['loss'],
  'profile_normal_stop_metadata':True,'db_export_exists':True,'profile_window_seconds':window,'profile_step_seconds':wall,
  'dependency_counts':next(x['counts'] for x in data if x['event']=='dependency_counts'),
  'event_method_sha256':next(x['source_sha256'] for x in data if x['event']=='dependency_source'),
  'steps':result['steps'],'parameter_manifest':manifest})
assert ranks[0]['parameter_manifest']==ranks[1]['parameter_manifest']
info=subprocess.check_output(['npu-smi','info'],text=True)
assert all(f'No running processes found in NPU {i}' in info for i in [2,3])
v={'completed':True,'same_inputs_order_lengths_and_validation_split':True,'same_selected_configuration_fp32_state_and_logical_transfer_bytes':True,'same_parameter_manifest_both_ranks':True,
 'both_devices_released':True,'other_known_service_present':'1424210' in info,'ranks':ranks,
 'source_sha256':{p.name:hashlib.sha256(p.read_bytes().replace(b'\r\n',b'\n')).hexdigest() for p in r.glob('*.py') if 'private' not in p.name},
 'limits':'Additional event labels and two profiled ranks perturb timings. No speed ranking or long-term numeric equivalence claim.'}
(r/'capture-verification.json').write_text(json.dumps(v,indent=2));print(json.dumps({k:x for k,x in v.items() if k not in ['ranks','source_sha256']}))
