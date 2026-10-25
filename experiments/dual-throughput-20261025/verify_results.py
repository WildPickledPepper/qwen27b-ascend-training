"""Audit saved aggregate results and the historical matched-batch comparison."""
import json,math
from pathlib import Path
repo=Path(__file__).resolve().parents[2]
data=json.loads((repo/'evidence/2026-10-25/dual-throughput-results.json').read_text())
old=json.loads((repo/'evidence/2026-10-25/zero3-offload-results.json').read_text())
cases={c['name']:c for c in data['cases']}
reference=None
for case in cases.values():
 assert case['parameters']==26895998464 and case['fp32_master'] and case['fp32_m_v']
 assert math.isfinite(case['validation_after']['loss'])
 for step in case['steps']:
  assert step['fp32_m_v_bytes_total']==case['parameters']*8
  assert step['seconds']>0 and step['padded_input_tokens']>=step['useful_input_tokens']
 events=[e for e in case['events'] if e['event']=='data']
 assert len(events)==2
 assert events[0]['training_hashes']==events[1]['training_hashes']
 assert not set(events[0]['training_hashes'])&set(events[0]['validation_hashes'])
 if case['global_batch']==32:
  assert len(events[0]['training_hashes'])==96 and len(case['steps'])==3
  hashes=events[0]['training_hashes']
  if reference is None:reference=hashes
  assert hashes==reference
 else:
  assert case['global_batch']==2 and len(case['steps'])==4
  assert events[0]['training_hashes']==old['dual']['training_hashes']
  assert events[0]['validation_hashes']==old['dual']['validation_hashes']
matched=cases['matched-g2-optimized']
result={'completed':True,'successful_cases_checked':len(cases),
 'same_batch32_data_and_update_count':True,'batch2_matches_historical_data_and_updates':True,
 'fp32_m_v_bytes_all_updates':26895998464*8,
 'matched_batch2_optimized_seconds':matched['warm_mean_seconds'],
 'matched_batch2_speedup_vs_previous_dual':old['dual']['steps2_4_mean_seconds']/matched['warm_mean_seconds'],
 'matched_batch2_speedup_vs_previous_single':old['single']['steps2_4_mean_seconds']/matched['warm_mean_seconds'],
 'matched_validation_loss_delta':matched['validation_after']['loss']-old['dual']['validation_after']['loss'],
 'batch32_speedup_micro4_to_best':cases['mb4-g32-v2']['warm_mean_seconds']/cases['cpu24-throughput']['warm_mean_seconds']}
(repo/'evidence/2026-10-25/dual-throughput-verification.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
