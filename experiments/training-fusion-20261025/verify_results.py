"""Audit same-data comparisons, FP32 optimizer capacity and numerical gates."""
import json,math
from pathlib import Path
r=Path(__file__).resolve().parents[2]
new=json.loads((r/'evidence/2026-10-25/training-fusion-results.json').read_text())
old=json.loads((r/'evidence/2026-10-25/dual-throughput-results.json').read_text())
prior={c['name']:c for c in old['cases']};cases={c['name']:c for c in new['cases']}
assert new['supervisor']['stage']=='complete'
for case in cases.values():
 assert case['fp32_master'] and case['fp32_m_v'] and case['parameters']==26895998464
 assert math.isfinite(case['validation_after']['loss'])
 for step in case['steps']:assert step['fp32_m_v_bytes_total']==215167987712
 data=[e for e in case['events'] if e['event']=='data']
 assert len(data)==2 and data[0]['training_hashes']==data[1]['training_hashes']
 assert not set(data[0]['training_hashes'])&set(data[0]['validation_hashes'])
 baseline=prior['matched-g2-optimized' if case['global_batch']==2 else 'cpu24-throughput']
 previous=next(e for e in baseline['events'] if e['event']=='data')
 assert data[0]['training_hashes']==previous['training_hashes'] and data[0]['validation_hashes']==previous['validation_hashes']
 assert len(case['steps'])==len(baseline['steps'])
 if not case['fuse_gdn']:
  assert abs(case['validation_after']['loss']-baseline['validation_after']['loss'])<1e-5
  assert case['validation_after']['correct']==baseline['validation_after']['correct']
 else:
  assert abs(case['validation_after']['loss']-baseline['validation_after']['loss'])<.02
  assert abs(case['validation_after']['correct']-baseline['validation_after']['correct'])<=8
for name,key in [('identity-probe','identity_bit_differences'),('cast-probe','bf16_bit_differences'),('pipeline-probe','bf16_bit_differences')]:
 assert new['probes'][name]['completed'] and new['probes'][name][key]==0
assert not any(new['probes']['identity-probe']['after_three_updates_bit_differences'].values())
actual=new['probes']['gdn-actual-fp64-probe'];assert actual['completed']
for case in actual['cases']:
 for metric in [case['output'],*case['gradients'].values()]:
  assert metric['finite'] and metric['relative_l2']<.03 and .995<metric['cosine']<=1+1e-8
g2=cases['fusion-g2'];g32=min((c for c in cases.values() if c['global_batch']==32 and c['ranking_eligible']),key=lambda c:c['warm_mean_seconds'])
result={'completed':True,'successful_cases':len(cases),'same_data_and_update_counts':True,
 'master_and_m_v_fp32':True,'m_v_bytes_total':215167987712,'copy_and_identity_probes_bit_exact':True,
 'fused_gdn_fp64_metric_recheck_passed':True,
 'batch2_before_seconds':prior['matched-g2-optimized']['warm_mean_seconds'],'batch2_after_seconds':g2['warm_mean_seconds'],
 'batch2_throughput_speedup':prior['matched-g2-optimized']['warm_mean_seconds']/g2['warm_mean_seconds'],
 'batch32_before_seconds':prior['cpu24-throughput']['warm_mean_seconds'],'batch32_after_seconds':g32['warm_mean_seconds'],
 'batch32_throughput_speedup':prior['cpu24-throughput']['warm_mean_seconds']/g32['warm_mean_seconds'],
 'limits':['Numerical gates are short-run checks, not long-term convergence proof',
 'Fused GDN BF16 intermediate math is not bit-identical to FP32 decomposition']}
(r/'evidence/2026-10-25/training-fusion-verification.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
