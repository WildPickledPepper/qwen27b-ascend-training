"""Keep lightweight, path-free evidence and compare with the previous fixed-batch run."""
import json
from pathlib import Path
r=Path(__file__).resolve().parent;root=r.parents[1];out=root/'evidence/2026-10-25'
ranks=[json.loads((r/f'rank{i}-result.private.json').read_text()) for i in [0,1]]
prior=json.loads((out/'sql-batch-tuning-results.json').read_text())['runs'][0]
assert all(x['completed'] and x['global_batch']==256 and x['micro_batch']==16 for x in ranks)
assert all(x['fp32_master'] and x['fp32_m_v'] for x in ranks)
assert ranks[0]['validation_after']['correct']==204
cases=[]
for i,x in enumerate(ranks):
 cases.append({'rank':i,'parameters':x['parameters'],'steps':x['steps'],'validation_before':x['validation_before'],'validation_after':x['validation_after']})
v={'completed':True,'global_batch':256,'micro_batch_per_rank':16,'accumulation':8,'cases':cases,
 'original_unprofiled_seconds_per_256':107.9709251700,
 'same_validation_loss_as_prior_g256':abs(ranks[0]['validation_after']['loss']-prior['validation_checkpoints'][-1]['metrics']['loss'])<1e-7,
 'scope':'Full 64-layer text model; three updates, last rank0 update profiled; original checkpoint read-only; no checkpoint saved',
 'transfer_interpretation':'Actual tensor copy submissions inside gradient partition/offload paths, source/destination byte counts; host API seconds are not DMA durations.',
 'transfers_per_rank_per_update':{'h2d_bytes':215167987712,'d2h_bytes':268959984640,'total_bytes':484127972352},
 'limits':['Short validation does not prove long-run training quality.','Instrumentation run does not replace the prior unprofiled throughput ranking.']}
(out/'sql-bottleneck-observation.json').write_text(json.dumps(v,indent=2))
print(json.dumps({'final_validation':ranks[0]['validation_after'],'steps':[[s['seconds'] for s in x['steps']] for x in ranks]}))
