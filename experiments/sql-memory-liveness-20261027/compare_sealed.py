"""Read observations only after predictions have been completed and sealed."""
import argparse,hashlib,json
from pathlib import Path

p=argparse.ArgumentParser()
p.add_argument('--predictions',type=Path,required=True)
p.add_argument('--observations',type=Path,required=True)
p.add_argument('--breakdown',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args()
seal=json.loads((a.predictions/'prediction-seal.json').read_text())
assert all(j['state']=='complete' for j in seal['jobs'].values())
inputs=json.loads((a.predictions/'inputs.json').read_text())
assert hashlib.sha256((a.predictions/'inputs.json').read_bytes()).hexdigest()==seal['inputs_sha256']
# Every prediction is verified BEFORE opening any observation file.
pred={}
for key,j in seal['jobs'].items():
    case,rank=key.rsplit('-',1)
    file=a.predictions/f'{case}-rank{rank}.prediction.private.json'
    assert hashlib.sha256(file.read_bytes()).hexdigest()==j['prediction_sha256']
    row=json.loads(file.read_text());assert row['sequences']==inputs[case]
    assert not row['device_initialized'] and not row['calibration_read'] and not row['weights_loaded']
    assert row['counts']['parameter_gradient_hooks']==(425 if row['rank']==0 else 426)*16
    assert row['counts']['resident_adds']==29*16
    pred[key]=row
observed=json.loads(a.observations.read_text(encoding='utf-8-sig'))
breakdown=json.loads(a.breakdown.read_text(encoding='utf-8-sig'))
comparisons=[]
for key,row in pred.items():
    case,rank=key.rsplit('-',1);rank=int(rank);step=int(case.removeprefix('step'))
    actual=observed['rank_results'][rank]['steps'][step-1]['allocated_peak']
    delta=row['peak_bytes']-actual
    comp=dict(case=case,rank=rank,physical_card=rank+2,
              predicted_bytes=row['peak_bytes'],observed_bytes=actual,
              predicted_gib=row['peak_bytes']/2**30,observed_gib=actual/2**30,
              signed_error_mib=delta/2**20,signed_error_percent=100*delta/actual,
              absolute_error_percent=100*abs(delta)/actual,
              peak_context=row['peak_context'],categories=row['categories'],
              parameter_gradient_hooks=row['counts']['parameter_gradient_hooks'],
              resident_adds=row['counts']['resident_adds'],
              storage_allocations=row['allocations'],alias_results=row['alias_results'],
              skipped_kernel_launches=row['skipped_kernel_launches'],
              prediction_sha256=seal['jobs'][key]['prediction_sha256'])
    if step==1:
        pg={r['category']:r['bytes'] for r in row['categories']}
        og={r['category']:r['bytes'] for r in breakdown['ranks'][rank]['categories']}
        comparisons_by_category=[]
        mapping=[('weight',['weight'],['weight']),('resident_gradient',['resident_gradient'],['resident_gradient']),
                 ('ffn_forward',['ffn_forward_intermediates'],['ffn_forward_intermediates']),
                 ('rmsnorm_forward',['rmsnorm_forward_intermediates'],['rmsnorm_forward_intermediates']),
                 ('attention_forward',['attention_forward_intermediates'],['attention_forward_intermediates']),
                 ('checkpoint_recompute',['checkpoint_recompute_intermediates'],['checkpoint_recompute_intermediates']),
                 ('loss_forward',['loss_forward_intermediates'],['loss_forward_intermediates']),
                 ('backward_temporary',['backward_temporary'],['matmul_backward_temporary_gradients','parameter_gradient_accumulation_temporary','activation_gradient_merge','loss_temporary_gradients']),
                 ('other_boundary_inputs',['hidden_states_and_boundaries','inputs_and_pipeline','other_explicit_tensor','model_buffer'],['hidden_states_checkpoint_boundaries_and_pipeline','inputs_positions_masks_small_control','model_buffer'])]
        for name,pk,ok in mapping:
            pv=sum(pg.get(k,0) for k in pk);ov=sum(og.get(k,0) for k in ok)
            comparisons_by_category.append(dict(category=name,predicted_bytes=pv,observed_bytes=ov,signed_error_mib=(pv-ov)/2**20))
        comp['composition_at_respective_peaks']=comparisons_by_category
    comparisons.append(comp)
report=dict(method='Actual meta/autograd/checkpoint storage lifetimes; predictions sealed before comparison',
            predictions_read_observations=False,calibration_coefficients_used=False,
            cases=comparisons,maximum_absolute_error_percent=max(c['absolute_error_percent'] for c in comparisons),
            source_hashes={k:r['source_hashes'] for k,r in pred.items()},
            limits=['Accuracy is against historical NPU framework allocated peaks, not complete physical HBM.',
                    'Category comparison uses each implementation\'s own peak; it is not a per-address event alignment.',
                    'Small total error does not prove cancellation-free per-category accuracy.',
                    'CANN/internal compiler workspace, runtime/reserve and arbitrary multi-stream timing remain uncovered.',
                    'Native NPU backward layouts/copies can differ from meta. Any discrepancies are retained rather than fitted.'])
a.output.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:v for k,v in report.items() if k not in ['cases','source_hashes']},indent=2))
for c in comparisons:print(json.dumps({k:v for k,v in c.items() if k not in ['categories','composition_at_respective_peaks']}))
