"""Export prospective differences after verifying sealed CPU/meta outputs."""
import argparse,hashlib,json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
seal=json.loads((a.root/'prediction-seal.private.json').read_text())
assert hashlib.sha256((a.root/'inputs.json').read_bytes()).hexdigest()==seal['inputs_sha256']
raw={}
for name,job in seal['jobs'].items():
    assert job['state']=='complete'
    file=a.root/(name+'.prediction.private.json')
    assert hashlib.sha256(file.read_bytes()).hexdigest()==job['prediction_sha256']
    row=json.loads(file.read_text())
    assert not row['device_initialized'] and not row['weights_loaded'] and not row['calibration_read']
    assert row['counts']['parameter_gradient_hooks']==row['parameter_tensors']*16
    raw[name]=row
baseline=raw['baseline'];rows=[]
for name,row in raw.items():
    summary={k:v for k,v in row.items() if k!='peak_blocks'}
    summary.update(case=name,saved_bytes=baseline['peak_bytes']-row['peak_bytes'],saved_gib=(baseline['peak_bytes']-row['peak_bytes'])/2**30,saved_percent=100*(baseline['peak_bytes']-row['peak_bytes'])/baseline['peak_bytes'],prediction_sha256=seal['jobs'][name]['prediction_sha256'])
    summary['largest_runtime_blocks_at_peak']=[{k:b[k] for k in ['shape','stride','dtype','bytes','allocated','category','phase','op']} for b in sorted((b for b in row['peak_blocks'] if not b['persistent']),key=lambda b:-b['bytes'])[:8]]
    rows.append(summary)
report=dict(scope='Prospective framework tensor memory only, physical card3/rank1; not a trained/deployed NPU candidate.',micro_batch=16,global_batch=256,resident_gib=4,checkpoint_layers=48,input_sequences=json.loads((a.root/'inputs.json').read_text())['step1'],vocab_size=248320,hidden_size=5120,rows=rows,cpu_numerical_tests=json.loads((a.root/'tests.private.json').read_text()),limits=['Baseline and chunked use identical allocator granularity rules inherited from historical trace, not a new physical allocator model.','No chunked-loss NPU measurement, full-model numeric validation, timing or CANN workspace estimate.','Chunked backward recomputes logits/CE, retains full BF16 output-weight gradient, accumulates BF16 partial gradients and can change rounding.','Card2 unchanged; per-card savings cannot be applied to card2 or the maximum across both cards.','No forward attention/context splitting, no lower precision or optimizer state changes.'])
a.output.write_text(json.dumps(report,indent=2)+'\n')
for r in rows:print(r['case'],r['peak_gib'],r['saved_gib'],r['peak_context'])
