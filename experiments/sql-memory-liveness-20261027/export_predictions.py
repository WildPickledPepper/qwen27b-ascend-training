"""Export sealed summaries and tests; raw per-storage traces stay private."""
import argparse,json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
seal=json.loads((a.root/'prediction-seal.json').read_text());rows=[]
for key,j in seal['jobs'].items():
    case,rank=key.rsplit('-',1)
    raw=json.loads((a.root/f'{case}-rank{rank}.prediction.private.json').read_text())
    summary={k:v for k,v in raw.items() if k!='peak_blocks'}
    summary.update(case=case,prediction_sha256=j['prediction_sha256'])
    rows.append(summary)
result=dict(predictions=rows,inputs=json.loads((a.root/'inputs.json').read_text()),
            seal_inputs_sha256=seal['inputs_sha256'],
            runtime=json.loads((a.root/'runtime-provenance.json').read_text()),
            upstream_source_sha256=json.loads((a.root/'upstream-source-hashes.json').read_text()))
a.output.write_text(json.dumps(result,indent=2)+'\n')
