"""Export aggregate device results without data, weights, logs or private paths."""
import argparse,hashlib,json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
report=dict(scope='Matched 3/4-card SQL full-parameter short runs; original 768 samples/padding/global256/micro16/48 checkpoints/4GiB BF16 cache/CPU FP32 Adam',runs={},loss_probe=json.loads((a.root/'loss-probe.private.json').read_text()),initialization=[json.loads((a.root/f'init-rank{rank}.private.json').read_text()) for rank in [0,1]],tiny_comparison=json.loads((a.root/'tiny-comparison.private.json').read_text()),source_sha256=json.loads((a.root/'source-manifest.private.json').read_text()))
inputs={}
for name in ['baseline','chunk512']:
 ranks=[json.loads((a.root/(name+'.private')/f'rank{rank}-result.json').read_text()) for rank in [0,1]]
 data=[]
 for rank in [0,1]:
  events=[json.loads(x) for x in (a.root/(name+'.private')/f'rank{rank}.jsonl').read_text().splitlines()]
  d=next(x for x in events if x['event']=='data');data.append({k:d[k] for k in ['training_hashes','training_lengths','validation_hashes']})
  manifest=next(x for x in events if x['event']=='model_ready')
  assert manifest['checkpoint_layers']==24
  assert next(x for x in events if x['event']=='resident_plan')['allocated_bytes']==4*2**30
  assert manifest['physical_card']==[3,4][rank]
 inputs[name]=data
 seconds=[max(r['steps'][s]['seconds'] for r in ranks) for s in range(3)]
 report['runs'][name]=dict(rank_results=ranks,step_seconds=seconds,warm_mean_seconds=sum(seconds[1:])/2,peak_allocated_gib=[max(s['allocated_peak'] for s in r['steps'])/2**30 for r in ranks],peak_reserved_gib=[max(s['reserved_peak'] for s in r['steps'])/2**30 for r in ranks])
assert inputs['baseline']==inputs['chunk512']
b,c=report['runs']['baseline'],report['runs']['chunk512']
report['same_data_and_order']=True
report['allocated_saving_gib_per_card']=[x-y for x,y in zip(b['peak_allocated_gib'],c['peak_allocated_gib'])]
report['elapsed_change_percent']=100*(c['warm_mean_seconds']/b['warm_mean_seconds']-1)
report['validation_loss_difference']=c['rank_results'][0]['validation_after_768']['loss']-b['rank_results'][0]['validation_after_768']['loss']
report['limits']=['Three updates only, first learning rate zero; not long-term convergence or sustained performance.','Fresh physical3/4 topology differs from historical2/3; only current matched runs used for performance ratio.','BF16 partial weight-gradient sums change rounding; no lossless claim.','allocated/reserved peaks are framework measurements; driver/communication runtime outside allocated may differ.']
a.output.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:v for k,v in report.items() if k not in ['runs','source_sha256','loss_probe','initialization','tiny_comparison']}))
