"""Export aggregate device results without data, weights, logs or private paths."""
import argparse,hashlib,json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--candidate',choices=['chunk512','chunk512-contiguous'],default='chunk512');a=p.parse_args()
contiguous=a.candidate.endswith('-contiguous')
report=dict(scope='Matched 3/4-card SQL full-parameter short runs; original 768 samples/padding/global256/micro16/48 checkpoints/4GiB BF16 cache/CPU FP32 Adam',runs={},loss_probe=json.loads((a.root/('loss-contiguous-probe.private.json' if contiguous else 'loss-probe.private.json')).read_text()),initialization=[json.loads((a.root/f'init-rank{rank}.private.json').read_text()) for rank in [0,1]],tiny_comparison=json.loads((a.root/('tiny-contiguous-comparison.private.json' if contiguous else 'tiny-comparison.private.json')).read_text()),source_sha256=json.loads((a.root/('source-contiguous-manifest.private.json' if contiguous else 'source-manifest.private.json')).read_text()))
# The training/math files must still match the launch-time snapshot.
for name in [('train_chunked_contiguous.py' if contiguous else 'train_chunked.py'),('chunked_loss_contiguous.py' if contiguous else 'chunked_loss.py'),'resident_gradients.py','community_gdn.py']:
 assert hashlib.sha256((a.root/name).read_bytes()).hexdigest()==report['source_sha256'][name],name
report['reporting_source_sha256']={name:hashlib.sha256((a.root/name).read_bytes()).hexdigest() for name in ['collect_results.py','compare_tiny.py']}
inputs={}
for name in ['baseline',a.candidate]:
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
assert inputs['baseline']==inputs[a.candidate]
b,c=report['runs']['baseline'],report['runs'][a.candidate]
for name,run in report['runs'].items():
 assert sum(r['parameters'] for r in run['rank_results'])==26895998464
 assert sum(r['state_bytes'] for r in run['rank_results'])==26895998464*8
 assert all(r['loss_chunk_tokens']==(0 if name=='baseline' else 512) for r in run['rank_results'])
for rank in [0,1]:
 assert b['rank_results'][rank]['validation_before']==c['rank_results'][rank]['validation_before']
 for bs,cs in zip(b['rank_results'][rank]['steps'],c['rank_results'][rank]['steps']):
  for key in ['input_tokens','padded_tokens','gradient_d2h_bytes','resident_add_calls','resident_add_bytes']:
   assert bs[key]==cs[key],(rank,bs['step'],key)
report['unchanged_workload_state_and_transfer_counts']=True
report['candidate_name']=a.candidate
report['same_data_and_order']=True
report['allocated_saving_gib_per_card']=[x-y for x,y in zip(b['peak_allocated_gib'],c['peak_allocated_gib'])]
report['elapsed_change_percent']=100*(c['warm_mean_seconds']/b['warm_mean_seconds']-1)
report['validation_loss_difference']=c['rank_results'][0]['validation_after_768']['loss']-b['rank_results'][0]['validation_after_768']['loss']
report['limits']=['Three updates only, first learning rate zero; not long-term convergence or sustained performance.','Fresh physical3/4 topology differs from historical2/3; only current matched runs used for performance ratio.','BF16 partial weight-gradient sums change rounding; no lossless claim.','allocated/reserved peaks are framework measurements; driver/communication runtime outside allocated may differ.']
a.output.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:v for k,v in report.items() if k not in ['runs','source_sha256','loss_probe','initialization','tiny_comparison']}))
