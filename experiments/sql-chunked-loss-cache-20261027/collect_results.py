"""Compare 4/8GiB against the completed contiguous chunked-loss 4/4GiB run."""
import argparse,hashlib,json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--reference',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
manifest=json.loads((a.root/'source-manifest.private.json').read_text())
for name in ['train_cache.py','resident_gradients.py','chunked_loss_contiguous.py','community_gdn.py']:
 assert hashlib.sha256((a.root/name).read_bytes()).hexdigest()==manifest[name],name
reference_manifest=json.loads((a.reference/'source-contiguous-manifest.private.json').read_text())
for name in ['chunked_loss_contiguous.py','community_gdn.py']:assert manifest[name]==reference_manifest[name],name
runs={};events={}
for name,folder in [('reference',a.reference/'chunk512-contiguous.private'),('cache4-8',a.root/'cache4-8.private')]:
 ranks=[json.loads((folder/f'rank{rank}-result.json').read_text()) for rank in [0,1]]
 ev=[[json.loads(x) for x in (folder/f'rank{rank}.jsonl').read_text().splitlines()] for rank in [0,1]]
 events[name]=ev
 assert sum(r['parameters'] for r in ranks)==26895998464
 assert sum(r['state_bytes'] for r in ranks)==26895998464*8
 assert all(r['loss_chunk_tokens']==512 for r in ranks)
 for rank in [0,1]:assert next(x for x in ev[rank] if x['event']=='model_ready')['checkpoint_layers']==24
 seconds=[max(r['steps'][step]['seconds'] for r in ranks) for step in range(3)]
 plans=[next(x for x in ev[rank] if x['event']=='resident_plan') for rank in [0,1]]
 runs[name]=dict(rank_results=ranks,step_seconds=seconds,warm_mean_seconds=sum(seconds[1:])/2,plans=[{k:x[k] for k in ['budget_bytes','allocated_bytes','selected_parameters','total_parameters','dtype','order','flush']} for x in plans],peak_allocated_gib=[max(s['allocated_peak'] for s in r['steps'])/2**30 for r in ranks],peak_reserved_gib=[max(s['reserved_peak'] for s in r['steps'])/2**30 for r in ranks])
b,c=runs['reference'],runs['cache4-8'];checks=[]
for rank in [0,1]:
 bd=next(x for x in events['reference'][rank] if x['event']=='data');cd=next(x for x in events['cache4-8'][rank] if x['event']=='data')
 for key in ['training_hashes','training_lengths','validation_hashes']:assert bd[key]==cd[key]
 assert b['rank_results'][rank]['validation_before']==c['rank_results'][rank]['validation_before']
 assert next(x for x in events['reference'][rank] if x['event']=='model_ready')['manifest']==next(x for x in events['cache4-8'][rank] if x['event']=='model_ready')['manifest']
 expected_saved=(c['plans'][rank]['allocated_bytes']-b['plans'][rank]['allocated_bytes'])*15
 assert c['plans'][rank]['allocated_bytes']==[4,8][rank]*2**30
 norm_exact=[]
 for bs,cs in zip(b['rank_results'][rank]['steps'],c['rank_results'][rank]['steps']):
  for key in ['input_tokens','padded_tokens']:assert bs[key]==cs[key]
  assert bs['gradient_d2h_bytes']-cs['gradient_d2h_bytes']==expected_saved
  assert cs['resident_add_calls']==c['plans'][rank]['selected_parameters']*16
  norm_exact.append(bs['grad_norm']==cs['grad_norm'])
 checks.append(dict(rank=rank,physical_card=[3,4][rank],d2h_saved_bytes_per_update=expected_saved,d2h_saved_gib_per_update=expected_saved/2**30,grad_norm_exact_each_step=norm_exact,validation_exact=b['rank_results'][rank]['validation_after_768']==c['rank_results'][rank]['validation_after_768']))
report=dict(scope='Reuse just-completed 4/4GiB contiguous-loss reference; new physical3/4 4/8GiB run, same768 SQL samples/global256/micro16/16micro/48checkpoint/BF16/CPU FP32 master-m-v',runs=runs,checks=checks,tiny_exact=json.loads((a.root/'tiny-exact.private.json').read_text()),source_sha256=manifest,reporting_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),elapsed_change_percent=100*(c['warm_mean_seconds']/b['warm_mean_seconds']-1),limits=['Three updates only, first learning rate zero; no long-term quality or sustained performance claim.','Tiny states compared completely bitwise; full model only norms/validation/workload verified, not all tensors.','Worker CPU-add/slot wait/update durations can overlap; do not sum them as a critical-path partition.','Framework allocated/reserved peaks exclude some runtime/driver allocations.','Only physical4/rank1 cache expanded; physical3/rank0 unchanged.'])
a.output.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:v for k,v in report.items() if k not in ['runs','tiny_exact','source_sha256']}))
