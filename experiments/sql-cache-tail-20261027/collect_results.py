"""Host-only tail breakdown, no inference of pure device DMA time."""
import json,hashlib,argparse,subprocess,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
r=a.root
manifest=json.loads((r/'source.private.json').read_text())
for name,h in manifest.items():assert hashlib.sha256((r/name).read_bytes()).hexdigest()==h,name
job=json.loads((r/'job.private.json').read_text());assert job['state']=='complete'
runs={};events={}
for name in ['cache4-4','cache4-8']:
 folder=r/(name+'.private');ranks=[json.loads((folder/f'rank{i}-result.json').read_text()) for i in [0,1]]
 ev=[[json.loads(line) for line in (folder/f'rank{i}.jsonl').read_text().splitlines()] for i in [0,1]];events[name]=ev
 assert sum(x['parameters'] for x in ranks)==26895998464
 for i,x in enumerate(ranks):
  assert len(x['steps'])==3
  for step in x['steps']:
   assert abs(sum(step['tail_seconds'].values())-step['cpu_update_return_seconds'])<.5
   assert sum(c['elements'] for c in step['tail_chunk_times'])==x['parameters']
 warm=[]
 for i,x in enumerate(ranks):
  ss=x['steps'][1:];keys=list(ss[0]['tail_seconds'])
  warm.append(dict(rank=i,physical_card=[3,4][i],tail_mean={k:sum(s['tail_seconds'][k] for s in ss)/2 for k in keys},update_tail_mean=sum(s['cpu_update_return_seconds'] for s in ss)/2,fb_drain_mean=sum(s['forward_backward_and_drain_seconds'] for s in ss)/2,norm_mean=sum(s['global_norm_seconds'] for s in ss)/2))
 runs[name]=dict(rank_results=ranks,warm_summary=warm,step_seconds=[max(x['steps'][i]['seconds'] for x in ranks) for i in range(3)],warm_seconds=sum(max(x['steps'][i]['seconds'] for x in ranks) for i in [1,2])/2)
checks=[]
for rank in [0,1]:
 def find(name,event):return next(x for x in events[name][rank] if x['event']==event)
 assert find('cache4-4','model_ready')['manifest']==find('cache4-8','model_ready')['manifest']
 for k in ['training_hashes','training_lengths','validation_hashes']:assert find('cache4-4','data')[k]==find('cache4-8','data')[k]
 b=runs['cache4-4']['rank_results'][rank];c=runs['cache4-8']['rank_results'][rank]
 assert b['validation_before']==c['validation_before']
 for bs,cs in zip(b['steps'],c['steps']):
  for k in ['input_tokens','padded_tokens','grad_norm']:assert bs[k]==cs[k],(rank,k)
  assert bs['gradient_d2h_bytes']-cs['gradient_d2h_bytes']==([0,60][rank]*2**30)
 checks.append(dict(rank=rank,validation_exact=b['validation_after_768']==c['validation_after_768'],all_norms_exact=True,per_update_d2h_reduction_gib=[0,60][rank]))
smi=subprocess.check_output(['npu-smi','info'],text=True)
idle={str(i):f'No running processes found in NPU {i}' in smi for i in [3,4]};assert all(idle.values())
(r/'post-smi.private.txt').write_text(smi)
report=dict(runs=runs,checks=checks,source_sha256=manifest,cleanup=dict(physical_card_no_processes=idle,checked_unix=time.time()),limits=['Host wall timers, not device-only DMA time; H2D call may include blocking.','Last barrier is separately timed; CPUAdam/return may overlap with device work.','Only two warm steps per configuration; historical extra3.87s cannot be causally attributed if not reproduced.','Full model norms/short validation checked, not every state tensor.'])
a.output.write_text(json.dumps(report,indent=2))
for n,x in runs.items():print(json.dumps(dict(name=n,warm_seconds=x['warm_seconds'],warm_summary=x['warm_summary'])))
