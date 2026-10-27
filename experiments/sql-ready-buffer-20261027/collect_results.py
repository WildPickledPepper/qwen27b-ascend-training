import argparse,json,hashlib,subprocess,time,bisect
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--reference',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();r=a.root
manifest=json.loads((r/'source-manifest.private.json').read_text())
for name,h in manifest.items():assert hashlib.sha256((r/name).read_bytes()).hexdigest()==h,name
job=json.loads((r/'job.private.json').read_text());assert job['state']=='complete'
new=r/'buffered4-8.private';old=a.reference/'cache4-8.private'
runs={};events={}
for name,folder in [('reference',old),('buffered',new)]:
 ranks=[json.loads((folder/f'rank{i}-result.json').read_text()) for i in [0,1]]
 events[name]=[[json.loads(x) for x in (folder/f'rank{i}.jsonl').read_text().splitlines()] for i in [0,1]]
 assert sum(x['parameters'] for x in ranks)==26895998464
 assert sum(x['state_bytes'] for x in ranks)==215167987712
 seconds=[max(x['steps'][i]['seconds'] for x in ranks) for i in range(3)]
 runs[name]=dict(rank_results=ranks,step_seconds=seconds,warm_seconds=sum(seconds[1:])/2,peak_allocated_gib=[max(s['allocated_peak'] for s in x['steps'])/2**30 for x in ranks],peak_reserved_gib=[max(s['reserved_peak'] for s in x['steps'])/2**30 for x in ranks])
checks=[];traces=[]
for rank in [0,1]:
 def ev(name,event):return next(x for x in events[name][rank] if x['event']==event)
 assert ev('reference','model_ready')['manifest']==ev('buffered','model_ready')['manifest']
 assert ev('reference','model_ready')['checkpoint_layers']==ev('buffered','model_ready')['checkpoint_layers']==24
 for k in ['training_hashes','training_lengths','validation_hashes']:assert ev('reference','data')[k]==ev('buffered','data')[k]
 b=runs['reference']['rank_results'][rank];c=runs['buffered']['rank_results'][rank]
 assert b['validation_before']==c['validation_before']
 norms=[]
 for i,(bs,cs) in enumerate(zip(b['steps'],c['steps'])):
  for k in ['input_tokens','padded_tokens','gradient_d2h_bytes','resident_add_calls','resident_add_bytes']:assert bs[k]==cs[k],(rank,i,k)
  norms.append(bs['grad_norm']==cs['grad_norm'])
  m=cs['buffer_metrics'];assert m['device_backlog_peak_bytes']<=m['device_backlog_budget_bytes']
  assert m['cpu_receiving_slot_bytes']==4*2**30
  path=new/f'host-trace-rank{rank}-step{i+1}.private.json';trace=json.loads(path.read_text())['traceEvents']
  ready=sorted(x['ts']+x['dur'] for x in trace if x['name']=='D2H/wait_completion_includes_dependencies')
  adds=sorted(x['ts'] for x in trace if x['name']=='CPU/BF16_gradient_add')
  assert len(adds)==len(ready)==m['cpu_add_count']
  hist={}
  for consumed,t in enumerate(adds):
   depth=bisect.bisect_right(ready,t)-consumed;assert depth>=1,depth
   hist[str(depth)]=hist.get(str(depth),0)+1
  waits=[x['dur']/1e6 for x in trace if x['name']=='CPU/wait_ready_data']
  traces.append(dict(rank=rank,step=i+1,sha256=hashlib.sha256(path.read_bytes()).hexdigest(),completed_chunks=len(ready),ready_queue_depth_before_add_histogram=hist,approximate_max_ready_chunks=max(map(int,hist)),ready_depth_note='Completion-event timestamps slightly precede actual ready-queue insertion; histogram is a timestamp-based upper estimate.',cpu_ready_get_total_wait_seconds=sum(waits),cpu_ready_get_wait_over_1ms_seconds=sum(x for x in waits if x>.001),cpu_ready_get_max_wait_seconds=max(waits,default=0)))
 checks.append(dict(rank=rank,grad_norm_exact_each_step=norms,validation_exact=b['validation_after_768']==c['validation_after_768']))
smi=subprocess.check_output(['npu-smi','info'],text=True);idle={str(i):f'No running processes found in NPU {i}' in smi for i in [3,4]};assert all(idle.values());(r/'post-smi.private.txt').write_text(smi)
report=dict(runs=runs,checks=checks,host_trace_summary=traces,tiny_exact=json.loads((r/'tiny-exact.private.json').read_text()),source_sha256=manifest,reporting_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),cleanup=dict(no_processes=idle,checked_unix=time.time()),elapsed_change_percent=100*(runs['buffered']['warm_seconds']/runs['reference']['warm_seconds']-1),limits=['Original16 slots already form a multi-buffer receiver; lack of buffering was an incorrect description.','Combined32 slots/private copy stream/512MiB backlog/ready-only worker experiment; no isolated attribution to buffer capacity.','CPU inter-add gaps include scheduler and Python overhead; event-monitor wait is not pure DMA time.','CPU busy ratio excludes initial fill; elapsed training time is primary outcome.','Two warm updates; no sustained performance or long-term quality claim.','Full model norms and short validation checked, not all full model states bitwise.'])
a.output.write_text(json.dumps(report,indent=2))
print(json.dumps(dict(elapsed_change_percent=report['elapsed_change_percent'],checks=checks,warm_seconds={k:v['warm_seconds'] for k,v in runs.items()},peaks={k:v['peak_allocated_gib'] for k,v in runs.items()})))
