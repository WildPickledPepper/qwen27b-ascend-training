"""Export matched four-update original/large-window FP32 results."""
import argparse,json,statistics,hashlib
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--output',required=True);a=p.parse_args();r=Path(a.root)
runs={};summary={}
for label,name in [('baseline','fp32-npuadam-run1'),('optimized','fp32-largewindow-numa-run1')]:
 d=json.loads((r/name/'status.json').read_text());assert d['completed'] and d['full_parameter_steps']==4 and not d.get('error')
 rows=[];previous={}
 for v in d['logs']:
  if v['event']!='step_complete':continue
  assert v['updated_parameters']==26895998464 and v['remaining_parameter_gradients']==0
  delta={k:x-previous.get(k,0) for k,x in v['host_timing'].items()};previous=v['host_timing']
  rows.append({'step':v['step'],'seconds':v['seconds'],'transfer_h2d_seconds':delta['device_copy_seconds'],'transfer_d2h_seconds':delta['host_copy_seconds'],'explicit_adam_seconds':delta['npu_adam_seconds']})
 mem=[m for e in d['logs'] for m in e.get('memory',[])]
 summary[label]={'steps':rows,'warm_means':{k:statistics.mean(v[k] for v in rows[1:]) for k in ['seconds','transfer_h2d_seconds','transfer_d2h_seconds','explicit_adam_seconds']},'validation_before':d['validation_before'],'validation_after':d['validation_after'],'peak_allocated_gib':[max(m['peak_allocated'] for m in mem if m['logical_device']==i)/2**30 for i in [0,1]],'host_peak_rss_gib':max(v['host_max_rss_kib'] for v in d['logs'] if v['event']=='step_complete')/2**20,'cpu_state_offload':d['cpu_state_offload'],'optimizer_window_coordinates':d.get('optimizer_window_coordinates',1048576),'host_affinity':d.get('host_affinity','not recorded in original run')}
 runs[label]=d
assert runs['baseline']['data']==runs['optimized']['data']
assert abs(runs['baseline']['validation_before']['loss']-runs['optimized']['validation_before']['loss'])<1e-5
result={'scope':'Same initial model, complete four-step FP32 text training, same independent 8-example validation; old completed baseline reused, not rerun.','changes':['Per-state offload window4MiB to64MiB; original FP32 Adam formulas and pageable copy path unchanged','Process CPU affinity96-99 near physical2/3 NUMA4; actual anonymous memory also spilled to node5, not claimed fully local'],'summary':summary,'warm_speedup':summary['baseline']['warm_means']['seconds']/summary['optimized']['warm_means']['seconds'],'sampled_trajectory_analysis':json.loads((r/'largewindow-analysis.json').read_text()),'probes':{n:json.loads((r/n).read_text()) for n in ['staged-offload-probe.json','staged-window-probe.json']},'source_sha256':{n:hashlib.sha256((r/n).read_bytes()).hexdigest() for n in ['bench_staged_offload.py','probe_staged_offload.py','probe_staged_windows.py','train_fp32_large_window.py','bench_resident_npu.py','train_compare_npu.py','summarize_largewindow.py']},'notes':['Small probe gives exact full weight/m/v bit comparison across3 updates including partial tail windows. Whole-model samples are52928coordinates per step, not exhaustive full-model state equality.','NUMA affinity and larger window are applied together in full model; controlled same-process2x2 microprobe isolates window and staging factors. No full-model factorial claim.','Historical baseline and new run share server with other workloads; warm-step individual timings retained. No inference about absolute maximum throughput.','Transfer region includes allocations/synchronization; explicit Adam region is not whole backward.','Staged two-slot pinned prototype only microtested, not adopted in full-model candidate. No BF16/quantization change.']}
Path(a.output).write_text(json.dumps(result,indent=2));print(json.dumps({'speedup':result['warm_speedup'],'groups':{k:{'warm':v['warm_means'],'loss':v['validation_after']['loss'],'token_accuracy':v['validation_after']['assistant_token_accuracy']} for k,v in summary.items()}},indent=2))
