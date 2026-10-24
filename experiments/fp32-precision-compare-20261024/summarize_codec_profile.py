"""Export phase accounting from one observed complete compressed update."""
import argparse,hashlib,json,statistics,subprocess,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--output',required=True);a=p.parse_args();r=Path(a.root)
d=json.loads((r/'compressed-scope-profile1/status.json').read_text());assert d['completed'] and d['full_parameter_steps']==1
s=json.loads((r/'compressed-scope-profile1/scope-step1.json').read_text())
steps=[v for v in d['logs'] if v['event']=='step_complete'];assert len(steps)==1
assert steps[0]['updated_parameters']==26895998464 and steps[0]['remaining_parameter_gradients']==0
b=next(v for v in d['logs'] if v['event']=='step_begin');f=next(v for v in d['logs'] if v['event']=='forward_done')
summary={}
for key,v in s['groups'].items():
 name=key.split('/',1)[1];g=summary.setdefault(name,{'calls':0,'raw_bytes':0,'wall_seconds':0.,'event_seconds':0.,'exclusive_wall_seconds':0.,'exclusive_event_seconds':0.})
 for k in g:g[k]+=v[k]
ref=json.loads((r/'fp32-npuadam-run1/status.json').read_text());previous={};transfers=[]
for row in ref['logs']:
 if row['event']!='step_complete':continue
 now=row['host_timing'];delta={k:v-previous.get(k,0) for k,v in now.items()};previous=now
 transfers.append({'step':row['step'],'total_seconds':row['seconds'],'cpu_to_npu_seconds':delta['device_copy_seconds'],'npu_to_cpu_seconds':delta['host_copy_seconds'],'npu_adam_seconds':delta['npu_adam_seconds']})
old=json.loads((r/'compressed-run1/parameter-probes.json').read_text());new=json.loads((r/'compressed-scope-profile1/parameter-probes.json').read_text())
def coordinates(rows):
 flat={}
 for v in rows:
  if v['step']!=1:continue
  for j,idx in enumerate(v['indices']):
   key=(v['name'],idx);assert key not in flat
   flat[key]={k:v[k][j] for k in ['weight_before','gradient','weight_after']}
 return flat
old=coordinates(old);new=coordinates(new)
assert old.keys()==new.keys();probe={'sampled_coordinates':len(new),'tensors':len({n for n,i in new}),'first_update_only':True}
for k in ['weight_before','gradient','weight_after']:
 probe[k+'_max_abs_delta']=max(abs(old[n][k]-new[n][k]) for n in old)
result={'scope':'One newly observed full compressed update of first original training example, not retrospectively decomposed old four-step timings. No numerical algorithm change.','step_total_seconds':steps[0]['seconds'],'forward_wall_seconds':f['elapsed']-b['elapsed'],'sampled_trajectory_observation_check':probe,'codec_and_moments':summary,'phase_groups':s['groups'],'timing_semantics':s['timing'],'reference_original_four_steps':transfers,'reference_warm_means':{k:statistics.mean(v[k] for v in transfers[1:]) for k in ['total_seconds','cpu_to_npu_seconds','npu_to_cpu_seconds','npu_adam_seconds']},'source_sha256':{n:hashlib.sha256((r/n).read_bytes()).hexdigest() for n in ['profile_codec_scopes.py','train_profile_codec.py','smoke_codec_scopes.py']},'notes':['Codec scopes use nested accounting; compression total includes its verification child. Do not add inclusive parent and child.','Event spans exclude work queued before their start, but include internal host-submission gaps; not pure kernel busy time.','Host wall and device event spans overlap. Do not add the two kinds of time.','Quantized moment scope includes FP32 Adam math, decode/requantization/packing and existing sample/progress audits. It is not solely entropy coding.','Reference transfer phase includes allocations, window management and waits; not a peak PCIe/DMA bandwidth measurement.']}
Path(a.output).write_text(json.dumps(result,indent=2))
print(json.dumps({'step_seconds':result['step_total_seconds'],'groups':summary,'trajectory':probe},indent=2))
