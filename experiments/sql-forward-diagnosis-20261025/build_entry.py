"""Instrument the anomalous third update without changing training mathematics."""
from pathlib import Path
r=Path(__file__).resolve().parent;base=r.parent/'sql-cpu-fusion-20261025'
for name in ['cpu_accumulation.py','community_gdn.py','bounded_grad_copy.py','zero3-sql.json','gradient_scopes.py','gradient_cache.py','cpu_update_reference.py','cpu_update_fused.py','analyze_profile.py','analyze_gradient.py','communication_overlap.py','analyze_boundary.py']:
 (r/name).write_text((base/name).read_text(),encoding='utf-8',newline='\n')
s=(base/'train_fused.py').read_text()
s=s.replace('import argparse,datetime,', 'import cProfile,pstats,resource\nimport argparse,datetime,')
start="  phase_started=time.monotonic()\n  if a.tiny:"
assert s.count(start)==1
s=s.replace(start,"""  phase_started=time.monotonic()
  before_memory={'allocated':torch.npu.memory_allocated(),'reserved':torch.npu.memory_reserved()}
  usage_before=resource.getrusage(resource.RUSAGE_SELF)
  cpu_profile=cProfile.Profile() if step==2 and micro==0 else None
  if cpu_profile:cpu_profile.enable()
  forward_parts={};part_started=time.monotonic()
  if a.tiny:""")
s=s.replace("   batch=make_batch(group);padded_tokens+=batch['input_ids'].numel()\n   loss=engine(**batch,use_cache=False).loss", """   with torch.profiler.record_function(f'diagnosis/batch/{step+1}/{micro+1}'):
    batch=make_batch(group);padded_tokens+=batch['input_ids'].numel()
   forward_parts['batch_seconds']=time.monotonic()-part_started;part_started=time.monotonic()
   with torch.profiler.record_function(f'diagnosis/engine_forward/{step+1}/{micro+1}'):
    loss=engine(**batch,use_cache=False).loss
   forward_parts['engine_forward_seconds']=time.monotonic()-part_started;part_started=time.monotonic()""")
anchor="  phase['forward_wall']+=time.monotonic()-phase_started"
assert s.count(anchor)==1
s=s.replace(anchor,"""  forward_parts['loss_scale_and_read_seconds']=time.monotonic()-part_started
  phase['forward_wall']+=time.monotonic()-phase_started
  if cpu_profile:
   cpu_profile.disable();stats=pstats.Stats(cpu_profile)
   entries=[]
   for (filename,line,function),(primitive,calls,self_seconds,total_seconds,callers) in stats.stats.items():
    entries.append({'file':Path(filename).name,'line':line,'function':function,'calls':calls,'self_seconds':self_seconds,'cumulative_seconds':total_seconds})
   log('forward_cpu_profile',step=step+1,micro=micro+1,top_cumulative=sorted(entries,key=lambda x:x['cumulative_seconds'],reverse=True)[:40],top_self=sorted(entries,key=lambda x:x['self_seconds'],reverse=True)[:40])
  usage_after=resource.getrusage(resource.RUSAGE_SELF)
  log('forward_parts',step=step+1,micro=micro+1,parts=forward_parts,before_memory=before_memory,after_memory={'allocated':torch.npu.memory_allocated(),'reserved':torch.npu.memory_reserved()},host_usage_delta={k:getattr(usage_after,k)-getattr(usage_before,k) for k in ['ru_utime','ru_stime','ru_minflt','ru_majflt','ru_nvcsw','ru_nivcsw']})""")
(r/'train_diagnosis.py').write_text(s,encoding='utf-8',newline='\n')
print('Prepared scoped forward timing, memory and first-third-micro CPU call profiling')
