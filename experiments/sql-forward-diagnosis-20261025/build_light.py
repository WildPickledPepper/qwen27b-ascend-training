"""Preserve original forward path; add allocator/GC counters without profiler."""
from pathlib import Path
r=Path(__file__).resolve().parent;b=r.parent/'sql-cpu-fusion-20261025'
s=(b/'train_fused.py').read_text().replace('import argparse,datetime,','import gc,collections\nimport argparse,datetime,')
# Keep --profile's original validation-after2 behavior, but do not start a trace.
s=s.replace('if a.profile and rank==0:', 'if False and a.profile and rank==0:')
s=s.replace("'profiled':bool(a.profile and step==a.steps-1)", "'profiled':False")
anchor='for step in range(a.steps):'
setup="""gc_time=collections.defaultdict(float);gc_count=collections.defaultdict(int);gc_starts={};gc_long=[]
def observe_gc(phase,info):
 generation=info['generation']
 if phase=='start':gc_starts[generation]=time.monotonic()
 else:
  elapsed=time.monotonic()-gc_starts.pop(generation)
  gc_time[generation]+=elapsed;gc_count[generation]+=1
  if elapsed>.02 and len(gc_long)<20:gc_long.append({'generation':generation,'seconds':elapsed})
gc.callbacks.append(observe_gc)
def allocator_counters():
 return {k:v for k,v in torch.npu.memory_stats().items() if k.startswith('num_') or 'retry' in k or 'oom' in k}
"""
assert s.count(anchor)==1;s=s.replace(anchor,setup+anchor)
anchor='  phase_started=time.monotonic()\n  if a.tiny:'
assert s.count(anchor)==1;s=s.replace(anchor,"""  allocator_before=allocator_counters();gc_before=dict(gc_time)
  before_memory={'allocated':torch.npu.memory_allocated(),'reserved':torch.npu.memory_reserved()}
  phase_started=time.monotonic();part_started=phase_started;forward_parts={}
  if a.tiny:""")
s=s.replace("   batch=make_batch(group);padded_tokens+=batch['input_ids'].numel()\n   loss=engine(**batch,use_cache=False).loss", """   batch=make_batch(group);padded_tokens+=batch['input_ids'].numel()
   forward_parts['batch_seconds']=time.monotonic()-part_started;part_started=time.monotonic()
   loss=engine(**batch,use_cache=False).loss
   forward_parts['engine_forward_seconds']=time.monotonic()-part_started;part_started=time.monotonic()""")
anchor="  phase['forward_wall']+=time.monotonic()-phase_started"
assert s.count(anchor)==1;s=s.replace(anchor,"""  forward_parts['loss_scale_and_read_seconds']=time.monotonic()-part_started
  forward_seconds=time.monotonic()-phase_started;phase['forward_wall']+=forward_seconds
  allocator_after=allocator_counters()
  log('forward_light',step=step+1,micro=micro+1,seconds=forward_seconds,parts=forward_parts,before_memory=before_memory,after_memory={'allocated':torch.npu.memory_allocated(),'reserved':torch.npu.memory_reserved()},allocator_delta={k:v-allocator_before.get(k,0) for k,v in allocator_after.items()},gc_seconds_by_generation={str(k):v-gc_before.get(k,0) for k,v in gc_time.items()},long_gc_seen=list(gc_long))""")
(r/'train_light.py').write_text(s,encoding='utf-8',newline='\n')
s=(r/'run_diagnosis.py').read_text().replace('job.json','light-job.json').replace("'run.private.log'","'light.private.log'").replace('train_diagnosis.py','train_light.py').replace("'cpu-full'","'light-full'")
s=s.replace("r=Path(__file__).resolve().parent;os.chdir(r);assert", "r=Path(__file__).resolve().parent;os.chdir(r);assert json.loads((r/'job.json').read_text())['stage']=='complete'\nassert")
(r/'run_light.py').write_text(s,encoding='utf-8',newline='\n')
print('Prepared original path with allocator retries and generation-specific GC counters')
