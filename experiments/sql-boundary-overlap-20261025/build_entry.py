"""Defer boundary norm/offload until all backward launches have completed."""
from pathlib import Path
r=Path(__file__).resolve().parent;b=r.parent/'sql-reduce-schedule-20261025'
for name in ['community_gdn.py','cpu_update.py','bounded_grad_copy.py','zero3-sql.json','gradient_cache.py','compare_probes.py','export_evidence.py','status.py']:
 (r/name).write_text((b/name).read_text(encoding='utf-8'),encoding='utf-8',newline='\n')
s=(b/'cpu_accumulation.py').read_text(encoding='utf-8')
old="""  if self.is_gradient_accumulation_boundary():
   start=time.monotonic();future.result();stats['boundary_wait_seconds']+=time.monotonic()-start
   return target.to(part.device,non_blocking=True)
"""
assert s.count(old)==1;s=s.replace(old,'')
old="lines[i:i+1]=[indent+'if self.is_gradient_accumulation_boundary():',indent+'    '+lines[i].strip()]"
assert s.count(old)==1;s=s.replace(old,"lines[i:i+1]=[indent+'# Keep the CPU accumulation buffer; native norm/offload is replayed after backward.']")
anchor=" # Compile inside the original class name"
assert s.count(anchor)==1
code=''' # Replace only the boundary call, preserving all native per-parameter
 # bookkeeping, gradient release and reduction behavior.
 import re
 boundary_pattern=r'self\\._offload_grad_partition_at_boundary\\(param, grad_buffer, offload_fp32_gradients,\\s+offload_fp32_offsets\\)'
 source,count=re.subn(boundary_pattern,'self._defer_boundary_param(param)',source)
 assert count==1,'Installed boundary call contract differs'
 assert 'self.reduce_and_partition_stream.synchronize()' in inspect.getsource(unwrap(optimizer.independent_gradient_partition_epilogue))
 deferred=[];seen=set()
 def defer(self,param):
  assert param.ds_id not in seen,'Multiple boundary contributions require a different contract'
  seen.add(param.ds_id);deferred.append(param)
 optimizer._defer_boundary_param=types.MethodType(defer,optimizer)
 def finish(self):
  if not deferred:return
  start=time.monotonic()
  # All D2H producers are complete after the native reduce-stream epilogue.
  for job in pending:
   if job is not None:job.result()
  stats['deferred_tail_wait_seconds']+=time.monotonic()-start
  offload_gradients={};offload_offsets={}
  start=time.monotonic()
  with torch.no_grad(),torch.npu.stream(self.reduce_and_partition_stream):
   for param in deferred:
    target=mapping[param.ds_id]
    # Identical native norm/cast/offload operations, original reduction-ready order.
    buffer=target.to(f'npu:{device}',non_blocking=True)
    self._offload_grad_partition_at_boundary(param,buffer,offload_gradients,offload_offsets)
    stats['deferred_boundary_parameters']+=1
   self._swap_out_offload_fp32_gradients(offload_gradients,offload_offsets)
  self.reduce_and_partition_stream.synchronize()
  stats['deferred_native_finalize_seconds']+=time.monotonic()-start
  deferred.clear();seen.clear()
 optimizer._finish_deferred_boundary=types.MethodType(finish,optimizer)
 epilogue=optimizer.independent_gradient_partition_epilogue
 def boundary_epilogue(self,*args,**kwargs):
  result=epilogue(*args,**kwargs)
  self._finish_deferred_boundary()
  return result
 optimizer.independent_gradient_partition_epilogue=types.MethodType(boundary_epilogue,optimizer)
'''
s=s.replace(anchor,code+anchor)
s=s.replace("'microbatch_addition_order_preserved':True", "'boundary_policy':'native norm/offload after reduce epilogue, original parameter-ready order','microbatch_addition_order_preserved':True")
(r/'cpu_accumulation.py').write_text(s,encoding='utf-8',newline='\n')
s=(b/'gradient_scopes.py').read_text(encoding='utf-8')
s=s.replace("methods={'partition_grads':'partition_gradients',", "methods={'_finish_deferred_boundary':'deferred_boundary','partition_grads':'partition_gradients',")
s=s.replace("['partition_gradients','boundary_gradient_offload']", "['partition_gradients','boundary_gradient_offload','deferred_boundary']")
(r/'gradient_scopes.py').write_text(s,encoding='utf-8',newline='\n')
(r/'train_boundary.py').write_text((b/'train_reduce.py').read_text(encoding='utf-8'),encoding='utf-8',newline='\n')
s=(b/'run_experiment.py').read_text(encoding='utf-8').replace('train_reduce.py','train_boundary.py')
begin=s.index(" state['stage']='waiting-for-prior-prefetch-run'")
end=s.index(" for name,cache",begin)
s=s[:begin]+s[end:]
(r/'run_experiment.py').write_text(s,encoding='utf-8',newline='\n')
if (b/'launch.private.sh').exists():
 s=(b/'launch.private.sh').read_text(encoding='utf-8').replace('qwen-sql-reduce-schedule-20261025','qwen-sql-boundary-overlap-20261025')
 (r/'launch.private.sh').write_text(s,encoding='utf-8',newline='\n')
print('Generated deferred native boundary replay; reduction/event limit and all batch settings unchanged')
