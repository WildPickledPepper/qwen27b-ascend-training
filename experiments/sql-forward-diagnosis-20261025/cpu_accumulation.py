"""CPU BF16 accumulation with a bounded tiered pinned receive queue.

Keep every microbatch addition and the native boundary norm/FP32 offload path.
The replacement is guarded against the inspected installed Stage3 source.
"""
import collections,concurrent.futures,hashlib,inspect,textwrap,time,types
import torch

def unwrap(fn):
 fn=getattr(fn,'__func__',fn)
 for _ in range(8):
  if fn.__module__!='deepspeed.utils.nvtx':break
  fn=inspect.getclosurevars(fn).nonlocals['func']
 return fn

def install(optimizer):
 assert optimizer.offload_optimizer and not optimizer.swap_optimizer
 assert optimizer.gradient_accumulation_dtype==torch.bfloat16
 mapping=optimizer._DeepSpeedZeroOptimizer_Stage3__param_id_to_grad_partition
 assert all(v.device.type=='cpu' and v.is_pinned() and v.dtype==torch.bfloat16 for v in mapping.values())
 maximum=max(v.numel() for v in mapping.values());device=torch.npu.current_device()
 capacities=[min(maximum,64*2**20)]*30+[maximum]*2
 slots=[torch.empty(n,dtype=torch.bfloat16,pin_memory=True) for n in capacities]
 pending=[None]*len(slots);issued=[-1]*len(slots);counter=[0];stats=collections.defaultdict(float)
 pool=concurrent.futures.ThreadPoolExecutor(max_workers=1,thread_name_prefix='gradient-cpu-add')
 def add_after_copy(event,target,incoming):
  torch.npu.set_device(device)
  start=time.monotonic();event.synchronize();stats['worker_copy_wait_seconds']+=time.monotonic()-start
  start=time.monotonic();target.add_(incoming);stats['cpu_add_seconds']+=time.monotonic()-start
 def accumulate(self,target,part):
  eligible=[i for i,n in enumerate(capacities) if n>=part.numel()]
  minimum=min(capacities[i] for i in eligible)
  eligible=[i for i in eligible if capacities[i]==minimum]
  ready=[i for i in eligible if pending[i] is None or pending[i].done()]
  slot=ready[0] if ready else min(eligible,key=lambda i:issued[i])
  issued[slot]=counter[0];counter[0]+=1
  if pending[slot] is not None:
   start=time.monotonic();pending[slot].result();stats['slot_wait_seconds']+=time.monotonic()-start
  incoming=slots[slot][:part.numel()]
  incoming.copy_(part,non_blocking=True)
  event=torch.npu.Event();event.record(torch.npu.current_stream())
  future=pool.submit(add_after_copy,event,target,incoming);pending[slot]=future
  stats['cpu_add_calls']+=1;stats['incoming_bytes']+=part.numel()*part.element_size()
  return target
 optimizer._accumulate_cpu_resident=types.MethodType(accumulate,optimizer)
 original=unwrap(optimizer.partition_grads);source=textwrap.dedent(inspect.getsource(original));source=source[source.index('def partition_grads('):]
 lines=source.splitlines()
 indices=[i for i,line in enumerate(lines) if line.strip()=='grad_buffer = grad_buffer.to(grad_partition.device, non_blocking=True)']
 assert len(indices)==1
 i=indices[0];indent=lines[i][:-len(lines[i].lstrip())]
 lines[i:i+1]=[indent+'# Keep the CPU accumulation buffer; native norm/offload is replayed after backward.']
 starts=[i for i,line in enumerate(lines) if line.strip()=='cuda_grad_buffer = grad_buffer.to(grad_partition.device, non_blocking=True)']
 ends=[i for i,line in enumerate(lines) if line.strip()=='grad_buffer = cuda_grad_buffer']
 assert len(starts)==len(ends)==1 and ends[0]>starts[0]
 begin,end=starts[0],ends[0];indent=lines[begin][:-len(lines[begin].lstrip())]
 assert any('cuda_grad_buffer.add_' in line for line in lines[begin:end])
 assert any('grad_buffer.copy_(cuda_grad_buffer' in line for line in lines[begin:end])
 lines[begin:end+1]=[indent+'grad_buffer = self._accumulate_cpu_resident(grad_buffer, grad_partition.to(self.gradient_accumulation_dtype).view(grad_buffer.shape))']
 source='\n'.join(lines)+'\n'
 # Replace only the boundary call, preserving all native per-parameter
 # bookkeeping, gradient release and reduction behavior.
 import re
 boundary_pattern=r'self\._offload_grad_partition_at_boundary\(param, grad_buffer, offload_fp32_gradients,\s+offload_fp32_offsets\)'
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
 # Compile inside the original class name so Python mangles private attributes
 # exactly as in the installed native class; do not replace the shared class.
 wrapped_source='class DeepSpeedZeroOptimizer_Stage3:\n'+textwrap.indent(source,'    ')
 scope={};exec(compile(wrapped_source,'<bounded-cpu-gradient-accumulation>','exec'),original.__globals__,scope)
 optimizer.partition_grads=types.MethodType(scope['DeepSpeedZeroOptimizer_Stage3'].partition_grads,optimizer)
 def flush():
  for job in pending:
   if job is not None:job.result()
 optimizer._cpu_gradient_flush=flush
 plan={'accumulation_dtype':'torch.bfloat16','pinned_staging_bytes':sum(capacities)*2,'slots':len(slots),'small_slot_elements':capacities[0],
       'native_partition_source_sha256':hashlib.sha256(inspect.getsource(original).encode()).hexdigest(),
       'patched_partition_source_sha256':hashlib.sha256(source.encode()).hexdigest(),
       'boundary_policy':'native norm/offload after reduce epilogue, original parameter-ready order','microbatch_addition_order_preserved':True,'cpu_fp32_master_m_v_unchanged':True}
 return plan,stats
