"""CPU BF16 accumulation with two bounded pinned incoming buffers.

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
 slots=[torch.empty(maximum,dtype=torch.bfloat16,pin_memory=True) for _ in range(2)]
 pending=[None,None];counter=[0];stats=collections.defaultdict(float)
 pool=concurrent.futures.ThreadPoolExecutor(max_workers=1,thread_name_prefix='gradient-cpu-add')
 def add_after_copy(event,target,incoming):
  torch.npu.set_device(device)
  start=time.monotonic();event.synchronize();stats['worker_copy_wait_seconds']+=time.monotonic()-start
  start=time.monotonic();target.add_(incoming);stats['cpu_add_seconds']+=time.monotonic()-start
 def accumulate(self,target,part):
  slot=counter[0]%2;counter[0]+=1
  if pending[slot] is not None:
   start=time.monotonic();pending[slot].result();stats['slot_wait_seconds']+=time.monotonic()-start
  incoming=slots[slot][:part.numel()]
  incoming.copy_(part,non_blocking=True)
  event=torch.npu.Event();event.record(torch.npu.current_stream())
  future=pool.submit(add_after_copy,event,target,incoming);pending[slot]=future
  stats['cpu_add_calls']+=1;stats['incoming_bytes']+=part.numel()*part.element_size()
  if self.is_gradient_accumulation_boundary():
   start=time.monotonic();future.result();stats['boundary_wait_seconds']+=time.monotonic()-start
   return target.to(part.device,non_blocking=True)
  return target
 optimizer._accumulate_cpu_resident=types.MethodType(accumulate,optimizer)
 original=unwrap(optimizer.partition_grads);source=textwrap.dedent(inspect.getsource(original));source=source[source.index('def partition_grads('):]
 lines=source.splitlines()
 indices=[i for i,line in enumerate(lines) if line.strip()=='grad_buffer = grad_buffer.to(grad_partition.device, non_blocking=True)']
 assert len(indices)==1
 i=indices[0];indent=lines[i][:-len(lines[i].lstrip())]
 lines[i:i+1]=[indent+'if self.is_gradient_accumulation_boundary():',indent+'    '+lines[i].strip()]
 starts=[i for i,line in enumerate(lines) if line.strip()=='cuda_grad_buffer = grad_buffer.to(grad_partition.device, non_blocking=True)']
 ends=[i for i,line in enumerate(lines) if line.strip()=='grad_buffer = cuda_grad_buffer']
 assert len(starts)==len(ends)==1 and ends[0]>starts[0]
 begin,end=starts[0],ends[0];indent=lines[begin][:-len(lines[begin].lstrip())]
 assert any('cuda_grad_buffer.add_' in line for line in lines[begin:end])
 assert any('grad_buffer.copy_(cuda_grad_buffer' in line for line in lines[begin:end])
 lines[begin:end+1]=[indent+'grad_buffer = self._accumulate_cpu_resident(grad_buffer, grad_partition.to(self.gradient_accumulation_dtype).view(grad_buffer.shape))']
 source='\n'.join(lines)+'\n'
 # Compile inside the original class name so Python mangles private attributes
 # exactly as in the installed native class; do not replace the shared class.
 wrapped_source='class DeepSpeedZeroOptimizer_Stage3:\n'+textwrap.indent(source,'    ')
 scope={};exec(compile(wrapped_source,'<bounded-cpu-gradient-accumulation>','exec'),original.__globals__,scope)
 optimizer.partition_grads=types.MethodType(scope['DeepSpeedZeroOptimizer_Stage3'].partition_grads,optimizer)
 def flush():
  for job in pending:
   if job is not None:job.result()
 optimizer._cpu_gradient_flush=flush
 plan={'accumulation_dtype':'torch.bfloat16','pinned_staging_bytes':2*maximum*2,'slots':2,
       'native_partition_source_sha256':hashlib.sha256(inspect.getsource(original).encode()).hexdigest(),
       'patched_partition_source_sha256':hashlib.sha256(source.encode()).hexdigest(),
       'microbatch_addition_order_preserved':True,'cpu_fp32_master_m_v_unchanged':True}
 return plan,stats
