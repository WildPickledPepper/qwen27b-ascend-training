"""Process-local identity-scaling removal and optimizer phase instrumentation."""
import collections,inspect,time,types

def install(optimizer,skip_identity=False,pipeline=False):
 stats=collections.defaultdict(float)
 original=optimizer.unscale_and_clip_grads
 source=inspect.getsource(original)
 assert 'grad.mul_(1. / combined_scale)' in source
 def unscale(self,group,norm):
  if skip_identity and self.clip_grad==0 and isinstance(self.loss_scale,(float,int)) and self.loss_scale==1:
   stats['identity_scale_skipped_calls']+=1
   return
  return original(group,norm)
 optimizer.unscale_and_clip_grads=types.MethodType(unscale,optimizer)
 if pipeline:
  import torch
  from deepspeed.accelerator import get_accelerator
  assert optimizer.offload_optimizer and not optimizer.swap_optimizer
  assert all(p is not None and p.dtype==torch.bfloat16 and p.device.type=='npu' for p in optimizer.fp16_partitioned_groups_flat)
  maximum=max(p.numel() for p in optimizer.fp32_partitioned_groups_flat)
  buffers=[get_accelerator().pin_memory(torch.empty(maximum,dtype=torch.bfloat16,device='cpu')) for _ in range(2)]
  assert all(b.is_pinned() for b in buffers)
  events=[None,None];counter=[0]
  def reassign(self,group):
   slot=counter[0]%2;counter[0]+=1
   if events[slot] is not None:
    start=time.monotonic();events[slot].synchronize();stats['staging_reuse_wait_seconds']+=time.monotonic()-start
   source=self.fp32_partitioned_groups_flat[group].detach()
   target=self.fp16_partitioned_groups_flat[group].detach()
   assert source.device.type=='cpu' and source.dtype==torch.float32 and source.numel()<=maximum
   staging=buffers[slot][:source.numel()]
   start=time.monotonic();staging.copy_(source);stats['cpu_bf16_cast_seconds']+=time.monotonic()-start
   start=time.monotonic();target.copy_(staging,non_blocking=True)
   event=torch.npu.Event();event.record(torch.npu.current_stream());events[slot]=event
   stats['async_copy_submit_seconds']+=time.monotonic()-start
   stats['async_copy_bytes']+=source.numel()*2
   self._unflatten_partitioned_parameters(group)
  optimizer._reassign_or_swap_out_partitioned_parameters=types.MethodType(reassign,optimizer)
  post=optimizer._post_step
  def post_step(self,*args,**kwargs):
   start=time.monotonic()
   for event in events:
    if event is not None:event.synchronize()
   stats['staging_final_wait_seconds']+=time.monotonic()-start
   return post(*args,**kwargs)
  optimizer._post_step=types.MethodType(post_step,optimizer)
 for name,label in [('_optimizer_step','cpu_adam_step'),('_reassign_or_swap_out_partitioned_parameters','master_to_compute_copy'),
                    ('unscale_and_clip_grads','gradient_unscale'),('_prepare_sub_group','prepare_group'),('_release_sub_group','release_group')]:
  method=getattr(optimizer,name)
  def timed(self,*args,_fn=method,_label=label,**kwargs):
   started=time.monotonic()
   try:return _fn(*args,**kwargs)
   finally:stats[_label+'_seconds']+=time.monotonic()-started;stats[_label+'_calls']+=1
  setattr(optimizer,name,types.MethodType(timed,optimizer))
 return stats
