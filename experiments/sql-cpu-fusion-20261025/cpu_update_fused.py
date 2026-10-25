"""Use original CPUAdam state bookkeeping with a fused per-group native update."""
import collections,hashlib,importlib.util,inspect,json,textwrap,time,types
from pathlib import Path
import torch
from cpu_update_reference import verified_source

def install(optimizer,skip_identity=False,pipeline=False):
 assert pipeline and optimizer.offload_optimizer and not optimizer.swap_optimizer
 assert not optimizer.use_muon and not optimizer.zenflow and not optimizer.torch_autocast_gradscaler
 assert all(optimizer.subgroup_to_device[i]=='cpu' for i in range(len(optimizer.fp16_groups)))
 cpu=optimizer.optimizer;assert cpu.adam_w_mode and cpu.fp32_optimizer_states
 r=Path(__file__).resolve().parent;binary=r/'build.private/cpu_adam_fused.so'
 assert json.loads((r/'cpu-probe.json').read_text())['passed']
 spec=importlib.util.spec_from_file_location('cpu_adam_fused',str(binary))
 fused=importlib.util.module_from_spec(spec);spec.loader.exec_module(fused)
 base=cpu.ds_opt_adam;ident=cpu.opt_id;group=cpu.param_groups[0]
 fused.create_adam(ident,group['lr'],*group['betas'],group['eps'],group['weight_decay'],True,False)
 stats=collections.defaultdict(float);context={};scales={};optimizer._fused_cpu_scale_by_group=scales
 maximum=max(p.numel() for p in optimizer.fp32_partitioned_groups_flat)
 buffers=[torch.empty(maximum,dtype=torch.bfloat16,pin_memory=True) for _ in range(2)]
 assert all(x.is_pinned() for x in buffers)
 events=[None,None];counter=[0]
 def update(*args):
  assert context and args[0]==ident and args[8].numel()==context['out'].numel()
  with torch.profiler.record_function('optimizer/CPUAdam'):
   result=fused.adam_update_fused(*args,context['scale'],context['out'])
  context['calls']+=1;stats['fused_scale_adam_bf16_calls']+=1
  return result
 def destroy(which):
  fused.destroy_adam(which);base.destroy_adam(which)
 cpu.ds_opt_adam=types.SimpleNamespace(adam_update=update,destroy_adam=destroy)
 original_unscale=optimizer.unscale_and_clip_grads
 original_source=verified_source(original_unscale)
 fn=getattr(original_unscale,'__func__',original_unscale)
 for _ in range(8):
  if fn.__module__!='deepspeed.utils.nvtx':break
  fn=inspect.getclosurevars(fn).nonlocals['func']
 source=textwrap.dedent(inspect.getsource(fn));source=source[source.index('def unscale_and_clip_grads('):]
 call='self.fp32_partitioned_groups_flat[sub_group_id].grad.mul_(1. / combined_scale)'
 assert source.count(call)==1
 source=source.replace(call,'self._fused_cpu_scale_by_group[sub_group_id] = float(1. / combined_scale)')
 scope={};exec(compile(source,'<same-native-clip-scalar>','exec'),fn.__globals__,scope)
 optimizer.unscale_and_clip_grads=types.MethodType(scope['unscale_and_clip_grads'],optimizer)
 original_step=optimizer._optimizer_step
 def step(self,index):
  assert not context
  slot=counter[0]%2;counter[0]+=1
  if events[slot] is not None:
   start=time.monotonic();events[slot].synchronize();stats['staging_reuse_wait_seconds']+=time.monotonic()-start
  master=self.fp32_partitioned_groups_flat[index]
  assert master.dtype==torch.float32 and master.grad.dtype==torch.float32
  context.update(group=index,slot=slot,scale=scales.pop(index),out=buffers[slot][:master.numel()],calls=0)
  result=original_step(index)
  assert context['calls']==1
  return result
 optimizer._optimizer_step=types.MethodType(step,optimizer)
 def reassign(self,index):
  assert context['group']==index
  target=self.fp16_partitioned_groups_flat[index].detach();assert target.dtype==torch.bfloat16
  assert target.numel()==context['out'].numel()
  start=time.monotonic();target.copy_(context['out'],non_blocking=True)
  event=torch.npu.Event();event.record(torch.npu.current_stream());events[context['slot']]=event
  stats['async_copy_submit_seconds']+=time.monotonic()-start
  stats['async_copy_bytes']+=target.numel()*2;stats['bf16_output_fused_calls']+=1
  self._unflatten_partitioned_parameters(index);context.clear()
 optimizer._reassign_or_swap_out_partitioned_parameters=types.MethodType(reassign,optimizer)
 original_post=optimizer._post_step
 def post(self,*args,**kwargs):
  assert not context and not scales
  start=time.monotonic()
  for event in events:
   if event is not None:event.synchronize()
  stats['staging_final_wait_seconds']+=time.monotonic()-start
  return original_post(*args,**kwargs)
 optimizer._post_step=types.MethodType(post,optimizer)
 for name,label in [('_optimizer_step','cpu_adam_step'),('_reassign_or_swap_out_partitioned_parameters','master_to_compute_copy'),('unscale_and_clip_grads','gradient_unscale'),('_prepare_sub_group','prepare_group'),('_release_sub_group','release_group')]:
  method=getattr(optimizer,name)
  def timed(self,*args,_fn=method,_label=label,**kwargs):
   start=time.monotonic()
   try:return _fn(*args,**kwargs)
   finally:stats[_label+'_seconds']+=time.monotonic()-start;stats[_label+'_calls']+=1
  setattr(optimizer,name,types.MethodType(timed,optimizer))
 optimizer._cpu_fusion_plan={'native_adam_simd_math_reused':True,'master_m_v_dtype':'torch.float32','bf16_compute_output_fused':True,
  'native_clip_scalar_source_sha256':hashlib.sha256(original_source.encode()).hexdigest(),
  'extension_sha256':hashlib.sha256(binary.read_bytes()).hexdigest(),
  'scaled_gradient_not_written_back':'Used once by optimizer; next active boundary overwrites FP32 gradients. Validated only for the fully active tested model.'}
 return stats
