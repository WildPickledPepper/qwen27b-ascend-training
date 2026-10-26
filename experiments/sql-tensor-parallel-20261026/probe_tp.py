"""Actual two-NPU AutoTP hybrid-Qwen forward/backward and gradient check.

CPU FP32 Adam via native ZeRO-1 offload. This is a compatibility probe, not a
27B throughput result. Full reference gradients retained only for the tiny model.
"""
import argparse,copy,ctypes,datetime,hashlib,importlib.util,json,math,os,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--output',required=True)
p.add_argument('--checkpoint',choices=['off','all'],default='off')
p.add_argument('--zero-stage',type=int,choices=[1,2],default=1);a=p.parse_args()
rank=int(os.environ['LOCAL_RANK']);assert os.environ['ASCEND_RT_VISIBLE_DEVICES']=='2,3'
os.sched_setaffinity(0,list(range(96+rank*24,120+rank*24)))
import torch,torch_npu,deepspeed
import torch.distributed as dist
torch.npu.set_device(rank);torch.set_num_threads(24)
torch.npu.config.allow_internal_format=False;torch.npu.matmul.allow_hf32=False
device=f'npu:{rank}'
deepspeed.init_distributed(dist_backend='hccl',timeout=datetime.timedelta(seconds=300))
out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
def log(event,**kw):
    row=dict(event=event,rank=rank,**kw)
    with (out/f'rank{rank}.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
    print(json.dumps(row),flush=True)
spec=importlib.util.spec_from_file_location('cpu_adam',os.environ['CPU_ADAM_EXTENSION'])
ext=importlib.util.module_from_spec(spec);spec.loader.exec_module(ext)
from deepspeed.ops.adam.cpu_adam import CPUAdamBuilder
CPUAdamBuilder.load=lambda self,verbose=True:ext
from transformers import AutoConfig
from transformers.models.qwen3_5 import modeling_qwen3_5 as qm
from community_gdn import chunk_gated_delta_rule
qm.chunk_gated_delta_rule=chunk_gated_delta_rule
cfg=AutoConfig.from_pretrained(a.model,local_files_only=True).text_config
cfg.hidden_size=128;cfg.intermediate_size=256;cfg.vocab_size=256
cfg.num_hidden_layers=4;cfg.num_attention_heads=4;cfg.num_key_value_heads=2;cfg.head_dim=32
cfg.linear_num_key_heads=2;cfg.linear_num_value_heads=4;cfg.linear_key_head_dim=32;cfg.linear_value_head_dim=32
cfg.layer_types=['linear_attention']*3+['full_attention'];cfg.tie_word_embeddings=False
cfg.use_cache=False;cfg._attn_implementation='eager'
def build():
    model=qm.Qwen3_5ForCausalLM(copy.deepcopy(cfg)).bfloat16()
    model.model.rotary_emb=qm.Qwen3_5TextRotaryEmbedding(config=model.config,device=device)
    with torch.no_grad():
        for name,param in model.named_parameters():
            gen=torch.Generator().manual_seed(int(hashlib.sha256(name.encode()).hexdigest()[:8],16))
            param.copy_(torch.randn(param.shape,generator=gen)*.02)
    if a.checkpoint=='all':model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    return model
def batch(m):
    ids=(torch.arange(16*128,device=device).reshape(16,128)+m*13)%256
    return dict(input_ids=ids,labels=ids,attention_mask=torch.ones_like(ids),use_cache=False)
reference=build().to(device);reference.train()
grads={name:torch.zeros(p.shape,dtype=torch.bfloat16) for name,p in reference.named_parameters()}
for name,param in reference.named_parameters():
    def take(p,name=name):grads[name].add_(p.grad.detach().cpu());p.grad=None
    param.register_post_accumulate_grad_hook(take)
with torch.no_grad():
    r=reference(**batch(0));ref_loss=float(r.loss);ref_logits=r.logits.detach().cpu();del r
for micro in range(16):
    (reference(**batch(micro)).loss/16).backward()
log('serial_reference',loss=ref_loss,parameters=sum(p.numel() for p in reference.parameters()))
del reference;torch.npu.empty_cache()
from deepspeed.utils import groups,safe_get_full_grad
groups._init_tp_mesh_device(tensor_model_parallel_size=2)
model=build().to(device)
original_shapes={name:list(p.shape) for name,p in model.named_parameters()}
ds=dict(train_batch_size=256,train_micro_batch_size_per_gpu=16,gradient_accumulation_steps=16,
        bf16=dict(enabled=True),gradient_clipping=1.,communication_data_type='fp32',steps_per_print=1000000,
        tensor_parallel=dict(autotp_size=2,vocab_parallel_lm_head=True),
        zero_optimization=dict(stage=a.zero_stage,offload_optimizer=dict(device='cpu',pin_memory=True),
                               contiguous_gradients=True,overlap_comm=True,reduce_bucket_size=1048576),
        optimizer=dict(type='AdamW',params=dict(lr=1e-3,betas=[.9,.999],eps=1e-8,weight_decay=.01)))
engine,opt,_,_=deepspeed.initialize(model=model,model_parameters=model.parameters(),config=ds,mpu=groups)
log('initialized',data_parallel_size=groups.get_data_parallel_world_size(),
    tensor_parallel_size=groups.get_tensor_model_parallel_world_size(),gradient_dtype=str(opt.gradient_accumulation_dtype),
    local_shapes={name:list(p.shape) for name,p in engine.module.named_parameters()},original_shapes=original_shapes)
log('replicated_norm_hooks',parameters={name:dict(registered=bool(getattr(p,'_ds_grad_allreduce_registered',False)),
    hook_count=len(p._backward_hooks or {})) for name,p in engine.module.named_parameters() if '.q_norm.' in name or '.k_norm.' in name})
from replicated_norm_hooks import install
reattached_calls,reattached_handles=install(engine.module,groups.get_tensor_model_parallel_group())
norm_observations={}
for name,param in engine.module.named_parameters():
    if '.q_norm.' not in name and '.k_norm.' not in name:continue
    norm_observations[name]=[]
    for key,fn in (list((param._backward_hooks or {}).items()) if os.environ.get('TP_OBSERVE_HOOKS')=='1' else []):
        def observed(grad,fn=fn,name=name):
            before=grad.detach().cpu().clone()
            result=fn(grad)
            after=result.detach().cpu().clone()
            norm_observations[name].append((before,after))
            return result
        param._backward_hooks[key]=observed
assert groups.get_data_parallel_world_size()==1 and groups.get_tensor_model_parallel_world_size()==2
assert all(p.dtype==torch.float32 and p.device.type=='cpu' for p in opt.single_partition_of_fp32_groups)
engine.eval()
with torch.no_grad():
    output=engine(**batch(0));loss=float(output.loss)
    local=output.logits.contiguous();parts=[torch.empty_like(local) for _ in range(2)]
    dist.all_gather(parts,local,group=groups.get_tensor_model_parallel_group())
    full=torch.cat(parts,dim=-1).cpu();assert full.shape==ref_logits.shape,(full.shape,ref_logits.shape)
    diff=full.float()-ref_logits.float()
    log('forward_check',reference_loss=ref_loss,tp_loss=loss,max_logit_abs=float(diff.abs().max()),
        logit_relative_l2=float(diff.norm()/ref_logits.float().norm()))
    assert abs(loss-ref_loss)<.02 and float(diff.norm()/ref_logits.float().norm())<.05
    del output,local,parts,full,diff
engine.train()
for micro in range(16):
    output=engine(**batch(micro));engine.backward(output.loss)
    if micro<15:
        engine.step()
        if micro==0:
            storages={}
            for param in engine.module.parameters():
                for attr in ['grad','grad_accum']:
                    value=getattr(param,attr,None)
                    if isinstance(value,torch.Tensor) and value.device.type=='npu':
                        storage=value.untyped_storage();storages[storage.data_ptr()]=storage.nbytes()
            log('nonboundary_grad_residence',zero_stage=a.zero_stage,npu_gradient_storage_bytes=sum(storages.values()),
                npu_allocated_bytes=torch.npu.memory_allocated())
torch.npu.synchronize()  # CPU offload buffers must finish before diagnostic reads.
from deepspeed.module_inject.layers import GatherReplacedLayerParams
checks=[]
norm_buffers={}
for name,param in engine.module.named_parameters():
    grad=safe_get_full_grad(param)
    assert grad is not None,name
    original=param.data
    try:
        param.data=grad.detach().to(device).view_as(original)
        with GatherReplacedLayerParams([param],engine.module):
            full=param.detach().cpu().float().clone()
    finally:param.data=original
    expected=grads[name].float()
    if name in norm_observations:norm_buffers[name]=dict(actual=full,expected=expected,history=norm_observations[name])
    assert full.shape==expected.shape,(name,full.shape,expected.shape)
    delta=full-expected; rel=float(delta.norm()/expected.norm().clamp_min(1e-20))
    checks.append(dict(name=name,relative_l2=rel,max_abs=float(delta.abs().max())))
log('gradient_check',checks=checks)
torch.save(norm_buffers,out/f'norm-diagnostic-rank{rank}.private.pt')
log('norm_hook_observations',calls={name:len(items) for name,items in norm_observations.items()})
log('reattached_hook_calls',calls=reattached_calls)
assert all(x['relative_l2']<.08 or x['max_abs']<1e-6 for x in checks),[x for x in checks if x['relative_l2']>=.08 and x['max_abs']>=1e-6]
engine.step();torch.npu.synchronize()
states=list(opt.optimizer.state.values())
assert states and all(s[k].dtype==torch.float32 and s[k].device.type=='cpu' for s in states for k in ['exp_avg','exp_avg_sq'])
result=dict(passed=True,rank=rank,forward_loss=loss,reference_loss=ref_loss,gradient_checks=checks,
            fp32_cpu_master_m_v=True,hybrid_layers=cfg.layer_types,config=ds,
            scope='tiny hybrid Qwen; one 256-sample update, not 27B speed')
(out/f'rank{rank}-result.json').write_text(json.dumps(result,indent=2));log('complete',passed=True)
dist.destroy_process_group()
