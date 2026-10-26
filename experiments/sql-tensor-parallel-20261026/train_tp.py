"""Native AutoTP2 + ZeRO-1 CPU FP32 Adam, matched 256-sample SQL task."""
import argparse,ctypes,datetime,hashlib,importlib.util,json,math,os,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--data',required=True)
p.add_argument('--output',required=True);p.add_argument('--steps',type=int,default=4);p.add_argument('--profile',action='store_true')
p.add_argument('--checkpoint',choices=['selective','all'],default='selective');a=p.parse_args()
rank=int(os.environ['LOCAL_RANK']);assert os.environ['ASCEND_RT_VISIBLE_DEVICES']=='2,3'
os.sched_setaffinity(0,list(range(96+24*rank,120+24*rank)))
lib=ctypes.CDLL('libnuma.so.1',use_errno=True);lib.set_mempolicy.argtypes=[ctypes.c_int,ctypes.POINTER(ctypes.c_ulong),ctypes.c_ulong]
mask=ctypes.c_ulong(1<<(4+rank));assert lib.set_mempolicy(1,ctypes.byref(mask),8)==0
import torch,torch_npu,deepspeed
import torch.distributed as dist
torch.npu.set_device(rank);torch.set_num_threads(24);device=f'npu:{rank}'
torch.npu.config.allow_internal_format=False;torch.npu.matmul.allow_hf32=False
deepspeed.init_distributed(dist_backend='hccl',timeout=datetime.timedelta(seconds=600))
out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
def log(event,**kw):
    row=dict(event=event,rank=rank,unix=time.time(),**kw)
    with (out/f'rank{rank}.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
    print(json.dumps(row),flush=True)
spec=importlib.util.spec_from_file_location('cpu_adam',os.environ['CPU_ADAM_EXTENSION'])
ext=importlib.util.module_from_spec(spec);spec.loader.exec_module(ext)
from deepspeed.ops.adam.cpu_adam import CPUAdamBuilder
CPUAdamBuilder.load=lambda self,verbose=True:ext
from transformers import AutoConfig,AutoTokenizer
from transformers.models.qwen3_5 import modeling_qwen3_5 as qm
from community_gdn import chunk_gated_delta_rule
qm.chunk_gated_delta_rule=chunk_gated_delta_rule
cfg=AutoConfig.from_pretrained(a.model,local_files_only=True).text_config
cfg.use_cache=False;cfg._attn_implementation='eager'
with torch.device('meta'):model=qm.Qwen3_5ForCausalLM(cfg)
model.to_empty(device='cpu');model.bfloat16()
model.model.rotary_emb=qm.Qwen3_5TextRotaryEmbedding(config=cfg,device=device)
from safetensors import safe_open
mapping=json.loads((Path(a.model)/'model.safetensors.index.json').read_text())['weight_map']
original_shapes={name:list(p.shape) for name,p in model.named_parameters()}
assert sum(p.numel() for p in model.parameters())==26895998464
for i,(name,param) in enumerate(model.named_parameters()):
    key=name.replace('model.','model.language_model.',1) if name.startswith('model.') else name
    with safe_open(str(Path(a.model)/mapping[key]),framework='pt',device='cpu') as f:
        value=f.get_tensor(key)
        with torch.no_grad():param.copy_(value)
        del value
    if i%64==0:log('load',tensors=i+1)
# Attach AutoTP callbacks only after initial device placement; reattach norm
# callbacks again after optimizer setup, as validated by the tiny probe.
model.to(device)
model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
for i,layer in enumerate(model.model.layers):
    if a.checkpoint=='selective' and i%4==3:layer.gradient_checkpointing=False
from deepspeed.utils import groups
groups._init_tp_mesh_device(tensor_model_parallel_size=2)
ds=dict(train_batch_size=256,train_micro_batch_size_per_gpu=16,gradient_accumulation_steps=16,
        bf16=dict(enabled=True),gradient_clipping=1.,communication_data_type='fp32',steps_per_print=1000000,
        tensor_parallel=dict(autotp_size=2,vocab_parallel_lm_head=True),
        zero_optimization=dict(stage=1,offload_optimizer=dict(device='cpu',pin_memory=True),
                               contiguous_gradients=True,overlap_comm=True,reduce_bucket_size=16777216),
        optimizer=dict(type='AdamW',params=dict(lr=0.,betas=[.9,.999],eps=1e-8,weight_decay=.01)))
engine,opt,_,_=deepspeed.initialize(model=model,model_parameters=model.parameters(),config=ds,mpu=groups)
assert groups.get_data_parallel_world_size()==1 and groups.get_tensor_model_parallel_world_size()==2
assert engine.train_batch_size()==256 and engine.gradient_accumulation_steps()==16
assert opt.gradient_accumulation_dtype==torch.bfloat16
assert all(p.dtype==torch.float32 and p.device.type=='cpu' for p in opt.single_partition_of_fp32_groups)
from replicated_norm_hooks import install
hook_calls,hook_handles=install(engine.module,groups.get_tensor_model_parallel_group())
log('model_ready',config=ds,original_shapes=original_shapes,
    local_shapes={n:list(p.shape) for n,p in engine.module.named_parameters()},
    local_parameters=sum(p.numel() for p in engine.module.parameters()),rebound_norm_parameters=len(hook_calls),
    checkpoint_policy=a.checkpoint,checkpointed_layers=sum(bool(x.gradient_checkpointing) for x in model.model.layers),
    allocated_after_init=torch.npu.memory_allocated(),reserved_after_init=torch.npu.memory_reserved())
tok=AutoTokenizer.from_pretrained(a.model,local_files_only=True)
train=[json.loads(x) for x in (Path(a.data)/'train.jsonl').read_text().splitlines()]
val=[json.loads(x) for x in (Path(a.data)/'validation.jsonl').read_text().splitlines()]
assert not {x['hash'] for x in train}&{x['hash'] for x in val}
log('data',training_hashes=[train[i%len(train)]['hash'] for i in range(a.steps*256)],
    training_lengths=[train[i%len(train)]['length'] for i in range(a.steps*256)],validation_hashes=[r['hash'] for r in val])
def selected(step):return [[train[(step*256+(m*16+j)*2+r)%len(train)] for j in range(16)] for m in range(8) for r in range(2)]
def batch(rows,exact=False):
    length=max(x['length'] for x in rows);size=length if exact else max(128,math.ceil(length/64)*64)
    pad=tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    return dict(input_ids=torch.tensor([x['ids']+[pad]*(size-x['length']) for x in rows],device=device),
                labels=torch.tensor([x['labels']+[-100]*(size-x['length']) for x in rows],device=device),
                attention_mask=torch.tensor([[1]*x['length']+[0]*(size-x['length']) for x in rows],device=device),use_cache=False)
def evaluate():
    engine.eval();totals=torch.zeros(3,device=device)
    with torch.no_grad():
        for row in val[:8]:
            data=batch([row],exact=True);data.pop('attention_mask');output=engine(**data)
            labels=data['labels'][:,1:];valid=labels!=-100;n=valid.sum()
            values,indices=output.logits[:,:-1].max(-1)
            indices=indices+rank*(cfg.vocab_size//2)
            vv=[torch.empty_like(values) for _ in range(2)];ii=[torch.empty_like(indices) for _ in range(2)]
            dist.all_gather(vv,values);dist.all_gather(ii,indices)
            winner=torch.stack(vv).argmax(0);pred=torch.stack(ii).gather(0,winner.unsqueeze(0)).squeeze(0)
            totals[0]+=output.loss*n;totals[1]+=n;totals[2]+=((pred==labels)&valid).sum()
    metrics=dict(loss=float(totals[0]/totals[1]),assistant_tokens=int(totals[1]),correct=int(totals[2]))
    assert math.isfinite(metrics['loss']);log('validation',metrics=metrics);return metrics
before=evaluate();steps=[]
for step in range(a.steps):
    engine.train();dist.barrier();torch.npu.synchronize();torch.npu.reset_peak_memory_stats()
    prof=None
    if a.profile and step==a.steps-1:
        prof=torch_npu.profiler.profile(activities=[torch_npu.profiler.ProfilerActivity.CPU,torch_npu.profiler.ProfilerActivity.NPU],
            record_shapes=False,profile_memory=False,with_stack=False,
            experimental_config=torch_npu.profiler._ExperimentalConfig(profiler_level=torch_npu.profiler.ProfilerLevel.Level1,export_type=[torch_npu.profiler.ExportType.Db]),
            on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(str(out/f'profile-rank{rank}')))
        prof.start()
    started=time.monotonic();rows=selected(step)
    count=sum(sum(y!=-100 for y in r['labels'][1:]) for g in rows for r in g)
    for group in opt.optimizer.param_groups:group['lr']=2e-5*step/35
    for name in hook_calls:hook_calls[name]=0
    phase=dict(forward_wall=0.,backward_wall=0.,step_wall=0.)
    for micro,group in enumerate(rows):
        t=time.monotonic();data=batch(group);output=engine(**data)
        loss=output.loss*((data['labels'][:,1:]!=-100).sum()*16/count)
        value=float(loss.detach());assert math.isfinite(value)
        phase['forward_wall']+=time.monotonic()-t
        t=time.monotonic();engine.backward(loss);phase['backward_wall']+=time.monotonic()-t
        t=time.monotonic();engine.step();phase['step_wall']+=time.monotonic()-t
        log('micro',step=step+1,micro=micro+1,loss=value)
        del output,loss,data
    torch.npu.synchronize();dist.barrier();seconds=time.monotonic()-started
    assert all(c==16 for c in hook_calls.values()),hook_calls
    row=dict(step=step+1,seconds=seconds,profiled=prof is not None,phase=phase,
             grad_norm=float(engine.get_global_grad_norm()),allocated_peak=torch.npu.max_memory_allocated(),reserved_peak=torch.npu.max_memory_reserved(),
             input_tokens=sum(r['length'] for g in rows for r in g),
             padded_tokens=sum(16*max(128,math.ceil(max(r['length'] for r in g)/64)*64) for g in rows),
             norm_hook_calls=dict(hook_calls))
    steps.append(row);log('step',**row)
    if prof:prof.stop()
    if step==2:after=evaluate()
states=list(opt.optimizer.state.values())
assert states and all(s[k].dtype==torch.float32 and s[k].device.type=='cpu' for s in states for k in ['exp_avg','exp_avg_sq'])
result=dict(steps=steps,validation_before=before,validation_after_768=after,config=ds,
    fp32_cpu_master_m_v=True,state_bytes=sum(s[k].numel()*s[k].element_size() for s in states for k in ['exp_avg','exp_avg_sq']),
    local_parameters=sum(p.numel() for p in engine.module.parameters()),
    checkpoint_policy=a.checkpoint,
    limits=['Native AutoTP arithmetic/reduction tree differs from pipeline; no bitwise claim.',
            'Native ZeRO-1 CPU offload; not using the optimized pipeline FIFO/cache.'])
(out/f'rank{rank}-result.json').write_text(json.dumps(result,indent=2));log('complete')
dist.destroy_process_group()
