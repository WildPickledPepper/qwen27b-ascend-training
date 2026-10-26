"""Two-stage synchronous 1F1B, CPU FP32 AdamW; isolated experimental entry.

No parameter/gradient collectives: only stage activations and their gradients.
Original ZeRO-3 grouping, padding, token weighting and update boundaries kept.
BF16 accumulation order changes and is explicitly not claimed bitwise equivalent.
"""
import argparse, contextlib, copy, ctypes, datetime, hashlib, importlib.util
import json, math, os, queue, threading, time, traceback
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument('--model', required=True); p.add_argument('--data', required=True)
p.add_argument('--output', required=True); p.add_argument('--steps', type=int, default=4)
p.add_argument('--tiny', action='store_true'); p.add_argument('--profile', action='store_true')
p.add_argument('--slots', type=int, default=16)
p.add_argument('--resident-gib',type=float,default=0.)
p.add_argument('--norm-device',choices=['cpu','npu'],default='cpu')
a = p.parse_args()
rank = int(os.environ['RANK']); assert int(os.environ['WORLD_SIZE']) == 2
assert os.environ['ASCEND_RT_VISIBLE_DEVICES'] == '2,3'
cores = list(range(96 + rank*24, 120 + rank*24))
assert set(cores) <= os.sched_getaffinity(0)
os.sched_setaffinity(0, cores)
lib = ctypes.CDLL('libnuma.so.1', use_errno=True)
lib.set_mempolicy.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_ulong), ctypes.c_ulong]
mask = ctypes.c_ulong(1 << (4+rank)); assert lib.set_mempolicy(1, ctypes.byref(mask), 8) == 0
import torch, torch_npu
import torch.distributed as dist
from torch.profiler import record_function as scope
torch.set_num_threads(24); torch.npu.set_device(rank)
torch.npu.config.allow_internal_format = False; torch.npu.matmul.allow_hf32 = False
device = f'npu:{rank}'
dist.init_process_group('hccl', timeout=datetime.timedelta(seconds=600))
out = Path(a.output); out.mkdir(parents=True, exist_ok=True)
def log(event, **kw):
    row = dict(event=event, rank=rank, unix=time.time(), **kw)
    with (out/f'rank{rank}.jsonl').open('a') as f: f.write(json.dumps(row)+'\n')
    print(json.dumps(row), flush=True)

from transformers import AutoConfig, AutoTokenizer
from transformers.models.qwen3_5 import modeling_qwen3_5 as qm
from community_gdn import chunk_gated_delta_rule
qm.chunk_gated_delta_rule = chunk_gated_delta_rule
config = AutoConfig.from_pretrained(a.model, local_files_only=True).text_config
config.use_cache = False; config._attn_implementation = 'eager'
if a.tiny:
    config.hidden_size=128; config.intermediate_size=256; config.vocab_size=256
    config.num_hidden_layers=4; config.num_attention_heads=4; config.num_key_value_heads=2
    config.head_dim=32; config.layer_types=['full_attention']*4
    config.tie_word_embeddings=False
split = config.num_hidden_layers//2
reference_config=copy.deepcopy(config)
torch.manual_seed(42)
with torch.device('meta'):
    full = qm.Qwen3_5ForCausalLM(config)
# Preserve original state names and globally indexed checkpoint decisions.
if rank == 0:
    full.model.layers = torch.nn.ModuleList(full.model.layers[:split])
    full.model.norm = torch.nn.Identity(); full.lm_head = torch.nn.Identity()
else:
    full.model.layers = torch.nn.ModuleList(full.model.layers[split:])
    full.model.embed_tokens = torch.nn.Identity()
config.num_hidden_layers = split
config.layer_types = config.layer_types[:split] if rank == 0 else config.layer_types[split:]
# to_empty also empties rotary buffers; reconstruct rotary module afterwards.
full.to_empty(device='cpu'); full.to(dtype=torch.bfloat16)
full.model.rotary_emb = qm.Qwen3_5TextRotaryEmbedding(config=config,device=device)
mapping = json.loads((Path(a.model)/'model.safetensors.index.json').read_text())['weight_map']
from safetensors import safe_open
manifest=[]
for i,(name, param) in enumerate(full.named_parameters()):
    original=name
    if rank == 1 and name.startswith('model.layers.'):
        parts=name.split('.'); parts[2]=str(int(parts[2])+split); original='.'.join(parts)
    key=original.replace('model.', 'model.language_model.', 1) if original.startswith('model.') else original
    if a.tiny:
        # Name-seeded initialization identical in split and unsplit probes.
        gen=torch.Generator().manual_seed(int(hashlib.sha256(original.encode()).hexdigest()[:8],16))
        with torch.no_grad():
            param.copy_(torch.randn(param.shape, generator=gen)*.02)
    else:
        with safe_open(str(Path(a.model)/mapping[key]),framework='pt',device='cpu') as f:
            weight=f.get_tensor(key); assert tuple(weight.shape)==tuple(param.shape)
            with torch.no_grad(): param.copy_(weight)
            del weight
    manifest.append(dict(name=original,shape=list(param.shape),numel=param.numel(),dtype=str(param.dtype)))
    if i%64==0: log('load', tensors=i+1)
full.to(device)
full.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
for i,layer in enumerate(full.model.layers):
    if (i+rank*split)%4==3: layer.gradient_checkpointing=False
model=full
params=list(model.parameters())
log('model_ready',parameters=sum(p.numel() for p in params),manifest=manifest,
    checkpoint_layers=sum(bool(x.gradient_checkpointing) for x in model.model.layers))

class Offload:
    """Bounded D2H FIFO. One BF16 CPU accumulator per owned parameter."""
    def __init__(self):
        self.chunk=65536 if a.tiny else 64*1024*1024
        self.free=queue.Queue(); self.jobs=queue.Queue(); self.error=None
        self.accum=[torch.zeros(p.numel(),dtype=torch.bfloat16) for p in params]
        self.slots=[torch.empty(self.chunk,dtype=torch.bfloat16,pin_memory=True) for _ in range(a.slots)]
        for i in range(a.slots): self.free.put(i)
        self.bytes=0; self.wait=0.; self.copy_wait=0.; self.cpu_add=0.
        self.thread=threading.Thread(target=self.worker,daemon=True); self.thread.start()
        self.hooks=[p.register_post_accumulate_grad_hook(self.hook(i)) for i,p in enumerate(params)]
    def worker(self):
        torch.npu.set_device(rank)
        while True:
            job=self.jobs.get()
            if job is None: self.jobs.task_done(); return
            slot,idx,start,n,event=job
            try:
                t=time.monotonic(); event.synchronize(); self.copy_wait+=time.monotonic()-t
                t=time.monotonic()
                self.accum[idx][start:start+n].add_(self.slots[slot][:n])
                self.cpu_add+=time.monotonic()-t
            except BaseException:
                self.error=traceback.format_exc()
            finally: self.free.put(slot); self.jobs.task_done()
    def hook(self,idx):
        def take(param):
            grad=param.grad.detach().view(-1)
            for start in range(0,grad.numel(),self.chunk):
                then=time.monotonic(); slot=self.free.get(); self.wait+=time.monotonic()-then
                assert self.error is None,self.error
                n=min(self.chunk,grad.numel()-start)
                self.slots[slot][:n].copy_(grad[start:start+n],non_blocking=True)
                ev=torch.npu.Event(); ev.record(torch.npu.current_stream())
                self.jobs.put((slot,idx,start,n,ev)); self.bytes+=n*2
            param.grad=None
        return take
    def drain(self):
        self.jobs.join(); assert self.error is None,self.error
    def reset(self):
        self.drain()
        for buf in self.accum: buf.zero_()
        self.bytes=0; self.wait=0.; self.copy_wait=0.; self.cpu_add=0.

from resident_gradients import create_offload_class
from async_offload import make_async_base
Offload=make_async_base(Offload,params,rank,device,a,scope)
Offload=create_offload_class(Offload,params,rank,device,a,scope)

spec=importlib.util.spec_from_file_location('cpu_adam',os.environ['CPU_ADAM_EXTENSION'])
ext=importlib.util.module_from_spec(spec); spec.loader.exec_module(ext)
from deepspeed.ops.adam.cpu_adam import CPUAdamBuilder, DeepSpeedCPUAdam
CPUAdamBuilder.load=lambda self,verbose=True:ext
# Bounded CPU FP32 gradient/update chunks; same native CPUAdam math as reference.
chunk=65536 if a.tiny else 64*1024*1024
masters=[]; layout=[]
for idx,param in enumerate(params):
    cpu=param.detach().cpu().view(-1)
    for start in range(0,param.numel(),chunk):
        master=torch.nn.Parameter(cpu[start:start+chunk].float())
        masters.append(master); layout.append((idx,start,master.numel()))
    del cpu
adam=DeepSpeedCPUAdam(masters,lr=0.,betas=(.9,.999),eps=1e-8,weight_decay=.01,
                     adamw_mode=True,fp32_optimizer_states=True)
off=Offload()
log('resident_plan',**off.resident_plan)
staging=[torch.empty(min(chunk,max(p.numel() for p in params)),dtype=torch.bfloat16,pin_memory=True) for _ in range(2)]
events=[None,None]
tok=AutoTokenizer.from_pretrained(a.model,local_files_only=True)
if a.tiny:
    train=[dict(length=16,hash=f'tiny-{i}',ids=[(i+j)%200 for j in range(16)],labels=[(i+j)%200 for j in range(16)]) for i in range(256)]
    val=train[:8]
else:
    train=[json.loads(x) for x in (Path(a.data)/'train.jsonl').read_text().splitlines()]
    val=[json.loads(x) for x in (Path(a.data)/'validation.jsonl').read_text().splitlines()]
    assert not {x['hash'] for x in train}&{x['hash'] for x in val}
log('data',training_hashes=[train[i%len(train)]['hash'] for i in range(a.steps*256)],
    training_lengths=[train[i%len(train)]['length'] for i in range(a.steps*256)],
    validation_hashes=[x['hash'] for x in val],global_batch=256,micro_batch=16,
    pipeline_micros=16,source_data_ranks=2,padding='original rank groups dynamic64 minimum128')
def groups(step):
    return [[train[(step*256+(m*16+j)*2+r)%len(train)] for j in range(16)] for m in range(8) for r in range(2)]
def batch(rows,exact=False):
    length=max(x['length'] for x in rows)
    size=length if exact or a.tiny else max(128,math.ceil(length/64)*64)
    pad=0 if a.tiny else (tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id)
    ids=torch.tensor([x['ids']+[pad]*(size-x['length']) for x in rows],device=device)
    labels=torch.tensor([x['labels']+[-100]*(size-x['length']) for x in rows],device=device)
    mask=torch.tensor([[1]*x['length']+[0]*(size-x['length']) for x in rows],device=device)
    return ids,labels,mask
def p2p(send=None,recv=None,forward=True):
    peer=1-rank; ops=[]
    # FIFO, no tag support assumed. Forward and backward have separate directions.
    if send is not None: ops.append(dist.P2POp(dist.isend,send.detach().contiguous(),peer))
    if recv is not None: ops.append(dist.P2POp(dist.irecv,recv,peer))
    with scope('pipeline/P2P'):
        for req in dist.batch_isend_irecv(ops): req.wait()

def forward(ids,labels,mask,activation=None,denom=None):
    with scope(f'pipeline/stage{rank}/forward'):
        if rank==0: return model.model(input_ids=ids,attention_mask=mask,use_cache=False).last_hidden_state
        result=model(inputs_embeds=activation,attention_mask=mask,labels=labels,use_cache=False)
        # Native loss mean * count / global assistant token count.
        if denom is None: return result
        return result.loss*((labels[:,1:]!=-100).sum()/denom)

def evaluate():
    model.eval(); totals=torch.zeros(3,device=device)
    with torch.no_grad():
        for row in val[:8]:
            ids,labels,mask=batch([row],exact=True)
            mask=None  # Exact original validation call: no padding mask for single rows.
            if rank==0:
                act=forward(ids,labels,mask); p2p(send=act)
            else:
                act=torch.empty((1,ids.shape[1],config.hidden_size),device=device,dtype=torch.bfloat16)
                p2p(recv=act); result=forward(ids,labels,mask,act)
                labels=labels[:,1:]; valid=labels!=-100; n=valid.sum()
                totals[0]+=result.loss*n; totals[1]+=n
                totals[2]+=((result.logits[:,:-1].argmax(-1)==labels)&valid).sum()
    dist.broadcast(totals,src=1)
    metrics=dict(loss=float(totals[0]/totals[1]),assistant_tokens=int(totals[1]),correct=int(totals[2]))
    assert math.isfinite(metrics['loss']); log('validation',metrics=metrics)
    return metrics

before=evaluate(); steps=[]
if not a.tiny:
    assert before['correct']==202 and abs(before['loss']-1.0108281373977661)<=1e-6,before
if a.tiny:
    if rank==1:
        ref=qm.Qwen3_5ForCausalLM(reference_config).to(dtype=torch.bfloat16)
        ref.model.rotary_emb=qm.Qwen3_5TextRotaryEmbedding(config=reference_config,device=device)
        with torch.no_grad():
            for name,param in ref.named_parameters():
                gen=torch.Generator().manual_seed(int(hashlib.sha256(name.encode()).hexdigest()[:8],16))
                param.copy_(torch.randn(param.shape,generator=gen)*.02)
        ref.to(device); ref.train()
        refgrads={name:torch.zeros(p.shape,dtype=torch.bfloat16) for name,p in ref.named_parameters()}
        hooks=[]
        for name,param in ref.named_parameters():
            def take(p,name=name):
                refgrads[name].add_(p.grad.detach().cpu()); p.grad=None
            hooks.append(param.register_post_accumulate_grad_hook(take))
        selected=groups(0)
        denom=sum(sum(y!=-100 for y in x['labels'][1:]) for g in selected for x in g)
        for g in selected:
            ids,labels,mask=batch(g)
            loss=ref(input_ids=ids,attention_mask=mask,labels=labels,use_cache=False).loss
            (loss*((labels[:,1:]!=-100).sum()/denom)).backward()
        torch.save(refgrads,out/'tiny-reference.private.pt')
        del ref,refgrads,loss; torch.npu.empty_cache()
    dist.barrier()
for step in range(a.steps):
    model.train(); dist.barrier(); torch.npu.synchronize(); torch.npu.reset_peak_memory_stats()
    prof=None
    if a.profile and step==a.steps-1:
        prof=torch_npu.profiler.profile(activities=[torch_npu.profiler.ProfilerActivity.CPU,torch_npu.profiler.ProfilerActivity.NPU],
             record_shapes=False,profile_memory=False,with_stack=False,
             experimental_config=torch_npu.profiler._ExperimentalConfig(profiler_level=torch_npu.profiler.ProfilerLevel.Level1,export_type=[torch_npu.profiler.ExportType.Db]),
             on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(str(out/f'profile-rank{rank}')))
        prof.start()
    started=time.monotonic()
    with scope('pipeline/gradient_zero'): off.reset()
    selected=groups(step)
    denom=sum(sum(y!=-100 for y in row['labels'][1:]) for group in selected for row in group)
    phase=dict(forward_wall=0.,backward_wall=0.,p2p_wall=0.)
    def fw(m):
        ids,labels,mask=batch(selected[m]); then=time.monotonic()
        if rank==0: result=forward(ids,labels,mask)
        else:
            activation=torch.empty((16,ids.shape[1],config.hidden_size),device=device,dtype=torch.bfloat16)
            t=time.monotonic(); p2p(recv=activation); phase['p2p_wall']+=time.monotonic()-t
            activation.requires_grad_(); result=forward(ids,labels,mask,activation,denom)
        phase['forward_wall']+=time.monotonic()-then
        return result if rank==0 else (result,activation)
    if rank==0:
        previous=fw(0)
        t=time.monotonic(); p2p(send=previous); phase['p2p_wall']+=time.monotonic()-t
        for m in range(1,16):
            current=fw(m); grad=torch.empty_like(previous)
            t=time.monotonic(); p2p(send=current,recv=grad); phase['p2p_wall']+=time.monotonic()-t
            t=time.monotonic()
            with scope('pipeline/stage0/backward'): previous.backward(grad)
            phase['backward_wall']+=time.monotonic()-t
            previous=current; log('micro',step=step+1,micro=m)
        grad=torch.empty_like(previous); t=time.monotonic(); p2p(recv=grad)
        phase['p2p_wall']+=time.monotonic()-t
        with scope('pipeline/stage0/backward'): previous.backward(grad)
        del previous,current,grad
    else:
        for m in range(16):
            loss,activation=fw(m); t=time.monotonic()
            with scope('pipeline/stage1/backward'): loss.backward()
            phase['backward_wall']+=time.monotonic()-t
            t=time.monotonic(); p2p(send=activation.grad); phase['p2p_wall']+=time.monotonic()-t
            log('micro',step=step+1,micro=m+1,loss=float(loss.detach()))
            del loss,activation
    with scope('pipeline/gradient_drain'): off.finish()
    torch.npu.synchronize(); fb_end=time.monotonic()
    # Full-model global clipping: sum both stage squared norms before any update.
    with scope('optimizer/global_norm'):
        if a.norm_device=='cpu':
            square=0.
            for idx,start,n in layout:
                square+=float(off.accum[idx][start:start+n].float().double().square().sum())
            sq=torch.tensor(square,device=device,dtype=torch.float32)
        else:
            norms=[]
            for idx,start,n in layout:
                buf=off.accum[idx][start:start+n].to(device)
                norms.append(torch.linalg.vector_norm(buf.float(),ord=2))
                del buf
            sq=torch.linalg.vector_norm(torch.stack(norms),ord=2).square()
        dist.all_reduce(sq)
        norm=math.sqrt(float(sq)); assert math.isfinite(norm)
        scale=max(1.,(norm+1e-6))
    progress=step; factor=progress/35 if progress<35 else .5*(1+math.cos(math.pi*(progress-35)/(698-35)))
    for group in adam.param_groups: group['lr']=2e-5*factor
    update_start=time.monotonic()
    with scope('optimizer/CPUAdam_and_return'):
        allmasters=adam.param_groups[0]['params']
        for i,(master,(idx,start,n)) in enumerate(zip(masters,layout)):
            master.grad=off.accum[idx][start:start+n].float().div_(scale)
            adam.param_groups[0]['params']=[master]
            adam.step(); master.grad=None
            slot=i%2
            if events[slot] is not None: events[slot].synchronize()
            staging[slot][:n].copy_(master.detach())
            params[idx].detach().view(-1)[start:start+n].copy_(staging[slot][:n],non_blocking=True)
            ev=torch.npu.Event(); ev.record(); events[slot]=ev
        adam.param_groups[0]['params']=allmasters
        for ev in events:
            if ev is not None: ev.synchronize()
    torch.npu.synchronize(); dist.barrier(); elapsed=time.monotonic()-started
    row=dict(step=step+1,seconds=elapsed,profiled=prof is not None,phase=phase,
             forward_backward_and_drain_seconds=fb_end-started,cpu_update_return_seconds=time.monotonic()-update_start,
             grad_norm=norm,norm_device=a.norm_device,global_norm_seconds=update_start-fb_end,
             gradient_d2h_bytes=off.bytes,offload_slot_wait=off.wait,
             worker_copy_wait_seconds=off.copy_wait,worker_cpu_add_seconds=off.cpu_add,
             resident_add_calls=off.resident_add_calls,resident_add_bytes=off.resident_add_bytes,
             offload_parameter_stats=off.parameter_stats,
             allocated_peak=torch.npu.max_memory_allocated(),reserved_peak=torch.npu.max_memory_reserved(),
             input_tokens=sum(x['length'] for g in selected for x in g),
             padded_tokens=sum(16*(max(x['length'] for x in g) if a.tiny else max(128,math.ceil(max(x['length'] for x in g)/64)*64)) for g in selected))
    steps.append(row); log('step',**row)
    if prof: prof.stop()
    if step==2: after=evaluate()
    if a.tiny:
        if step==0:
            refgrads=torch.load(out/'tiny-reference.private.pt',weights_only=True)
            checks=[]
            for info,grad in zip(manifest,off.accum):
                expected=refgrads[info['name']].view(-1)
                diff=float((grad.float()-expected.float()).abs().max())
                checks.append(dict(name=info['name'],exact=torch.equal(grad,expected),max_abs=diff))
                assert torch.allclose(grad.float(),expected.float(),rtol=.02,atol=1e-7),checks[-1]
            log('tiny_gradient_comparison',checks=checks)
            del refgrads
        torch.save(dict(master=[x.detach() for x in masters],states=adam.state_dict(),grad=off.accum),out/f'probe-rank{rank}-step{step+1}.private.pt')
state_bytes=sum(s[k].numel()*s[k].element_size() for s in adam.state.values() for k in ['exp_avg','exp_avg_sq'])
assert all(s[k].dtype==torch.float32 and s[k].device.type=='cpu' for s in adam.state.values() for k in ['exp_avg','exp_avg_sq'])
result=dict(steps=steps,validation_before=before,validation_after_768=after if len(steps)>=3 else None,
            parameters=sum(x.numel() for x in params),state_bytes=state_bytes,
            normal_mean=sum(x['seconds'] for x in steps[1:3])/len(steps[1:3]),
            protocol='global256/micro16/16 pipeline micros; original paired DP groups and padding',
            gradient_precision='BF16 accumulation, FP32 Adam; changed reduction order')
(out/f'rank{rank}-result.json').write_text(json.dumps(result,indent=2))
log('complete',state_bytes=state_bytes)
off.jobs.put(None); off.jobs.join(); off.thread.join(); dist.destroy_process_group()
