"""Trace actual model/autograd/checkpoint metadata, without calibration or weights."""
import argparse,contextlib,copy,hashlib,json,os,sys,time
from pathlib import Path

p=argparse.ArgumentParser()
p.add_argument('--config',required=True)
p.add_argument('--vendor-root',required=True)
p.add_argument('--rank',type=int,choices=[0,1],required=True)
p.add_argument('--sequences',default=','.join(['256']*16))
p.add_argument('--micro-batch',type=int,default=16)
p.add_argument('--resident-gib',type=float,default=4)
p.add_argument('--output',required=True)
p.add_argument('--tiny',action='store_true')
a=p.parse_args();root=Path(__file__).resolve().parent
# Prediction cannot read previous observations or the old fitted estimator.
def audit(event,args):
    if event=='open' and isinstance(args[0],(str,bytes)):
        name=os.fsdecode(args[0]).replace('\\','/').lower()
        if '/evidence/' in name or 'calibration.json' in name or name.endswith(('.safetensors','.private.pickle','.db')):
            raise RuntimeError('Forbidden prediction input: '+Path(name).name)
sys.addaudithook(audit)
os.environ['VEOMNI_SOURCE']=a.vendor_root
import torch
torch.set_num_threads(1)
# Metadata run must never initialize a device.
import torch_npu
def deny_device(*args,**kwargs):raise RuntimeError('Device initialization forbidden in metadata prediction')
assert not torch.npu.is_initialized()
assert torch_npu.version.git_version=='5dd8ef3f9b375b5ae4a83538d5785754148c3302','Backend rule requires its pinned TorchNPU version'
torch.npu._lazy_init=deny_device;torch.cuda._lazy_init=deny_device
from torch.utils import checkpoint as cp
cp._infer_device_type=lambda *args:'cpu'  # Meta has no RNG/autocast device module.
original_autocast=torch.autocast
def meta_autocast(device_type,*args,**kwargs):
    return contextlib.nullcontext() if device_type=='meta' else original_autocast(device_type,*args,**kwargs)
torch.autocast=meta_autocast

from transformers.models.qwen3_5 import modeling_qwen3_5 as qm
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
from storage_tracker import StorageTracker
from launch_stub import install
from npu_linear_meta import install as install_npu_linear,OP_PLUGIN_COMMIT
from community_gdn import chunk_gated_delta_rule
launches=install();install_npu_linear();qm.chunk_gated_delta_rule=chunk_gated_delta_rule
cdata=json.loads(Path(a.config).read_text());cdata=cdata.get('text_config',cdata)
c=Qwen3_5TextConfig(**cdata);c.use_cache=False;c._attn_implementation='eager'
if a.tiny:
    c.hidden_size=128;c.intermediate_size=256;c.vocab_size=256
    c.num_hidden_layers=4;c.num_attention_heads=4;c.num_key_value_heads=2;c.head_dim=32
    c.layer_types=['full_attention']*4;c.tie_word_embeddings=False
split=c.num_hidden_layers//2
with torch.device('meta'):
    model=qm.Qwen3_5ForCausalLM(c).to(dtype=torch.bfloat16)
    if a.rank==0:
        model.model.layers=torch.nn.ModuleList(model.model.layers[:split])
        model.model.norm=torch.nn.Identity();model.lm_head=torch.nn.Identity()
    else:
        model.model.layers=torch.nn.ModuleList(model.model.layers[split:])
        model.model.embed_tokens=torch.nn.Identity()
    model.model.rotary_emb=qm.Qwen3_5TextRotaryEmbedding(c,device='meta')
c.num_hidden_layers=split;c.layer_types=c.layer_types[a.rank*split:(a.rank+1)*split]
model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
for i,layer in enumerate(model.model.layers):
    if (i+a.rank*split)%4==3:layer.gradient_checkpointing=False
model.train();params=list(model.parameters());tracker=StorageTracker()
for name,t in model.named_parameters():tracker.register(t,'weight',name,True)
for name,t in model.named_buffers():tracker.register(t,'model_buffer',name,True)
resident={};budget=int(a.resident_gib*2**30);used=0
for idx,t in sorted(enumerate(params),key=lambda kv:(-kv[1].numel(),kv[0])):
    n=t.numel()*2
    if n<=budget-used:
        v=torch.zeros(t.numel(),dtype=torch.bfloat16,device='meta');resident[idx]=v;used+=n
        tracker.register(v,'resident_gradient',str(idx),True)
counts=dict(parameter_gradient_hooks=0,resident_adds=0,offloaded_parameters=0)
def take(idx):
    def callback(param):
        counts['parameter_gradient_hooks']+=1
        g=param.grad
        if idx in resident:
            resident[idx].add_(g.detach().view(-1));counts['resident_adds']+=1
        else:
            # The original D2H source view aliases g. Host slots are not HBM.
            source=g.detach().view(-1);counts['offloaded_parameters']+=1
        param.grad=None
    return callback
handles=[v.register_post_accumulate_grad_hook(take(i)) for i,v in enumerate(params)]
sequences=[int(s) for s in a.sequences.split(',')]
assert len(sequences)==16 and all(s>0 for s in sequences)
b=a.micro_batch
def forward(m):
    s=sequences[m]
    with tracker.scope('input',m):
        ids=torch.empty((b,s),dtype=torch.long,device='meta')
        labels=torch.empty_like(ids)
        mask=torch.empty_like(ids)
        # Shape-explicit mask branch avoids inspecting nonexistent token values.
        causal=torch.empty((b,1,s,s),dtype=torch.bfloat16,device='meta')
        masks={'full_attention':causal,'linear_attention':mask}
        activation=None
        if a.rank==1:activation=torch.empty((b,s,c.hidden_size),dtype=torch.bfloat16,device='meta',requires_grad=True)
    with tracker.scope('forward',m):
        if a.rank==0:return model.model(input_ids=ids,attention_mask=masks,use_cache=False).last_hidden_state
        r=model(inputs_embeds=activation,attention_mask=masks,labels=labels,use_cache=False)
        # Same scalar-shaped weighting operations; numerical values are irrelevant.
        loss=r.loss*((labels[:,1:]!=-100).sum()/1)
        return loss,activation

started=time.monotonic()
with tracker:
    if a.rank==0:
        previous=forward(0)
        for m in range(1,16):
            current=forward(m)
            with tracker.scope('input',m):grad=torch.empty_like(previous)
            with tracker.scope('backward',m-1):previous.backward(grad)
            previous=current
            print(json.dumps(dict(event='micro',rank=a.rank,micro=m,peak_gib=tracker.peak/2**30)),flush=True)
        with tracker.scope('input',15):grad=torch.empty_like(previous)
        with tracker.scope('backward',15):previous.backward(grad)
        del previous,current,grad
    else:
        for m in range(16):
            loss,activation=forward(m)
            with tracker.scope('backward',m):loss.backward()
            # Same send source retains activation.grad until the blocking send completes.
            source=activation.grad
            del source,loss,activation
            print(json.dumps(dict(event='micro',rank=a.rank,micro=m+1,peak_gib=tracker.peak/2**30)),flush=True)
    with tracker.scope('gradient_flush'):
        for idx in sorted(resident):source=resident[idx].detach().view(-1)
        del source
    with tracker.scope('global_norm'):
        norms=[];chunk=64*1024*1024
        for param in params:
            for start in range(0,param.numel(),chunk):
                n=min(chunk,param.numel()-start)
                buf=torch.empty(n,dtype=torch.bfloat16,device='meta')
                norms.append(torch.linalg.vector_norm(buf.float(),ord=2));del buf
        sq=torch.linalg.vector_norm(torch.stack(norms),ord=2).square()
        del norms,sq
result=tracker.summary();tracker.close()
assert not torch.npu.is_initialized()
result.update(rank=a.rank,micro_batch=b,sequences=sequences,resident_bytes=used,
              counts=counts,seconds=time.monotonic()-started,skipped_kernel_launches=len(launches),
              device_initialized=False,calibration_read=False,weights_loaded=False,
              parameter_count=sum(t.numel() for t in params),parameter_tensors=len(params),resident_tensors=len(resident),op_plugin_rule_commit=OP_PLUGIN_COMMIT,
              source_hashes={f:hashlib.sha256((root/f).read_bytes()).hexdigest() for f in ['predict_meta.py','storage_tracker.py','launch_stub.py','community_gdn.py','npu_linear_meta.py']},
              config_sha256=hashlib.sha256(Path(a.config).read_bytes()).hexdigest(),
              limitations=['Counts meta/ATen tensor storages, not native CANN workspace or allocator reserve/runtime.',
                           'Uses real model/autograd/checkpoint code with a meta-only device-context adapter.',
                           'Triton host wrappers allocate real meta tensors; launch bodies are skipped, compiler/internal workspace remains unknown.',
                           'NPU-specific backward layouts/copies may differ from meta kernels; discrepancies are not fitted away.',
                           'Explicit mask shapes assume the padded eager-attention branch.',
                           'Copy completion is modeled as same-stream storage reuse; no timing claim for arbitrary asynchronous streams.'])
Path(a.output).write_text(json.dumps(result,indent=2))
print(json.dumps({k:v for k,v in result.items() if k!='peak_blocks'}),flush=True)
