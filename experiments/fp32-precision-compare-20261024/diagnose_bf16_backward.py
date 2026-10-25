"""Explicit BF16 linear compute with CPU FP32 master/Adam; up to 20 full-parameter steps."""
import argparse,collections,gc,hashlib,json,os,time,traceback,faulthandler
from math import prod as math_prod
from pathlib import Path
faulthandler.enable();faulthandler.dump_traceback_later(600,repeat=True)
import ctypes
os.sched_setaffinity(0,set(range(96,120)) & os.sched_getaffinity(0))
_process_libc=ctypes.CDLL(None,use_errno=True)
assert _process_libc.prctl(41,1,0,0,0)==0, 'Cannot disable THP for this process'
import torch,torch_npu
from transformers import AutoConfig,AutoTokenizer
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM,Qwen3_5TextRotaryEmbedding
from bench_resident_npu import Codec,WeightStore,CompressedLinear,CompressedEmbedding,load_frames,configure,precision_context,evaluation_context,CpuMoments,TIMES,PROBES,probe_before,probe_after
import resource
import bench_cpu_master
from bench_cpu_master import DETAIL as CPU_MASTER_DETAIL
import bench_cpu_pipeline
import bench_bf16_master
from bench_bf16_master import precision_context,evaluation_context


class DiagnosticMoments(bench_bf16_master.MixedMoments):
    def __init__(self,n,device,block=128,lr=1e-5,wd=.01,window=16777216):
        self.n=n;self.lr=lr;self.wd=wd;self.window=window
        self.master=torch.empty(n,dtype=torch.float32);self.initialized=False
        self.updated=0;self.progress=None
    def bytes(self):return self.n*4
    def update(self,start,w,g,step):
        assert torch.isfinite(g).all()
        token=probe_before(self.name,self.n,start,w.reshape(-1).float(),g.reshape(-1).float(),step)
        if token:
            chosen,ix,before,gradient=token
            PROBES.append({'name':self.name,'step':step,'indices':chosen,'weight_before':before,'gradient':gradient,'weight_after':before})
        self.updated+=g.numel()
def install_diagnostic_moments():
    import bench_resident
    bench_resident.base.QuantMoments=DiagnosticMoments

p=argparse.ArgumentParser()
p.add_argument('--model',required=True);p.add_argument('--archives',required=True)
p.add_argument('--data',required=True);p.add_argument('--output',required=True)
p.add_argument('--mode',required=True,choices=['fp32','bf16','compressed'])
p.add_argument('--eval-count',type=int,default=8)
p.add_argument('--block',type=int,choices=[64,128],default=128)
p.add_argument('--steps',type=int,default=4);p.add_argument('--lr',type=float,default=1e-5)
p.add_argument('--window',type=int,default=1048576)
p.add_argument('--memory-fraction',type=float,default=.995)
a=p.parse_args();assert a.mode=="bf16" and a.window==16777216 and 4<=a.steps<=20;configure("fp32");bench_bf16_master.install();install_diagnostic_moments()
assert os.environ.get('ASCEND_RT_VISIBLE_DEVICES')=='2,3'
root=Path(__file__).resolve().parent;out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
torch.npu.config.allow_internal_format=False
torch.npu.matmul.allow_hf32=False;torch.npu.conv.allow_hf32=False
torch.set_default_dtype(torch.float32);torch.set_num_threads(24)
torch.manual_seed(20261024);torch.npu.manual_seed_all(20261024)
for dev in [0,1]:torch.npu.set_device(dev);torch.npu.set_per_process_memory_fraction(a.memory_fraction,dev)
torch.npu.set_device(0)
started=time.time();stores={};native={};model=None;codec=Codec(root/'libonline_codec.so')
status={'completed':False,'full_parameter_steps':0,'block':a.block,'lr':a.lr,'weight_decay':.01,
        'mode':a.mode,'compute_dtype':'explicit BF16 linear matmuls, FP32 outputs/other operations; persistent BF16 linear weights + CPU FP32 master' if a.mode=='bf16' else 'float32','bf16_model_copy':True,'cpu_state_offload':a.mode!='compressed',
        'gradient_accumulation':1,'global_gradient_clipping':False,'gradient_policy':'consume in custom backward; native tiny-parameter post-accumulate hooks',
        'moment_format':'NF4 m + zero-preserving log v, 4+4 bit, stochastic rounding; FP32 native tiny tensors',
        'quantizer_probability_arithmetic':'FP32 NPU port, not bit-identical NumPy FP64 SR',
        'seed':20261024,'physical_devices':[2,3],'logs':[]}
status['optimizer_window_coordinates']=a.window
status['optimizer_window_bytes_per_state']=a.window*4
status['host_affinity']=sorted(os.sched_getaffinity(0))
status['optimization']='CPU FP32 master/m/v; GIL-free unmodified DeepSpeed CPUAdam math; 4-slot gradient/update/weight pipeline; persistent device frames'
status['rounding']='FP32 library AdamW, not bitwise identical to reference NPU operations'
status['process_thp_disabled']=bool(_process_libc.prctl(42,0,0,0,0))
status['allocator_memory_fraction']=a.memory_fraction
status['comparison']='same custom autograd harness; persistent CPU master and installed DeepSpeed CPUAdam, not a DeepSpeed training engine'
status['optimizer_compute_device']='cpu (DeepSpeed CPUAdam); tiny native tensors retain reference NPU update'
status['full_state_offload_precision']='FP32 raw copies; no quantization' if a.mode!='compressed' else None
status['validation_precision']='canonical FP32 for all groups, plus execution dtype for BF16'
if a.mode!='compressed':status['moment_format']='full FP32 m/v and persistent master on CPU; DeepSpeed CPUAdam'
def memory():
    return [{'logical_device':d,'allocated':torch.npu.memory_allocated(d),'reserved':torch.npu.memory_reserved(d),
             'peak_allocated':torch.npu.max_memory_allocated(d),'peak_reserved':torch.npu.max_memory_reserved(d)} for d in [0,1]]
def log(event,**kw):
    row={'event':event,'unix':time.time(),'elapsed':time.time()-started,**kw}
    print(json.dumps(row),flush=True)
    with (out/'events.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
    status['logs'].append(row)
    (out/'status.json').write_text(json.dumps(status,indent=2))
def update_progress(store):
    row={'tensor':store.name,'step':store.step,'updated_coordinates':store.moments.updated,
         'tensor_parameters':store.n,'unix':time.time(),'memory':memory()}
    (out/'update-progress.json').write_text(json.dumps(row))
def rename(key):return key.replace('model.language_model.','model.',1)
def replace(module,name,value):
    parent,_,leaf=name.rpartition('.');obj=module.get_submodule(parent) if parent else module
    setattr(obj,leaf,value)
native_step=[0]
native_changes=collections.Counter()
def install_native_hook(name,param):
    native[name]=(param,{})
    def hook(p):
        if p.grad is None:return
        token=probe_before(name,p.numel(),0,p.detach().view(-1),p.grad.float().view(-1),1)
        if token:
            chosen,ix,before,gradient=token
            PROBES.append({'name':name,'step':1,'indices':chosen,'weight_before':before,'gradient':gradient,'weight_after':before})
        p.grad=None
    param.register_post_accumulate_grad_hook(hook)

def get_examples(tokenizer,filename,count):
    result=[]
    for line in Path(filename).read_text().splitlines():
        entry=json.loads(line);messages=entry['messages']
        ids=tokenizer.apply_chat_template(messages,tokenize=True,enable_thinking=False,return_dict=False)
        prefix=tokenizer.apply_chat_template(messages[:-1],tokenize=True,add_generation_prompt=True,enable_thinking=False,return_dict=False)
        if ids[:len(prefix)]!=prefix:continue
        if len(ids)>128 or len(ids)<=len(prefix):continue
        labels=[-100]*len(prefix)+ids[len(prefix):]
        digest=hashlib.sha256(json.dumps(messages,ensure_ascii=False,sort_keys=True).encode()).hexdigest()
        result.append((len(ids),digest,ids,labels))
    result.sort(key=lambda x:(x[0],x[1]));assert len(result)>=min(count,4)
    return result[:count]
def tensors(example):
    return {'input_ids':torch.tensor([example[2]],dtype=torch.long,device='npu:0'),
            'labels':torch.tensor([example[3]],dtype=torch.long,device='npu:0'),
            'use_cache':False}
def run_eval(examples,canonical=True):
    model.eval();weighted=0.;count=0;correct=0;rows=[]
    with torch.no_grad(),evaluation_context(canonical):
        for example in examples:
            batch=tensors(example);output=model(**batch);value=float(output.loss)
            assert math_isfinite(value)
            labels=batch['labels'][:,1:];mask=labels!=-100;n=int(mask.sum())
            pred=output.logits[:,:-1].argmax(-1)
            hits=int(((pred==labels)&mask).sum())
            weighted+=value*n;count+=n;correct+=hits
            rows.append({'hash':example[1],'tokens':example[0],'assistant_tokens':n,'loss':value,'correct':hits})
            del output,batch,pred
    return {'loss':weighted/count,'assistant_token_accuracy':correct/count,'assistant_tokens':count,'examples':rows,'canonical_fp32':canonical}

def math_isfinite(x):return x==x and abs(x)!=float('inf')

try:
    log('begin',memory=memory(),torch=torch.__version__,torch_npu=torch_npu.__version__)
    config=AutoConfig.from_pretrained(a.model,local_files_only=True).text_config
    config.dtype=torch.float32;config.use_cache=False;config._attn_implementation='eager'
    with torch.device('meta'):model=Qwen3_5ForCausalLM(config)
    model.model.rotary_emb=Qwen3_5TextRotaryEmbedding(config,device=torch.device('npu:0'))
    original_shapes={n:tuple(p.shape) for n,p in model.named_parameters()}
    reports=[]
    for physical in [2,3]:
        folder=Path(a.archives)/f'full-card{physical}'
        report=json.loads((folder/'report.json').read_text());assert report['complete']
        verification=json.loads((folder/'archive_verification.json').read_text());assert verification['passed']
        reports.append((folder,report,f'npu:{physical-2}'))
    entries=[]
    for folder,report,device in reports:
        for key,shape in report['tensor_index'].items():entries.append((key,shape,folder,report,device))
    entries.sort(key=lambda x:(-math_prod(x[1]),x[0]))
    loaded=0;declared=0;native_values={}
    for index,(key,shape,folder,report,device) in enumerate(entries):
        name=rename(key);assert tuple(shape)==original_shapes[name],(name,shape,original_shapes.get(name))
        module_name,_,leaf=name.rpartition('.');old=model.get_submodule(module_name)
        use_compressed=isinstance(old,(torch.nn.Linear,torch.nn.Embedding)) and leaf=='weight'
        frames=load_frames(folder,report,key,device if use_compressed else 'npu:0')
        n=math_prod(shape);declared+=n
        if use_compressed:
            store=WeightStore(name,shape,frames,codec,device,a.block,a.lr,.01,a.window)
            bench_bf16_master.install_store(store,is_embedding=isinstance(old,torch.nn.Embedding))
            stores[name]=store;store.progress=lambda s=store:update_progress(s)
            store.moments.progress=store.progress
            replacement=CompressedLinear(store) if isinstance(old,torch.nn.Linear) else CompressedEmbedding(store)
            replace(model,module_name,replacement)
            loaded+=store.stored_bytes()+store.moments.bytes()
        else:
            with torch.npu.device(0):
                value=torch.empty(n,dtype=torch.float32,device='npu:0');offset=0
                for frame in frames:codec.decode(frame,value[offset:offset+frame['n']]);offset+=frame['n']
                param=torch.nn.Parameter(value.reshape(shape));replace(model,name,param)
                install_native_hook(name,param)
                loaded+=n*12
        del frames,old
        if index%32==0 or index==len(entries)-1:
            gc.collect()
            for d in [0,1]:
                with torch.npu.device(d):torch.npu.empty_cache()
            log('load_progress',tensors=index+1,total_tensors=len(entries),logical_owned_bytes=loaded,memory=memory())
    assert declared==26_895_998_464,declared
    assert not any(p.is_meta for p in model.parameters())
    assert all(p.dtype==torch.float32 for p in model.parameters())
    assert not any(b.is_meta for b in model.buffers())
    status['full_text_parameters']=declared;status['wrapped_parameters']=sum(s.n for s in stores.values())
    status['native_parameters']=sum(p.numel() for p,_ in native.values())
    assert status['wrapped_parameters']+status['native_parameters']==declared
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':True})
    tokenizer=AutoTokenizer.from_pretrained(a.model,local_files_only=True)
    train_pool=get_examples(tokenizer,Path(a.data)/'train.jsonl',a.steps)
    train=[train_pool[i%len(train_pool)] for i in range(a.steps)]
    status['training_unique_examples']=len(train_pool)
    status['training_repeats_if_needed']=len(train_pool)<a.steps
    val=get_examples(tokenizer,Path(a.data)/'validation.jsonl',a.eval_count)
    status['data']={'source':'Existing independent Alpaca-GPT4 Chinese 32/8 split; shortest complete examples selected deterministically',
        'train_hashes':[x[1] for x in train],'validation_hashes':[x[1] for x in val],
        'train_lengths':[x[0] for x in train],'validation_lengths':[x[0] for x in val],
        'gsm8k_used':False,'loss':'causal assistant tokens only'}
    assert not set(status['data']['validation_hashes'])&set(status['data']['train_hashes'])
    log('model_ready',parameters=declared,wrapped_parameters=status['wrapped_parameters'],memory=memory(),data=status['data'])
    diagnostic={'stages':{}}
    for canonical in [True,False]:
        label='fp32' if canonical else 'bf16';snap={};handles=[];PROBES.clear()
        def capture(name):
            def forward(module,inputs,output):
                if not output.requires_grad:return
                snap[name+'_output']=output.detach().cpu().clone()
                snap[name+'_input']=inputs[0].detach().cpu().clone()
                output.register_hook(lambda g:snap.__setitem__(name+'_output_gradient',g.detach().cpu().clone()))
                if inputs[0].requires_grad:
                    inputs[0].register_hook(lambda g:snap.__setitem__(name+'_input_gradient',g.detach().cpu().clone()))
            return forward
        for part in ['gate_proj','up_proj','down_proj']:
            mod=model.get_submodule('model.layers.63.mlp.'+part)
            handles.append(mod.register_forward_hook(capture(part)))
        for st in stores.values():st.enabled=True;st.step=1
        native_step[0]=1;model.train()
        with evaluation_context(canonical):
            loss=model(**tensors(train[0])).loss
            diagnostic['stages'][label]={'loss':float(loss)}
            loss.backward()
        for d in [0,1]:torch.npu.synchronize(d)
        for h in handles:h.remove()
        torch.save(snap,out/(label+'-last-ffn-private.pt'))
        (out/(label+'-probes-private.json')).write_text(json.dumps(PROBES))
        g=snap['gate_proj_output'];up=snap['up_proj_output'];dy=snap['down_proj_input_gradient']
        sigmoid=torch.sigmoid(g);expected=dy*up*(sigmoid+g*sigmoid*(1-sigmoid))
        actual=snap['gate_proj_output_gradient']
        diagnostic['stages'][label]['silu_gradient']={
            'gate_input_max_abs':float(g.abs().max()),'up_output_max_abs':float(up.abs().max()),
            'actual_l2':float(actual.norm()),'expected_l2':float(expected.norm()),
            'relative_l2':float((actual-expected).norm()/expected.norm().clamp_min(1e-30)),
            'actual_max_abs':float(actual.abs().max()),'expected_max_abs':float(expected.abs().max())}
        del loss,snap;gc.collect()
        log('backward_diagnostic_stage',stage=label,metrics=diagnostic['stages'][label])
        (out/'backward-diagnostic.json').write_text(json.dumps(diagnostic,indent=2))
    status['diagnostic_complete']=True;log('diagnostic_complete',metrics=diagnostic)

except Exception:
    status['error']=traceback.format_exc();log('failed',error=status['error'],memory=memory());raise
finally:
    status['elapsed_seconds']=time.time()-started
    (out/'status.json').write_text(json.dumps(status,indent=2))
