"""Four-step full-parameter controlled FP32/offload, BF16/offload and compressed comparison."""
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

p=argparse.ArgumentParser()
p.add_argument('--model',required=True);p.add_argument('--archives',required=True)
p.add_argument('--data',required=True);p.add_argument('--output',required=True)
p.add_argument('--mode',required=True,choices=['fp32','bf16','compressed'])
p.add_argument('--eval-count',type=int,default=8)
p.add_argument('--block',type=int,choices=[64,128],default=128)
p.add_argument('--steps',type=int,default=4);p.add_argument('--lr',type=float,default=1e-5)
p.add_argument('--window',type=int,default=1048576)
p.add_argument('--memory-fraction',type=float,default=.995)
a=p.parse_args();assert a.mode=="fp32" and a.window==16777216 and a.steps==4;configure(a.mode);bench_cpu_master.install()
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
        'mode':a.mode,'compute_dtype':'bf16 autocast with FP32 master weights' if a.mode=='bf16' else 'float32','bf16_model_copy':False,'cpu_state_offload':a.mode!='compressed',
        'gradient_accumulation':1,'global_gradient_clipping':False,'gradient_policy':'consume in custom backward; native tiny-parameter post-accumulate hooks',
        'moment_format':'NF4 m + zero-preserving log v, 4+4 bit, stochastic rounding; FP32 native tiny tensors',
        'quantizer_probability_arithmetic':'FP32 NPU port, not bit-identical NumPy FP64 SR',
        'seed':20261024,'physical_devices':[2,3],'logs':[]}
status['optimizer_window_coordinates']=a.window
status['optimizer_window_bytes_per_state']=a.window*4
status['host_affinity']=sorted(os.sched_getaffinity(0))
status['optimization']='Persistent CPU FP32 master/m/v; DeepSpeed CPUAdam NEON; 24 local cores; bounded pinned staging'
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
    state_device=param.device if a.mode=='compressed' else torch.device('cpu')
    state={'m':torch.zeros(param.shape,dtype=torch.float32,device=state_device),'v':torch.zeros(param.shape,dtype=torch.float32,device=state_device)};native[name]=(param,state)
    def hook(p):
        if p.grad is None:return
        with torch.no_grad():
            g=p.grad.float()
            if not torch.isfinite(g).all().item():raise FloatingPointError('Nonfinite gradient '+name)
            step=native_step[0];assert step>0
            token=probe_before(name,p.numel(),0,p.detach().view(-1),p.grad.float().view(-1),step)
            w=p
            m=state['m'].to(p.device);v=state['v'].to(p.device);m.mul_(.9).add_(g,alpha=.1);v.mul_(.999).addcmul_(g,g,value=.001)
            w.mul_(1-a.lr*.01).addcdiv_(m/(1-.9**step),(v/(1-.999**step)).sqrt().add_(1e-8),value=-a.lr)
            if a.mode!='compressed':
                state['m'].copy_(m);state['v'].copy_(v)
            probe_after(name,step,p.detach().view(-1),token)
            native_changes[name]+=1;p.grad=None
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
    result.sort(key=lambda x:(x[0],x[1]));assert len(result)>=count
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
            store.moments.initialize_frames(store.frames)
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
    train=get_examples(tokenizer,Path(a.data)/'train.jsonl',a.steps)
    val=get_examples(tokenizer,Path(a.data)/'validation.jsonl',a.eval_count)
    status['data']={'source':'Existing independent Alpaca-GPT4 Chinese 32/8 split; shortest complete examples selected deterministically',
        'train_hashes':[x[1] for x in train],'validation_hashes':[x[1] for x in val],
        'train_lengths':[x[0] for x in train],'validation_lengths':[x[0] for x in val],
        'gsm8k_used':False,'loss':'causal assistant tokens only'}
    assert not set(status['data']['validation_hashes'])&set(status['data']['train_hashes'])
    log('model_ready',parameters=declared,wrapped_parameters=status['wrapped_parameters'],memory=memory(),data=status['data'])
    baseline=run_eval(val);status['validation_before']=baseline
    log('validation_before',metrics=baseline,memory=memory())
    for step,example in enumerate(train,1):
        native_step[0]=step
        for s in stores.values():s.step=step;s.enabled=True
        model.train();start=time.monotonic()
        for d in [0,1]:torch.npu.reset_peak_memory_stats(d)
        log('step_begin',step=step,tokens=example[0])
        with precision_context():loss=model(**tensors(example)).loss
        value=float(loss.detach())
        if not math_isfinite(value):raise FloatingPointError('Nonfinite model loss')
        log('forward_done',step=step,loss=value,memory=memory())
        loss.backward()
        for d in [0,1]:torch.npu.synchronize(d)
        del loss
        assert all(s.changed_steps==step for s in stores.values()),[(n,s.changed_steps) for n,s in stores.items() if s.changed_steps!=step]
        assert all(s.moments.updated==s.n*step for s in stores.values())
        assert all(p.grad is None for p in model.parameters())
        missing=[n for n in native if native_changes[n]!=step]
        assert not missing,missing
        status['full_parameter_steps']=step
        log('step_complete',step=step,loss=value,seconds=time.monotonic()-start,
            updated_parameters=declared,remaining_parameter_gradients=0,
            weights_and_guards_bytes=sum(s.stored_bytes() for s in stores.values()),
            moment_bytes=sum(s.n*8 for s in stores.values())+sum(p.numel()*8 for p,_ in native.values()),
            cpu_master_bytes=sum(s.n*4 for s in stores.values()),cpu_master_timing=dict(CPU_MASTER_DETAIL),
            codec_new_fallbacks=codec.fallbacks,memory=memory(),host_timing=dict(TIMES),host_max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        (out/'parameter-probes.json').write_text(json.dumps(PROBES))
        for s in stores.values():s.enabled=False
    final=run_eval(val);status['validation_after']=final
    log('validation_after',metrics=final,delta=final['loss']-baseline['loss'],memory=memory())
    if a.mode=='bf16':
        status['validation_after_bf16_execution']=run_eval(val,canonical=False)
        log('validation_after_bf16_execution',metrics=status['validation_after_bf16_execution'])
    status['completed']=True
    log('complete',memory=memory())
except Exception:
    status['error']=traceback.format_exc();log('failed',error=status['error'],memory=memory());raise
finally:
    status['elapsed_seconds']=time.time()-started
    (out/'status.json').write_text(json.dumps(status,indent=2))
