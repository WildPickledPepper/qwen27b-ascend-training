"""Pure-FP32 text full-parameter short SFT with resident compressed weights.

No CPU/NVMe runtime state offload, BF16 model copy, gradient accumulation or global
clipping. m/v are approximate packed 4-bit values; tiny native tensors retain FP32
moments. Compression checks are bitwise. Original checkpoints are read-only.
"""
import argparse,collections,gc,hashlib,json,os,time,traceback,faulthandler
from math import prod as math_prod
from pathlib import Path
faulthandler.enable();faulthandler.dump_traceback_later(600,repeat=True)
import torch,torch_npu
from transformers import AutoConfig,AutoTokenizer
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM,Qwen3_5TextRotaryEmbedding
from resident import Codec,WeightStore,CompressedLinear,CompressedEmbedding,load_frames

p=argparse.ArgumentParser()
p.add_argument('--model',required=True);p.add_argument('--archives',required=True)
p.add_argument('--data',required=True);p.add_argument('--output',required=True)
p.add_argument('--block',type=int,choices=[64,128],default=128)
p.add_argument('--steps',type=int,default=2);p.add_argument('--lr',type=float,default=1e-5)
p.add_argument('--window',type=int,default=1048576)
p.add_argument('--memory-fraction',type=float,default=.995)
p.add_argument('--save',action='store_true');a=p.parse_args()
assert os.environ.get('ASCEND_RT_VISIBLE_DEVICES')=='2,3'
root=Path(__file__).resolve().parent;out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
torch.npu.config.allow_internal_format=False
torch.npu.matmul.allow_hf32=False;torch.npu.conv.allow_hf32=False
torch.set_default_dtype(torch.float32);torch.set_num_threads(4)
torch.manual_seed(20261024);torch.npu.manual_seed_all(20261024)
for dev in [0,1]:torch.npu.set_device(dev);torch.npu.set_per_process_memory_fraction(a.memory_fraction,dev)
torch.npu.set_device(0)
started=time.time();stores={};native={};model=None;codec=Codec(root/'libonline_codec.so')
status={'completed':False,'full_parameter_steps':0,'block':a.block,'lr':a.lr,'weight_decay':.01,
        'compute_dtype':'float32','bf16_model_copy':False,'cpu_state_offload':False,
        'gradient_accumulation':1,'global_gradient_clipping':False,'gradient_policy':'consume in custom backward; native tiny-parameter post-accumulate hooks',
        'moment_format':'NF4 m + zero-preserving log v, 4+4 bit, stochastic rounding; FP32 native tiny tensors',
        'quantizer_probability_arithmetic':'FP32 NPU port, not bit-identical NumPy FP64 SR',
        'seed':20261024,'physical_devices':[2,3],'logs':[]}
status['allocator_memory_fraction']=a.memory_fraction
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
    state={'m':torch.zeros_like(param),'v':torch.zeros_like(param)};native[name]=(param,state)
    def hook(p):
        if p.grad is None:return
        with torch.no_grad():
            g=p.grad
            if not torch.isfinite(g).all().item():raise FloatingPointError('Nonfinite gradient '+name)
            step=native_step[0];assert step>0
            m=state['m'];v=state['v'];m.mul_(.9).add_(g,alpha=.1);v.mul_(.999).addcmul_(g,g,value=.001)
            p.mul_(1-a.lr*.01).addcdiv_(m/(1-.9**step),(v/(1-.999**step)).sqrt().add_(1e-8),value=-a.lr)
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
def run_eval(example):
    model.eval()
    with torch.no_grad():
        loss=model(**tensors(example)).loss
        val=float(loss);assert math_isfinite(val)
        del loss
    return val
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
    status['full_text_parameters']=declared;status['compressed_parameters']=sum(s.n for s in stores.values())
    status['native_parameters']=sum(p.numel() for p,_ in native.values())
    assert status['compressed_parameters']+status['native_parameters']==declared
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':True})
    tokenizer=AutoTokenizer.from_pretrained(a.model,local_files_only=True)
    train=get_examples(tokenizer,Path(a.data)/'train.jsonl',a.steps)
    val=get_examples(tokenizer,Path(a.data)/'validation.jsonl',1)[0]
    status['data']={'source':'Existing independent Alpaca-GPT4 Chinese 32/8 split; shortest complete examples selected deterministically',
        'train_hashes':[x[1] for x in train],'validation_hash':val[1],
        'train_lengths':[x[0] for x in train],'validation_length':val[0],
        'gsm8k_used':False,'loss':'causal assistant tokens only'}
    assert val[1] not in status['data']['train_hashes']
    log('model_ready',parameters=declared,compressed_parameters=status['compressed_parameters'],memory=memory(),data=status['data'])
    baseline=run_eval(val);status['validation_before']=baseline
    log('validation_before',loss=baseline,memory=memory())
    for step,example in enumerate(train,1):
        native_step[0]=step
        for s in stores.values():s.step=step;s.enabled=True
        model.train();start=time.monotonic()
        for d in [0,1]:torch.npu.reset_peak_memory_stats(d)
        log('step_begin',step=step,tokens=example[0])
        loss=model(**tensors(example)).loss;value=float(loss.detach())
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
            moment_bytes=sum(s.moments.bytes() for s in stores.values())+sum(p.numel()*8 for p,_ in native.values()),
            codec_new_fallbacks=codec.fallbacks,memory=memory())
        for s in stores.values():s.enabled=False
    final=run_eval(val);status['validation_after']=final
    log('validation_after',loss=final,delta=final-baseline,memory=memory())
    if a.save:
        log('save_begin')
        # Checkpoint output only, after all training; never a runtime state backing store.
        for d in [0,1]:
            index=[]
            with (out/f'checkpoint-device{d}.bin').open('wb') as f:
                for name,s in stores.items():
                    if s.device.index!=d:continue
                    rows=[]
                    for frame in s.frames:
                        offset=f.tell();raw=frame['data'][:frame['bytes']].cpu().numpy().tobytes();f.write(raw)
                        rows.append({'kind':frame['kind'],'n':frame['n'],'bytes':len(raw),'offset':offset,
                                     'header_hex':frame['header'].hex() if frame['header'] else None})
                    offsets=[]
                    for tensor in [s.moments.codes,s.moments.scales]:
                        raw=tensor.cpu().numpy().tobytes();offsets.append({'offset':f.tell(),'bytes':len(raw),'shape':list(tensor.shape),'dtype':str(tensor.dtype)});f.write(raw)
                    index.append({'name':name,'shape':s.shape,'step':s.step,'frames':rows,'moments':offsets})
            (out/f'checkpoint-device{d}.json').write_text(json.dumps(index))
        torch.save({n:{'weight':p.detach().cpu(),'m':state['m'].cpu(),'v':state['v'].cpu()} for n,(p,state) in native.items()},out/'native-small.pt')
        log('save_complete')
    status['completed']=True
    log('complete',memory=memory())
except Exception:
    status['error']=traceback.format_exc();log('failed',error=status['error'],memory=memory());raise
finally:
    status['elapsed_seconds']=time.time()-started
    (out/'status.json').write_text(json.dumps(status,indent=2))

