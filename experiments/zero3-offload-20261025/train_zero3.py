"""Native two-process ZeRO-3; original weights read-only, aggregate evidence only."""
import argparse,datetime,faulthandler,hashlib,importlib.util,json,math,os,time
from pathlib import Path
faulthandler.enable();faulthandler.dump_traceback_later(600,repeat=True)
p=argparse.ArgumentParser()
p.add_argument('--config',default='zero3.json');p.add_argument('--output',required=True)
p.add_argument('--model');p.add_argument('--data');p.add_argument('--tiny',action='store_true')
p.add_argument('--steps',type=int,default=4)
a=p.parse_args()
rank=int(os.environ['RANK']); local=int(os.environ['LOCAL_RANK']); world=int(os.environ['WORLD_SIZE'])
assert world==2 and os.environ['ASCEND_RT_VISIBLE_DEVICES']=='2,3'
allowed=sorted(os.sched_getaffinity(0)); chosen=[i for i in range(96+local*12,108+local*12) if i in allowed]
assert chosen;os.sched_setaffinity(0,chosen)
import torch,torch_npu
import torch.distributed as dist
import deepspeed
# Reuse our already-built, unmodified CPUAdam extension. The old experiment's
# JIT cache has a stale lock; do not delete shared cache files or rebuild it.
if os.environ.get('CPU_ADAM_EXTENSION'):
 from deepspeed.ops.adam.cpu_adam import CPUAdamBuilder
 spec=importlib.util.spec_from_file_location('cpu_adam',os.environ['CPU_ADAM_EXTENSION'])
 cpu_adam_extension=importlib.util.module_from_spec(spec);spec.loader.exec_module(cpu_adam_extension)
 CPUAdamBuilder.load=lambda self,verbose=True:cpu_adam_extension
torch.set_num_threads(len(chosen));torch.npu.set_device(local)
torch.npu.config.allow_internal_format=False
torch.npu.matmul.allow_hf32=False
torch.manual_seed(20261024)
deepspeed.init_distributed(dist_backend='hccl',timeout=datetime.timedelta(seconds=300))
out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
def log(event,**kw):
 row={'event':event,'rank':rank,'unix':time.time(),**kw}
 with (out/f'rank{rank}.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
 print(json.dumps(row),flush=True)
cfg=json.loads(Path(a.config).read_text())
log('begin',affinity=chosen,task_queue_enable=os.environ.get('TASK_QUEUE_ENABLE'),tiny=a.tiny)
with deepspeed.zero.Init(config_dict_or_path=cfg,dtype=torch.bfloat16):
 if a.tiny:
  model=torch.nn.Sequential(torch.nn.Linear(128,256,bias=False),torch.nn.GELU(),torch.nn.Linear(256,128,bias=False))
 else:
  from transformers import AutoConfig
  from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM
  config=AutoConfig.from_pretrained(a.model,local_files_only=True).text_config
  config.use_cache=False;config._attn_implementation='eager'
  model=Qwen3_5ForCausalLM(config)
params=sum(p.ds_numel for p in model.parameters())
if not a.tiny:
 from safetensors import safe_open
 mapping=json.loads((Path(a.model)/'model.safetensors.index.json').read_text())['weight_map']
 for i,(name,param) in enumerate(model.named_parameters()):
  key=name.replace('model.','model.language_model.',1) if name.startswith('model.') else name
  assert key in mapping,key
  with deepspeed.zero.GatheredParameters([param],modifier_rank=0):
   if rank==0:
    with safe_open(str(Path(a.model)/mapping[key]),framework='pt',device='cpu') as f:
     weight=f.get_tensor(key);assert tuple(weight.shape)==tuple(param.shape)
     param.data.copy_(weight);del weight
  if i%64==0:log('load_progress',tensor_count=i+1)
 assert params==26895998464,params
 model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
log('model_ready',parameters=params)
engine,optimizer,_,_=deepspeed.initialize(model=model,model_parameters=model.parameters(),config=cfg)
log('engine_ready',master_dtypes=sorted(set(str(p.dtype) for p in optimizer.fp32_partitioned_groups_flat)),
    master_devices=sorted(set(str(p.device) for p in optimizer.fp32_partitioned_groups_flat)))
assert all(p.dtype==torch.float32 and p.device.type=='cpu' for p in optimizer.fp32_partitioned_groups_flat)
if not a.tiny:
 from transformers import AutoTokenizer
 tok=AutoTokenizer.from_pretrained(a.model,local_files_only=True)
 def examples(filename):
  rows=[]
  for line in (Path(a.data)/filename).read_text().splitlines():
   messages=json.loads(line)['messages']
   ids=tok.apply_chat_template(messages,tokenize=True,enable_thinking=False,return_dict=False)
   prefix=tok.apply_chat_template(messages[:-1],tokenize=True,add_generation_prompt=True,enable_thinking=False,return_dict=False)
   if len(ids)>128 or ids[:len(prefix)]!=prefix or len(ids)<=len(prefix):continue
   digest=hashlib.sha256(json.dumps(messages,ensure_ascii=False,sort_keys=True).encode()).hexdigest()
   rows.append((len(ids),digest,ids,[-100]*len(prefix)+ids[len(prefix):]))
  return sorted(rows,key=lambda r:(r[0],r[1]))
 train=examples('train.jsonl');val=examples('validation.jsonl')
 assert len(train)>=a.steps*world
 assert not set(r[1] for r in train)&set(r[1] for r in val)
 log('data',training_hashes=[r[1] for r in train[:a.steps*world]],validation_hashes=[r[1] for r in val])
def optimizer_state_check():
 states=list(optimizer.optimizer.state.values())
 tensors=[s[k] for s in states for k in ['exp_avg','exp_avg_sq'] if k in s]
 assert tensors and all(t.dtype==torch.float32 and t.device.type=='cpu' for t in tensors)
 assert all(bool(torch.isfinite(t.view(-1)[:16]).all()) for t in tensors)
 return sum(t.numel()*t.element_size() for t in tensors)
def evaluate():
 engine.eval(); totals=torch.zeros(3,device=f'npu:{local}',dtype=torch.float32)
 with torch.no_grad():
  for start in range(0,8,world):
   row=val[start+rank]
   batch={k:torch.tensor([v],device=f'npu:{local}') for k,v in [('input_ids',row[2]),('labels',row[3])]}
   output=engine(**batch,use_cache=False)
   labels=batch['labels'][:,1:];mask=labels!=-100;n=mask.sum()
   totals[0]+=output.loss.float()*n;totals[1]+=n
   totals[2]+=((output.logits[:,:-1].argmax(-1)==labels)&mask).sum()
   del output,batch
 dist.all_reduce(totals)
 result={'loss':float(totals[0]/totals[1]),'assistant_tokens':int(totals[1]),'correct':int(totals[2]),'execution':'native BF16 model'}
 assert math.isfinite(result['loss']);return result
before=None
if not a.tiny:
 before=evaluate();log('validation_before',metrics=before)
steps=[]
for step in range(a.steps):
 engine.train();dist.barrier();torch.npu.synchronize();torch.npu.reset_peak_memory_stats()
 started=time.monotonic()
 if a.tiny:
  torch.manual_seed(100+step*world+rank)
  x=torch.randn(1,16,128,device=f'npu:{local}',dtype=torch.bfloat16)
  loss=engine(x).float().square().mean();tokens=16
 else:
  row=train[step*world+rank];tokens=row[0]
  batch={k:torch.tensor([v],device=f'npu:{local}') for k,v in [('input_ids',row[2]),('labels',row[3])]}
  loss=engine(**batch,use_cache=False).loss
  # Normalize by the global number of supervised tokens, not equal rank means.
  local_count=(batch['labels'][:,1:]!=-100).sum()
  total_count=local_count.clone();dist.all_reduce(total_count)
  loss=loss*(local_count*world/total_count)
 value=float(loss.detach());assert math.isfinite(value)
 engine.backward(loss);engine.step();torch.npu.synchronize();dist.barrier()
 seconds=time.monotonic()-started
 assert not optimizer.overflow
 state_bytes=optimizer_state_check()
 row={'step':step+1,'seconds':seconds,'weighted_local_loss':value,'input_tokens':tokens,
      'peak_allocated':torch.npu.max_memory_allocated(),'peak_reserved':torch.npu.max_memory_reserved(),
      'fp32_m_v_bytes':state_bytes,'global_batch_size':world}
 steps.append(row);log('step_complete',**row)
after=None
if not a.tiny:
 after=evaluate();log('validation_after',metrics=after)
result={'completed':True,'rank':rank,'steps':steps,'parameters':params,'fp32_master':True,'fp32_m_v':True,
        'validation_before':before,'validation_after':after,
        'state_cpu_offload':True,'checkpoint_saved':False,'tiny':a.tiny}
(out/f'rank{rank}-result.json').write_text(json.dumps(result,indent=2))
log('complete');dist.barrier();dist.destroy_process_group()
