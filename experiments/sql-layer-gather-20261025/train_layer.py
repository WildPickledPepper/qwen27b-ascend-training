"""Native two-process ZeRO-3; original weights read-only, aggregate evidence only."""
import argparse,datetime,faulthandler,hashlib,importlib.util,json,math,os,time
from pathlib import Path
faulthandler.enable()
p=argparse.ArgumentParser()
p.add_argument('--config',default='zero3.json');p.add_argument('--output',required=True)
p.add_argument('--model');p.add_argument('--data');p.add_argument('--tiny',action='store_true')
p.add_argument('--steps',type=int,default=3)
p.add_argument('--pad-mode',choices=['dynamic64','fixed512'],default='dynamic64')
p.add_argument('--micro-batch',type=int,default=1)
p.add_argument('--global-batch',type=int,default=32)
p.add_argument('--checkpoint',choices=['on','off','selective'],default='on')
p.add_argument('--cpu-threads',type=int,default=12)
p.add_argument('--profile',action='store_true')
p.add_argument('--cpu-mode',choices=['observe','skip','pipeline'],default='observe')
p.add_argument('--fuse-gdn',action='store_true')
p.add_argument('--numa-spread',action='store_true')
p.add_argument('--gradient-cache-gib',type=float,default=0)
p.add_argument('--cpu-gradient-accum',action='store_true')
a=p.parse_args()
if a.profile:os.environ['HCCL_EXEC_TIMEOUT']='600'
rank=int(os.environ['RANK']); local=int(os.environ['LOCAL_RANK']); world=int(os.environ['WORLD_SIZE'])
assert world in [1,2] and os.environ['ASCEND_RT_VISIBLE_DEVICES']=='2,3'
assert a.global_batch%(world*a.micro_batch)==0
micros=a.global_batch//(world*a.micro_batch)
allowed=sorted(os.sched_getaffinity(0)); chosen=[i for i in (range(96,96+a.cpu_threads) if world==1 else range(96+local*12,96+local*12+a.cpu_threads)) if i in allowed]
if a.numa_spread:
 import ctypes
 chosen=[i for i in range(96+local*24,96+local*24+a.cpu_threads) if i in allowed]
 libnuma=ctypes.CDLL('libnuma.so.1',use_errno=True)
 mask=ctypes.c_ulong(1<<(4+local))
 libnuma.set_mempolicy.argtypes=[ctypes.c_int,ctypes.POINTER(ctypes.c_ulong),ctypes.c_ulong]
 assert libnuma.set_mempolicy(1,ctypes.byref(mask),8)==0,('NUMA preferred policy unavailable',ctypes.get_errno())
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
 if a.profile:
  real_adam_update=cpu_adam_extension.adam_update
  def profiled_adam_update(*args,**kwargs):
   with torch.profiler.record_function('optimizer/CPUAdam'):
    return real_adam_update(*args,**kwargs)
  cpu_adam_extension.adam_update=profiled_adam_update
 CPUAdamBuilder.load=lambda self,verbose=True:cpu_adam_extension
from bounded_grad_copy import install as install_bounded_grad_copy
install_bounded_grad_copy()
torch.set_num_threads(len(chosen));torch.npu.set_device(local)
torch.npu.config.allow_internal_format=False
torch.npu.matmul.allow_hf32=False
torch.manual_seed(42)
deepspeed.init_distributed(dist_backend='hccl',timeout=datetime.timedelta(seconds=300))
out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
def log(event,**kw):
 row={'event':event,'rank':rank,'unix':time.time(),**kw}
 with (out/f'rank{rank}.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
 print(json.dumps(row),flush=True)
cfg=json.loads(Path(a.config).read_text())
cfg['gradient_accumulation_steps']=micros
cfg['train_micro_batch_size_per_gpu']=a.micro_batch
cfg['train_batch_size']=a.global_batch
log('begin',affinity=chosen,task_queue_enable=os.environ.get('TASK_QUEUE_ENABLE'),tiny=a.tiny,micro_batch=a.micro_batch,global_batch=a.global_batch,accumulation=micros,checkpoint=a.checkpoint,numa_spread=a.numa_spread)
if a.fuse_gdn:
 from community_gdn import chunk_gated_delta_rule as fused_training_gdn
 from transformers.models.qwen3_5 import modeling_qwen3_5
 modeling_qwen3_5.chunk_gated_delta_rule=fused_training_gdn
 log('fusion_enabled',backend='VeOmni pinned Triton-Ascend forward and backward')
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
 if a.checkpoint!='off':
  model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
  if a.checkpoint=='selective':
   for i,layer in enumerate(model.model.layers):
    if i%4==3:layer.gradient_checkpointing=False
 log('checkpoint_policy',layers=len(model.model.layers),enabled_layers=sum(bool(getattr(layer,'gradient_checkpointing',False)) for layer in model.model.layers))
log('model_ready',parameters=params)

if a.tiny:
 grouped=[model];expected=1
else:
 grouped=list(model.model.layers);expected=64
assert len(grouped)==expected
layer_sizes=[sum(p.ds_numel for p in module.parameters()) for module in grouped]
largest=max(layer_sizes)
assert 2*largest<cfg['zero_optimization']['stage3_max_live_parameters']
previous_prefetch=cfg['zero_optimization']['stage3_prefetch_bucket_size']
cfg['zero_optimization']['stage3_prefetch_bucket_size']=largest
layer_plan={'layer_class':type(grouped[0]).__name__,'layers':len(grouped),'leaf_modules':0,'native_gradient_hooks_preserved':True,'largest_layer_parameters':largest,'smallest_layer_parameters':min(layer_sizes),'old_prefetch_elements':previous_prefetch,'prefetch_elements':largest,'native_config_key':'stage3_prefetch_bucket_size','zero_quantized_weights':False}
log('layer_schedule',**layer_plan)

engine,optimizer,_,_=deepspeed.initialize(model=model,model_parameters=model.parameters(),config=cfg)
log('engine_ready',master_dtypes=sorted(set(str(p.dtype) for p in optimizer.fp32_partitioned_groups_flat)),
    master_devices=sorted(set(str(p.device) for p in optimizer.fp32_partitioned_groups_flat)))
assert all(p.dtype==torch.float32 and p.device.type=='cpu' for p in optimizer.fp32_partitioned_groups_flat)
from cpu_update import install as install_cpu_update
cpu_stats=install_cpu_update(optimizer,skip_identity=a.cpu_mode!='observe',pipeline=a.cpu_mode=='pipeline')
log('cpu_update_policy',mode=a.cpu_mode,loss_scale=optimizer.loss_scale,clip_grad=optimizer.clip_grad)
from gradient_cache import install as install_gradient_cache
cache_plan=install_gradient_cache(optimizer,a.gradient_cache_gib)
log('gradient_cache',**cache_plan)
plans=[None]*world;dist.all_gather_object(plans,cache_plan['plan_sha256']);assert len(set(plans))==1
cpu_grad_plan=None;cpu_grad_stats={}
if a.cpu_gradient_accum:
 from cpu_accumulation import install as install_cpu_accumulation
 cpu_grad_plan,cpu_grad_stats=install_cpu_accumulation(optimizer)
 log('cpu_gradient_accumulation',**cpu_grad_plan)
from gradient_scopes import install as install_gradient_scopes
gradient_stats=install_gradient_scopes(optimizer)
if not a.tiny:
 from transformers import AutoTokenizer
 tok=AutoTokenizer.from_pretrained(a.model,local_files_only=True)
 def examples(filename):
  rows=[]
  for line in (Path(a.data)/filename).read_text().splitlines():
   row=json.loads(line)
   assert 0<row['length']<=512 and len(row['ids'])==len(row['labels'])==row['length']
   rows.append((row['length'],row['hash'],row['ids'],row['labels']))
  return rows
 train=examples('train.jsonl');val=examples('validation.jsonl')
 assert len(train)>=16
 assert not set(r[1] for r in train)&set(r[1] for r in val)
 log('data',training_hashes=[train[i%len(train)][1] for i in range(a.steps*a.global_batch)],training_lengths=[train[i%len(train)][0] for i in range(a.steps*a.global_batch)],training_corpus_size=len(train),repeats=(a.steps*a.global_batch)>len(train),validation_hashes=[r[1] for r in val])
def make_batch(rows):
 size=512 if a.pad_mode=='fixed512' else max(128,math.ceil(max(row[0] for row in rows)/64)*64)
 pad=tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
 ids=[row[2]+[pad]*(size-row[0]) for row in rows]
 labels=[row[3]+[-100]*(size-row[0]) for row in rows]
 mask=[[1]*row[0]+[0]*(size-row[0]) for row in rows]
 return {k:torch.tensor(v,device=f'npu:{local}') for k,v in [('input_ids',ids),('labels',labels),('attention_mask',mask)]}

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
prof=None
if a.profile and rank==0:
 prof=torch_npu.profiler.profile(schedule=torch_npu.profiler.schedule(wait=0,warmup=0,active=1,repeat=1),activities=[torch_npu.profiler.ProfilerActivity.CPU,torch_npu.profiler.ProfilerActivity.NPU],
  record_shapes=False,profile_memory=False,with_stack=False,
  experimental_config=torch_npu.profiler._ExperimentalConfig(profiler_level=torch_npu.profiler.ProfilerLevel.Level1,export_type=[torch_npu.profiler.ExportType.Db]),
  on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(str(out/'profile')))
for step in range(a.steps):
 engine.train();dist.barrier();torch.npu.synchronize();torch.npu.reset_peak_memory_stats()
 if prof and step==a.steps-1:prof.start()
 started=time.monotonic()
 # Match the author's 698-update cosine schedule with 5% warmup (35 steps).
 warmup=35;total_updates=698
 progress=step*a.global_batch/256
 factor=progress/warmup if progress<warmup else .5*(1+math.cos(math.pi*(progress-warmup)/(total_updates-warmup)))
 for group in optimizer.optimizer.param_groups:group['lr']=2e-5*factor
 log('learning_rate',step=step+1,learning_rate=2e-5*factor)
 cpu_stats.clear();gradient_stats.clear();cpu_grad_stats.clear()
 phase={'forward_wall':0.,'backward_wall':0.,'step_wall':0.}
 if not a.tiny:
  selected=[[train[(step*a.global_batch+(micro*a.micro_batch+j)*world+rank)%len(train)] for j in range(a.micro_batch)] for micro in range(micros)]
  total_count=torch.tensor(sum(sum(y!=-100 for y in row[3][1:]) for group in selected for row in group),device=f'npu:{local}')
  dist.all_reduce(total_count)
 tokens=0; padded_tokens=0;value=0.
 for micro in range(micros):
  phase_started=time.monotonic()
  if a.tiny:
   torch.manual_seed(100+step*2+micro*world+rank)
   x=torch.randn(1,16,128,device=f'npu:{local}',dtype=torch.bfloat16)
   loss=engine(x).float().square().mean();tokens+=16
  else:
   group=selected[micro];tokens+=sum(row[0] for row in group)
   batch=make_batch(group);padded_tokens+=batch['input_ids'].numel()
   loss=engine(**batch,use_cache=False).loss
   local_count=(batch['labels'][:,1:]!=-100).sum()
   loss=loss*(local_count*a.global_batch/a.micro_batch/total_count)
  local_value=float(loss.detach());assert math.isfinite(local_value);value+=local_value/micros
  phase['forward_wall']+=time.monotonic()-phase_started
  phase_started=time.monotonic()
  engine.backward(loss)
  phase['backward_wall']+=time.monotonic()-phase_started
  phase_started=time.monotonic()
  if a.tiny and engine.is_gradient_accumulation_boundary():
   torch.npu.synchronize()
   torch.save([p.grad.detach().cpu().clone() for p in optimizer.fp32_partitioned_groups_flat],out/f'probe-gradient-rank{rank}-step{step}.private.pt')
  engine.step()
  phase['step_wall']+=time.monotonic()-phase_started
  log('micro_complete',step=step+1,micro=micro+1,total_micros=micros)
 torch.npu.synchronize();dist.barrier()
 seconds=time.monotonic()-started
 if prof and step==a.steps-1:prof.stop()
 assert not optimizer.overflow
 state_bytes=optimizer_state_check()
 row={'step':step+1,'seconds':seconds,'weighted_local_loss':value,'input_tokens':tokens,
      'peak_allocated':torch.npu.max_memory_allocated(),'peak_reserved':torch.npu.max_memory_reserved(),
      'fp32_m_v_bytes':state_bytes,'global_batch_size':a.global_batch,'micro_batch_size':a.micro_batch,'accumulation':micros,'phase_wall_seconds':phase,'padded_input_tokens':padded_tokens,'profiled':bool(a.profile and step==a.steps-1)}
 row['cpu_update_stats']=dict(cpu_stats)
 row['gradient_observer_stats']=dict(gradient_stats)
 row['cpu_gradient_accumulation_stats']=dict(cpu_grad_stats)
 if a.tiny:
  torch.save({'master':[p.detach().clone() for p in optimizer.fp32_partitioned_groups_flat],'state':[[v[k].detach().clone() for k in ['exp_avg','exp_avg_sq']] for v in optimizer.optimizer.state.values()]},out/f'probe-state-rank{rank}-step{step}.private.pt')
 steps.append(row);log('step_complete',**row)
 if not a.tiny and (step+1)*a.global_batch in [256,512,768]:
  log('sample_checkpoint',samples=(step+1)*a.global_batch,training_seconds=sum(x['seconds'] for x in steps),metrics=evaluate())
 if a.profile and step==1:
  log('validation_after2',metrics=evaluate())
 if step==a.steps-1:
  import re
  numa_pages={}
  for line in Path('/proc/self/numa_maps').read_text().splitlines():
   if 'anon=' not in line and 'heap ' not in line:continue
   for node,n in re.findall(r' N(\d+)=(\d+)',line):numa_pages[node]=numa_pages.get(node,0)+int(n)
  log('numa_placement',anonymous_resident_gib={node:pages*os.sysconf('SC_PAGE_SIZE')/2**30 for node,pages in numa_pages.items()})
after=None
if not a.tiny:
 after=evaluate();log('validation_after',metrics=after)
result={'completed':True,'rank':rank,'steps':steps,'parameters':params,'fp32_master':True,'fp32_m_v':True,
        'validation_before':before,'validation_after':after,
        'benchmark':'sql-batch-tuning-fixed-768-samples','lr_schedule_basis':'processed samples, 256-sample reference update','pad_mode':a.pad_mode,'max_sequence_length':512,'seed':42,
        'cpu_mode':a.cpu_mode,'fuse_gdn':a.fuse_gdn,'numa_spread':a.numa_spread,'config':cfg,'micro_batch':a.micro_batch,'global_batch':a.global_batch,'checkpoint':a.checkpoint,'state_cpu_offload':True,'checkpoint_saved':False,'tiny':a.tiny,'gradient_cache':cache_plan,'cpu_gradient_accumulation':cpu_grad_plan,'layer_schedule':layer_plan}
(out/f'rank{rank}-result.json').write_text(json.dumps(result,indent=2))
if a.cpu_gradient_accum:optimizer._cpu_gradient_flush()
log('complete');dist.barrier();dist.destroy_process_group()
