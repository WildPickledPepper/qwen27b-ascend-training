"""Derive an isolated batching harness from the completed ZeRO-3 experiment."""
from pathlib import Path
r=Path(__file__).resolve().parent
s=(r.parent/'zero3-offload-20261025/train_zero3_bounded.py').read_text()
s=s.replace("p.add_argument('--steps',type=int,default=4)", """p.add_argument('--steps',type=int,default=3)
p.add_argument('--micro-batch',type=int,default=1)
p.add_argument('--global-batch',type=int,default=32)
p.add_argument('--checkpoint',choices=['on','off'],default='on')
p.add_argument('--cpu-threads',type=int,default=12)
p.add_argument('--profile',action='store_true')""")
s=s.replace('micros=2//world', "assert a.global_batch%(world*a.micro_batch)==0\nmicros=a.global_batch//(world*a.micro_batch)")
s=s.replace("range(96,120) if world==1 else range(96+local*12,108+local*12)", "range(96,96+a.cpu_threads) if world==1 else range(96+local*12,96+local*12+a.cpu_threads)")
s=s.replace("cfg['gradient_accumulation_steps']=micros", """cfg['gradient_accumulation_steps']=micros
cfg['train_micro_batch_size_per_gpu']=a.micro_batch
cfg['train_batch_size']=a.global_batch""")
s=s.replace("tiny=a.tiny)", "tiny=a.tiny,micro_batch=a.micro_batch,global_batch=a.global_batch,accumulation=micros,checkpoint=a.checkpoint)",1)
s=s.replace(" model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})", " if a.checkpoint=='on':model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})")
s=s.replace(" assert len(train)>=a.steps*2", " assert len(train)>=16")
s=s.replace("training_hashes=[r[1] for r in train[:a.steps*2]]", "training_hashes=[train[i%len(train)][1] for i in range(a.steps*a.global_batch)],training_lengths=[train[i%len(train)][0] for i in range(a.steps*a.global_batch)],training_corpus_size=len(train),repeats=(a.steps*a.global_batch)>len(train)")
marker='def optimizer_state_check():'
insert='''def make_batch(rows):
 size=max(row[0] for row in rows)
 pad=tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
 ids=[row[2]+[pad]*(size-row[0]) for row in rows]
 labels=[row[3]+[-100]*(size-row[0]) for row in rows]
 mask=[[1]*row[0]+[0]*(size-row[0]) for row in rows]
 return {k:torch.tensor(v,device=f'npu:{local}') for k,v in [('input_ids',ids),('labels',labels),('attention_mask',mask)]}

'''
s=s.replace(marker,insert+marker)
s=s.replace('steps=[]\nfor step', '''steps=[]
prof=None
if a.profile and rank==0:
 prof=torch_npu.profiler.profile(activities=[torch_npu.profiler.ProfilerActivity.CPU,torch_npu.profiler.ProfilerActivity.NPU],
  record_shapes=False,profile_memory=False,with_stack=False,
  experimental_config=torch_npu.profiler._ExperimentalConfig(profiler_level=torch_npu.profiler.ProfilerLevel.Level1,export_type=[torch_npu.profiler.ExportType.Db]),
  on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(str(out/'profile')))
for step''')
s=s.replace(' started=time.monotonic()\n if not a.tiny:', " if prof and step==a.steps-1:prof.start()\n started=time.monotonic()\n phase={'forward_wall':0.,'backward_wall':0.,'step_wall':0.}\n if not a.tiny:")
s=s.replace("  selected=[train[step*2+micro*world+rank] for micro in range(micros)]", "  selected=[[train[(step*a.global_batch+(micro*a.micro_batch+j)*world+rank)%len(train)] for j in range(a.micro_batch)] for micro in range(micros)]")
s=s.replace("sum(sum(y!=-100 for y in row[3][1:]) for row in selected)", "sum(sum(y!=-100 for y in row[3][1:]) for group in selected for row in group)")
s=s.replace(' tokens=0; value=0.', ' tokens=0; padded_tokens=0;value=0.')
s=s.replace(' for micro in range(micros):\n  if a.tiny:', " for micro in range(micros):\n  phase_started=time.monotonic()\n  if a.tiny:")
s=s.replace('   row=selected[micro];tokens+=row[0]\n   batch={k:torch.tensor([v],device=f\'npu:{local}\') for k,v in [(\'input_ids\',row[2]),(\'labels\',row[3])]}', "   group=selected[micro];tokens+=sum(row[0] for row in group)\n   batch=make_batch(group);padded_tokens+=batch['input_ids'].numel()")
s=s.replace('loss=loss*(local_count*2/total_count)', 'loss=loss*(local_count*a.global_batch/a.micro_batch/total_count)')
s=s.replace('  engine.backward(loss);engine.step()', """  phase['forward_wall']+=time.monotonic()-phase_started
  phase_started=time.monotonic()
  engine.backward(loss)
  phase['backward_wall']+=time.monotonic()-phase_started
  phase_started=time.monotonic()
  engine.step()
  phase['step_wall']+=time.monotonic()-phase_started""")
s=s.replace(' seconds=time.monotonic()-started', ' seconds=time.monotonic()-started\n if prof and step==a.steps-1:prof.stop()')
s=s.replace("'global_batch_size':2", "'global_batch_size':a.global_batch,'micro_batch_size':a.micro_batch,'accumulation':micros,'phase_wall_seconds':phase,'padded_input_tokens':padded_tokens,'profiled':bool(a.profile and step==a.steps-1)")
s=s.replace("'state_cpu_offload':True", "'config':cfg,'micro_batch':a.micro_batch,'global_batch':a.global_batch,'checkpoint':a.checkpoint,'state_cpu_offload':True")
# Existing normalisation multiplies by world*accumulation = global_batch/micro_batch.
# DeepSpeed divides by accumulation and averages ranks; the global token mean follows.
assert 'selected=[train[step*2' not in s and 'local_count*2' not in s
(r/'train_throughput.py').write_text(s,encoding='utf-8',newline='\n')
import ast
ast.parse(s)
print('Built isolated batch/throughput harness')
