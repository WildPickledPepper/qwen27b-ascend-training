"""Bounded extra reduction lookahead; all collective/math and batch settings unchanged."""
from pathlib import Path
r=Path(__file__).resolve().parent;b=r.parent/'sql-gradient-pipeline-20261025'
for name in ['cpu_accumulation.py','community_gdn.py','cpu_update.py','bounded_grad_copy.py','zero3-sql.json','gradient_scopes.py','gradient_cache.py','compare_probes.py','export_evidence.py','status.py']:
 (r/name).write_text((b/name).read_text(encoding='utf-8'),encoding='utf-8',newline='\n')
s=(b/'train_cpu.py').read_text(encoding='utf-8')
anchor='from gradient_scopes import install as install_gradient_scopes'
code="""from deepspeed.accelerator import get_accelerator
assert not get_accelerator().handles_memory_backpressure()
assert optimizer.max_param_reduce_events==2
optimizer.max_param_reduce_events=4
reduce_plan={'previous_event_limit':2,'event_limit':4,'native_condition':'len(events) > limit','maximum_queued_events':5,'native_collectives_and_math_preserved':True,'handles_memory_backpressure':False}
log('reduce_schedule',**reduce_plan)
"""
assert s.count(anchor)==1;s=s.replace(anchor,code+anchor)
s=s.replace("'cpu_gradient_accumulation':cpu_grad_plan}","'cpu_gradient_accumulation':cpu_grad_plan,'reduce_schedule':reduce_plan}")
(r/'train_reduce.py').write_text(s,encoding='utf-8',newline='\n')
s=(b/'run_experiment.py').read_text(encoding='utf-8').replace('train_cpu.py','train_reduce.py').replace("'waiting-for-prior-cpu-run'","'waiting-for-prior-prefetch-run'").replace("prior=r.parent/'qwen-sql-gradient-cpu-20261025'","prior=r.parent/'qwen-sql-layer-gather-20261025'")
(r/'run_experiment.py').write_text(s,encoding='utf-8',newline='\n')
s=(b/'launch.private.sh').read_text(encoding='utf-8').replace('qwen-sql-gradient-pipeline-20261025','qwen-sql-reduce-schedule-20261025')
(r/'launch.private.sh').write_text(s,encoding='utf-8',newline='\n')
s=(b/'collect_result.py').read_text(encoding='utf-8').replace('sql-gradient-pipeline-results.json','sql-reduce-schedule-results.json').replace('cpu_accumulation_seconds_per_256','scheduled_seconds_per_256')
s=s.replace("assert result['gradient_cache']['resident_bytes']==0", "assert result['gradient_cache']['resident_bytes']==0\n assert result['reduce_schedule']['event_limit']==4 and result['reduce_schedule']['previous_event_limit']==2")
s=s.replace("'cpu_accumulation_plan':x['cpu_gradient_accumulation'],", "'cpu_accumulation_plan':x['cpu_gradient_accumulation'],'reduce_schedule':x['reduce_schedule'],")
(r/'collect_result.py').write_text(s,encoding='utf-8',newline='\n')
print('Generated reducer limit 2->4 with bounded CPU receive pipeline')
