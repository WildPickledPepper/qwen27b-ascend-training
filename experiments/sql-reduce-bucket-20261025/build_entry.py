"""Flush an oversized singleton as soon as ready; preserve native packet groups."""
from pathlib import Path
r=Path(__file__).resolve().parent;b=r.parent/'sql-boundary-overlap-20261025'
for name in ['cpu_accumulation.py','community_gdn.py','cpu_update.py','bounded_grad_copy.py','zero3-sql.json','gradient_scopes.py','gradient_cache.py','compare_probes.py','export_evidence.py','progress.py','analyze_profile.py','analyze_gradient.py','communication_overlap.py']:
 (r/name).write_text((b/name).read_text(encoding='utf-8'),encoding='utf-8',newline='\n')
s=(b/'train_boundary.py').read_text(encoding='utf-8')
s=s.replace("p.add_argument('--cpu-gradient-accum',action='store_true')", "p.add_argument('--cpu-gradient-accum',action='store_true')\np.add_argument('--ready-reduce',action='store_true')")
anchor="log('begin',affinity=chosen"
# Force the oversized path for both tiny-control and tiny-candidate probes.
assert s.count(anchor)==1;s=s.replace(anchor,"if a.tiny:cfg['zero_optimization']['reduce_bucket_size']=4096\n"+anchor)
anchor='from gradient_scopes import install as install_gradient_scopes'
code="""ready_plan=None;ready_stats={}
if a.ready_reduce:
 from ready_reduce import install as install_ready_reduce
 ready_plan,ready_stats=install_ready_reduce(optimizer)
 log('ready_reduce',**ready_plan)
"""
assert s.count(anchor)==1;s=s.replace(anchor,code+anchor)
s=s.replace('cpu_stats.clear();gradient_stats.clear();cpu_grad_stats.clear()','cpu_stats.clear();gradient_stats.clear();cpu_grad_stats.clear();ready_stats.clear()')
s=s.replace("row['cpu_gradient_accumulation_stats']=dict(cpu_grad_stats)","row['cpu_gradient_accumulation_stats']=dict(cpu_grad_stats)\n row['ready_reduce_stats']=dict(ready_stats)")
s=s.replace("'reduce_schedule':reduce_plan}","'reduce_schedule':reduce_plan,'ready_reduce':ready_plan}")
(r/'train_ready.py').write_text(s,encoding='utf-8',newline='\n')
s=(b/'launch.private.sh').read_text(encoding='utf-8').replace('qwen-sql-boundary-overlap-20261025','qwen-sql-reduce-bucket-20261025') if (b/'launch.private.sh').exists() else None
if s:(r/'launch.private.sh').write_text(s,encoding='utf-8',newline='\n')
print('Generated singleton immediate-reduction candidate; no bucket size change in full model')
