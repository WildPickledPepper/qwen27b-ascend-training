from pathlib import Path
r=Path(__file__).resolve().parent;b=r.parent/'sql-gradient-cache-20261025'
s=(b/'train_cache.py').read_text(encoding='utf-8')
s=s.replace("p.add_argument('--gradient-cache-gib',type=float,default=0)","p.add_argument('--gradient-cache-gib',type=float,default=0)\np.add_argument('--cpu-gradient-accum',action='store_true')")
anchor='from gradient_scopes import install as install_gradient_scopes'
s=s.replace(anchor,"cpu_grad_plan=None;cpu_grad_stats={}\nif a.cpu_gradient_accum:\n from cpu_accumulation import install as install_cpu_accumulation\n cpu_grad_plan,cpu_grad_stats=install_cpu_accumulation(optimizer)\n log('cpu_gradient_accumulation',**cpu_grad_plan)\n"+anchor)
s=s.replace('cpu_stats.clear();gradient_stats.clear()','cpu_stats.clear();gradient_stats.clear();cpu_grad_stats.clear()')
s=s.replace(" row['gradient_observer_stats']=dict(gradient_stats)"," row['gradient_observer_stats']=dict(gradient_stats)\n row['cpu_gradient_accumulation_stats']=dict(cpu_grad_stats)")
s=s.replace("'gradient_cache':cache_plan}","'gradient_cache':cache_plan,'cpu_gradient_accumulation':cpu_grad_plan}")
s=s.replace("log('complete');dist.barrier();dist.destroy_process_group()","if a.cpu_gradient_accum:optimizer._cpu_gradient_flush()\nlog('complete');dist.barrier();dist.destroy_process_group()")
(r/'train_cpu.py').write_text(s,encoding='utf-8',newline='\n')
for name in ['community_gdn.py','cpu_update.py','bounded_grad_copy.py','zero3-sql.json','gradient_scopes.py','gradient_cache.py','compare_probes.py','export_evidence.py']:
 text=(b/name).read_text(encoding='utf-8')
 if name=='export_evidence.py':text=text.replace("'cache12-full'","'cpu-full'")
 (r/name).write_text(text,encoding='utf-8',newline='\n')
# A separate controller records native-reference reuse and all new run PIDs.
s=(b/'run_experiment.py').read_text(encoding='utf-8').replace('train_cache.py','train_cpu.py').replace("'--cpu-mode','pipeline']","'--cpu-mode','pipeline','--cpu-gradient-accum']")
s=s.replace("for name,cache in [('tiny-native','0'),('tiny-cache','12')]:","for name,cache in [('tiny-cache','0')]:")
s=s.replace("'cache12-full'","'cpu-full'").replace("'--gradient-cache-gib','12'","'--gradient-cache-gib','0'")
s=s.replace("# First three updates", "reference=r.parent/'qwen-sql-gradient-cache-20261025'/'tiny-native'\n assert reference.exists()\n (r/'tiny-native').symlink_to(reference,target_is_directory=True)\n # First three updates")
# The reference symlink must precede compare_probes, not the full run.
s=s.replace(" run('compare-probes',[sys.executable,'compare_probes.py'],90)\n "," ")
s=s.replace(" (r/'tiny-native').symlink_to(reference,target_is_directory=True)\n", " (r/'tiny-native').symlink_to(reference,target_is_directory=True)\n run('compare-probes',[sys.executable,'compare_probes.py'],90)\n")
(r/'run_experiment.py').write_text(s,encoding='utf-8',newline='\n')
print('Generated CPU accumulation entry')
