from pathlib import Path
r=Path(__file__).resolve().parent;b=r.parent/'sql-boundary-overlap-20261025'
for name in ['cpu_accumulation.py','community_gdn.py','bounded_grad_copy.py','zero3-sql.json','gradient_scopes.py','gradient_cache.py','compare_probes.py','export_evidence.py','progress.py','analyze_profile.py','analyze_gradient.py','communication_overlap.py','analyze_boundary.py']:
 text=(b/name).read_text(encoding='utf-8')
 if name=='compare_probes.py':text=text.replace('torch.equal(x,y)','torch.equal(x.contiguous().view(torch.uint8),y.contiguous().view(torch.uint8))')
 (r/name).write_text(text,encoding='utf-8',newline='\n')
(r/'cpu_update_reference.py').write_text((b/'cpu_update.py').read_text(encoding='utf-8'),encoding='utf-8',newline='\n')
s=(b/'train_boundary.py').read_text(encoding='utf-8')
s=s.replace("p.add_argument('--cpu-gradient-accum',action='store_true')", "p.add_argument('--cpu-gradient-accum',action='store_true')\np.add_argument('--fused-cpu-update',action='store_true')")
s=s.replace('from cpu_update import install as install_cpu_update', "from cpu_update_reference import install as install_cpu_update\nif a.fused_cpu_update:\n from cpu_update_fused import install as install_cpu_update")
s=s.replace("log('cpu_update_policy',", "if a.fused_cpu_update:log('cpu_fusion',**optimizer._cpu_fusion_plan)\nlog('cpu_update_policy',")
s=s.replace("'reduce_schedule':reduce_plan}", "'reduce_schedule':reduce_plan,'cpu_fusion':getattr(optimizer,'_cpu_fusion_plan',None)}")
(r/'train_fused.py').write_text(s,encoding='utf-8',newline='\n')
s=(b/'run_experiment.py').read_text(encoding='utf-8').replace('train_boundary.py','train_fused.py')
start=s.index(" for name,cache in [('tiny-cache','0')]:");end=s.index(" run('compare-probes'",start)
s=s[:start]+" for name,flag in [('tiny-native',[]),('tiny-cache',['--fused-cpu-update'])]:\n  run(name,base+['--tiny','--steps','3','--global-batch','16','--micro-batch','1','--output',name,'--gradient-cache-gib','0']+flag,300)\n"+s[end:]
s=s.replace("run('cpu-full',base+['--model'", "run('cpu-full',base+['--fused-cpu-update','--model'")
s=s.replace("r=Path(__file__).resolve().parent;os.chdir(r);assert not (r/'job.json').exists()", "r=Path(__file__).resolve().parent;os.chdir(r);assert not (r/'job.json').exists()\nassert json.loads((r/'cpu-job.json').read_text())['stage']=='complete'\nassert json.loads((r/'cpu-probe.json').read_text())['passed']")
(r/'run_experiment.py').write_text(s,encoding='utf-8',newline='\n')
if (b/'launch.private.sh').exists():
 s=(b/'launch.private.sh').read_text(encoding='utf-8').replace('qwen-sql-boundary-overlap-20261025','qwen-sql-cpu-fusion-20261025')
 (r/'launch.private.sh').write_text(s,encoding='utf-8',newline='\n')
print('Generated native ZeRO-state integration, reference/fused tiny controls and full-model run')
