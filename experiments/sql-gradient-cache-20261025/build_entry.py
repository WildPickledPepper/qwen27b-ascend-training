from pathlib import Path
r=Path(__file__).resolve().parent;b=r.parent/'sql-bottleneck-20261025'
s=(b/'train_profile.py').read_text(encoding='utf-8')
s=s.replace("p.add_argument('--numa-spread',action='store_true')","p.add_argument('--numa-spread',action='store_true')\np.add_argument('--gradient-cache-gib',type=float,default=0)")
anchor='from gradient_scopes import install as install_gradient_scopes'
s=s.replace(anchor,"from gradient_cache import install as install_gradient_cache\ncache_plan=install_gradient_cache(optimizer,a.gradient_cache_gib)\nlog('gradient_cache',**cache_plan)\nplans=[None]*world;dist.all_gather_object(plans,cache_plan['plan_sha256']);assert len(set(plans))==1\n"+anchor)
anchor='  engine.step()'
s=s.replace(anchor,"  if a.tiny and engine.is_gradient_accumulation_boundary():\n   torch.npu.synchronize()\n   torch.save([p.grad.detach().cpu().clone() for p in optimizer.fp32_partitioned_groups_flat],out/f'probe-gradient-rank{rank}-step{step}.private.pt')\n"+anchor)
anchor=" steps.append(row);log('step_complete',**row)"
s=s.replace(anchor," if a.tiny:\n  torch.save({'master':[p.detach().clone() for p in optimizer.fp32_partitioned_groups_flat],'state':[[v[k].detach().clone() for k in ['exp_avg','exp_avg_sq']] for v in optimizer.optimizer.state.values()]},out/f'probe-state-rank{rank}-step{step}.private.pt')\n"+anchor)
s=s.replace("'checkpoint_saved':False,'tiny':a.tiny}","'checkpoint_saved':False,'tiny':a.tiny,'gradient_cache':cache_plan}")
s=s.replace('in [256,512]:','in [256,512,768]:')
assert s!= (b/'train_profile.py').read_text(encoding='utf-8')
(r/'train_cache.py').write_text(s,encoding='utf-8',newline='\n')
for name in ['community_gdn.py','cpu_update.py','bounded_grad_copy.py','zero3-sql.json','gradient_scopes.py']:
 (r/name).write_text((b/name).read_text(encoding='utf-8'),encoding='utf-8',newline='\n')
print('Generated cache training entry')
