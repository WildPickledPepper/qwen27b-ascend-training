"""Add observation scopes to the measured GBS256/micro16 SQL training configuration."""
from pathlib import Path
r=Path(__file__).resolve().parent;base=r.parent/'sql-batch-tuning-20261025'
s=(base/'train_batch.py').read_text(encoding='utf-8')
anchor="log('cpu_update_policy',mode=a.cpu_mode,loss_scale=optimizer.loss_scale,clip_grad=optimizer.clip_grad)"
assert s.count(anchor)==1
s=s.replace(anchor,anchor+"\nfrom gradient_scopes import install as install_gradient_scopes\ngradient_stats=install_gradient_scopes(optimizer)")
s=s.replace(' cpu_stats.clear()',' cpu_stats.clear();gradient_stats.clear()')
s=s.replace(" row['cpu_update_stats']=dict(cpu_stats)"," row['cpu_update_stats']=dict(cpu_stats)\n row['gradient_observer_stats']=dict(gradient_stats)")
(r/'train_profile.py').write_text(s,encoding='utf-8',newline='\n')
for name in ['community_gdn.py','cpu_update.py','bounded_grad_copy.py','zero3-sql.json']:
 (r/name).write_text((base/name).read_text(encoding='utf-8'),encoding='utf-8',newline='\n')
# Keep the skill CTEs unchanged, including their documented JOIN expressions.
(r/'analyze_profile.py').write_text((r.parent/'training-fusion-20261025'/'analyze_profile.py').read_text(encoding='utf-8'),encoding='utf-8',newline='\n')
print('Profile entry generated')
