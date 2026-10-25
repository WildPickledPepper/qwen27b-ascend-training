"""Derive one dependency capture from the unchanged selected training setup."""
from pathlib import Path
r=Path(__file__).resolve().parent;b=r.parent/'sql-boundary-overlap-20261025'
for name in ['cpu_accumulation.py','community_gdn.py','bounded_grad_copy.py','zero3-sql.json','gradient_scopes.py','gradient_cache.py','cpu_update.py']:
 (r/name).write_text((b/name).read_text(),encoding='utf-8',newline='\n')
s=(b/'train_boundary.py').read_text()
s=s.replace('gradient_stats=install_gradient_scopes(optimizer)', "gradient_stats=install_gradient_scopes(optimizer)\nfrom event_dependencies import install as install_event_dependencies\nevent_plan=install_event_dependencies(optimizer)\nlog('dependency_source',source_sha256=event_plan['source_sha256'])")
s=s.replace('if a.profile and rank==0:', 'if a.profile:')
s=s.replace(' if prof and step==a.steps-1:prof.start()', " if prof and step==a.steps-1:\n  prof.start();event_plan['active']=True")
s=s.replace(' if prof and step==a.steps-1:prof.stop()', " if prof and step==a.steps-1:\n  event_plan['active']=False\n  prof.stop()\n  log('dependency_counts',counts=dict(event_plan['counts']))")
s=s.replace("log('model_ready',parameters=params)","log('model_ready',parameters=params)\nlog('parameter_manifest',parameters=[{'id':p.ds_id,'name':name,'shape':list(p.ds_shape),'numel':p.ds_numel} for name,p in model.named_parameters()])")
(r/'train_capture.py').write_text(s,encoding='utf-8',newline='\n')
print('Prepared event dependency labels, host gradient hooks and exact bucket IDs')
