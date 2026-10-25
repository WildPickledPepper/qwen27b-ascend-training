"""Fixed-data batch comparison with a sample-based LR schedule."""
from pathlib import Path
import json
r=Path(__file__).resolve().parent;base=r.parent/'sql-throughput-20261025'
s=(base/'train_sql.py').read_text(encoding='utf-8')
old='factor=step/warmup if step<warmup else .5*(1+math.cos(math.pi*(step-warmup)/(total_updates-warmup)))'
assert s.count(old)==1
s=s.replace(old,'progress=step*a.global_batch/256\n factor=progress/warmup if progress<warmup else .5*(1+math.cos(math.pi*(progress-warmup)/(total_updates-warmup)))')
s=s.replace("'benchmark':'public-text-to-sql'","'benchmark':'sql-batch-tuning-fixed-768-samples','lr_schedule_basis':'processed samples, 256-sample reference update'")
s=s.replace(" steps.append(row);log('step_complete',**row)"," steps.append(row);log('step_complete',**row)\n if not a.tiny and (step+1)*a.global_batch in [256,512]:\n  log('sample_checkpoint',samples=(step+1)*a.global_batch,training_seconds=sum(x['seconds'] for x in steps),metrics=evaluate())")
(r/'train_batch.py').write_text(s,encoding='utf-8',newline='\n')
for name in ['community_gdn.py','cpu_update.py','bounded_grad_copy.py','zero3-sql.json']:
 (r/name).write_text((base/name).read_text(encoding='utf-8'),encoding='utf-8',newline='\n')
print('Generated fixed-data batch-tuning entry')
