from pathlib import Path
import subprocess,sys
r=Path(__file__).resolve().parent;b=r.parent/'sql-layer-gather-20261025'
subprocess.run([sys.executable,str(b/'build_entry.py'),'--ffn-leaf'],check=True)
for name in ['cpu_accumulation.py','community_gdn.py','cpu_update.py','bounded_grad_copy.py','zero3-sql.json','gradient_scopes.py','gradient_cache.py','compare_probes.py','export_evidence.py','status.py']:
 (r/name).write_text((b/name).read_text(encoding='utf-8'),encoding='utf-8',newline='\n')
(r/'train_ffn.py').write_text((b/'train_ffn.py').read_text(encoding='utf-8'),encoding='utf-8',newline='\n')
s=(b/'run_experiment.py').read_text(encoding='utf-8').replace('train_layer.py','train_ffn.py').replace("'waiting-for-prior-pipeline-run'","'waiting-for-prior-prefetch-run'").replace("prior=r.parent/'qwen-sql-gradient-pipeline-20261025'","prior=r.parent/'qwen-sql-layer-gather-20261025'")
(r/'run_experiment.py').write_text(s,encoding='utf-8',newline='\n')
s=(b/'launch.private.sh').read_text(encoding='utf-8').replace('qwen-sql-layer-gather-20261025','qwen-sql-ffn-gather-20261025')
(r/'launch.private.sh').write_text(s,encoding='utf-8',newline='\n')
# Restore the still-running prefetch experiment's exact entry source.
subprocess.run([sys.executable,str(b/'build_entry.py')],check=True)
print('FFN grouping isolated from the existing prefetch run')
