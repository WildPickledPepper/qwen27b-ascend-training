"""Group each FFN with native ZeRO leaf policy; derive prefetch from FFN size."""
from pathlib import Path
import sys
r=Path(__file__).resolve().parent;b=r.parent/'sql-gradient-pipeline-20261025'
for name in ['cpu_accumulation.py','community_gdn.py','cpu_update.py','bounded_grad_copy.py','zero3-sql.json','gradient_scopes.py','gradient_cache.py','compare_probes.py','export_evidence.py','status.py']:
 (r/name).write_text((b/name).read_text(encoding='utf-8'),encoding='utf-8',newline='\n')
s=(b/'train_cpu.py').read_text(encoding='utf-8')
anchor="log('model_ready',parameters=params)"
insertion="""
from deepspeed.utils import set_z3_leaf_modules
if a.tiny:
 grouped=set_z3_leaf_modules(model,[torch.nn.Sequential]);expected=1
else:
 grouped=set_z3_leaf_modules(model,[type(model.model.layers[0].mlp)]);expected=64
assert len(grouped)==expected
layer_sizes=[sum(p.ds_numel for p in module.parameters()) for module in grouped]
largest=max(layer_sizes)
assert 2*largest<cfg['zero_optimization']['stage3_max_live_parameters']
previous_prefetch=cfg['zero_optimization']['stage3_prefetch_bucket_size']
cfg['zero_optimization']['stage3_prefetch_bucket_size']=largest
layer_plan={'group_class':type(grouped[0]).__name__,'groups':len(grouped),'leaf_modules':len(grouped),'gradient_hooks':'native ZeRO leaf policy for FFN','largest_group_parameters':largest,'smallest_group_parameters':min(layer_sizes),'old_prefetch_elements':previous_prefetch,'prefetch_elements':largest,'native_public_api':'deepspeed.utils.set_z3_leaf_modules','native_config_key':'stage3_prefetch_bucket_size','zero_quantized_weights':False}
log('layer_schedule',**layer_plan)
"""
if '--ffn-leaf' not in sys.argv:
 insertion="""
if a.tiny:
 grouped=[model];expected=1
else:
 grouped=list(model.model.layers);expected=64
assert len(grouped)==expected
layer_sizes=[sum(p.ds_numel for p in module.parameters()) for module in grouped]
largest=max(layer_sizes)
assert 2*largest<cfg['zero_optimization']['stage3_max_live_parameters']
previous_prefetch=cfg['zero_optimization']['stage3_prefetch_bucket_size']
cfg['zero_optimization']['stage3_prefetch_bucket_size']=largest
layer_plan={'layer_class':type(grouped[0]).__name__,'layers':len(grouped),'leaf_modules':0,'native_gradient_hooks_preserved':True,'largest_layer_parameters':largest,'smallest_layer_parameters':min(layer_sizes),'old_prefetch_elements':previous_prefetch,'prefetch_elements':largest,'native_config_key':'stage3_prefetch_bucket_size','zero_quantized_weights':False}
log('layer_schedule',**layer_plan)
"""
assert s.count(anchor)==1
s=s.replace(anchor,anchor+'\n'+insertion)
s=s.replace("'cpu_gradient_accumulation':cpu_grad_plan}","'cpu_gradient_accumulation':cpu_grad_plan,'layer_schedule':layer_plan}")
(r/('train_ffn.py' if '--ffn-leaf' in sys.argv else 'train_layer.py')).write_text(s,encoding='utf-8',newline='\n')
s=(b/'run_experiment.py').read_text(encoding='utf-8').replace('train_cpu.py','train_layer.py').replace("'waiting-for-prior-cpu-run'","'waiting-for-prior-pipeline-run'").replace("prior=r.parent/'qwen-sql-gradient-cpu-20261025'","prior=r.parent/'qwen-sql-gradient-pipeline-20261025'")
(r/'run_experiment.py').write_text(s,encoding='utf-8',newline='\n')
s=(b/'launch.private.sh').read_text(encoding='utf-8').replace('qwen-sql-gradient-pipeline-20261025','qwen-sql-layer-gather-20261025')
(r/'launch.private.sh').write_text(s,encoding='utf-8',newline='\n')
print('Generated isolated native prefetch or FFN-grouping entry')
