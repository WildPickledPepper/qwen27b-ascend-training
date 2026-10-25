"""Build a separate entry from the previous successful, frozen training driver."""
from pathlib import Path
r=Path(__file__).resolve().parent
s=(r.parent/'dual-throughput-20261025/train_advanced.py').read_text()
s=s.replace("p.add_argument('--profile',action='store_true')", "p.add_argument('--profile',action='store_true')\np.add_argument('--cpu-mode',choices=['observe','skip','pipeline'],default='observe')\np.add_argument('--fuse-gdn',action='store_true')")
s=s.replace("with deepspeed.zero.Init(config_dict_or_path=cfg,dtype=torch.bfloat16):", """if a.fuse_gdn:
 from community_gdn import chunk_gated_delta_rule as fused_training_gdn
 from transformers.models.qwen3_5 import modeling_qwen3_5
 modeling_qwen3_5.chunk_gated_delta_rule=fused_training_gdn
 log('fusion_enabled',backend='VeOmni pinned Triton-Ascend forward and backward')
with deepspeed.zero.Init(config_dict_or_path=cfg,dtype=torch.bfloat16):""")
s=s.replace("assert all(p.dtype==torch.float32 and p.device.type=='cpu' for p in optimizer.fp32_partitioned_groups_flat)", """assert all(p.dtype==torch.float32 and p.device.type=='cpu' for p in optimizer.fp32_partitioned_groups_flat)
from cpu_update import install as install_cpu_update
cpu_stats=install_cpu_update(optimizer,skip_identity=a.cpu_mode!='observe',pipeline=a.cpu_mode=='pipeline')
log('cpu_update_policy',mode=a.cpu_mode,loss_scale=optimizer.loss_scale,clip_grad=optimizer.clip_grad)""")
s=s.replace("started=time.monotonic()\n phase=", "started=time.monotonic()\n cpu_stats.clear()\n phase=")
s=s.replace("steps.append(row);log('step_complete',**row)", "row['cpu_update_stats']=dict(cpu_stats)\n steps.append(row);log('step_complete',**row)")
s=s.replace("'numa_spread':a.numa_spread,'config':cfg,", "'cpu_mode':a.cpu_mode,'fuse_gdn':a.fuse_gdn,'numa_spread':a.numa_spread,'config':cfg,")
s=s.replace("prof=torch_npu.profiler.profile(activities=", "prof=torch_npu.profiler.profile(schedule=torch_npu.profiler.schedule(wait=0,warmup=0,active=1,repeat=1),activities=")
(r/'train_fusion.py').write_text(s,encoding='utf-8',newline='\n')
import ast
ast.parse(s)
print('Built separate CPU/fusion entry')
