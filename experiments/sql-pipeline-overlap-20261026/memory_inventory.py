"""CPU-only module inventory from the actual per-rank tensor manifests and peaks."""
import argparse,json
from collections import defaultdict
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--output',required=True);a=p.parse_args()
root=Path(a.root);rows=[]
def category(name):
    if name=='model.embed_tokens.weight':return 'token_embedding'
    if name=='lm_head.weight':return 'output_lm_head'
    if '.mlp.gate_proj.' in name or '.mlp.up_proj.' in name:return 'ffn_gate_up'
    if '.mlp.down_proj.' in name:return 'ffn_down'
    if '.self_attn.' in name:return 'self_attention'
    if '.linear_attn.' in name:return 'gdn_linear_attention'
    if 'norm' in name:return 'other_norm'
    return 'other_parameters'
for rank in [0,1]:
    logs=[json.loads(x) for x in (root/f'rank{rank}.jsonl').read_text().splitlines()]
    ready=next(x for x in logs if x['event']=='model_ready');manifest=ready['manifest']
    plan=next(x for x in logs if x['event']=='resident_plan')
    selected=set(plan['parameter_indices']);groups=defaultdict(lambda:dict(parameter_tensors=0,weight_bytes=0,resident_gradient_bytes=0))
    for idx,x in enumerate(manifest):
        assert x['dtype']=='torch.bfloat16'
        group=groups[category(x['name'])];size=x['numel']*2
        group['parameter_tensors']+=1;group['weight_bytes']+=size
        if idx in selected:group['resident_gradient_bytes']+=size
    result=json.loads((root/f'rank{rank}-result.json').read_text())
    weights=sum(g['weight_bytes'] for g in groups.values())
    assert weights==result['parameters']*2
    assert sum(g['resident_gradient_bytes'] for g in groups.values())==plan['allocated_bytes']
    allocated=max(x['allocated_peak'] for x in result['steps']);reserved=max(x['reserved_peak'] for x in result['steps'])
    rows.append(dict(rank=rank,physical_card=rank+2,module_groups=dict(groups),weight_bytes=weights,
        resident_gradient_bytes=plan['allocated_bytes'],framework_peak_allocated_bytes=allocated,
        framework_peak_reserved_bytes=reserved,
        residual_at_allocated_peak_bytes=allocated-weights-plan['allocated_bytes'],
        cpu_fp32_master_bytes=result['parameters']*4,cpu_fp32_m_v_bytes=result['state_bytes']))
out=dict(ranks=rows,unit='All sizes exact bytes; divide by 2**30 for GiB',
    interpretation=['Weights and 4GiB cache are persistent during the measured steps and are included in allocated/reserved.',
        'Residual subtracts persistent weights/cache from allocated peak; it is one combined amount, not separate module peaks.',
        'Activations, temporary gradients, operator tensors and model buffers were not individually tracked.',
        'Max reserved and max allocated can occur at different times; their difference is not an exact simultaneous idle-cache measurement.',
        'CPU master/m/v and pinned receive slots are host RAM, not HBM.',
        'These are completed training peaks, not current live usage; no model is launched by this script.'])
Path(a.output).write_text(json.dumps(out,indent=2));print(json.dumps(out))
