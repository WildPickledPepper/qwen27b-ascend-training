"""CPU-only, shape-based memory accounting for the pinned two-stage Qwen recipe.

No tensor framework, checkpoint loading, SSH, or accelerator is required.
Peak lifetimes use explicitly calibrated templates; this is not a general simulator.
"""
import argparse
import json
from pathlib import Path

GIB = 2**30
ROOT = Path(__file__).resolve().parent
CONFIG = ROOT.parents[1] / 'research/2026-10-24-adam-state-memory/stage11/qwen_audit_sources/qwen3_5_27b_config.json'


def weight_elements(c, rank):
    """Count all parameters from architecture, including the small GDN parameters."""
    h, i, v = c['hidden_size'], c['intermediate_size'], c['vocab_size']
    q = c['num_attention_heads'] * c['head_dim']
    k = c['num_key_value_heads'] * c['head_dim']
    lk = c['linear_num_key_heads'] * c['linear_key_head_dim']
    lv = c['linear_num_value_heads'] * c['linear_value_head_dim']
    layers = c['layer_types'][rank*32:(rank+1)*32]
    p = h*v + (h if rank == 1 else 0)
    for layer in layers:
        p += 3*h*i + 2*h  # FFN and input/post-attention norms.
        if layer == 'full_attention':
            p += h*(3*q + 2*k) + 2*c['head_dim']
        elif layer == 'linear_attention':
            p += h*(2*lk + 3*lv + 2*c['linear_num_value_heads'])
            p += (2*lk+lv)*c['linear_conv_kernel_dim']
            p += 2*c['linear_num_value_heads'] + c['linear_value_head_dim']
        else:
            raise ValueError('Unsupported layer implementation')
    return p


def layer_norm_bytes(b, s, h):
    # Saved/living FP32 intermediates (8*B*S*H), BF16 result (2*B*S*H),
    # per-token statistic and FP32 weight expression. Storage views are not copied.
    return 10*b*s*h + 4*b*s + 4*h


def qk_norm_bytes(b, s, heads, dim):
    # The BF16 final norm result is consumed/freed by rotary in this pinned graph;
    # FP32 intermediates survive. This is a lifetime rule, not only a shape rule.
    return 8*b*s*heads*dim + 4*b*s*heads + 4*dim


def predict(b=16, s=256, resident_gib=4.0, runtime_gib=None,
            allocator_slack_gib=None, extra_workspace_gib=0.0):
    if b <= 0 or s < 128 or s % 64 or s > 512:
        raise ValueError('Require B>0 and padded S in {128,192,...,512}')
    if resident_gib < 0 or extra_workspace_gib < 0:
        raise ValueError('Memory budgets must be nonnegative')
    if any(x is not None and x < 0 for x in [runtime_gib, allocator_slack_gib]):
        raise ValueError('Overheads must be nonnegative')
    c = json.loads(CONFIG.read_text(encoding='utf-8'))['text_config']
    cal = json.loads((ROOT/'calibration.json').read_text(encoding='utf-8'))
    assert c['num_hidden_layers'] == 64
    h, i, v = c['hidden_size'], c['intermediate_size'], c['vocab_size']
    q, k, d = c['num_attention_heads'], c['num_key_value_heads'], c['head_dim']
    rms = 2*layer_norm_bytes(b,s,h) + qk_norm_bytes(b,s,q,d) + qk_norm_bytes(b,s,k,d)
    ffn = 8*b*s*i  # gate, up, SiLU, gated product, four BF16 storages.
    attn = 12*b*s*q*d + 6*b*q*s*s  # linear terms plus FP32/BF16 probabilities.
    scale = (b*s)/(cal['reference_micro_batch']*cal['reference_padded_sequence'])
    ranks = []
    for rank in [0,1]:
        rc = cal['ranks'][rank]
        n = rc['noncheckpoint_attention_sets']
        p = weight_elements(c,rank)
        rows = []
        def put(name, nbytes, basis, formula):
            rows.append(dict(category=name,bytes=nbytes,gib=nbytes/GIB,basis=basis,formula=formula))
        put('weight',2*p,'architecture_exact','2*parameter_elements')
        put('resident_gradient',resident_gib*GIB,'budget_input','resident_GiB*2**30')
        put('ffn_forward_intermediates',n*ffn,'shape_formula_with_calibrated_lifetime',f'{n} * 8*B*S*I')
        put('rmsnorm_forward_intermediates',n*rms+(layer_norm_bytes(b,s,h) if rank else 0),
            'shape_formula_with_calibrated_lifetime',f'{n}*(2*Norm(H)+QNorm+KNorm) + final_norm_if_rank1')
        put('attention_forward_intermediates',n*attn,'shape_formula_with_calibrated_lifetime',f'{n}*(12*B*S*Q*D+6*B*Q*S**2)')
        put('hidden_states_checkpoint_boundaries_and_pipeline',rc['hidden_vector_count']*2*b*s*h,
            'shape_formula_with_calibrated_lifetime',f'{rc["hidden_vector_count"]}*2*B*S*H')
        if rank == 0:
            put('checkpoint_recompute_intermediates',rc['recompute_reference_bytes']*scale,
                'empirical_BL_scaling_not_derived_kernel_workspace','reference_recompute_bytes*(B*S)/(16*256)')
            put('temporary_gradients',4*h*i+2*b*s*i+2*b*s*h,
                'shape_formula_with_calibrated_lifetime','two_BF16_HxI_weight_buffers + BF16_BxSxI + BF16_BxSxH')
        else:
            put('loss_forward_intermediates',4*b*s*v,'shape_formula_with_calibrated_lifetime','4*B*S*V')
            put('temporary_gradients',8*b*s*v,'shape_formula_with_calibrated_lifetime','2*4*B*S*V; output gradients, not parameter gradients')
        put('small_controls_and_buffers',rc['small_control_reference_bytes']*scale,
            'empirical_small_term','reference_small_bytes*(B*S)/(16*256)')
        put('extra_workspace_allowance',extra_workspace_gib*GIB,'explicit_user_allowance','extra_workspace_GiB*2**30')
        total=sum(x['bytes'] for x in rows)
        physical=None
        if runtime_gib is not None and allocator_slack_gib is not None:
            physical=total/GIB+runtime_gib+allocator_slack_gib
        ranks.append(dict(rank=rank,physical_card=rank+2,categories=rows,
                          estimated_allocated_peak_template_gib=total/GIB,
                          estimated_physical_with_supplied_overheads_gib=physical,
                          cpu_fp32_master_gib=4*p/GIB,cpu_fp32_m_v_gib=8*p/GIB))
    return dict(model='qwen35-two-stage-calibrated-memory-v0',micro_batch=b,padded_sequence=s,
                pipeline_microbatches=16,global_batch=16*b,ranks=ranks,
                status='reference_reconstruction' if (b,s,resident_gib,extra_workspace_gib)==(16,256,4,0) else 'unvalidated_what_if',
                constraints=['Pinned BF16 weights/accumulation, CPU FP32 master/m/v, eager attention, unfused RMSNorm/loss, 48 checkpoint layers, two-stage schedule.',
                             'Lifetime multiplicities 15/8 and hidden-state counts 51/24 are calibrated to the measured peak scenarios; they are not inferred for arbitrary schedules.',
                             'Changing shape, checkpointing, fusion, sequence distribution or schedule can move the peak; extrapolations require later validation.',
                             'Recompute/small terms are empirical BL scaling; unknown backend workspace must be supplied or modeled separately.',
                             'Allocated is not reserved or device total. Missing runtime/allocator overheads leave physical prediction unset.',
                             'Changing resident budget only estimates a storage delta; it does not recompute whole-parameter cache selection or temporary lifetimes.'])


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--micro-batch',type=int,default=16)
    p.add_argument('--sequence',type=int,default=256)
    p.add_argument('--resident-gib',type=float,default=4)
    p.add_argument('--runtime-gib',type=float)
    p.add_argument('--allocator-slack-gib',type=float)
    p.add_argument('--extra-workspace-gib',type=float,default=0)
    p.add_argument('--output',type=Path)
    a=p.parse_args()
    result=predict(a.micro_batch,a.sequence,a.resident_gib,a.runtime_gib,a.allocator_slack_gib,a.extra_workspace_gib)
    text=json.dumps(result,indent=2,ensure_ascii=False)
    if a.output:a.output.write_text(text+'\n',encoding='utf-8')
    else:print(text)
