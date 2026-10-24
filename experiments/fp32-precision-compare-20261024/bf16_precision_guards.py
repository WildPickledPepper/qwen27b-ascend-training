"""Keep Qwen3.5's explicitly FP32 gated-delta fallback outside outer autocast.

Only patches function references inside this benchmark process. No installed
Transformers files are changed; projections/FFN remain BF16 in the AMP group.
"""
import functools
import torch

COUNTS = {'chunk': 0, 'recurrent': 0}


def install():
    import transformers.models.qwen3_5.modeling_qwen3_5 as module
    assert module.chunk_gated_delta_rule is None, 'Expected audited PyTorch fallback; re-audit other kernels'
    assert module.fused_recurrent_gated_delta_rule is None
    for name, key in [('torch_chunk_gated_delta_rule', 'chunk'),
                      ('torch_recurrent_gated_delta_rule', 'recurrent')]:
        original = getattr(module, name)

        def make_guard(function, counter):
            @functools.wraps(function)
            def guarded(*args, **kwargs):
                COUNTS[counter] += 1
                query = args[0] if args else kwargs['query']
                device_type = query.device.type
                with torch.autocast(device_type, enabled=False):
                    assert not torch.is_autocast_enabled(device_type)
                    return function(*args, **kwargs)
            return guarded

        setattr(module, name, make_guard(original, key))
    return module
