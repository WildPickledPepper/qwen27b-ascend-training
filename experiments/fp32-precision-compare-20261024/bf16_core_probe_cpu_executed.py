"""Audit gated-delta core matmul dtypes without allocating a full model."""
import argparse
import ctypes
import json
from pathlib import Path
import torch
from torch.utils._python_dispatch import TorchDispatchMode
import transformers.models.qwen3_5.modeling_qwen3_5 as model
from bf16_precision_guards import install

p = argparse.ArgumentParser()
p.add_argument('--device', choices=['cpu', 'npu'], required=True)
p.add_argument('--output', required=True)
a = p.parse_args()
if a.device == 'npu':
    import torch_npu
    torch.npu.config.allow_internal_format = False
    torch.npu.matmul.allow_hf32 = False
    torch.npu.set_device(0)
ctypes.CDLL(None).prctl(41, 1, 0, 0, 0)
torch.set_num_threads(4)
torch.manual_seed(20261024)
device = 'npu:0' if a.device == 'npu' else 'cpu'
q, k, v = [torch.randn(1, 16, 2, 16, device=device).to(torch.bfloat16) for _ in range(3)]
g = -torch.rand(1, 16, 2, device=device)*.2
beta = torch.rand(1, 16, 2, device=device).to(torch.bfloat16)
original = model.torch_chunk_gated_delta_rule


class Trace(TorchDispatchMode):
    def __init__(self):
        super().__init__()
        self.rows = []

    def __torch_dispatch__(self, function, types, args=(), kwargs=None):
        result = function(*args, **(kwargs or {}))
        if function in [torch.ops.aten.mm.default, torch.ops.aten.bmm.default, torch.ops.aten.addmm.default]:
            self.rows.append({'op': str(function), 'inputs': [str(x.dtype) for x in args if isinstance(x, torch.Tensor)],
                              'output': str(result.dtype)})
        return result


kw = {'g': g, 'beta': beta, 'use_qk_l2norm_in_kernel': True}
with torch.no_grad():
    baseline = original(q, k, v, **kw)[0]
    trace_bad = Trace()
    with torch.autocast(a.device, dtype=torch.bfloat16), trace_bad:
        naive = original(q, k, v, **kw)[0]
    install()
    trace_good = Trace()
    with torch.autocast(a.device, dtype=torch.bfloat16), trace_good:
        guarded = model.torch_chunk_gated_delta_rule(q, k, v, **kw)[0]
    exact = torch.equal(baseline.view(torch.int16), guarded.view(torch.int16))
    assert exact and trace_good.rows
    assert all(row['output'] == 'torch.float32' for row in trace_good.rows)
    assert any(row['output'] == 'torch.bfloat16' for row in trace_bad.rows)
    result = {'device': a.device, 'passed': True, 'guarded_equals_explicit_fp32_core_bits': exact,
              'naive_vs_reference_max_abs': float((naive.float()-baseline.float()).abs().max()),
              'naive_core_matmuls': trace_bad.rows, 'guarded_core_matmuls': trace_good.rows}
Path(a.output).write_text(json.dumps(result, indent=2))
print(json.dumps({k: v for k, v in result.items() if not k.endswith('matmuls')}), flush=True)
