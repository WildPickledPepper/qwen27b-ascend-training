"""Check routing, complete FP32 CPU Adam state and BF16 compute before full loading."""
import argparse
import json
from pathlib import Path
import torch
import torch_npu
import bench_resident as bench

p = argparse.ArgumentParser()
p.add_argument('--output', required=True)
a = p.parse_args()
torch.npu.config.allow_internal_format = False
torch.npu.matmul.allow_hf32 = False
torch.npu.conv.allow_hf32 = False
torch.manual_seed(20261024)
rows = []
for mode in ['fp32', 'bf16', 'compressed']:
    bench.configure(mode)
    for device in ['npu:0', 'npu:1']:
        with torch.npu.device(device):
            w = torch.randn(256, 256, device=device)*.02
            codec = bench.Codec(Path(__file__).resolve().parent/'libonline_codec.so') if mode == 'compressed' else bench.RawCodec()
            s = bench.WeightStore('smoke.weight', w.shape, [codec.encode(w)], codec, device, 128, 1e-5, .01, 65536)
            module = bench.CompressedLinear(s)
            x = torch.randn(3, 256, device='npu:0', requires_grad=True)
            before = s.read_all()
            reference_x = x.detach().to(device).requires_grad_()
            reference_w = before.detach().clone().requires_grad_()
            dtype = torch.bfloat16 if mode == 'bf16' else torch.float32
            reference_y = torch.nn.functional.linear(reference_x.to(dtype), reference_w.to(dtype))
            reference_y.float().square().mean().backward()
            actual = module(x)
            actual.float().square().mean().backward()
            forward_error = float((actual-reference_y.to('npu:0')).abs().max())
            gradient_error = float((x.grad-reference_x.grad.to('npu:0')).abs().max())
            assert forward_error < 1e-6 and gradient_error < 1e-6
            x.grad = None
            s.enabled = True
            s.step = 1
            module(x).float().square().mean().backward()
            expected = before.cpu()*(1-1e-5*.01)-1e-5*(reference_w.grad.cpu()/(reference_w.grad.cpu().abs()+1e-8))
            error = float((s.read_all().cpu()-expected).abs().max())
            assert error < 2e-7
            s.step = 2
            x.grad = None
            module(x).float().square().mean().backward()
            assert s.changed_steps == 2 and s.moments.updated == s.n*2
            assert torch.isfinite(s.read_all()).all()
            if mode != 'compressed':
                assert s.moments.m.device.type == 'cpu' and s.moments.v.dtype == torch.float32
            rows.append({'mode': mode, 'device': device, 'forward_max_abs': forward_error,
                         'input_gradient_max_abs': gradient_error, 'first_update_max_abs': error,
                         'complete_steps': 2, 'moment_bytes': s.moments.bytes()})
            del s,module,x,before,reference_x,reference_w,reference_y,actual,w,expected,codec
            torch.npu.synchronize(device)
            torch.npu.empty_cache()
Path(a.output).write_text(json.dumps({'passed': True, 'tests': rows}, indent=2))
print('ALL_PASSED', flush=True)
