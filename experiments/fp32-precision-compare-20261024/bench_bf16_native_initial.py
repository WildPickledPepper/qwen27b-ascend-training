"""Use native BF16 linear autograd, with an FP32 CPU master/Adam backend.

BF16 temporary weight leaves are kept by autograd until their backward completes.
Each leaf's post-accumulate hook consumes its gradient into the existing pipeline.
This removes the custom GEMM backward from the mixed-compute path.
"""
import contextlib
import torch
import torch.nn.functional as F
import bench_resident as bench
import bench_bf16_master as mixed

COUNTS=mixed.COUNTS
MixedMoments=mixed.MixedMoments
install_store=mixed.install_store
precision_context=mixed.precision_context


class NativeLinear:
    @staticmethod
    def apply(x,proxy,s):
        with torch.npu.device(s.device),torch.autocast('npu',enabled=False):
            z=x.to(device=s.device,dtype=torch.bfloat16)
            w=s.read_all().detach().requires_grad_(torch.is_grad_enabled())
            assert w.dtype==torch.bfloat16
            if w.requires_grad:
                def consume(parameter):
                    gradient=parameter.grad
                    if s.enabled:
                        s.update_all(parameter.detach().float(),gradient.float())
                        COUNTS['backward_bf16_matmuls']+=2
                    parameter.grad=None
                w.register_post_accumulate_grad_hook(consume)
            y=F.linear(z,w).float()
            COUNTS['forward_bf16_matmuls']+=1
            return y.to(x.device)


@contextlib.contextmanager
def evaluation_context(canonical=True):
    with mixed.evaluation_context(canonical):
        if not canonical:bench.base.LinearFunction=NativeLinear
        yield


def install():
    mixed.install()
    bench.base.LinearFunction=NativeLinear
