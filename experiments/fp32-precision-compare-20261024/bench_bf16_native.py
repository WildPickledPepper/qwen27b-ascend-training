"""Native BF16 autograd directly on the persistent BF16 compute-weight cache.

Device frames are views of the same Parameter storage, not a second weight copy.
FP32 master/m/v remain on CPU; each post-accumulate hook consumes one weight grad.
"""
import contextlib
import torch
import torch.nn.functional as F
import bench_resident as bench
import bench_bf16_master as mixed

COUNTS=mixed.COUNTS
SUBMIT_LOCK=mixed.SUBMIT_LOCK
MixedMoments=mixed.MixedMoments
precision_context=mixed.precision_context


@torch.no_grad()
def bind_compute_weight(s):
    with mixed.SUBMIT_LOCK,torch.npu.device(s.device):
        if hasattr(s,'compute_weight'):return
        data=torch.cat([f['data'].view(torch.bfloat16) for f in s.frames]).reshape(s.shape)
        s.compute_weight=torch.nn.Parameter(data)
        offset=0
        for frame in s.frames:
            frame['data']=s.compute_weight.detach().view(-1)[offset:offset+frame['n']].view(torch.uint8)
            offset+=frame['n']
        def consume(parameter):
            with mixed.SUBMIT_LOCK,torch.npu.device(s.device),torch.no_grad():
                gradient=parameter.grad
                if s.enabled:
                    # Conversions occur within bounded frame updates on CPU/NPU.
                    s.update_all(parameter.detach(),gradient)
                    COUNTS['backward_bf16_matmuls']+=2
                parameter.grad=None
        s.compute_weight.register_post_accumulate_grad_hook(consume)


def install_store(store,is_embedding=False):
    mixed.install_store(store,is_embedding)
    if not is_embedding:bind_compute_weight(store)


class NativeLinear:
    @staticmethod
    def apply(x,proxy,s):
        with torch.npu.device(s.device),torch.autocast('npu',enabled=False):
            z=x.to(device=s.device,dtype=torch.bfloat16)
            if not hasattr(s,'compute_weight'):bind_compute_weight(s)
            w=s.compute_weight
            assert w.dtype==torch.bfloat16
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
