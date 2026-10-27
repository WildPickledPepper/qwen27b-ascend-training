"""Shape/lifetime translation of the pinned NPU matmul forward/backward path.

Source: Ascend/op-plugin 8b9c8534fa41eff367a41c155843daa530ab3a08
ops/opapi/MatmulKernelNpuOpApi.cpp, MatmulBackwardKernelNpuOpApi.cpp;
config/derivatives.yaml. Upstream is BSD-3-Clause, Huawei Technologies.
No device compute or workspace assumption is made here. This module only
models the dense 2D/3D-by-2D, bias-free linear path used by this model.
"""
import torch

OP_PLUGIN_COMMIT='8b9c8534fa41eff367a41c155843daa530ab3a08'


def grad_other(x,other,g):
    # Pinned C++ allocates a dense result with other.sizes(), even when other
    # itself is a transposed weight view. Keep the real reshape/transpose ops.
    out=torch.empty(tuple(other.shape),dtype=x.dtype,device='meta')
    left=x.reshape(-1,x.shape[-1]).transpose(-2,-1)
    right=g.reshape(-1,g.shape[-1])
    assert left.shape[-1]==right.shape[-2]
    return out


def grad_self(x,other,g):
    right=other.transpose(-2,-1)
    shape=(*g.shape[:-1],right.shape[-1])
    out=torch.empty(shape,dtype=g.dtype,device='meta')
    assert out.shape==x.shape
    return out


class NpuDenseMatmul(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x,other):
        assert x.device.type==other.device.type=='meta'
        assert x.dim() in [2,3] and other.dim()==2
        assert x.shape[-1]==other.shape[0]
        ctx.save_for_backward(x,other)
        return torch.empty((*x.shape[:-1],other.shape[-1]),dtype=x.dtype,device='meta')

    @staticmethod
    def backward(ctx,g):
        x,other=ctx.saved_tensors
        # Actual pinned backend computes the second input gradient first.
        go=grad_other(x,other,g) if ctx.needs_input_grad[1] else None
        gx=grad_self(x,other,g) if ctx.needs_input_grad[0] else None
        return gx,go


def install():
    original=torch.nn.functional.linear
    def linear(x,weight,bias=None):
        if x.device.type!='meta':return original(x,weight,bias)
        if bias is not None:raise RuntimeError('NPU meta rule supports only current bias-free dense linears')
        # Transpose remains an autograd view. AccumulateGrad, not a manual
        # hard-coded clone, enforces the parameter gradient layout contract.
        return NpuDenseMatmul.apply(x,weight.t())
    torch.nn.functional.linear=linear
    return original
