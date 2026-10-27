"""Token-chunked linear + CE with recomputation, for prospective memory modeling.

Ordinary PyTorch operations; no claim of NPU deployment or a fused kernel.
Only logits/CE are recomputed, not the transformer. BF16 partial weight
 gradients are accumulated in BF16, so numerical order differs from full GEMM.
"""
import torch
records=[]
def record(stage,start,dw=None,gw=None):
    torch.npu.synchronize()
    def layout(t):
        return None if t is None else dict(shape=list(t.shape),stride=list(t.stride()),contiguous=t.is_contiguous(),dtype=str(t.dtype))
    records.append(dict(stage=stage,start=start,allocated_bytes=torch.npu.memory_allocated(),peak_bytes=torch.npu.max_memory_allocated(),dw=layout(dw),gw=layout(gw)))
import torch.nn.functional as F


def shifted_targets(labels):
    return F.pad(labels, (0, 1), value=-100)[..., 1:].contiguous().view(-1)


def sum_loss(x, weight, target):
    logits = F.linear(x, weight)
    return F.cross_entropy(logits.float(), target, ignore_index=-100, reduction='sum')


class ChunkedLinearCE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hidden, weight, labels, chunk_tokens):
        assert hidden.is_contiguous() and weight.is_contiguous()
        assert chunk_tokens > 0
        flat = hidden.view(-1, hidden.shape[-1])
        targets = shifted_targets(labels)
        count = (targets != -100).sum().clamp_min(1)
        total = hidden.new_zeros((), dtype=torch.float32)
        for start in range(0, flat.shape[0], chunk_tokens):
            total.add_(sum_loss(flat[start:start+chunk_tokens], weight, targets[start:start+chunk_tokens]))
        ctx.save_for_backward(hidden, weight, targets, count)
        ctx.chunk_tokens = chunk_tokens
        return total / count

    @staticmethod
    def backward(ctx, grad_output):
        hidden, weight, targets, count = ctx.saved_tensors
        flat = hidden.view(-1, hidden.shape[-1])
        # One full hidden gradient; a full output-weight gradient is unavoidable
        # for this simple implementation, and another partial is live on addition.
        dx = torch.empty_like(flat)
        dw = None
        for start in range(0, flat.shape[0], ctx.chunk_tokens):
            stop = start + ctx.chunk_tokens
            with torch.enable_grad():
                x = flat[start:stop].detach().requires_grad_(True)
                w = weight.detach().requires_grad_(True)
                loss = sum_loss(x, w, targets[start:stop]) / count
                gx, gw = torch.autograd.grad(loss, (x, w), grad_outputs=grad_output)
            record('before_contiguous',start,dw,gw)
            gw=gw.contiguous()
            record('after_gradient',start,dw,gw)
            dx[start:stop].copy_(gx)
            if dw is None:
                dw = gw
            else:
                record('before_add',start,dw,gw)
                dw.add_(gw)
                record('after_add',start,dw,gw)
            del x, w, loss, gx, gw
        return dx.view_as(hidden), dw, None, None


def chunked_linear_ce(hidden, weight, labels, chunk_tokens):
    return ChunkedLinearCE.apply(hidden, weight, labels, chunk_tokens)
