"""CPU numerical checks including ignored-token chunks and CE normalization."""
import json
import torch
import torch.nn.functional as F
from chunked_loss import chunked_linear_ce, shifted_targets

torch.set_num_threads(1)
torch.manual_seed(20261027)
results=[]
for dtype in [torch.float32, torch.bfloat16]:
    hidden=torch.randn(2,9,8,dtype=dtype,requires_grad=True)
    weight=torch.randn(31,8,dtype=dtype,requires_grad=True)
    labels=torch.randint(0,31,(2,9));labels[0,:5]=-100;labels[1,-2:]=-100
    reference=F.cross_entropy(F.linear(hidden,weight).float().view(-1,31),shifted_targets(labels),ignore_index=-100)
    gref=torch.autograd.grad(reference,(hidden,weight))
    for chunk in [1,4,7,18,32]:
        loss=chunked_linear_ce(hidden,weight,labels,chunk)
        grads=torch.autograd.grad(loss,(hidden,weight))
        rel=[float((x.float()-y.float()).norm()/y.float().norm()) for x,y in zip(grads,gref)]
        assert torch.allclose(loss,reference,rtol=2e-6,atol=2e-6)
        assert max(rel)<(2e-6 if dtype==torch.float32 else .03),(dtype,chunk,rel)
        results.append(dict(dtype=str(dtype),chunk_tokens=chunk,loss_abs_error=float((loss-reference).detach().abs()),hidden_gradient_relative_l2=rel[0],weight_gradient_relative_l2=rel[1]))
print(json.dumps(dict(device='cpu',cases=results,scope='Small tensors only; BF16 accumulation order changes. Not full-model/NPU quality validation.'),indent=2))
