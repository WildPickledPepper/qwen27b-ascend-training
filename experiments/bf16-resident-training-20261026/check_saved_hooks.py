"""CPU-only check that the production pass-through hook preserves gradients."""
import json,os
os.environ['TORCH_DEVICE_BACKEND_AUTOLOAD']='0'
import torch
from compressed_ffn import CompressedFFN
torch.set_num_threads(1);torch.manual_seed(19)
x=torch.randn(3,8,requires_grad=True);w=torch.randn(4,8,requires_grad=True)
loss=torch.nn.functional.linear(x,w).relu().square().sum();loss.backward()
ref=[x.grad.clone(),w.grad.clone()];x.grad=None;w.grad=None
obj=CompressedFFN.__new__(CompressedFFN);obj.live={}
with torch.autograd.graph.saved_tensors_hooks(obj.pack,obj.unpack):
    loss=torch.nn.functional.linear(x,w).relu().square().sum()
loss.backward()
assert all(torch.equal(a,b) for a,b in zip(ref,[x.grad,w.grad]))
packed=obj.pack(x*2);assert packed.grad_fn is None
print(json.dumps(dict(passed=True,nonweight_gradients_exact=True,pass_through_detaches_graph_reference=True)))
