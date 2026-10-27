"""Check source-derived layout/copy rules without retaining gradient tensors."""
import json
import torch
from npu_linear_meta import install,OP_PLUGIN_COMMIT
install()
cases=[]
for shape in [(2,3),(2,4,3)]:
    x=torch.empty(shape,device='meta',requires_grad=True)
    w=torch.empty((5,3),device='meta',requires_grad=True)
    seen={}
    def incoming(g):
        seen['incoming_shape']=list(g.shape);seen['incoming_stride']=list(g.stride())
        seen['incoming_storage']=g.untyped_storage()._cdata
    def accumulated(p):
        seen['accumulated_stride']=list(p.grad.stride())
        seen['copy_created']=p.grad.untyped_storage()._cdata!=seen.pop('incoming_storage')
    w.register_hook(incoming);w.register_post_accumulate_grad_hook(accumulated)
    y=torch.nn.functional.linear(x,w)
    assert tuple(y.shape)==(*shape[:-1],5)
    y.sum().backward()
    assert seen['incoming_stride']==[1,5]
    assert seen['accumulated_stride']==[3,1]
    assert seen['copy_created']
    assert tuple(x.grad.shape)==shape
    cases.append(dict(input_shape=list(shape),**seen))
print(json.dumps(dict(op_plugin_commit=OP_PLUGIN_COMMIT,cases=cases,
                      checks=dict(native_autograd_creates_layout_copy=True,shape_rules_2d_and_3d=True),
                      no_numeric_kernel_executed=True),indent=2))
