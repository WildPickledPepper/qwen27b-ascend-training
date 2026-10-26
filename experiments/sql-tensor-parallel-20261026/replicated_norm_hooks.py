"""Process-local reattachment after optimizer setup; no shared library edits."""
import torch.distributed as dist

def install(model,tp_group):
    calls={};handles=[];names=[]
    for name,param in model.named_parameters():
        if not name.endswith(('.self_attn.q_norm.weight','.self_attn.k_norm.weight')):continue
        existing=param._backward_hooks or {}
        assert getattr(param,'_ds_grad_allreduce_registered',False),name
        assert len(existing)==1,(name,len(existing))
        fn=next(iter(existing.values()))
        assert fn.__module__=='deepspeed.module_inject.auto_tp',(name,fn.__module__)
        # The existing Python hook dictionary reports registered, but observed
        # callback counts were zero after optimizer setup. Re-register through
        # the public API against the final Parameter, preserving SUM semantics.
        param._backward_hooks=None
        calls[name]=0
        def reduce(grad,name=name):
            calls[name]+=1
            result=grad.contiguous();dist.all_reduce(result,group=tp_group)
            return result
        handles.append(param.register_hook(reduce));names.append(name)
    assert names,'No replicated attention norm parameters found'
    return calls,handles
