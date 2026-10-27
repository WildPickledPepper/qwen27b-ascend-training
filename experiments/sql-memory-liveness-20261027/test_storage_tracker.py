"""Independent small cases for lifetime, aliases, copies, and saved variables."""
import gc,json
import torch
from storage_tracker import StorageTracker

checks={}
t=StorageTracker()
with t:
    x=torch.empty(2,3,device='meta',dtype=torch.float32)
    assert t.total==24
    view=x.transpose(0,1)
    assert t.total==24 and view.untyped_storage()._cdata==x.untyped_storage()._cdata
    copy=view.reshape(-1)
    assert t.total==48 and copy.untyped_storage()._cdata!=x.untyped_storage()._cdata
    del x;gc.collect();t.sweep();assert t.total==48
    del view;gc.collect();t.sweep();assert t.total==24
    del copy;gc.collect();t.sweep();assert t.total==0
checks['view_retains_base_and_noncontiguous_reshape_copies']=True
t.close()

t=StorageTracker()
with t:
    x=torch.empty(1,3,device='meta')
    e=x.expand(100,3)
    assert t.total==12 and e.untyped_storage().nbytes()==12
    e2=e.detach();del x,e;gc.collect();t.sweep();assert t.total==12
    del e2;gc.collect();t.sweep();assert t.total==0
checks['expand_and_detach_share_unique_storage']=True
t.close()

t=StorageTracker()
w=torch.empty(3,4,device='meta',requires_grad=True)
t.register(w,'weight','w',True)
with t:
    x=torch.empty(2,3,device='meta')
    sid=x.untyped_storage()._cdata
    y=x@w;loss=y.sum();del x,y;gc.collect();t.sweep()
    assert sid in t.live,'Autograd saved variable must retain x storage without Python x'
    loss.backward();t.sweep()
    assert sid not in t.live,'Backward must release saved x storage'
    assert w.grad is not None and tuple(w.grad.shape)==(3,4)
    grad_sid=w.grad.untyped_storage()._cdata
    w.grad=None;gc.collect();t.sweep()
    assert grad_sid not in t.live
checks['native_savedvariable_lifetime_and_gradient_release']=True
t.close()

t=StorageTracker()
with t:
    cache=torch.zeros(10,device='meta');t.register(cache,'resident_gradient','cache',True)
    g=torch.ones(10,device='meta');before=t.total
    ptr=cache.untyped_storage()._cdata
    cache.add_(g)
    assert cache.untyped_storage()._cdata==ptr and t.total==before
checks['inplace_accumulation_creates_no_result_storage']=True
t.close()
print(json.dumps(dict(checks=checks,device='meta',device_initialized=False),indent=2))
