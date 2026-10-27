"""Count distinct meta storages using native StorageImpl weak references.

No tensor/storage is retained by the tracker. SavedVariable references in the
autograd engine and checkpoint therefore determine their actual meta lifetime.
"""
import collections
import contextlib
import threading
import traceback
import torch
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_flatten


ALLOCATION_GRANULARITY=512


def allocated_bytes(n,persistent):
    """Bytes the allocator charges for one storage of ``n`` requested bytes.

    Two rules are read off the historical allocator trace (requested vs allocated
    columns), not fitted per case:

    * initialization blocks (weights, buffers, residency) are charged one whole
      512-byte boundary above the request, even when the request is aligned;
    * runtime intermediates are only rounded up to the next 512-byte boundary.

    The rules are reproduced so the peak can be compared with an allocated peak.
    They are not a claim about CANN-internal workspace.
    """
    if persistent:
        return (n//ALLOCATION_GRANULARITY+1)*ALLOCATION_GRANULARITY
    return (n+ALLOCATION_GRANULARITY-1)//ALLOCATION_GRANULARITY*ALLOCATION_GRANULARITY


class StorageTracker(TorchDispatchMode):
    def __init__(self):
        super().__init__()
        self.live={};self.dead=[];self.total=0;self.requested=0;self.peak=0;self.peak_blocks=[]
        self.event=0;self.allocations=0;self.aliases=0;self.operations=collections.Counter()
        self.state=threading.local();self.phase='initialization';self.micro=-1
        self.events=[];self.track_events=False

    @contextlib.contextmanager
    def scope(self,phase,micro=None):
        old=(self.phase,self.micro);self.phase=phase
        if micro is not None:self.micro=micro
        try:yield
        finally:self.phase,self.micro=old

    def sweep(self):
        for key in list(self.live):
            b=self.live[key]
            if not b['persistent'] and torch.UntypedStorage._expired(b['weak']):
                self.total-=b['allocated'];self.live.pop(key)
                torch.UntypedStorage._free_weak_ref(b['weak'])
                if self.track_events:self.events.append(dict(action='free',id=b['id'],at=self.event))

    def register(self,tensor,category=None,name=None,persistent=False,op=None):
        if not isinstance(tensor,torch.Tensor) or tensor.device.type!='meta':return
        st=tensor.untyped_storage();key=st._cdata
        if key in self.live:
            self.aliases+=1
            if persistent:self.live[key]['persistent']=True
            if category:self.live[key]['category']=category
            return
        n=st.nbytes()
        if not n:return
        alloc=allocated_bytes(n,persistent)
        self.allocations+=1
        stack=traceback.extract_stack(limit=45) if category is None else []
        frames=[dict(file=f.filename.replace('\\','/').split('/')[-1],line=f.lineno,name=f.name) for f in stack
                if f.filename.endswith('.py') and not f.filename.endswith('storage_tracker.py')]
        category=category or self.classify(frames,op)
        b=dict(id=self.allocations,bytes=n,allocated=alloc,weak=st._weak_ref(),persistent=persistent,
               category=category,name=name,shape=list(tensor.shape),stride=list(tensor.stride()),
               dtype=str(tensor.dtype),phase=self.phase,micro=self.micro,op=str(op),frames=frames)
        self.live[key]=b;self.total+=alloc;self.requested+=n
        if self.track_events:self.events.append(dict(action='alloc',id=b['id'],bytes=n,category=category,at=self.event))
        if self.total>self.peak:
            self.peak=self.total
            self.peak_blocks=[{k:v for k,v in x.items() if k!='weak'} for x in self.live.values()]
            self.peak_context=dict(phase=self.phase,micro=self.micro,op=str(op),event=self.event)

    def classify(self,frames,op):
        pairs={(f['file'],f['name']) for f in frames}
        qlines={f['line'] for f in frames if f['file']=='modeling_qwen3_5.py'}
        if self.phase=='global_norm':return 'global_norm_temporary'
        if self.phase=='gradient_flush':return 'gradient_flush'
        if ('checkpoint.py','recompute_fn') in pairs:return 'checkpoint_recompute_intermediates'
        if any(f['file']=='loss_utils.py' for f in frames):return 'loss_forward_intermediates'
        if 736 in qlines:return 'ffn_forward_intermediates'
        if qlines & {747,750,753,754}:return 'rmsnorm_forward_intermediates'
        if any(600<=x<=721 for x in qlines):return 'attention_forward_intermediates'
        if any(f['file']=='community_gdn.py' for f in frames):return 'gdn_forward_intermediates'
        if 811 in qlines or 1174 in qlines:return 'hidden_states_and_boundaries'
        if self.phase=='backward':return 'backward_temporary'
        if self.phase=='input':return 'inputs_and_pipeline'
        return 'other_explicit_tensor'

    def __torch_dispatch__(self,func,types,args=(),kwargs=None):
        self.event+=1;self.operations[str(func)]+=1;self.sweep()
        result=func(*args,**(kwargs or {}))
        for t in tree_flatten(result)[0]:self.register(t,op=func)
        return result

    def summary(self):
        self.sweep();groups=collections.defaultdict(lambda:dict(bytes=0,requested_bytes=0,storages=0))
        for b in self.peak_blocks:
            g=groups[b['category']];g['bytes']+=b['allocated'];g['requested_bytes']+=b['bytes'];g['storages']+=1
        return dict(peak_bytes=self.peak,peak_gib=self.peak/2**30,
                    peak_requested_bytes=sum(b['bytes'] for b in self.peak_blocks),
                    peak_context=self.peak_context,live_end_bytes=self.total,
                    categories=[dict(category=k,**v,gib=v['bytes']/2**30) for k,v in groups.items()],
                    allocations=self.allocations,alias_results=self.aliases,
                    operation_count=sum(self.operations.values()),peak_blocks=self.peak_blocks)

    def close(self):
        for b in self.live.values():torch.UntypedStorage._free_weak_ref(b['weak'])
        self.live.clear()
