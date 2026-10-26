"""Bounded BF16 gradient residency; original microbatch addition order retained."""
import types
import torch

def create_offload_class(base,params,rank,device,args,scope):
    class ResidentOffload(base):
        def __init__(self):
            budget=int(args.resident_gib*2**30)
            assert 0<=budget<=4*2**30,'Only a bounded experimental cache is authorized'
            used=0;chosen=[]
            for idx,param in sorted(enumerate(params),key=lambda x:(-x[1].numel(),x[0])):
                size=param.numel()*2
                if size<=budget-used:
                    chosen.append(idx);used+=size
            self.resident={idx:torch.zeros(params[idx].numel(),device=device,dtype=torch.bfloat16) for idx in chosen}
            self.ready=set();self.resident_add_calls=0;self.resident_add_bytes=0
            self.resident_plan=dict(budget_bytes=budget,allocated_bytes=used,parameter_indices=sorted(chosen),
                                    selected_parameters=len(chosen),total_parameters=len(params),
                                    dtype='BF16',order='same per-parameter 16 microbatch additions',
                                    flush='once after all microbatches; no early optimizer update')
            super().__init__()
        def hook(self,idx):
            if idx not in self.resident:return super().hook(idx)
            def take(param):
                with scope('gradient/resident_add'):
                    self.resident[idx].add_(param.grad.detach().view(-1))
                self.ready.add(idx);self.resident_add_calls+=1
                self.resident_add_bytes+=param.numel()*2
                param.grad=None
            return take
        def finish(self):
            with scope('gradient/resident_flush'):
                for idx in sorted(self.ready):
                    # CPU target is zero for resident parameters. Reuse original
                    # bounded D2H slots/event lifetimes and FIFO worker unchanged.
                    base.hook(self,idx)(types.SimpleNamespace(grad=self.resident[idx]))
                self.ready.clear()
            self.drain()
        def reset(self):
            assert not self.ready,'Must finish accumulated gradients before resetting'
            super().reset()
            for buf in self.resident.values():buf.zero_()
            self.resident_add_calls=0;self.resident_add_bytes=0
    return ResidentOffload
