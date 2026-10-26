"""Separate producer and D2H streams; bounded pinned slots and ordered CPU adds."""
import queue,threading,time,traceback
import torch

def make_async_base(base,params,rank,device,args,scope):
    class AsyncOffload(base):
        def __init__(self):
            # Initialize before hooks can enqueue anything; base allocates slots.
            self.copy_stream=torch.npu.Stream(device=device)
            self.parameter_stats={}
            super().__init__()
        def worker(self):
            torch.npu.set_device(rank)
            while True:
                job=self.jobs.get()
                if job is None:self.jobs.task_done();return
                slot,idx,start,n,event=job
                try:
                    t=time.monotonic();event.synchronize();wait=time.monotonic()-t;self.copy_wait+=wait
                    t=time.monotonic()
                    with scope('gradient/cpu_add'):
                        self.accum[idx][start:start+n].add_(self.slots[slot][:n])
                    elapsed=time.monotonic()-t;self.cpu_add+=elapsed
                    stat=self.parameter_stats.setdefault(idx,dict(chunks=0,bytes=0,cpu_add_seconds=0.,copy_wait_seconds=0.))
                    stat['chunks']+=1;stat['bytes']+=n*2;stat['cpu_add_seconds']+=elapsed;stat['copy_wait_seconds']+=wait
                except BaseException:self.error=traceback.format_exc()
                finally:self.free.put(slot);self.jobs.task_done()
        def hook(self,idx):
            def take(param):
                grad=param.grad.detach().view(-1)
                ready=torch.npu.Event();ready.record(torch.npu.current_stream())
                self.copy_stream.wait_event(ready)
                for start in range(0,grad.numel(),self.chunk):
                    then=time.monotonic()
                    with scope('gradient/slot_wait'):slot=self.free.get()
                    self.wait+=time.monotonic()-then
                    assert self.error is None,self.error
                    n=min(self.chunk,grad.numel()-start)
                    with scope('gradient/d2h_enqueue'),torch.npu.stream(self.copy_stream):
                        self.slots[slot][:n].copy_(grad[start:start+n],non_blocking=True)
                        event=torch.npu.Event();event.record(self.copy_stream)
                    self.jobs.put((slot,idx,start,n,event));self.bytes+=n*2
                # Source storage must survive all queued copy-stream reads.
                grad.record_stream(self.copy_stream)
                param.grad=None
            return take
        def reset(self):
            super().reset();self.parameter_stats={}
    return AsyncOffload
