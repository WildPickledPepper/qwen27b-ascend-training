"""FP32 Adam with bounded pinned staging and two reusable NPU slots.

CPU moments remain full, uncompressed FP32. Same explicit Adam arithmetic as
NpuOffloadedMoments. No CPU optimizer and no whole-state pinning. Slots wait on
per-stage events before host reuse; update returns after every CPU state commit.
"""
import time
import threading
import torch
import bench_resident as bench
from bench_resident import CpuMoments,probe_before,probe_after,TIMES

POOLS={}
DETAIL={'host_stage_in_seconds':0.,'host_stage_out_seconds':0.,'host_slot_wait_seconds':0.,
        'h2d_event_seconds':0.,'adam_event_seconds':0.,'d2h_event_seconds':0.,'windows':0}

class Pool:
    def __init__(self,device,window):
        self.device=torch.device(device);self.window=window
        with torch.npu.device(self.device):
            self.h2d=torch.npu.Stream();self.d2h=torch.npu.Stream()
            self.slots=[]
            for _ in range(2):
                host=torch.empty((2,window),dtype=torch.float32,pin_memory=True)
                assert host.is_pinned()
                self.slots.append({'host':host,'device':torch.empty((2,window),device=self.device,dtype=torch.float32),
                    'events':[torch.npu.Event(enable_timing=True) for _ in range(6)],'pending':None})

def pool(device,window):
    key=(str(device),window,threading.get_ident())
    if key not in POOLS:POOLS[key]=Pool(device,window)
    return POOLS[key]

class StagedNpuMoments(CpuMoments):
    @torch.no_grad()
    def update(self,start,w,g,step):
        token=probe_before(self.name,self.n,start,w,g,step)
        if not torch.isfinite(g).all().item():raise FloatingPointError('Nonfinite offloaded-state gradient')
        with torch.npu.device(w.device):
            buffers=pool(w.device,self.window)
            compute=torch.npu.current_stream(w.device)
            def commit(slot):
                if slot['pending'] is None:return
                begin,length=slot['pending'];events=slot['events']
                t=time.monotonic();events[5].synchronize()
                DETAIL['host_slot_wait_seconds']+=time.monotonic()-t
                t=time.monotonic()
                self.m[begin:begin+length].copy_(slot['host'][0,:length])
                self.v[begin:begin+length].copy_(slot['host'][1,:length])
                DETAIL['host_stage_out_seconds']+=time.monotonic()-t
                for key,i in [('h2d_event_seconds',0),('adam_event_seconds',2),('d2h_event_seconds',4)]:
                    DETAIL[key]+=events[i].elapsed_time(events[i+1])/1000
                DETAIL['windows']+=1
                self.updated+=length;slot['pending']=None
                if self.progress and (self.updated%(8<<20)==0 or self.updated%self.n==0):self.progress()
            assert all(s['pending'] is None for s in buffers.slots)
            for index,off in enumerate(range(0,w.numel(),self.window)):
                slot=buffers.slots[index%2];commit(slot)
                length=min(self.window,w.numel()-off);begin=start+off;ev=slot['events']
                t=time.monotonic()
                slot['host'][0,:length].copy_(self.m[begin:begin+length])
                slot['host'][1,:length].copy_(self.v[begin:begin+length])
                DETAIL['host_stage_in_seconds']+=time.monotonic()-t
                with torch.npu.stream(buffers.h2d):
                    ev[0].record();slot['device'][:,:length].copy_(slot['host'][:,:length],non_blocking=True);ev[1].record()
                compute.wait_event(ev[1])
                with torch.npu.stream(compute):
                    ev[2].record()
                    m=slot['device'][0,:length];v=slot['device'][1,:length]
                    chunk=w[off:off+length];grad=g[off:off+length].float()
                    m.mul_(.9).add_(grad,alpha=.1)
                    v.mul_(.999).addcmul_(grad,grad,value=.001)
                    chunk.mul_(1-self.lr*self.wd).addcdiv_(m/(1-.9**step),(v/(1-.999**step)).sqrt().add_(1e-8),value=-self.lr)
                    ev[3].record()
                buffers.d2h.wait_event(ev[3])
                with torch.npu.stream(buffers.d2h):
                    ev[4].record();slot['host'][:,:length].copy_(slot['device'][:,:length],non_blocking=True);ev[5].record()
                slot['pending']=(begin,length)
            for slot in buffers.slots:commit(slot)
        probe_after(self.name,step,w,token)


def install():
    bench.base.QuantMoments=StagedNpuMoments
