"""Observe existing codec/optimizer scopes without changing numerical work.

Event spans exclude operations queued before the start event. Wall spans include
Python/ACL waits; nested scopes have exclusive accounting. Event spans are not
pure kernel busy time and must not be added to overlapping host wall time.
"""
import collections
import contextlib
import functools
import json
import threading
import time
import torch

ACTIVE=False
PHASE='disabled'
ROWS=[]
LOCAL=threading.local()

def stack():
    if not hasattr(LOCAL,'stack'):LOCAL.stack=[]
    return LOCAL.stack

@contextlib.contextmanager
def scope(name,device,nbytes=0):
    if not ACTIVE:
        yield
        return
    device=torch.device(device)
    with torch.npu.device(device):
        start=torch.npu.Event(enable_timing=True)
        end=torch.npu.Event(enable_timing=True)
        parent=stack()[-1] if stack() else None
        row={'name':name,'phase':PHASE,'device':device.index,'bytes':nbytes,
             'parent':parent,'start':start,'end':end}
        index=len(ROWS);ROWS.append(row);stack().append(index)
        t=time.perf_counter();start.record()
        try:yield
        finally:
            end.record();row['wall_seconds']=time.perf_counter()-t
            assert stack().pop()==index

def install():
    import resident
    import bench_resident as bench
    old_init=resident.Codec.__init__
    @functools.wraps(old_init)
    def init(self,*args,**kwargs):
        old_init(self,*args,**kwargs)
        compress=self.lib.online_compress
        decompress=self.lib.online_decompress
        def timed_compress(stream,device,source,n,*rest):
            with scope('enec_compress_bridge','npu:'+str(device),n):
                return compress(stream,device,source,n,*rest)
        def timed_decompress(stream,device,*rest):
            verifying=any(ROWS[i]['name']=='weight_encode_total' for i in stack())
            name='enec_verify_decompress_bridge' if verifying else 'enec_read_decompress_bridge'
            with scope(name,'npu:'+str(device)):
                return decompress(stream,device,*rest)
        self.lib.online_compress=timed_compress
        self.lib.online_decompress=timed_decompress
    resident.Codec.__init__=init
    old_decode=resident.Codec.decode
    @functools.wraps(old_decode)
    def decode(self,frame,*args,**kwargs):
        verifying=any(ROWS[i]['name']=='weight_encode_total' for i in stack())
        kind='verify' if verifying else 'read'
        with scope('weight_'+kind+'_decode_'+frame['kind'],frame['data'].device,frame['n']*4):
            return old_decode(self,frame,*args,**kwargs)
    resident.Codec.decode=decode
    old_encode=resident.Codec.encode
    @functools.wraps(old_encode)
    def encode(self,w):
        with scope('weight_encode_total',w.device,w.numel()*4):return old_encode(self,w)
    resident.Codec.encode=encode
    old_update=bench.TrackedQuantMoments.update
    @functools.wraps(old_update)
    def update(self,start,w,g,step):
        with scope('moment_update_quantization_and_audit',w.device,w.numel()*4):
            return old_update(self,start,w,g,step)
    bench.TrackedQuantMoments.update=update


def begin(phase='forward'):
    global ACTIVE,PHASE
    assert not ACTIVE
    ROWS.clear();PHASE=phase;ACTIVE=True

def finish(path):
    global ACTIVE
    ACTIVE=False
    for device in [0,1]:torch.npu.synchronize(device)
    for row in ROWS:
        row['event_seconds']=row['start'].elapsed_time(row['end'])/1000
        row['exclusive_wall_seconds']=row['wall_seconds']
        row['exclusive_event_seconds']=row['event_seconds']
    for row in ROWS:
        if row['parent'] is not None:
            parent=ROWS[row['parent']]
            parent['exclusive_wall_seconds']-=row['wall_seconds']
            parent['exclusive_event_seconds']-=row['event_seconds']
    grouped={}
    for row in ROWS:
        key=row['phase']+'/'+row['name'];g=grouped.setdefault(key,collections.Counter())
        g['calls']+=1;g['raw_bytes']+=row['bytes']
        for k in ['wall_seconds','event_seconds','exclusive_wall_seconds','exclusive_event_seconds']:g[k]+=row[k]
    result={'timing':'NPU event spans and nested exclusive host wall time. Event spans include device idle/host submission gaps inside the scope, not just kernel busy time. Across devices spans may overlap; do not sum event and wall times.','groups':grouped,'scopes':len(ROWS)}
    path.write_text(json.dumps(result,indent=2))
    return result
