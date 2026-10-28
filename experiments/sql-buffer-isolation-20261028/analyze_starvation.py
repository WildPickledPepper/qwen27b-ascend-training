"""Read existing host traces only; never initializes Torch or NPU."""
import argparse,bisect,json,hashlib
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();records=[]
for rank in [0,1]:
 for step in [2,3]:
  path=a.input/f'host-trace-rank{rank}-step{step}.private.json';es=json.loads(path.read_text())['traceEvents']
  g=lambda name:sorted([e for e in es if e['name']==name],key=lambda e:e['ts'])
  aa=g('CPU/BF16_gradient_add');ww=g('CPU/wait_ready_data');ee=g('D2H/enqueue_only');dd=g('D2H/wait_completion_includes_dependencies');ss=g('copy_dispatch/wait_cpu_slot')
  assert len(aa)==len(ww)==len(ee)==len(dd)==len(ss)
  ast=[e['ts'] for e in aa];ae=[e['ts']+e['dur'] for e in aa];endq=[e['ts']+e['dur'] for e in ee];don=[e['ts']+e['dur'] for e in dd];acq=[e['ts']+e['dur'] for e in ss]
  first=ast[0];last=ae[-1];tot=[0.]*4;long=[]
  for i,(w,e,d) in enumerate(zip(ww,ee,dd)):
   lo=max(first,w['ts']);hi=min(last,w['ts']+w['dur'])
   if hi<=lo:continue
   est=e['ts'];en=est+e['dur'];done=d['ts']+d['dur']
   vals=[max(0,min(hi,est)-lo),max(0,min(hi,en)-max(lo,est)),max(0,min(hi,done)-max(lo,en)),max(0,hi-max(lo,done))]
   assert abs(sum(vals)-(hi-lo))<1
   tot=[x+y/1e6 for x,y in zip(tot,vals)]
   if hi-lo>200000:
    t=(lo+hi)/2;consumed=bisect.bisect_right(ast,t);finished=bisect.bisect_right(ae,t);occupied=bisect.bisect_right(acq,t)-finished;ready=bisect.bisect_right(don,t)-consumed
    assert 0<=occupied<=32 and ready>=0
    long.append(dict(block_index=i,start_seconds=lo/1e6,end_seconds=hi/1e6,wait_seconds=(hi-lo)/1e6,enqueue_start_seconds=est/1e6,enqueue_end_seconds=en/1e6,completion_notice_seconds=done/1e6,wait_split_seconds=vals and [x/1e6 for x in vals],midpoint_slot_estimate=dict(ready=ready,occupied=occupied,empty=32-occupied,enqueued_but_unconfirmed=bisect.bisect_right(endq,t)-bisect.bisect_right(don,t))))
  records.append(dict(rank=rank,physical_card=rank+3,step=step,trace_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),steady_ready_wait_split_seconds=dict(zip(['before_needed_copy_call','inside_copy_enqueue_call','after_enqueue_until_completion_notice','after_notice_until_cpu_get'],tot)),long_waits=long))
report=dict(scope='Existing 4/8GiB ready-buffer experiment,32slots; first-to-last CPU add, excludes initial fill; slot states reconstructed at wait midpoint, not direct snapshots',records=records,limits=['Completion monitor timestamp can lag real DMA completion; after-enqueue wait includes producer dependencies/runtime and transfer, not pure link time.','Slot acquire/CPU-add-end timestamp differs from actual queue bookkeeping by small host overhead.','No job identity in trace; FIFO counts and equal event cardinalities provide chunk alignment.','Cannot identify exact NPU compute op from host spans alone.'])
a.output.write_text(json.dumps(report,indent=2))
print(json.dumps([dict(rank=x['rank'],step=x['step'],split=x['steady_ready_wait_split_seconds'],long_wait_count=len(x['long_waits'])) for x in records]))
