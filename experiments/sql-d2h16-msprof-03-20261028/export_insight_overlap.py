"""Export measured host add + native DMA fragments as a derived Insight trace.

Original profiler data is read-only. Host spans come from device-diagnostics;
their clock mapping is the same marker/hook alignment as analyze_msprof.py.
No synthetic scheduling or interpolated CPU work is introduced.
"""
import argparse, json, sqlite3, statistics
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument('--root', type=Path, required=True)
p.add_argument('--out', type=Path, required=True)
a = p.parse_args()
reports = json.loads((a.root/'msprof-analysis.private.json').read_text())['reports']
events = []
stats = []
def event(name, pid, tid, start, end, args, color):
    assert end >= start
    events.append(dict(name=name, cat='measured', ph='X', pid=pid, tid=tid,
                       ts=round(start*1e6,3), dur=round((end-start)*1e6,3),
                       args=args, cname=color))
def union(xs):
    out=[]
    for x,y in sorted(xs):
        if out and x<=out[-1][1]: out[-1][1]=max(out[-1][1],y)
        else: out.append([x,y])
    return out
def intersection(xs,ys):
    i=j=0; total=0
    while i<len(xs) and j<len(ys):
        total+=max(0,min(xs[i][1],ys[j][1])-max(xs[i][0],ys[j][0]))
        if xs[i][1]<ys[j][1]:i+=1
        else:j+=1
    return total
for r in reports:
    rank=r['rank'];card=r['physical_card'];pid=1000+card
    diag=json.loads((a.root/f'profile16.private/device-diagnostics-rank{rank}-step3.private.json').read_text())
    chunks=sorted(diag['chunks'],key=lambda x:(x['sequence'],x['element_offset']))
    assert diag['slots']==16 and len(chunks)==len(r['chunks'])
    dbpath=next((a.root/f'profile16.private/profile-rank{rank}').glob('*_ascend_pt/ASCEND_PROFILER_OUTPUT/ascend_pytorch_profiler_*.db'))
    db=sqlite3.connect('file:'+str(dbpath)+'?mode=ro',uri=True)
    db.execute('PRAGMA query_only=ON')
    markers=db.execute("SELECT MIN(CAST(p.startNs AS INTEGER)),n.value FROM PYTORCH_API p JOIN STRING_IDS n ON n.id=p.name WHERE n.value LIKE 'diag/gradient_hook/%' GROUP BY p.startNs,p.endNs,p.name,p.globalTid ORDER BY MIN(CAST(p.startNs AS INTEGER))").fetchall()
    owners={x['sequence']:x for x in chunks}
    owners=[owners[k] for k in sorted(owners)]
    assert len(markers)==len(owners)
    offsets=[]
    for m,o in zip(markers,owners):
        assert f"/p{o['param']}/" in m[1]
        offsets.append(m[0]-o['producer_hook_host']*1e9)
    base=statistics.median(offsets)
    native=db.execute('''SELECT MIN(t.startNs),MAX(t.endNs),MAX(m.size),MAX(t.connectionId)
        FROM MEMCPY_INFO m JOIN TASK t ON t.globalTaskId=m.globalTaskId
        WHERE t.streamId=? AND m.memcpyOperation=(SELECT id FROM ENUM_MEMCPY_OPERATION WHERE name='device to host')
        GROUP BY t.globalTaskId ORDER BY MIN(t.startNs)''',(r['copy_stream_id'],)).fetchall()
    assert sum(x[2] for x in native)==sum(x['bytes'] for x in chunks)
    for n,v in [('process_name',{'name':f'Physical card {card} | fixed16 measured'}),('process_sort_index',{'sort_index':card})]:
        events.append(dict(name=n,ph='M',pid=pid,tid=0,args=v))
    labels={1:'1. CPU gradient add',2:'2. D2H gradient DMA',3:'3. CPU waits for data',4:'4. All 16 slots occupied',5:'5. Copier waits for slot'}
    for tid,label in labels.items():
        events.extend([dict(name='thread_name',ph='M',pid=pid,tid=tid,args={'name':label}),dict(name='thread_sort_index',ph='M',pid=pid,tid=tid,args={'sort_index':tid})])
    cpu=[];dma=[]
    for d in chunks:
        x=d['cpu_add_start_host'];y=d['cpu_add_end_host'];cpu.append([x,y])
        args={'source':'measured host monotonic spans, mapped with profiler markers','sequence':d['sequence'],'parameter_index':d['param'],'bytes':d['bytes'],'slot':d['slot'],'element_offset':d['element_offset']}
        event('CPU add BF16 gradient',pid,1,x,y,args,'good')
        event('CPU waiting for chunk (not addition)',pid,3,d['cpu_get_wait_start_host'],d['cpu_get_return_host'],args,'rail_idle')
    for x,y,n,c in native:
        x=(x-base)/1e9;y=(y-base)/1e9;dma.append([x,y])
        event('D2H gradient DMA',pid,2,x,y,{'source':'native msprof MEMCPY_INFO + TASK','bytes':n,'connection_id':c},'thread_state_runnable')
    transitions=sorted([(d['copy_dispatch_host'],1) for d in chunks]+[(d['cpu_add_end_host'],-1) for d in chunks])
    occupied=0; full=[]
    for tm,delta in transitions:
        if occupied==16 and delta<0: full.append([full_start,tm])
        occupied+=delta
        assert 0<=occupied<=16
        if occupied==16: full_start=tm
    for x,y in full:
        event('ALL 16 receiving slots occupied',pid,4,x,y,{'source':'derived from measured slot reservation and CPU-add completion','meaning':'occupied includes copying, ready, or CPU processing; not 2 GiB of valid payload'},'terrible')
    host=json.loads((a.root/f'profile16.private/host-trace-rank{rank}-step3.private.json').read_text())['traceEvents']
    slot_wait=[e for e in host if e['name']=='copy_dispatch/wait_cpu_slot']
    for e in slot_wait:
        event('Copier waiting for free CPU slot',pid,5,e['ts']/1e6,(e['ts']+e['dur'])/1e6,{'source':'measured queue.get span','duration_includes_queue_call_overhead':True},'bad')
    longest=max(full,key=lambda x:x[1]-x[0]);mid=sum(longest)/2
    sample=[d for d in chunks if d['copy_dispatch_host']<=mid<d['cpu_add_end_host']]
    example=dict(start_seconds=longest[0],end_seconds=longest[1],occupied_slots=len(sample),ready_slots=sum(d['completion_notice_host']<=mid<d['cpu_add_start_host'] for d in sample),processing_slots=sum(d['cpu_add_start_host']<=mid for d in sample),valid_payload_mib=sum(d['bytes'] for d in sample)/2**20)
    stats.append(dict(physical_card=card,cpu_add_seconds=sum(y-x for x,y in cpu),dma_active_seconds=sum(y-x for x,y in dma),overlap_seconds=intersection(union(cpu),union(dma)),host_clock_offset_mad_us=r['host_clock_offset_mad_us'],dma_bytes=sum(x[2] for x in native),cpu_chunks=len(chunks),dma_fragments=len(native),all16_occupied_seconds=sum(y-x for x,y in full),all16_occupied_episodes=len(full),copier_wait_free_slot_seconds=sum(e['dur'] for e in slot_wait)/1e6,copier_wait_over1ms_seconds=sum(e['dur'] for e in slot_wait if e['dur']>1000)/1e6,longest_full_example=example))
    db.close()
a.out.mkdir(parents=True,exist_ok=True)
(a.out/'trace_view.json').write_text(json.dumps(events,separators=(',',':')))
(a.out/'summary.json').write_text(json.dumps({'statistics':stats,'sources':{'cpu':'per-chunk measured host spans (not native profiler CPU worker events)','dma':'unmodified native msprof hardware TASK spans'},'limits':['Derived diagnostic trace, not complete original model trace.','Clock alignment has the reported median absolute deviation; overlap is approximate.','CPU means BF16 gradient addition only, excludes CPUAdam.','Durations overlap and must not be added.'],'fixed_slots_per_rank':16},indent=2))
print(json.dumps(stats,indent=2))
