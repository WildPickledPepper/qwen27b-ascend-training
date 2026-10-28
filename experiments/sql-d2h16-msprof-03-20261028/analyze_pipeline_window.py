"""Read-only device computation analysis for the user's selected CPU starvation.

The window contains Insight marker 38.570s (step time +1.359057660s).
Native compute intervals, not CPU API durations, determine device coverage.
"""
import argparse,json,sqlite3,statistics
from pathlib import Path

p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
reports=json.loads((a.root/'msprof-analysis.private.json').read_text())['reports']
target=next(c for c in reports[1]['chunks'] if c['cpu_get_wait_start_seconds']<=39.929057660<=c['cpu_get_return_seconds'])
lo=target['cpu_get_wait_start_seconds'];hi=target['cpu_get_return_seconds']
events=json.loads((a.root/'insight-slot-proof.private/trace_view.json').read_text());summary=[]
macro='''WITH compute_view AS (
    SELECT c.globalTaskId, ROUND(t.endNs - t.startNs) AS duration_ns, n.value AS op_name, type_str.value AS op_type
    FROM COMPUTE_TASK_INFO c
    LEFT JOIN TASK t ON t.globalTaskId = c.globalTaskId
    LEFT JOIN STRING_IDS n ON n.id = c.name
    LEFT JOIN STRING_IDS type_str ON type_str.id = c.opType
)'''
def union(xs):
    out=[]
    for x,y in sorted(xs):
        if out and x<=out[-1][1]:out[-1][1]=max(out[-1][1],y)
        else:out.append([x,y])
    return out
def overlap(xs,ys):
    i=j=0;total=0
    while i<len(xs) and j<len(ys):
        total+=max(0,min(xs[i][1],ys[j][1])-max(xs[i][0],ys[j][0]))
        if xs[i][1]<ys[j][1]:i+=1
        else:j+=1
    return total
for r in reports:
    rank=r['rank'];pid=1000+r['physical_card']
    db=sqlite3.connect('file:'+str(next((a.root/f'profile16.private/profile-rank{rank}').glob('*_ascend_pt/ASCEND_PROFILER_OUTPUT/ascend_pytorch_profiler_*.db')))+'?mode=ro',uri=True);db.execute('PRAGMA query_only=ON')
    ds=json.loads((a.root/f'profile16.private/device-diagnostics-rank{rank}-step3.private.json').read_text())['chunks']
    owners={c['sequence']:c for c in sorted(ds,key=lambda c:c['element_offset'],reverse=True)}
    ms=db.execute("SELECT MIN(CAST(p.startNs AS INTEGER)),n.value FROM PYTORCH_API p JOIN STRING_IDS n ON n.id=p.name WHERE n.value LIKE 'diag/gradient_hook/%' GROUP BY p.startNs,p.endNs,p.name,p.globalTid ORDER BY MIN(CAST(p.startNs AS INTEGER))").fetchall()
    offsets=[]
    for m,k in zip(ms,sorted(owners)):
        assert f"/p{owners[k]['param']}/" in m[1]
        offsets.append(m[0]-owners[k]['producer_hook_host']*1e9)
    base=statistics.median(offsets);start=base+lo*1e9;end=base+hi*1e9
    compute=db.execute(macro+''' SELECT MIN(t.startNs),MAX(t.endNs),MAX(v.op_name) FROM compute_view v JOIN TASK t ON t.globalTaskId=v.globalTaskId WHERE t.startNs<? AND t.endNs>? GROUP BY v.globalTaskId,t.startNs,t.endNs ORDER BY MIN(t.startNs)''',(end,start)).fetchall()
    comm=db.execute('''SELECT MIN(c.startNs),MAX(c.endNs),MAX(n.value) FROM COMMUNICATION_OP c LEFT JOIN STRING_IDS n ON n.id=c.opName WHERE c.startNs<? AND c.endNs>? GROUP BY c.opId,c.opName ORDER BY MIN(c.startNs) LIMIT 20''',(end,start)).fetchall()
    scopes=db.execute('''SELECT MIN(CAST(p.startNs AS INTEGER)),MAX(CAST(p.endNs AS INTEGER)),n.value FROM PYTORCH_API p JOIN STRING_IDS n ON n.id=p.name WHERE n.value LIKE 'pipeline/%' AND CAST(p.startNs AS INTEGER)<? AND CAST(p.endNs AS INTEGER)>? GROUP BY p.startNs,p.endNs,p.name,p.globalTid ORDER BY MIN(CAST(p.startNs AS INTEGER)) LIMIT 20''',(end,start)).fetchall()
    merged=union([[max(lo,(x-base)/1e9),min(hi,(y-base)/1e9)] for x,y,n in compute]);gaps=[];prev=lo
    for x,y in merged:
        if x>prev:gaps.append([prev,x])
        prev=y
    if prev<hi:gaps.append([prev,hi])
    host=json.loads((a.root/f'profile16.private/host-trace-rank{rank}-step3.private.json').read_text())['traceEvents']
    budget=union([[max(lo,e['ts']/1e6),min(hi,(e['ts']+e['dur'])/1e6)] for e in host if e['name']=='main/wait_device_backlog_budget' and e['ts']/1e6<hi and (e['ts']+e['dur'])/1e6>lo])
    labels={6:'6. NPU compute (native TASK)',7:'7. NPU communication',8:'8. Host forward scope',9:'9. Host backward scope',10:'10. Main waits for backlog'}
    for tid,name in labels.items():
        events.extend([dict(ph='M',name='thread_name',pid=pid,tid=tid,args={'name':name}),dict(ph='M',name='thread_sort_index',pid=pid,tid=tid,args={'sort_index':tid})])
    def add(tid,name,x,y,source):
        events.append(dict(ph='X',cat='measured',name=name,pid=pid,tid=tid,ts=round(x*1e6,3),dur=round((y-x)*1e6,3),args={'source':source,'window_only':True}))
    for x,y,n in compute:add(6,n,(x-base)/1e9,(y-base)/1e9,'native COMPUTE_TASK_INFO/TASK')
    for x,y,n in comm:add(7,n,(x-base)/1e9,(y-base)/1e9,'native COMMUNICATION_OP')
    for x,y,n in scopes:
        if n.endswith('/forward'):add(8,n,(x-base)/1e9,(y-base)/1e9,'host profiler record_function; not device duration')
        if n.endswith('/backward'):add(9,n,(x-base)/1e9,(y-base)/1e9,'host profiler record_function; not device duration')
    for x,y in budget:add(10,'Main blocked by bounded device backlog',x,y,'measured host condition wait span')
    summary.append(dict(physical_card=r['physical_card'],window_seconds=[lo,hi],window_ms=(hi-lo)*1000,compute_union_ms=sum(y-x for x,y in merged)*1000,compute_coverage=sum(y-x for x,y in merged)/(hi-lo),compute_tasks=len(compute),communication=[dict(name=n,start_seconds=(x-base)/1e9,end_seconds=(y-base)/1e9) for x,y,n in comm],host_scopes=[dict(name=n,start_seconds=(x-base)/1e9,end_seconds=(y-base)/1e9) for x,y,n in scopes],main_backlog_wait_ms=sum(y-x for x,y in budget)*1000,compute_gaps_covered_by_backlog_wait_ms=overlap(gaps,budget)*1000,longest_compute_gaps=[dict(start_seconds=x,end_seconds=y,ms=(y-x)*1000,backlog_wait_overlap_ms=overlap([[x,y]],budget)*1000) for x,y in sorted(gaps,key=lambda x:x[1]-x[0],reverse=True)[:10]],max_compute_gap_ms=max((y-x)*1000 for x,y in gaps)))
    db.close()
a.out.mkdir(parents=True,exist_ok=True)
(a.out/'trace_view.json').write_text(json.dumps(events,separators=(',',':')))
(a.out/'summary.json').write_text(json.dumps(dict(selected_cpu_wait=target,device_windows=summary,queries={'compute':macro},limits=['Only selected screenshot window; no whole-step conclusion.','Host scopes show CPU submission phases, not exact device forward/backward duration.','Compute coverage denotes task presence, not full hardware utilization.','Cross-rank step-relative clocks use their respective measured step origins.']),indent=2))
print(json.dumps(summary,indent=2))
