"""Read-only official msprof DB + fixed16 per-chunk diagnostics.

Schema reference: ascend-profiler-db-explorer/scripts/get_schema.py.
Native MEMCPY_INFO/TASK determine DMA service; device events bound source readiness.
"""
import argparse, bisect, hashlib, json, math, re, sqlite3, statistics
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument('--folder', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
compute_macro = '''WITH compute_view AS (
    SELECT c.globalTaskId, ROUND(t.endNs - t.startNs) AS duration_ns, n.value AS op_name, type_str.value AS op_type
    FROM COMPUTE_TASK_INFO c
    LEFT JOIN TASK t ON t.globalTaskId = c.globalTaskId
    LEFT JOIN STRING_IDS n ON n.id = c.name
    LEFT JOIN STRING_IDS type_str ON type_str.id = c.opType
)'''
comm_macro = '''WITH comm_view AS (
    SELECT ROUND(c.endNs - c.startNs) AS duration_ns, n.value AS op_name, t.value AS op_type, g.value AS group_name
    FROM COMMUNICATION_OP c
    LEFT JOIN STRING_IDS n ON n.id = c.opName
    LEFT JOIN STRING_IDS t ON t.id = c.opType
    LEFT JOIN STRING_IDS g ON g.id = c.groupName
)'''
copy_sql = '''SELECT t.connectionId, MIN(t.startNs), MAX(t.endNs), SUM(m.size), MAX(t.streamId),
 MIN(c.startNs), MAX(c.endNs), MAX(n.value),COUNT(*),SUM(t.endNs-t.startNs)
 FROM MEMCPY_INFO m JOIN TASK t ON t.globalTaskId=m.globalTaskId
 LEFT JOIN CANN_API c ON c.connectionId=t.connectionId
 LEFT JOIN STRING_IDS n ON n.id=c.name
 WHERE m.memcpyOperation=?
 GROUP BY t.connectionId ORDER BY MIN(t.startNs)'''
marker_sql = '''SELECT MIN(CAST(p.startNs AS INTEGER)), MAX(CAST(p.endNs AS INTEGER)), n.value
 FROM PYTORCH_API p JOIN STRING_IDS n ON n.id=p.name
 WHERE n.value LIKE 'diag/gradient_hook/%'
 GROUP BY p.startNs,p.endNs,p.name,p.globalTid ORDER BY MIN(CAST(p.startNs AS INTEGER))'''

def quant(xs):
    v = sorted(xs)
    if not v:
        return {}
    return {str(q):v[round(q*(len(v)-1))] for q in [0,.5,.9,.99,1]}

reports = []
for rank in [0,1]:
    framework = next((a.folder/f'profile-rank{rank}').glob('*_ascend_pt'))
    output = framework/'ASCEND_PROFILER_OUTPUT'
    dbpath = next(output.glob('ascend_pytorch_profiler_*.db'))
    # Front check uses rank1, the primary all-slots-empty case.
    validation = None
    if rank == 1:
        info = json.loads(next(framework.glob('profiler_info*.json')).read_text())
        assert info.get('end_info'), 'Profiler did not stop'
        assert (output/'trace_view.json').exists() and (output/'kernel_details.csv').exists()
        validation = dict(type='framework PyTorch', stopped=True, parsed=True,
                          config=info['config'], db_present=True, trace_present=True, kernel_csv_present=True)
    db = sqlite3.connect('file:'+str(dbpath)+'?mode=ro', uri=True)
    db.execute('PRAGMA query_only=ON')
    db.row_factory = None
    comm_overview = db.execute(comm_macro+''' SELECT op_type,COUNT(*),SUM(duration_ns)
      FROM comm_view GROUP BY op_type ORDER BY SUM(duration_ns) DESC LIMIT 20''').fetchall()
    enums = dict(db.execute('SELECT id,name FROM ENUM_MEMCPY_OPERATION ORDER BY id LIMIT 20'))
    direction = next(k for k,v in enums.items() if v == 'device to host')
    diagpath = a.folder/f'device-diagnostics-rank{rank}-step3.private.json'
    diag = json.loads(diagpath.read_text())
    assert diag['slots'] == 16
    cs = sorted(diag['chunks'], key=lambda x:(x['sequence'],x['element_offset']))
    rows = [json.loads(x) for x in (a.folder/f'rank{rank}.jsonl').read_text().splitlines()]
    manifest = next(x for x in rows if x['event']=='model_ready')['manifest']
    step = next(x for x in rows if x['event']=='step' and x['step']==3)
    assert sum(x['bytes'] for x in cs) == step['gradient_d2h_bytes']
    owners = {x['sequence']:x for x in cs}
    owners = [owners[k] for k in sorted(owners)]
    markers = db.execute(marker_sql).fetchall()
    assert len(markers)==len(owners), (rank,len(markers),len(owners))
    offsets = []
    for m,o in zip(markers,owners):
        assert f"/p{o['param']}/" in m[2], (m,o['param'])
        offsets.append(m[0]-o['producer_hook_host']*1e9)
    host_base = statistics.median(offsets)
    host_mad = statistics.median(abs(x-host_base) for x in offsets)
    copies = db.execute(copy_sql,(direction,)).fetchall()
    streams = {}
    for x in copies:
        streams.setdefault(x[4],[]).append(x)
    matching = [(sid,xs) for sid,xs in streams.items() if len(xs)==len(cs) and sum(x[3] for x in xs)==sum(x['bytes'] for x in cs)]
    assert len(matching)==1, (rank,[(sid,len(xs),sum(x[3] for x in xs)) for sid,xs in streams.items()],len(cs))
    stream,copies = matching[0]
    copies.sort(key=lambda x:x[1])
    for c,d in zip(copies,cs):
        assert c[3]==d['bytes'], (c[3],d['bytes'])
    lower = [c[2]-d['device_copy_end_ms']*1e6 for c,d in zip(copies,cs)]
    upper = [c[1]-d['device_copy_start_ms']*1e6 for c,d in zip(copies,cs)]
    global_lo,global_hi = max(lower),min(upper)
    common = global_lo<=global_hi
    first_add = min(x['cpu_add_start_host'] for x in cs)
    waits = []; details = []
    for i,(c,d) in enumerate(zip(copies,cs)):
        taskid,start,end,n,sid,api_start,api_end,api_name = c[:8]
        lo,hi = lower[i],upper[i]
        assert hi>=lo-50000, ('event/native clock mismatch',lo,hi)
        base = (global_lo+global_hi)/2 if common else (lo+hi)/2
        uncertainty = max(50000,(global_hi-global_lo)/2 if common else abs(hi-lo)/2)
        source_ready = base+d['device_source_ready_ms']*1e6
        w0 = host_base+d['cpu_get_wait_start_host']*1e9
        w1 = host_base+d['cpu_get_return_host']*1e9
        native_ms = (end-start)/1e6
        item = dict(sequence=d['sequence'],param=d['param'],param_name=manifest[d['param']]['name'],
                    param_shape=manifest[d['param']]['shape'],element_offset=d['element_offset'],bytes=n,
                    device_dma_ms=native_ms,native_fragment_count=c[8],native_dma_active_sum_ms=c[9]/1e6,native_fragment_gap_ms=native_ms-c[9]/1e6,event_bounded_copy_ms=d['device_copy_service_ms'],
                    gib_per_second=n/2**30/(native_ms/1000) if native_ms>0 else None,
                    host_copy_call_ms=(api_end-api_start)/1e6 if api_start is not None else None,
                    runtime_api=api_name,runtime_api_start_seconds=(api_start-host_base)/1e9 if api_start is not None else None,runtime_api_end_seconds=(api_end-host_base)/1e9 if api_end is not None else None,
                    producer_hook_seconds=d['producer_hook_host'],producer_admit_seconds=d['producer_admit_host'],
                    source_ready_seconds=(source_ready-host_base)/1e9,
                    native_copy_start_seconds=(start-host_base)/1e9,native_copy_end_seconds=(end-host_base)/1e9,
                    cpu_get_wait_start_seconds=d['cpu_get_wait_start_host'],cpu_get_return_seconds=d['cpu_get_return_host'],
                    cpu_get_wait_ms=(w1-w0)/1e6,clock_source_uncertainty_ms=uncertainty/1e6,
                    snapshot=d.get('snapshot_before_get'))
        details.append(item)
        if d['cpu_get_wait_start_host']>=first_add and w1-w0>.1e9:
            # Pure service comes from native TASK; prior phases use source-ready marker.
            pr = min(source_ready,float(start))
            portions = [max(0,min(w1,pr)-w0),max(0,min(w1,start)-max(w0,pr)),
                        max(0,min(w1,end)-max(w0,start)),max(0,w1-max(w0,end))]
            assert abs(sum(portions)-(w1-w0))<4096
            item['wait_split_ms'] = dict(zip(['before_source_ready','source_ready_until_DMA','actual_DMA','DMA_done_until_CPU_get'],[x/1e6 for x in portions]))
            item['native_connection_id'] = taskid
            item['_absolute_wait_start'] = w0
            item['_absolute_source_ready'] = source_ready
            waits.append(item)
    # Probe long wait windows with the unmodified official Compute CTE.
    for item in sorted(waits,key=lambda x:x['cpu_get_wait_ms'],reverse=True)[:4]:
        start = int(item['_absolute_wait_start']); end = int(item['_absolute_source_ready'])
        if end>start:
            sql = compute_macro+''' SELECT v.op_name,COUNT(*),SUM(MAX(0,MIN(t.endNs,?)-MAX(t.startNs,?))) AS overlap_ns
             FROM compute_view v JOIN TASK t ON t.globalTaskId=v.globalTaskId
             WHERE t.startNs<? AND t.endNs>? GROUP BY v.op_name ORDER BY overlap_ns DESC LIMIT 10'''
            item['compute_tasks_before_source_ready'] = db.execute(sql,(end,start,end,start)).fetchall()
            interval_sql = compute_macro+''' SELECT MIN(t.startNs),MAX(t.endNs),MAX(v.op_name),COUNT(*)
             FROM compute_view v JOIN TASK t ON t.globalTaskId=v.globalTaskId
             WHERE t.startNs<? AND t.endNs>? GROUP BY v.globalTaskId,t.startNs,t.endNs
             ORDER BY MIN(t.startNs)'''
            intervals = db.execute(interval_sql,(end,start)).fetchall()
            clipped = sorted((max(start,x[0]),min(end,x[1])) for x in intervals if min(end,x[1])>max(start,x[0]))
            merged = []
            for x,y in clipped:
                if merged and x<=merged[-1][1]:
                    merged[-1][1] = max(merged[-1][1],y)
                else:
                    merged.append([x,y])
            item['compute_union_ms_before_source_ready'] = sum(y-x for x,y in merged)/1e6
            item['compute_intervals_relative_step'] = [[(x-host_base)/1e9,(y-host_base)/1e9,n] for x,y,n,_ in intervals]
            # Track B: explicit hardware time-window question; schema from skill reference.
            comm_sql = '''SELECT n.value,MIN(c.startNs),MAX(c.endNs),SUM(MAX(0,MIN(c.endNs,?)-MAX(c.startNs,?)))
             FROM COMMUNICATION_OP c LEFT JOIN STRING_IDS n ON n.id=c.opName
             WHERE c.startNs<? AND c.endNs>? GROUP BY c.opId,c.opName ORDER BY MIN(c.startNs) LIMIT 20'''
            comm_rows = db.execute(comm_sql,(end,start,end,start)).fetchall()
            item['communications_before_source_ready'] = [dict(name=n,start_seconds=(x-host_base)/1e9,end_seconds=(y-host_base)/1e9,overlap_ms=z/1e6) for n,x,y,z in comm_rows]
    for item in details:
        for k in [k for k in item if k.startswith('_')]:
            del item[k]
    totals = {k:sum(w['wait_split_ms'][k] for w in waits)/1000 for k in ['before_source_ready','source_ready_until_DMA','actual_DMA','DMA_done_until_CPU_get']}
    large = [x for x in details if x['bytes']==128*2**20]
    reports.append(dict(rank=rank,physical_card=[0,3][rank],validation=validation,slots=16,
                        db_sha256=hashlib.sha256(dbpath.read_bytes()).hexdigest(),diagnostic_sha256=hashlib.sha256(diagpath.read_bytes()).hexdigest(),
                        copy_count=len(copies),copy_bytes=sum(x['bytes'] for x in details),copy_stream_id=stream,
                        communication_overview=comm_overview,
                        host_clock_offset_mad_us=host_mad/1000,
                        host_clock_offset_spread_us=(max(offsets)-min(offsets))/1000,
                        common_device_clock_bounds=common,
                        device_clock_bound_width_ms=(global_hi-global_lo)/1e6 if common else None,
                        dma_ms_quantiles=quant([x['device_dma_ms'] for x in details]),
                        full128MiB_dma_ms_quantiles=quant([x['device_dma_ms'] for x in large]),
                        long_wait_count=len(waits),long_wait_split_totals_seconds=totals,
                        long_waits=waits,chunks=details))
    db.close()
payload = dict(reports=reports,queries=dict(copy=copy_sql,markers=marker_sql,compute_macro=compute_macro),
               limits=['Profiled iteration timing is not normal training throughput.',
                       'Source readiness is a stream event after gradient work, an upper bound on mathematical gradient readiness.',
                       'Host offset uses recorded scope/hook matching; device-clock mapping bounded by native DMA task and events.',
                       'Event-bounded copy span includes host enqueue gaps; native DMA tasks grouped by CANN connectionId reconstruct one copy, with fragment-active sum and envelope reported separately.',
                       'Compute top durations can overlap across tasks and do not sum to window length.'])
a.output.write_text(json.dumps(payload,indent=2))
for r in reports:
    print(json.dumps({k:v for k,v in r.items() if k not in ['chunks','long_waits','validation']},ensure_ascii=False))
