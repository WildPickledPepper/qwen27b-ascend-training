"""Read-only msprof/Insight DB aggregates, using the profiler skill's CTEs."""
import argparse,json,sqlite3
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--output',required=True);a=p.parse_args()
COMPUTE='''WITH compute_view AS (
    SELECT c.globalTaskId, ROUND(t.endNs - t.startNs) AS duration_ns, n.value AS op_name, type_str.value AS op_type
    FROM COMPUTE_TASK_INFO c
    LEFT JOIN TASK t ON t.globalTaskId = c.globalTaskId
    LEFT JOIN STRING_IDS n ON n.id = c.name
    LEFT JOIN STRING_IDS type_str ON type_str.id = c.opType
)
'''
COMM='''WITH comm_view AS (
    SELECT ROUND(c.endNs - c.startNs) AS duration_ns, n.value AS op_name, t.value AS op_type, g.value AS group_name
    FROM COMMUNICATION_OP c
    LEFT JOIN STRING_IDS n ON n.id = c.opName
    LEFT JOIN STRING_IDS t ON t.id = c.opType
    LEFT JOIN STRING_IDS g ON g.id = c.groupName
)
'''
def merge(rows):
    result=[]
    for s,e in sorted((int(s),int(e)) for s,e in rows):
        if e<=s:continue
        if result and s<=result[-1][1]:result[-1][1]=max(e,result[-1][1])
        else:result.append([s,e])
    return result
def duration(rows):return sum(e-s for s,e in merge(rows))/1e9
def intersection(left,right):
    left=merge(left);right=merge(right);i=j=0;result=[]
    while i<len(left) and j<len(right):
        s=max(left[i][0],right[j][0]);e=min(left[i][1],right[j][1])
        if e>s:result.append((s,e))
        if left[i][1]<right[j][1]:i+=1
        else:j+=1
    return result
root=Path(a.root);results=[]
for rank in [0,1]:
    folder=list((root/f'profile-rank{rank}').glob('*_ascend_pt'))[0]
    info=list(folder.glob('profiler_info*.json'));assert info,'No normal stop metadata'
    metadata=json.loads(info[0].read_text());assert metadata.get('end_info'),'Profiler did not stop normally'
    db=list(folder.rglob('*pytorch*profiler*.db'))[0]
    c=sqlite3.connect(f'file:{db}?mode=ro',uri=True);queries={}
    def query(name,sql,params=()):
        queries[name]=dict(sql=sql,params=params)
        cur=c.execute(sql,params);cols=[x[0] for x in cur.description]
        return [dict(zip(cols,row)) for row in cur.fetchall()]
    tables=query('tables',"SELECT COUNT(*) count,json_group_array(name) names FROM sqlite_master WHERE type='table'")
    top=query('compute_top',COMPUTE+'SELECT op_type,COUNT(*) calls,SUM(duration_ns)/1e9 summed_seconds FROM compute_view GROUP BY op_type ORDER BY summed_seconds DESC LIMIT 20')
    comm=query('communication',COMM+'SELECT op_type,COUNT(*) calls,SUM(duration_ns)/1e9 summed_seconds FROM comm_view GROUP BY op_type ORDER BY summed_seconds DESC LIMIT 20')
    window=query('window','SELECT MIN(startNs) s,MAX(endNs) e,COUNT(*) tasks FROM TASK')[0]
    # Aggregate arrays for interval arithmetic, no unbounded detail rows returned.
    ci=query('compute_intervals',COMPUTE+'SELECT json_group_array(json_array(t.startNs,t.endNs)) intervals FROM compute_view c JOIN TASK t ON t.globalTaskId=c.globalTaskId')[0]
    compute=json.loads(ci['intervals'])
    commi=query('comm_intervals','SELECT json_group_array(json_array(startNs,endNs)) intervals FROM COMMUNICATION_OP')[0]
    communication=json.loads(commi['intervals'])
    task_types=query('device_task_types',"SELECT n.value type,COUNT(*) calls,SUM(t.endNs-t.startNs)/1e9 summed_seconds FROM TASK t LEFT JOIN STRING_IDS n ON n.id=t.taskType GROUP BY n.value ORDER BY summed_seconds DESC LIMIT 20")
    memcpy=[]
    if 'MEMCPY_INFO' in json.loads(tables[0]['names']):
        memcpy=query('memcpy_totals','SELECT n.value operation,COUNT(*) calls,SUM(m.size) bytes FROM MEMCPY_INFO m LEFT JOIN STRING_IDS n ON n.id=m.memcpyOperation GROUP BY n.value ORDER BY bytes DESC LIMIT 20')
    mi=query('memcpy_intervals',"SELECT json_group_array(json_array(t.startNs,t.endNs)) intervals FROM TASK t JOIN STRING_IDS n ON n.id=t.taskType WHERE n.value='MEMCPY_ASYNC'")[0]
    copies=json.loads(mi['intervals'])
    # PYTORCH_API is absent from the skill's schema reference: documented fallback.
    schema=[x[1] for x in c.execute('PRAGMA table_info(PYTORCH_API)')]
    assert {'startNs','endNs','name'}<=set(schema)
    host=query('pipeline_host_ranges',"SELECT n.value name,COUNT(*) calls,SUM(p.endNs-p.startNs)/1e9 summed_seconds FROM PYTORCH_API p JOIN STRING_IDS n ON n.id=p.name WHERE n.value LIKE 'pipeline/%' OR n.value LIKE 'optimizer/%' OR n.value LIKE '%CPUAdam%' GROUP BY n.value ORDER BY summed_seconds DESC LIMIT 20")
    updates=query('update_intervals',"SELECT json_group_array(json_array(p.startNs,p.endNs)) intervals FROM PYTORCH_API p JOIN STRING_IDS n ON n.id=p.name WHERE n.value LIKE '%CPUAdam%'")[0]
    ui=json.loads(updates['intervals'])
    ni=query('norm_intervals',"SELECT json_group_array(json_array(p.startNs,p.endNs)) intervals FROM PYTORCH_API p JOIN STRING_IDS n ON n.id=p.name WHERE n.value='optimizer/global_norm'")[0]
    norm_intervals=json.loads(ni['intervals'])
    wall=(window['e']-window['s'])/1e9
    coverage=dict(device_window_seconds=wall,compute_union_seconds=duration(compute),
                  communication_union_seconds=duration(communication),
                  communication_without_compute_seconds=duration(communication)-duration(intersection(compute,communication)),
                  cpu_update_scope_available=bool(ui),cpu_update_scope_seconds=duration(ui),compute_during_cpu_update_seconds=duration(intersection(compute,ui)))
    coverage.update(memcpy_union_seconds=duration(copies),
                    memcpy_without_compute_seconds=duration(copies)-duration(intersection(copies,compute)),
                    no_compute_or_communication_seconds=wall-duration(compute+communication),
                    global_norm_scope_available=bool(norm_intervals),global_norm_scope_seconds=duration(norm_intervals),
                    compute_during_norm_seconds=duration(intersection(compute,norm_intervals)),
                    memcpy_during_norm_seconds=duration(intersection(copies,norm_intervals)))
    exclusive=dict(compute_seconds=duration(compute),
        communication_without_compute_seconds=duration(communication)-duration(intersection(communication,compute)),
        memcpy_without_compute_or_comm_seconds=duration(copies)-duration(intersection(copies,compute+communication)),
        cpu_adam_without_compute_comm_memcpy_seconds=duration(ui)-duration(intersection(ui,compute+communication+copies)),
        other_seconds=wall-duration(compute+communication+copies+ui))
    assert abs(sum(exclusive.values())-wall)<1e-6
    step=json.loads((root/f'rank{rank}-result.json').read_text())['steps'][-1]
    assert abs(wall-step['seconds'])<max(2.,step['seconds']*.12),(wall,step['seconds'])
    results.append(dict(rank=rank,normal_stop_metadata=True,db_basename=db.name,compute_top20=top,
                        communication=comm,host_ranges=host,coverage=coverage,exclusive_timeline=exclusive,task_types=task_types,memcpy=memcpy,
                        profiled_step_seconds=step['seconds'],queries=queries))
result=dict(ranks=results,execution='read-only SQLite; msprof_mcp unavailable',
            schema_reference='skill get_schema TASK and COMMUNICATION_OP; PYTORCH_API absent, explicit PRAGMA fallback',
            limits=['Both-rank profiling perturbs scheduling; use normal updates for ranking.',
                    'Communication duration includes waiting; not pure transfer time.',
                    'Host scopes may nest/overlap; never add them as independent costs.',
                    'No-compute intervals do not by themselves prove a pipeline bubble cause.'])
Path(a.output).write_text(json.dumps(result,indent=2));print(json.dumps([r['coverage'] for r in results]))
