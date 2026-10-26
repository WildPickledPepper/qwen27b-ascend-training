"""Read-only pipeline critical-path coverage; preserve skill Compute/Comm CTEs."""
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
    out=[]
    for s,e in sorted((int(s),int(e)) for s,e in rows if e>s):
        if out and s<=out[-1][1]:out[-1][1]=max(e,out[-1][1])
        else:out.append([s,e])
    return out
def duration(rows):return sum(e-s for s,e in merge(rows))/1e9
def intersect(left,right):
    left=merge(left);right=merge(right);i=j=0;out=[]
    while i<len(left) and j<len(right):
        s=max(left[i][0],right[j][0]);e=min(left[i][1],right[j][1])
        if e>s:out.append((s,e))
        if left[i][1]<right[j][1]:i+=1
        else:j+=1
    return out
root=Path(a.root);results=[]
for rank in [0,1]:
    folder=list((root/f'profile-rank{rank}').glob('*_ascend_pt'))[0]
    meta=json.loads(list(folder.glob('profiler_info*.json'))[0].read_text());assert meta.get('end_info')
    db=list(folder.rglob('*pytorch*profiler*.db'))[0]
    c=sqlite3.connect(f'file:{db}?mode=ro',uri=True);queries={}
    def query(name,sql,params=()):
        queries[name]=dict(sql=sql,params=params)
        cur=c.execute(sql,params);cols=[x[0] for x in cur.description]
        return [dict(zip(cols,row)) for row in cur.fetchall()]
    def intervals(name,sql,params=()):return json.loads(query(name,sql,params)[0]['intervals'])
    # Skill has no PYTORCH_API entry: explicit version fallback, not guessed fields.
    assert {'name','startNs','endNs'}<={x[1] for x in c.execute('PRAGMA table_info(PYTORCH_API)')}
    compute=intervals('compute',COMPUTE+'SELECT json_group_array(json_array(t.startNs,t.endNs)) intervals FROM compute_view c JOIN TASK t ON t.globalTaskId=c.globalTaskId')
    comm=intervals('comm','SELECT json_group_array(json_array(startNs,endNs)) intervals FROM COMMUNICATION_OP')
    copies=intervals('copy',"SELECT json_group_array(json_array(t.startNs,t.endNs)) intervals FROM TASK t JOIN STRING_IDS n ON n.id=t.taskType WHERE n.value='MEMCPY_ASYNC'")
    window=query('window','SELECT MIN(startNs) s,MAX(endNs) e,COUNT(*) tasks FROM TASK')[0]
    phases={}
    for label,pattern in [('forward',f'pipeline/stage{rank}/forward'),('backward',f'pipeline/stage{rank}/backward'),
                          ('norm','optimizer/global_norm'),('update','optimizer/CPUAdam_and_return'),
                          ('slot_wait','gradient/slot_wait'),('cpu_add','gradient/cpu_add'),
                          ('drain','pipeline/gradient_drain'),('resident_flush','gradient/resident_flush'),('weight_decode','weight/decode'),('weight_encode','weight/encode_verify')]:
        spans=intervals(label,'SELECT json_group_array(json_array(p.startNs,p.endNs)) intervals FROM PYTORCH_API p JOIN STRING_IDS n ON n.id=p.name WHERE n.value=?',(pattern,))
        phases[label]=dict(scope_observed=bool(spans),scope_seconds=duration(spans),compute_seconds=duration(intersect(spans,compute)),
            copy_seconds=duration(intersect(spans,copies)),communication_seconds=duration(intersect(spans,comm)),
            no_device_work_seconds=duration(spans)-duration(intersect(spans,compute+copies+comm)))
    stream=query('task_streams',"SELECT t.streamId,n.value task_type,COUNT(*) calls,SUM(t.endNs-t.startNs)/1e9 seconds FROM TASK t JOIN STRING_IDS n ON n.id=t.taskType WHERE n.value IN ('MEMCPY_ASYNC','AI_CORE','KERNEL_MIX_AIC','KERNEL_AICORE') GROUP BY t.streamId,n.value ORDER BY seconds DESC LIMIT 20")
    host=query('host_top',"SELECT n.value name,COUNT(*) calls,SUM(p.endNs-p.startNs)/1e9 summed_seconds FROM PYTORCH_API p JOIN STRING_IDS n ON n.id=p.name GROUP BY n.value ORDER BY summed_seconds DESC LIMIT 20")
    top=query('compute_top',COMPUTE+'SELECT op_type,COUNT(*) calls,SUM(duration_ns)/1e9 summed_seconds FROM compute_view GROUP BY op_type ORDER BY summed_seconds DESC LIMIT 20')
    codec=query('codec_compute',COMPUTE+"SELECT op_name,op_type,COUNT(*) calls,SUM(duration_ns)/1e9 summed_seconds FROM compute_view WHERE lower(op_name) LIKE '%decomp%' OR lower(op_name) LIKE '%compbf16%' OR lower(op_name) LIKE '%merge%' GROUP BY op_name,op_type ORDER BY summed_seconds DESC LIMIT 20")
    weight_host=query('weight_host',"SELECT n.value name,COUNT(*) calls,SUM(p.endNs-p.startNs)/1e9 summed_seconds FROM PYTORCH_API p JOIN STRING_IDS n ON n.id=p.name WHERE n.value LIKE 'weight/%' GROUP BY n.value ORDER BY summed_seconds DESC LIMIT 20")
    wall=(window['e']-window['s'])/1e9
    exclusive=dict(compute=duration(compute),communication_without_compute=duration(comm)-duration(intersect(comm,compute)),
        copy_without_compute_or_comm=duration(copies)-duration(intersect(copies,compute+comm)),
        other=wall-duration(compute+comm+copies))
    assert abs(sum(exclusive.values())-wall)<1e-6
    results.append(dict(rank=rank,normal_stop=True,window_seconds=wall,phases=phases,exclusive=exclusive,
        compute_copy_overlap_seconds=duration(intersect(compute,copies)),task_streams=stream,host_top20=host,compute_top20=top,codec_compute=codec,weight_host=weight_host,queries=queries))
out=dict(ranks=results,execution='Read-only SQLite; msprof_mcp unavailable',
    limits=['Device intervals may outlive asynchronous host scopes; phase intersections describe simultaneity only.',
            'Host phases overlap; do not sum independent rows.',
            'No missing scope is interpreted as zero cost. No memcpy direction table is inferred.',
            'Eliminating all uncovered time is an optimistic ceiling, not an achieved speedup.'])
Path(a.output).write_text(json.dumps(out,indent=2))
print(json.dumps([{k:v for k,v in x.items() if k not in ['queries','host_top20','task_streams']} for x in results]))
