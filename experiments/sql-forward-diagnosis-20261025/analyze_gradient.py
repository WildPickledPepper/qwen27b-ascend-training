"""Read-only range analysis; never add nested ranges as elapsed time."""
import json,runpy,sys
from pathlib import Path

ns=runpy.run_path(str(Path(__file__).with_name('analyze_profile.py')),run_name='__main__')
q=ns['query']; tables=ns['tables']; compute=ns['compute']; comm=ns['comm']
host=" FROM PYTORCH_API p LEFT JOIN STRING_IDS n ON n.id=p.name "
q('host_clock_windows',"SELECT n.value AS name,MIN(p.startNs) AS start_ns,MAX(p.endNs) AS end_ns"+host+"WHERE n.value IN ('optimizer/CPUAdam','ProfilerStep#0') GROUP BY n.value ORDER BY start_ns LIMIT 20")
q('gradient_host_scopes',"SELECT n.value AS name,COUNT(*) AS calls,SUM(p.endNs-p.startNs)/1e9 AS inclusive_seconds"+host+"WHERE n.value LIKE 'zero/%' OR n.value LIKE 'gradient_transfer/%' GROUP BY n.value ORDER BY inclusive_seconds DESC LIMIT 20")

def union(name,head,intervals):
 sql=head+", intervals AS ("+intervals+"""), ordered AS (
 SELECT s,e,MAX(e) OVER (ORDER BY s,e ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) AS pe FROM intervals
 ), marked AS (SELECT s,e,CASE WHEN pe IS NULL OR s>pe THEN 1 ELSE 0 END AS ng FROM ordered),
 grouped AS (SELECT s,e,SUM(ng) OVER (ORDER BY s,e ROWS UNBOUNDED PRECEDING) AS gid FROM marked),
 merged AS (SELECT MIN(s) AS s,MAX(e) AS e FROM grouped GROUP BY gid)
 SELECT COUNT(*) AS islands,SUM(e-s)/1e9 AS union_seconds FROM merged"""
 q(name,sql)

for label,condition in [('partition',"n.value='zero/partition_gradients'"),('transfer',"n.value LIKE 'gradient_transfer/%'"),('reduce',"n.value='zero/reduce_and_partition'"),('adam',"n.value='optimizer/CPUAdam'")]:
 union(label+'_host_union','WITH dummy AS (SELECT 1)',"SELECT CAST(p.startNs AS INTEGER) AS s,CAST(p.endNs AS INTEGER) AS e"+host+'WHERE '+condition)
union('communication_union',comm,'SELECT startNs AS s,endNs AS e FROM COMMUNICATION_OP')
union('compute_or_communication_union',compute,"SELECT t.startNs AS s,t.endNs AS e FROM compute_view c JOIN TASK t ON t.globalTaskId=c.globalTaskId UNION ALL SELECT startNs,endNs FROM COMMUNICATION_OP")
if 'MEMCPY_INFO' in tables:
 q('memcpy_totals','SELECT n.value AS operation,COUNT(*) AS calls,SUM(m.size) AS bytes FROM MEMCPY_INFO m LEFT JOIN STRING_IDS n ON n.id=m.memcpyOperation GROUP BY n.value ORDER BY bytes DESC LIMIT 20')
if 'CANN_API' in tables:
 q('cann_host_top20','SELECT n.value AS name,COUNT(*) AS calls,SUM(c.endNs-c.startNs)/1e9 AS host_seconds FROM CANN_API c LEFT JOIN STRING_IDS n ON n.id=c.name GROUP BY n.value ORDER BY host_seconds DESC LIMIT 20')
q('device_task_types','SELECT n.value AS type,COUNT(*) AS calls,SUM(t.endNs-t.startNs)/1e9 AS summed_seconds FROM TASK t LEFT JOIN STRING_IDS n ON n.id=t.taskType GROUP BY n.value ORDER BY summed_seconds DESC LIMIT 20')
q('exclusive_timeline',compute+''', intervals AS (
 SELECT t.startNs AS s,t.endNs AS e,1 AS mask FROM compute_view c JOIN TASK t ON t.globalTaskId=c.globalTaskId
 UNION ALL SELECT startNs,endNs,2 FROM COMMUNICATION_OP
 UNION ALL SELECT t.startNs,t.endNs,4 FROM TASK t JOIN STRING_IDS n ON n.id=t.taskType WHERE n.value='MEMCPY_ASYNC'
 UNION ALL SELECT CAST(p.startNs AS INTEGER),CAST(p.endNs AS INTEGER),8 FROM PYTORCH_API p JOIN STRING_IDS n ON n.id=p.name WHERE n.value='optimizer/CPUAdam'
 UNION ALL SELECT MIN(startNs),MAX(endNs),0 FROM TASK
), events AS (
 SELECT s AS ts,mask,1 AS delta FROM intervals UNION ALL SELECT e,mask,-1 FROM intervals
), points AS (
 SELECT ts,SUM(CASE WHEN mask=1 THEN delta ELSE 0 END) AS dc,SUM(CASE WHEN mask=2 THEN delta ELSE 0 END) AS dh,
 SUM(CASE WHEN mask=4 THEN delta ELSE 0 END) AS dm,SUM(CASE WHEN mask=8 THEN delta ELSE 0 END) AS da FROM events GROUP BY ts
), states AS (
 SELECT ts,LEAD(ts) OVER (ORDER BY ts) AS te,SUM(dc) OVER (ORDER BY ts) AS c,SUM(dh) OVER (ORDER BY ts) AS h,
 SUM(dm) OVER (ORDER BY ts) AS m,SUM(da) OVER (ORDER BY ts) AS a FROM points
), segments AS (
 SELECT ts,te,(CASE WHEN c>0 THEN 1 ELSE 0 END)+(CASE WHEN h>0 THEN 2 ELSE 0 END)+(CASE WHEN m>0 THEN 4 ELSE 0 END)+(CASE WHEN a>0 THEN 8 ELSE 0 END) AS mask FROM states
)
SELECT mask,COUNT(*) AS pieces,SUM(te-ts)/1e9 AS seconds FROM segments WHERE te>ts GROUP BY mask ORDER BY seconds DESC LIMIT 20''')
assert abs(sum(x['seconds'] for x in ns['outputs']['exclusive_timeline'])-ns['outputs']['device_window'][0]['window_seconds'])<1e-6
result=json.loads(Path(ns['a'].output).read_text())
result['queries']=ns['queries'];result['results']=ns['outputs']
result['limits'].append('Gradient transfer host ranges measure API elapsed time, not isolated DMA time; logical bytes are tracked separately.')
Path(ns['a'].output).write_text(json.dumps(result,indent=2))
print(json.dumps({'gradient_analysis':ns['outputs']}),flush=True)
