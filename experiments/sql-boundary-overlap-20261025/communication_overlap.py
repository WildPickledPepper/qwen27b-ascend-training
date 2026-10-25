"""Separate AllGather/ReduceScatter overlap using validated rank0 DB timestamps."""
import argparse,json,sqlite3
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--output',required=True);a=p.parse_args()
root=Path(a.root);assert list(root.rglob('profiler_info_0.json'))
db=next(root.rglob('ascend_pytorch_profiler_0.db'))
c=sqlite3.connect(f'file:{db}?mode=ro',uri=True)
sql='''WITH compute_view AS (
    SELECT c.globalTaskId, ROUND(t.endNs - t.startNs) AS duration_ns, n.value AS op_name, type_str.value AS op_type
    FROM COMPUTE_TASK_INFO c
    LEFT JOIN TASK t ON t.globalTaskId = c.globalTaskId
    LEFT JOIN STRING_IDS n ON n.id = c.name
    LEFT JOIN STRING_IDS type_str ON type_str.id = c.opType
), intervals AS (
 SELECT t.startNs AS s,t.endNs AS e,1 AS mask FROM compute_view c JOIN TASK t ON t.globalTaskId=c.globalTaskId
 UNION ALL SELECT startNs,endNs,CASE WHEN opType IN (SELECT id FROM STRING_IDS WHERE value='hcom_allGather_') THEN 2 WHEN opType IN (SELECT id FROM STRING_IDS WHERE value='hcom_reduceScatter_') THEN 4 ELSE 8 END FROM COMMUNICATION_OP
 UNION ALL SELECT MIN(startNs),MAX(endNs),0 FROM TASK
), events AS (SELECT s AS ts,mask,1 AS d FROM intervals UNION ALL SELECT e,mask,-1 FROM intervals),
points AS (SELECT ts,SUM(CASE WHEN mask=1 THEN d ELSE 0 END) AS dc,SUM(CASE WHEN mask=2 THEN d ELSE 0 END) AS dg,SUM(CASE WHEN mask=4 THEN d ELSE 0 END) AS dr,SUM(CASE WHEN mask=8 THEN d ELSE 0 END) AS da FROM events GROUP BY ts),
states AS (SELECT ts,LEAD(ts) OVER (ORDER BY ts) AS te,SUM(dc) OVER (ORDER BY ts) AS c,SUM(dg) OVER (ORDER BY ts) AS g,SUM(dr) OVER (ORDER BY ts) AS r,SUM(da) OVER (ORDER BY ts) AS a FROM points),
segments AS (SELECT ts,te,(CASE WHEN c>0 THEN 1 ELSE 0 END)+(CASE WHEN g>0 THEN 2 ELSE 0 END)+(CASE WHEN r>0 THEN 4 ELSE 0 END)+(CASE WHEN a>0 THEN 8 ELSE 0 END) AS mask FROM states)
SELECT mask,COUNT(*) AS pieces,SUM(te-ts)/1e9 AS seconds FROM segments WHERE te>ts GROUP BY mask ORDER BY seconds DESC LIMIT 20'''
rows=[{'mask':m,'pieces':n,'seconds':s} for m,n,s in c.execute(sql)]
summary={}
for name,bit in [('allgather',2),('reduce_scatter',4),('other_communication',8)]:
 summary[name]={'union_seconds':sum(x['seconds'] for x in rows if x['mask']&bit),'without_compute_seconds':sum(x['seconds'] for x in rows if x['mask']&bit and not x['mask']&1),'with_compute_seconds':sum(x['seconds'] for x in rows if x['mask']&bit and x['mask']&1)}
v={'sql':sql,'rank':'rank0','rows':rows,'summary':summary,'limit':'Communication ranges include waits; without compute does not prove physical link saturation or freely removable time.'}
Path(a.output).write_text(json.dumps(v,indent=2));print(json.dumps(summary))
