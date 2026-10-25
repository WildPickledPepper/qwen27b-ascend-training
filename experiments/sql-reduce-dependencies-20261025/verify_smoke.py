"""Check event labels link to actual hardware events before loading 27B."""
import json,sqlite3
from pathlib import Path
r=Path(__file__).resolve().parent;results=[]
dbs=list((r/'tiny-smoke/profile').rglob('ascend_pytorch_profiler_*.db'));assert len(dbs)==2
sql="""WITH scopes AS (SELECT CAST(p.startNs AS INTEGER) AS s,CAST(p.endNs AS INTEGER) AS e,p.globalTid,n.value AS name FROM PYTORCH_API p JOIN STRING_IDS n ON n.id=p.name WHERE n.value LIKE 'dep/E%') SELECT CASE WHEN p.name LIKE '%/record/%' THEN 'record' WHEN p.name LIKE '%/wait/%' THEN 'wait' ELSE 'sync' END AS kind,COUNT(DISTINCT p.name) AS labels,COUNT(t.globalTaskId) AS device_events FROM scopes p LEFT JOIN CANN_API a ON a.globalTid=p.globalTid AND a.startNs>=p.s AND a.endNs<=p.e AND a.name IN (SELECT id FROM STRING_IDS WHERE value IN ('aclrtRecordEvent','aclrtStreamWaitEvent')) LEFT JOIN TASK t ON t.connectionId=a.connectionId GROUP BY kind ORDER BY kind LIMIT 20"""
for db in dbs:
 c=sqlite3.connect(f'file:{db}?mode=ro',uri=True);c.row_factory=sqlite3.Row
 rows=[dict(x) for x in c.execute(sql)]
 assert all(any(x['kind']==kind and x['device_events']>0 for x in rows) for kind in ['record','wait']),rows
 results.append({'db_basename':db.name,'rows':rows})
v={'passed':True,'both_ranks':results,'sql':sql};(r/'smoke-verification.json').write_text(json.dumps(v,indent=2));print(json.dumps(v))
