"""Separate main reducer event waits from background CPU receive waits."""
import argparse,json,sqlite3
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--output',required=True);a=p.parse_args()
r=Path(a.root);assert list(r.rglob('profiler_info_0.json'))
c=sqlite3.connect(f"file:{next(r.rglob('ascend_pytorch_profiler_0.db'))}?mode=ro",uri=True)
# PYTORCH_API is absent from the official skill reference: documented fallback.
columns=[x[1] for x in c.execute('PRAGMA table_info(PYTORCH_API)')]
assert 'globalTid' in columns
sql="""WITH scopes AS (
 SELECT CAST(p.startNs AS INTEGER) AS s,CAST(p.endNs AS INTEGER) AS e,p.globalTid,n.value AS scope
 FROM PYTORCH_API p JOIN STRING_IDS n ON n.id=p.name
 WHERE n.value IN ('zero/reduce_and_partition','zero/partition_gradients','zero/deferred_boundary','zero/average_scatter')
), events AS (
 SELECT c.startNs AS s,c.endNs AS e,c.globalTid FROM CANN_API c JOIN STRING_IDS n ON n.id=c.name WHERE n.value='aclrtSynchronizeEvent'
)
SELECT p.scope,COUNT(*) AS event_calls,SUM(t.e-t.s)/1e9 AS event_host_seconds FROM events t JOIN scopes p
 ON t.globalTid=p.globalTid AND t.s>=p.s AND t.e<=p.e GROUP BY p.scope ORDER BY event_host_seconds DESC LIMIT 20"""
cur=c.execute(sql);keys=[x[0] for x in cur.description];rows=[dict(zip(keys,x)) for x in cur.fetchall()]
physical="SELECT n.value AS name,COUNT(*) AS calls,SUM(c.endNs-c.startNs)/1e9 AS host_seconds FROM CANN_API c JOIN STRING_IDS n ON n.id=c.name WHERE n.value IN ('aclrtMallocPhysical','aclrtFreePhysical') GROUP BY n.value ORDER BY host_seconds DESC LIMIT 20"
cur=c.execute(physical);keys=[x[0] for x in cur.description];memory=[dict(zip(keys,x)) for x in cur.fetchall()]
v={'schema_reference':'Official CANN_API schema; PYTORCH_API absent, PRAGMA fallback used','queries':{'scoped_event_waits':sql,'physical_memory':physical},'scoped_event_waits':rows,'physical_memory':memory,'limits':'Scopes may nest. SynchronizeEvent within reducer is consistent with inspected event-backpressure code, but must distinguish nested partition waits. Background thread waits are excluded by globalTid.'}
Path(a.output).write_text(json.dumps(v,indent=2));print(json.dumps(v,indent=2))
