"""Documented host/device connection IDs expose already-queued collectives."""
import argparse,json,sqlite3
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--output',required=True);a=p.parse_args()
r=Path(a.root);assert list(r.rglob('profiler_info_0.json'))
c=sqlite3.connect(f"file:{next(r.rglob('ascend_pytorch_profiler_0.db'))}?mode=ro",uri=True)
# Track B: official CANN_API and COMMUNICATION_OP connectionId links, timing only.
sql="""WITH paired AS (
 SELECT o.count,o.startNs-h.endNs AS queued_ns FROM COMMUNICATION_OP o
 JOIN STRING_IDS k ON k.id=o.opType
 JOIN CANN_API h ON h.connectionId=o.connectionId
 JOIN STRING_IDS n ON n.id=h.name
 WHERE k.value='hcom_reduceScatter_' AND n.value='hcom_reduceScatter_'
)
SELECT CASE WHEN count>8388608 THEN 'count>8M' ELSE 'count<=8M' END AS size_class,
 COUNT(*) AS matched_calls,AVG(queued_ns)/1e6 AS mean_device_start_minus_host_end_ms,
 SUM(CASE WHEN queued_ns>1000000 THEN 1 ELSE 0 END) AS queued_over_1ms_calls,
 MIN(queued_ns)/1e6 AS min_ms,MAX(queued_ns)/1e6 AS max_ms
FROM paired GROUP BY size_class ORDER BY matched_calls DESC LIMIT 20"""
cur=c.execute(sql);keys=[x[0] for x in cur.description];rows=[dict(zip(keys,x)) for x in cur.fetchall()]
v={'schema_reference':'Official CANN_API/COMMUNICATION_OP schema; attempted documented connectionId association','status':'matched' if rows else 'unmatched_do_not_infer_queue_latency','sql':sql,'results':rows,'limits':'Queue lead time is not input-gradient ready time. Host API may overlap device execution; negative differences are valid. Require matched counts before interpreting. Empty matches are a collection/linkage limitation, not zero queue latency.'}
Path(a.output).write_text(json.dumps(v,indent=2));print(json.dumps(v,indent=2))
