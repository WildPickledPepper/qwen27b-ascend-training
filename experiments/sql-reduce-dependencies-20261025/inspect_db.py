"""Read existing selected profile; inspect documented linkage and transport."""
import json,sqlite3
from pathlib import Path
r=Path(__file__).resolve().parent
db=next((r.parent/'qwen-sql-boundary-overlap-20261025/cpu-full/profile').rglob('ascend_pytorch_profiler_0.db'))
c=sqlite3.connect(f'file:{db}?mode=ro',uri=True);c.row_factory=sqlite3.Row
queries={
 'transport':"SELECT n.value AS task_type,k.isMaster,k.transportType,k.linkType,COUNT(*) AS calls,SUM(t.endNs-t.startNs)/1e9 AS seconds,SUM(k.size) AS task_bytes FROM COMMUNICATION_TASK_INFO k JOIN COMMUNICATION_OP o ON o.opId=k.opId JOIN TASK t ON t.globalTaskId=k.globalTaskId JOIN STRING_IDS n ON n.id=k.taskType JOIN STRING_IDS a ON a.id=o.opType WHERE a.value='hcom_reduceScatter_' GROUP BY n.value,k.isMaster,k.transportType,k.linkType ORDER BY seconds DESC LIMIT 20",
 'comm_ids':"SELECT COUNT(*) AS calls,COUNT(DISTINCT connectionId) AS unique_ids,MIN(connectionId) AS min_id,MAX(connectionId) AS max_id FROM COMMUNICATION_OP",
 'cann_ids':"SELECT n.value AS name,COUNT(*) AS calls,COUNT(DISTINCT a.connectionId) AS unique_ids,MIN(a.connectionId) AS min_id,MAX(a.connectionId) AS max_id FROM CANN_API a JOIN STRING_IDS n ON n.id=a.name WHERE n.value LIKE '%reduceScatter%' GROUP BY n.value ORDER BY calls DESC LIMIT 20",
 'connection_ids_schema':"SELECT sql FROM sqlite_master WHERE name='CONNECTION_IDS' ORDER BY name LIMIT 1",
 'host_schema':"SELECT sql FROM sqlite_master WHERE name='PYTORCH_API' ORDER BY name LIMIT 1",
 'op_samples':"SELECT o.opId,n.value AS op_name,o.connectionId,o.startNs,o.endNs,o.count FROM COMMUNICATION_OP o JOIN STRING_IDS n ON n.id=o.opName JOIN STRING_IDS a ON a.id=o.opType WHERE a.value='hcom_reduceScatter_' ORDER BY o.startNs LIMIT 6",
 'cann_samples':"SELECT a.connectionId,a.startNs,a.endNs,a.globalTid FROM CANN_API a JOIN STRING_IDS n ON n.id=a.name WHERE n.value='hcom_reduceScatter_' ORDER BY a.startNs LIMIT 6",
}
v={'queries':queries,'results':{k:[dict(x) for x in c.execute(sql)] for k,sql in queries.items()},'limits':'Host/device linkage must be verified; task byte sums are not unique physical traffic.'}
(r/'inspection.json').write_text(json.dumps(v,indent=2));print(json.dumps(v['results'],indent=2))
