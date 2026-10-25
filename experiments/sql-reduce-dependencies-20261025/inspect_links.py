import json,sqlite3
from pathlib import Path
r=Path(__file__).resolve().parent
c=sqlite3.connect(f"file:{next((r.parent/'qwen-sql-boundary-overlap-20261025/cpu-full/profile').rglob('ascend_pytorch_profiler_0.db'))}?mode=ro",uri=True);c.row_factory=sqlite3.Row
queries={
 'actual_comm_host_link':"SELECT n.value AS host_name,COUNT(*) AS matches,COUNT(DISTINCT o.opId) AS distinct_ops FROM COMMUNICATION_OP o JOIN STRING_IDS k ON k.id=o.opType JOIN CANN_API h ON h.connectionId=o.connectionId JOIN STRING_IDS n ON n.id=h.name WHERE k.value='hcom_reduceScatter_' GROUP BY n.value ORDER BY matches DESC LIMIT 20",
 'first_launch_context':"SELECT h.connectionId,n.value AS name,h.startNs,h.endNs,h.globalTid FROM CANN_API h JOIN STRING_IDS n ON n.id=h.name WHERE h.connectionId BETWEEN 1009735 AND 1009746 ORDER BY h.startNs LIMIT 20",
 'first_connections':"SELECT id,COUNT(*) AS ids,MIN(connectionId) AS min_connection,MAX(connectionId) AS max_connection FROM CONNECTION_IDS WHERE id IN (SELECT id FROM CONNECTION_IDS WHERE connectionId=1009743) GROUP BY id ORDER BY id LIMIT 20",
 'compute_streams':"SELECT t.streamId,COUNT(*) AS calls,SUM(t.endNs-t.startNs)/1e9 AS seconds FROM COMPUTE_TASK_INFO k JOIN TASK t ON t.globalTaskId=k.globalTaskId GROUP BY t.streamId ORDER BY seconds DESC LIMIT 20",
 'event_streams':"SELECT t.streamId,COUNT(*) AS calls,SUM(t.endNs-t.startNs)/1e9 AS seconds FROM TASK t JOIN STRING_IDS n ON n.id=t.taskType WHERE n.value='EVENT_WAIT' GROUP BY t.streamId ORDER BY seconds DESC LIMIT 20",
 'hccl_streams':"SELECT t.streamId,n.value AS task_type,COUNT(*) AS calls,SUM(t.endNs-t.startNs)/1e9 AS seconds FROM COMMUNICATION_TASK_INFO k JOIN COMMUNICATION_OP o ON o.opId=k.opId JOIN TASK t ON t.globalTaskId=k.globalTaskId JOIN STRING_IDS n ON n.id=k.taskType JOIN STRING_IDS a ON a.id=o.opType WHERE a.value='hcom_reduceScatter_' GROUP BY t.streamId,n.value ORDER BY seconds DESC LIMIT 20",
 'link_types':"SELECT * FROM ENUM_HCCL_LINK_TYPE ORDER BY 1 LIMIT 20",
 'transport_types':"SELECT * FROM ENUM_HCCL_TRANSPORT_TYPE ORDER BY 1 LIMIT 20"
}
v={'queries':queries,'results':{k:[dict(x) for x in c.execute(sql)] for k,sql in queries.items()}}
(r/'links.json').write_text(json.dumps(v,indent=2));print(json.dumps(v['results'],indent=2))
