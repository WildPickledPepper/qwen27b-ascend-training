"""Check compute coverage, communication waits, and native stream creation."""
import hashlib,inspect,json,sqlite3
from pathlib import Path
r=Path(__file__).resolve().parent;v={'ranks':[]}
for rank in [0,1]:
 db=next((r/'capture-full/profile').rglob(f'ascend_pytorch_profiler_{rank}.db'))
 c=sqlite3.connect(f'file:{db}?mode=ro',uri=True);c.row_factory=sqlite3.Row
 queries={
  'compute_count':'SELECT COUNT(*) AS calls FROM COMPUTE_TASK_INFO',
  'reduction_transport':"SELECT n.value AS type,COUNT(*) AS calls,SUM(t.endNs-t.startNs)/1e9 AS seconds FROM COMMUNICATION_TASK_INFO k JOIN COMMUNICATION_OP o ON o.opId=k.opId JOIN TASK t ON t.globalTaskId=k.globalTaskId JOIN STRING_IDS n ON n.id=k.taskType JOIN STRING_IDS a ON a.id=o.opType WHERE a.value='hcom_reduceScatter_' GROUP BY n.value ORDER BY seconds DESC LIMIT 20",
  'bucket_count':"SELECT COUNT(*) AS calls FROM PYTORCH_API p JOIN STRING_IDS n ON n.id=p.name WHERE n.value LIKE 'dep/bucket/%'",
  'event_task_count':"SELECT t.streamId,n.value AS type,COUNT(*) AS calls FROM TASK t JOIN STRING_IDS n ON n.id=t.taskType WHERE n.value IN ('EVENT_RECORD','EVENT_WAIT') GROUP BY t.streamId,n.value ORDER BY t.streamId,n.value LIMIT 20"
 }
 results={key:[dict(x) for x in c.execute(sql)] for key,sql in queries.items()}
 assert results['compute_count'][0]['calls']==219114
 assert results['bucket_count'][0]['calls']==4504
 v['ranks'].append({'rank':rank,'queries':queries,'results':results})
import deepspeed.runtime.zero.parameter_offload as offload
import deepspeed.runtime.zero.partitioned_param_coordinator as coordinator
source={'offload':inspect.getsource(offload),'coordinator':inspect.getsource(coordinator)}
(r/'parameter-offload-source.private.json').write_text(json.dumps(source))
v['native_source_sha256']={name:hashlib.sha256(text.encode()).hexdigest() for name,text in source.items()}
v['native_stream_role_source_lines']={name:[line.strip() for line in text.splitlines() if 'allgather_stream' in line or 'all_gather_stream' in line] for name,text in source.items()}
(r/'capture-roles.json').write_text(json.dumps(v,indent=2));print(json.dumps(v,indent=2))
