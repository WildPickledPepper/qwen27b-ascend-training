"""Identify reduction preparation stream from Cast launches inside bucket scopes."""
import bisect,collections,json,sqlite3
from pathlib import Path
r=Path(__file__).resolve().parent;out=[]
compute='''WITH compute_view AS (
    SELECT c.globalTaskId, ROUND(t.endNs - t.startNs) AS duration_ns, n.value AS op_name, type_str.value AS op_type
    FROM COMPUTE_TASK_INFO c
    LEFT JOIN TASK t ON t.globalTaskId = c.globalTaskId
    LEFT JOIN STRING_IDS n ON n.id = c.name
    LEFT JOIN STRING_IDS type_str ON type_str.id = c.opType
)
'''
for rank in [0,1]:
 db=next((r/'capture-full/profile').rglob(f'ascend_pytorch_profiler_{rank}.db'));c=sqlite3.connect(f'file:{db}?mode=ro',uri=True)
 queries={'buckets':"SELECT json_group_array(json_array(CAST(p.startNs AS INTEGER),CAST(p.endNs AS INTEGER),p.globalTid)) FROM PYTORCH_API p JOIN STRING_IDS n ON n.id=p.name WHERE n.value LIKE 'dep/bucket/%'",'casts':compute+"SELECT json_group_array(json_array(h.startNs,h.endNs,h.globalTid,t.streamId)) FROM compute_view c JOIN TASK t ON t.globalTaskId=c.globalTaskId JOIN CANN_API h ON h.connectionId=t.connectionId WHERE c.op_type='Cast' AND h.name IN (SELECT id FROM STRING_IDS WHERE value='launch')"}
 buckets=sorted(json.loads(c.execute(queries['buckets']).fetchone()[0]));starts=[x[0] for x in buckets]
 casts=json.loads(c.execute(queries['casts']).fetchone()[0]);counts=collections.Counter()
 for hs,he,tid,stream in casts:
  i=bisect.bisect_right(starts,hs)-1
  if i>=0 and buckets[i][1]>=he and buckets[i][2]==tid:counts[stream]+=1
 assert len(counts)==1 and sum(counts.values())>1000,counts
 events=json.loads((r/f'events-rank{rank}.json').read_text())
 main=events['main_compute_stream'];reduce=next(iter(counts))
 other={x for values in events['framework_stream_tokens'].values() for x in values}-{main,reduce}
 assert len(other)==1
 out.append({'rank':rank,'main_compute_stream':main,'reduction_prepare_stream':reduce,'bucket_cast_launches':dict(counts),'parameter_prefetch_stream':next(iter(other)),
 'role_assignment':'Reduction stream directly from bucket Cast launches; remaining Python stream from native offload/coordinator allgather-stream creation and wait_stream source.',
 'queries':queries})
v={'ranks':out,'limits':'Prefetch role combines native source and unique remaining Python stream; no claim that it is the HCCL worker stream.'}
(r/'stream-roles.json').write_text(json.dumps(v,indent=2));print(json.dumps(v,indent=2))
