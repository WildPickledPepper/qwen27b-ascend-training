"""Decompose selected-profile exposed reduction and verified submit linkage."""
import bisect,collections,json,sqlite3,statistics
from pathlib import Path
r=Path(__file__).resolve().parent
db=next((r.parent/'qwen-sql-boundary-overlap-20261025/cpu-full/profile').rglob('ascend_pytorch_profiler_0.db'))
c=sqlite3.connect(f'file:{db}?mode=ro',uri=True);queries={}
compute='''WITH compute_view AS (
    SELECT c.globalTaskId, ROUND(t.endNs - t.startNs) AS duration_ns, n.value AS op_name, type_str.value AS op_type
    FROM COMPUTE_TASK_INFO c
    LEFT JOIN TASK t ON t.globalTaskId = c.globalTaskId
    LEFT JOIN STRING_IDS n ON n.id = c.name
    LEFT JOIN STRING_IDS type_str ON type_str.id = c.opType
)
'''
def rows(name,sql):
 queries[name]=sql
 return json.loads(c.execute(sql).fetchone()[0] or '[]')
computation=rows('compute_intervals',compute+'SELECT json_group_array(json_array(t.startNs,t.endNs,t.streamId)) FROM compute_view c JOIN TASK t ON t.globalTaskId=c.globalTaskId')
ops=rows('reduce_ops',"SELECT json_group_array(json_array(o.opId,o.startNs,o.endNs,o.count,o.connectionId)) FROM COMMUNICATION_OP o JOIN STRING_IDS n ON n.id=o.opType WHERE n.value='hcom_reduceScatter_'")
transport=rows('reduce_transport',"SELECT json_group_array(json_array(t.startNs,t.endNs,n.value,k.size,t.streamId,k.opId)) FROM COMMUNICATION_TASK_INFO k JOIN COMMUNICATION_OP o ON o.opId=k.opId JOIN TASK t ON t.globalTaskId=k.globalTaskId JOIN STRING_IDS n ON n.id=k.taskType JOIN STRING_IDS a ON a.id=o.opType WHERE a.value='hcom_reduceScatter_'")
launches=rows('verified_launches',"SELECT json_group_array(json_array(o.opId,h.startNs,h.endNs,h.globalTid)) FROM COMMUNICATION_OP o JOIN STRING_IDS k ON k.id=o.opType JOIN CANN_API h ON h.connectionId=o.connectionId JOIN STRING_IDS n ON n.id=h.name WHERE k.value='hcom_reduceScatter_' AND n.value='launch'")
host=rows('host_scopes',"SELECT json_group_array(json_array(CAST(p.startNs AS INTEGER),CAST(p.endNs AS INTEGER),n.value,p.globalTid)) FROM PYTORCH_API p JOIN STRING_IDS n ON n.id=p.name WHERE n.value LIKE 'zero/%' OR n.value LIKE 'gradient_transfer/%'")
events=rows('device_waits',"SELECT json_group_array(json_array(t.startNs,t.endNs,t.streamId)) FROM TASK t JOIN STRING_IDS n ON n.id=t.taskType WHERE n.value='EVENT_WAIT'")
assert len(ops)==len(launches)==len({x[0] for x in launches})==4504

def merge(intervals):
 out=[]
 for s,e,*_ in sorted(intervals):
  if e<=s:continue
  if out and s<=out[-1][1]:out[-1][1]=max(e,out[-1][1])
  else:out.append([s,e])
 return out
def difference(intervals,remove):
 out=[];j=0
 for s,e in intervals:
  while j<len(remove) and remove[j][1]<=s:j+=1
  k=j;cursor=s
  while k<len(remove) and remove[k][0]<e:
   a,b=remove[k]
   if a>cursor:out.append([cursor,min(a,e)])
   cursor=max(cursor,b)
   if cursor>=e:break
   k+=1
  if cursor<e:out.append([cursor,e])
 return out
class Measure:
 def __init__(self,intervals):
  self.x=merge(intervals);self.s=[x[0] for x in self.x];self.e=[x[1] for x in self.x];self.prefix=[0]
  for s,e in self.x:self.prefix.append(self.prefix[-1]+e-s)
 def before(self,t):
  i=bisect.bisect_right(self.e,t);v=self.prefix[i]
  if i<len(self.x):v+=max(0,t-self.s[i])
  return v
 def inside(self,s,e):return self.before(e)-self.before(s)
 def union_inside(self,intervals):return sum(self.inside(s,e) for s,e in merge(intervals))
comp=merge(computation);reduces=merge([[x[1],x[2]] for x in ops]);exposed=difference(reduces,comp);m=Measure(exposed)
assert abs(m.prefix[-1]/1e9-13.361874859)<1e-6
main_stream=max(collections.Counter({stream:sum(e-s for s,e,st in computation if st==stream) for stream in {x[2] for x in computation}}),key=lambda k:sum(e-s for s,e,st in computation if st==k))
types={name:[[s,e] for s,e,n,*_ in transport if n==name] for name in {x[2] for x in transport}}
transport_summary={name:{'task_union_seconds':sum(e-s for s,e in merge(v))/1e9,'seconds_inside_exposed_reduction':m.union_inside(v)/1e9} for name,v in types.items()}
uncovered=difference(exposed,merge([[s,e] for s,e,*_ in transport]))
host_summary={name:m.union_inside([[s,e] for s,e,n,tid in host if n==name])/1e9 for name in sorted({x[2] for x in host})}
wait_summary={str(st):m.union_inside([[s,e] for s,e,stream in events if stream==st])/1e9 for st in sorted({x[2] for x in events})}
launchmap={x[0]:x[1:] for x in launches};by_count=collections.defaultdict(list)
op_details=[];previous_end=None
for opid,s,e,count,connection in sorted(ops,key=lambda x:x[1]):
 hs,he,tid=launchmap[opid];lead=s-he
 by_count[count].append({'lead_ns':lead,'exposed_ns':m.inside(s,e),'duration_ns':e-s})
 op_details.append({'op_id':opid,'count':count,'start_relative_seconds':(s-comp[0][0])/1e9,'duration_ms':(e-s)/1e6,'exposed_ms':m.inside(s,e)/1e6,'submit_to_device_ms':lead/1e6,'host_submit_ms':(he-hs)/1e6,'prior_reduction_end_to_start_ms':None if previous_end is None else (s-previous_end)/1e6})
 previous_end=e
count_summary=[]
for count,v in by_count.items():
 leads=sorted(x['lead_ns'] for x in v)
 count_summary.append({'count':count,'calls':len(v),'summed_seconds':sum(x['duration_ns'] for x in v)/1e9,'exposed_seconds':sum(x['exposed_ns'] for x in v)/1e9,'mean_submit_to_device_ms':statistics.mean(leads)/1e6,'p50_submit_to_device_ms':leads[len(leads)//2]/1e6,'max_submit_to_device_ms':max(leads)/1e6,'submitted_over_1ms_early_calls':sum(x>1000000 for x in leads)})
v={'read_only':True,'queries':queries,'matched_reduce_calls':4504,'verified_launch_name':'launch','exposed_reduction_seconds':m.prefix[-1]/1e9,'compute_main_stream':main_stream,
 'exposed_inside_transport':transport_summary,'exposed_not_in_transport_seconds':sum(e-s for s,e in uncovered)/1e9,
 'exposed_intersect_host_scope_seconds':host_summary,'exposed_intersect_device_event_wait_by_stream_seconds':wait_summary,
 'reduce_count_classes':sorted(count_summary,key=lambda x:x['exposed_seconds'],reverse=True),
 'top20_exposed_ops':sorted(op_details,key=lambda x:x['exposed_ms'],reverse=True)[:20],
 'limits':['Exposed means no COMPUTE_TASK_INFO interval, not proven critical-path wait.','Host scopes and different streams can overlap; do not add them.','Submit-to-device includes producer dependencies and preceding queued work; it is not pure launch overhead.','No event IDs are present in this DB schema to prove every wait producer.']}
(r/'dependencies.json').write_text(json.dumps(v,indent=2));print(json.dumps({k:x for k,x in v.items() if k not in ['queries','top20_exposed_ops']},indent=2))
