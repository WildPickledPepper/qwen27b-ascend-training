"""Link labeled framework event producers to their real device record/waits."""
import argparse,bisect,collections,json,re,sqlite3
from pathlib import Path
from intervals import merge,difference,Measure
p=argparse.ArgumentParser();p.add_argument('--rank',type=int,required=True);a=p.parse_args()
r=Path(__file__).resolve().parent
db=next((r/'capture-full/profile').rglob(f'ascend_pytorch_profiler_{a.rank}.db'))
c=sqlite3.connect(f'file:{db}?mode=ro',uri=True);queries={}
def rows(name,sql):
 queries[name]=sql;return json.loads(c.execute(sql).fetchone()[0] or '[]')
compute='''WITH compute_view AS (
    SELECT c.globalTaskId, ROUND(t.endNs - t.startNs) AS duration_ns, n.value AS op_name, type_str.value AS op_type
    FROM COMPUTE_TASK_INFO c
    LEFT JOIN TASK t ON t.globalTaskId = c.globalTaskId
    LEFT JOIN STRING_IDS n ON n.id = c.name
    LEFT JOIN STRING_IDS type_str ON type_str.id = c.opType
)
'''
comp=rows('compute',compute+'SELECT json_group_array(json_array(t.startNs,t.endNs,t.streamId)) FROM compute_view c JOIN TASK t ON t.globalTaskId=c.globalTaskId')
ops=rows('rs',"SELECT json_group_array(json_array(o.startNs,o.endNs)) FROM COMMUNICATION_OP o JOIN STRING_IDS n ON n.id=o.opType WHERE n.value='hcom_reduceScatter_'")
exposed=difference(merge(ops),merge(comp));m=Measure(exposed)
stream_durations=collections.Counter()
for s,e,stream in comp:stream_durations[stream]+=e-s
main=stream_durations.most_common(1)[0][0]
markers=rows('event_markers',"SELECT json_group_array(json_array(CAST(p.startNs AS INTEGER),CAST(p.endNs AS INTEGER),p.globalTid,n.value)) FROM PYTORCH_API p JOIN STRING_IDS n ON n.id=p.name WHERE n.value LIKE 'dep/E%'")
apis=rows('event_apis',"SELECT json_group_array(json_array(a.startNs,a.endNs,a.globalTid,a.connectionId,n.value)) FROM CANN_API a JOIN STRING_IDS n ON n.id=a.name WHERE n.value IN ('aclrtRecordEvent','aclrtStreamWaitEvent','aclrtSynchronizeEvent')")
tasks=rows('all_tasks',"SELECT json_group_array(json_array(t.startNs,t.endNs,t.streamId,t.connectionId,n.value,t.taskId)) FROM TASK t JOIN STRING_IDS n ON n.id=t.taskType")
by_tid=collections.defaultdict(list);by_connection=collections.defaultdict(list);by_stream=collections.defaultdict(list)
for item in apis:by_tid[item[2]].append(item)
for items in by_tid.values():items.sort()
tid_starts={tid:[x[0] for x in items] for tid,items in by_tid.items()}
for item in tasks:by_connection[item[3]].append(item);by_stream[item[2]].append(item)
for values in by_stream.values():values.sort()
stream_starts={st:[x[0] for x in values] for st,values in by_stream.items()}
pattern=re.compile(r'^dep/E(\d+)/R(\d+)/(record|wait|synchronize)(?:/S(\d+))?$')
records={};waits=[];coverage=collections.Counter();stream_tokens=collections.defaultdict(set);unlinked_waits=[];all_wait_api=[]
for s,e,tid,label in markers:
 match=pattern.match(label);assert match,label
 token,seq,kind,sid=match.groups();coverage[kind+'_markers']+=1
 api_name={'record':'aclrtRecordEvent','wait':'aclrtStreamWaitEvent','synchronize':'aclrtSynchronizeEvent'}[kind]
 i=bisect.bisect_left(tid_starts.get(tid,[]),s);found=[]
 while i<len(by_tid[tid]) and by_tid[tid][i][0]<e:
  item=by_tid[tid][i]
  if item[1]<=e and item[4]==api_name:found.append(item)
  i+=1
 if len(found)!=1:coverage[kind+'_without_unique_api']+=1;continue
 hs,he,_,connection,_=found[0];device=by_connection.get(connection,[]);all_device=list(device)
 if kind=='wait':all_wait_api.append({'token':token,'sequence':seq,'sid':sid,'host_start':hs,'host_end':he})
 wanted={'record':'EVENT_RECORD','wait':'EVENT_WAIT'}.get(kind)
 if wanted:device=[x for x in device if x[4]==wanted]
 if wanted and len(device)!=1:
  coverage[kind+'_without_unique_device']+=1
  if kind=='wait':unlinked_waits.append({'token':token,'sequence':seq,'sid':sid,'host_start':hs,'host_end':he,'device_types':[x[4] for x in all_device]})
  continue
 coverage[kind+'_matched_api']+=1
 if not wanted:continue
 ds,de,stream,_,_,taskid=device[0];stream_tokens[sid].add(stream)
 entry={'token':token,'sequence':seq,'host_start':hs,'host_end':he,'device_start':ds,'device_end':de,'stream':stream,'task_id':taskid}
 if kind=='record':
  assert (token,seq) not in records
  records[token,seq]=entry
 else:waits.append(entry)
unlinked_summary=collections.Counter()
for wait in unlinked_waits:
 record=records.get((wait['token'],wait['sequence']))
 unlinked_summary['device_types:'+','.join(sorted(wait['device_types']))]+=1
 if record is None:unlinked_summary['no_producer_record']+=1;continue
 targets=stream_tokens.get(wait['sid'],set())
 unlinked_summary['edge:'+','.join(str(x) for x in sorted(targets))+'<-'+str(record['stream'])]+=1
 if targets=={record['stream']}:unlinked_summary['same_producer_and_wait_stream']+=1
 elif record['device_end']<=wait['host_start']:unlinked_summary['producer_complete_before_wait_api']+=1
 else:unlinked_summary['different_stream_producer_not_yet_complete']+=1
edges=collections.defaultdict(list);expanded=[];examples=[]
for wait in waits:
 record=records.get((wait['token'],wait['sequence']))
 if not record:coverage['wait_without_captured_record']+=1;continue
 coverage['wait_with_captured_record']+=1
 key=f"{wait['stream']}<-{record['stream']}"
 edges[key].append([wait['device_start'],wait['device_end']])
 # An in-order stream cannot pass this wait before the producer record.
 # Reconstruct the pending interval after its preceding recorded command,
 # bounded by host submission and producer completion. Report it explicitly
 # as a dependency-derived interval, not a measured EVENT_WAIT duration.
 st=wait['stream'];index=bisect.bisect_left(stream_starts[st],wait['device_start'])
 preceding=by_stream[st][index-1] if index>0 else None
 if preceding is None:continue
 start=max(preceding[1],wait['host_end']);end=min(record['device_end'],wait['device_end'])
 if end>start:
  expanded.append([start,end,key]);seconds=m.inside(start,end)/1e9
  if seconds>.001:examples.append({'edge':key,'exposed_dependency_seconds':seconds,'pending_interval_seconds':(end-start)/1e9,'reported_event_wait_seconds':(wait['device_end']-wait['device_start'])/1e9,'event_token':wait['token'],'record_sequence':wait['sequence']})
main_launches=rows('main_compute_launches',compute+f"SELECT json_group_array(json_array(t.startNs,t.endNs,h.startNs,h.endNs)) FROM compute_view c JOIN TASK t ON t.globalTaskId=c.globalTaskId LEFT JOIN CANN_API h ON h.connectionId=t.connectionId AND h.name IN (SELECT id FROM STRING_IDS WHERE value='launch') WHERE t.streamId={main}")
assert all(x[2] is not None for x in main_launches)
by_device=sorted(main_launches);device_starts=[x[0] for x in by_device];queued_intervals=[]
for s,e in exposed:
 index=bisect.bisect_left(device_starts,s)
 if index<len(by_device):
  start=max(s,by_device[index][3])
  if e>start:queued_intervals.append([start,e])
queued=Measure(queued_intervals)
by_host=sorted(main_launches,key=lambda x:x[2]);host_starts=[x[2] for x in by_host];prefix_end=[0]
for ds,de,hs,he in by_host:prefix_end.append(max(prefix_end[-1],de))
constraints=collections.defaultdict(list)
for wait in all_wait_api:
 record=records.get((wait['token'],wait['sequence']));targets=stream_tokens.get(wait['sid'],set())
 if record is None or targets!={main}:continue
 index=bisect.bisect_left(host_starts,wait['host_start'])
 start=max(wait['host_end'],prefix_end[index]);end=record['device_end']
 if end>start:constraints[f"{main}<-{record['stream']}"].append([start,end])
constraint_results={edge:{'pending_inside_exposed_seconds':m.union_inside(v)/1e9,'pending_with_next_main_compute_already_queued_seconds':queued.union_inside(v)/1e9} for edge,v in constraints.items()}
edge_results=[]
for key,intervals in edges.items():
 pending=[[s,e] for s,e,k in expanded if k==key]
 edge_results.append({'edge':key,'wait_calls':len(intervals),'reported_wait_inside_rs_exposure_seconds':m.union_inside(intervals)/1e9,'dependency_pending_inside_rs_exposure_seconds':m.union_inside(pending)/1e9})
v={'rank':a.rank,'exposed_reduction_seconds':m.prefix[-1]/1e9,'main_compute_stream':main,'coverage':dict(coverage),'unlinked_wait_classification':dict(unlinked_summary),'framework_stream_tokens':{k:sorted(v) for k,v in stream_tokens.items()},
 'edges':sorted(edge_results,key=lambda x:x['dependency_pending_inside_rs_exposure_seconds'],reverse=True),
 'main_stream_dependency_pending_inside_exposed_seconds':m.union_inside([[s,e] for s,e,key in expanded if key.startswith(str(main)+'<-')])/1e9,
 'all_event_api_constraint_results':constraint_results,
 'exposed_with_next_main_compute_already_queued_seconds':queued.prefix[-1]/1e9,
 'queued_exposed_with_labeled_unfulfilled_event_seconds':queued.union_inside([x for v in constraints.values() for x in v])/1e9,
 'top20_dependencies':sorted(examples,key=lambda x:x['exposed_dependency_seconds'],reverse=True)[:20],'queries':queries,
 'limits':['Capture has extra labels and two profiled ranks; do not use as unprofiled throughput.',
 'Dependency-derived pending ranges are inferred from labeled in-order stream semantics and recorded command boundaries, not measured task durations.',
 'No captured producer for pre-capture or internal C++ events is not proof of no dependency.']}
(r/f'events-rank{a.rank}.json').write_text(json.dumps(v,indent=2));print(json.dumps({k:x for k,x in v.items() if k not in ['queries','top20_dependencies']},indent=2))
