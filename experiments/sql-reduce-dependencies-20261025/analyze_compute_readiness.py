"""Determine whether the next main-stream compute had already been submitted."""
import bisect,collections,json,runpy
from pathlib import Path
r=Path(__file__).resolve().parent;ns=runpy.run_path(str(r/'analyze_dependencies.py'))
rows=ns['rows'];c=ns['c'];main=ns['main_stream'];m=ns['m'];Measure=ns['Measure']
tasks=rows('compute_launches',ns['compute']+"SELECT json_group_array(json_array(t.startNs,t.endNs,t.streamId,c.globalTaskId,t.connectionId,h.startNs,h.endNs,c.op_type)) FROM compute_view c JOIN TASK t ON t.globalTaskId=c.globalTaskId LEFT JOIN CANN_API h ON h.connectionId=t.connectionId AND h.name IN (SELECT id FROM STRING_IDS WHERE value='launch')")
selected=sorted([x for x in tasks if x[2]==main]);starts=[x[0] for x in selected]
totals=collections.defaultdict(int);examples=[];categories=collections.defaultdict(list)
for s,e in ns['exposed']:
 i=bisect.bisect_left(starts,s)
 if i>=len(selected) or selected[i][5] is None:
  totals['unmatched_next_compute']+=e-s;categories['unmatched_next_compute'].append([s,e]);continue
 ds,de,stream,gid,conn,hs,he,kind=selected[i]
 totals['next_compute_not_yet_submitted']+=max(0,min(e,hs)-s)
 totals['next_compute_submit_in_progress']+=max(0,min(e,he)-max(s,hs))
 totals['next_compute_already_submitted']+=max(0,e-max(s,he))
 for key,begin,end in [('next_compute_not_yet_submitted',s,min(e,hs)),('next_compute_submit_in_progress',max(s,hs),min(e,he)),('next_compute_already_submitted',max(s,he),e)]:
  if end>begin:categories[key].append([begin,end])
 if e-s>50000000:examples.append({'exposed_gap_ms':(e-s)/1e6,'next_compute_type':kind,'next_compute_host_submit_relative_to_gap_ms':(hs-s)/1e6,'next_compute_device_start_relative_to_gap_ms':(ds-s)/1e6})
assert abs(sum(totals.values())-m.prefix[-1])<100
main_tid=ns['launches'][0][3]
sync=rows('main_sync_apis',f"SELECT json_group_array(json_array(a.startNs,a.endNs,n.value)) FROM CANN_API a JOIN STRING_IDS n ON n.id=a.name WHERE a.globalTid={main_tid} AND n.value LIKE 'aclrtSynchronize%'")
sync_summary={name:m.union_inside([[s,e] for s,e,n in sync if n==name])/1e9 for name in sorted({x[2] for x in sync})}
hostevents=rows('main_host_events',f"SELECT json_group_array(json_array(CAST(p.startNs AS INTEGER),CAST(p.endNs AS INTEGER),n.value)) FROM PYTORCH_API p JOIN STRING_IDS n ON n.id=p.name WHERE p.globalTid={main_tid} AND n.value NOT LIKE 'ProfilerStep%' AND CAST(p.endNs AS INTEGER)>CAST(p.startNs AS INTEGER)")
category_measure={key:Measure(v) for key,v in categories.items()}
main_tasks=rows('main_stream_tasks',f"SELECT json_group_array(json_array(t.startNs,t.endNs,n.value)) FROM TASK t JOIN STRING_IDS n ON n.id=t.taskType WHERE t.streamId={main}")
queued=category_measure['next_compute_already_submitted']
queued_device_types={name:queued.union_inside([[s,e] for s,e,n in main_tasks if n==name])/1e9 for name in sorted({x[2] for x in main_tasks})}
all_main=Measure([[s,e] for s,e,*_ in main_tasks])
queued_device_types['no_recorded_main_stream_task']=(queued.prefix[-1]-queued.union_inside(all_main.x))/1e9
# Exclusive innermost recorded scope on the host main thread. CPU/unmarked
# time is not necessarily CPU utilization; this is event attribution only.
import heapq
points=collections.defaultdict(list)
for index,(s,e,name) in enumerate(hostevents):points[s].append((s,e,index,name));points[e]
heap=[];last=None;label='unmarked';attribution=collections.defaultdict(int);by_category={key:collections.defaultdict(int) for key in categories}
for t,beginning in sorted(points.items()):
 if last is not None and t>last:
  attribution[label]+=m.inside(last,t)
  for key,measure in category_measure.items():by_category[key][label]+=measure.inside(last,t)
 for s,e,index,name in beginning:heapq.heappush(heap,(-s,e,index,name))
 while heap and heap[0][1]<=t:heapq.heappop(heap)
 label=heap[0][3] if heap else 'unmarked';last=t
v={'read_only':True,'queries':ns['queries'],'main_stream':main,'main_compute_tasks':len(selected),'main_compute_with_launch_match':sum(x[5] is not None for x in selected),
 'exposed_seconds_by_next_main_compute_submission':{k:v/1e9 for k,v in totals.items()},
 'exposed_inside_main_host_sync_seconds':sync_summary,
 'queued_compute_seconds_by_main_device_task_type':{k:v for k,v in queued_device_types.items() if v>0},
 'exposed_host_scope_top20_by_submission_state':{key:[{'name':name,'seconds':v/1e9} for name,v in sorted(values.items(),key=lambda x:x[1],reverse=True)[:20]] for key,values in by_category.items()},
 'exposed_by_innermost_main_host_record_top20':[{'name':k,'seconds':v/1e9} for k,v in sorted(attribution.items(),key=lambda x:x[1],reverse=True)[:20]],
 'long_gap_examples':sorted(examples,key=lambda x:x['exposed_gap_ms'],reverse=True)[:20],
 'limits':['Host not-yet-submitted indicates no future main-stream kernel command submitted yet; it does not alone identify why the host cannot advance.','Innermost recorded host scope identifies execution context, not isolated CPU usage or mathematical causality.','Queued compute may await communication, copy or other dependencies; exact event producer IDs are absent.']}
(r/'compute-readiness.json').write_text(json.dumps(v,indent=2));print(json.dumps({k:x for k,x in v.items() if k not in ['queries','long_gap_examples']},indent=2))
