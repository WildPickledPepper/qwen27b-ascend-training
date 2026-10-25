"""Verified launch-to-device lead split by prior communication and event waits."""
import json,runpy,statistics,collections,bisect
from pathlib import Path
r=Path(__file__).resolve().parent
ns=runpy.run_path(str(r/'analyze_dependencies.py'))
c=ns['c'];rows=ns['rows'];Measure=ns['Measure'];merge=ns['merge'];m=ns['m']
allops=rows('all_collectives',"SELECT json_group_array(json_array(o.opId,o.startNs,o.endNs,n.value)) FROM COMMUNICATION_OP o JOIN STRING_IDS n ON n.id=o.opType")
alltasks=rows('communication_stream_tasks',"SELECT json_group_array(json_array(t.startNs,t.endNs,t.streamId,n.value,k.opId)) FROM COMMUNICATION_TASK_INFO k JOIN TASK t ON t.globalTaskId=k.globalTaskId JOIN STRING_IDS n ON n.id=k.taskType")
stream_counts=collections.Counter(x[2] for x in alltasks);stream=stream_counts.most_common(1)[0][0]
assert {x[2] for x in alltasks}=={stream},'Do not assume one HCCL stream'
ag=Measure([[s,e] for _,s,e,kind in allops if kind=='hcom_allGather_'])
rs=Measure([[s,e] for _,s,e,kind in allops if kind=='hcom_reduceScatter_'])
wait=Measure([[s,e] for s,e,st in ns['events'] if st==stream])
both=sum(rs.inside(s,e) for s,e in ag.x)
summary=collections.defaultdict(list)
for opid,s,e,count,connection in ns['ops']:
 hs,he,tid=ns['launchmap'][opid]
 if s<he:continue
 item={'lead':s-he,'prior_ag':ag.inside(he,s),'prior_rs':rs.inside(he,s),'stream_event_wait':wait.inside(he,s)}
 item['other']=item['lead']-item['prior_ag']-item['prior_rs']-item['stream_event_wait']
 assert item['other']>=-100,'Unexpected overlap; queue labels are not exclusive'
 summary[count].append(item)
groups=[]
for count,values in summary.items():
 groups.append({'count':count,'calls':len(values),**{k+'_mean_ms':statistics.mean(x[k] for x in values)/1e6 for k in ['lead','prior_ag','prior_rs','stream_event_wait','other']}})
# Partition the exposed timeline according to top-level host reducer phases.
partition=Measure([[s,e] for s,e,n,tid in ns['host'] if n=='zero/partition_gradients'])
average=Measure([[s,e] for s,e,n,tid in ns['host'] if n=='zero/average_scatter'])
parent=Measure([[s,e] for s,e,n,tid in ns['host'] if n=='zero/reduce_and_partition'])
v={'read_only':True,'queries':ns['queries'],'all_hccl_transport_tasks_on_stream':stream,
 'allgather_reduce_device_overlap_seconds':both/1e9,
 'queue_lead_split':sorted(groups,key=lambda x:x['count'],reverse=True),
 'exposed_seconds_by_exclusive_host_phase':{'average_scatter':m.union_inside(average.x)/1e9,'partition_gradients':m.union_inside(partition.x)/1e9,'other_inside_reducer':(m.union_inside(parent.x)-m.union_inside(average.x)-m.union_inside(partition.x))/1e9,'outside_reducer':(m.prefix[-1]-m.union_inside(parent.x))/1e9},
 'limits':['Per-op queue leads overlap across operations; their sums are not elapsed wall time.','EVENT_WAIT indicates a stream dependency but this DB lacks event IDs for its exact producer.','Shared HCCL execution stream and no overlap are observed, not a proof that changing process groups improves training.']}
(r/'queue.json').write_text(json.dumps(v,indent=2));print(json.dumps({k:v for k,v in v.items() if k!='queries'},indent=2))
