"""Check pending parameter gathers during the original queued-compute gaps."""
import json,runpy,statistics
from pathlib import Path
r=Path(__file__).resolve().parent;ns=runpy.run_path(str(r/'analyze_compute_readiness.py'))
base=ns['ns'];rows=base['rows'];Measure=base['Measure'];m=base['m']
ag=rows('ag_launches',"SELECT json_group_array(json_array(o.startNs,o.endNs,h.startNs,h.endNs,o.count)) FROM COMMUNICATION_OP o JOIN STRING_IDS n ON n.id=o.opType JOIN CANN_API h ON h.connectionId=o.connectionId JOIN STRING_IDS a ON a.id=h.name WHERE n.value='hcom_allGather_' AND a.value='launch'")
assert len(ag)==4721
allops=rows('all_ops_for_ag_queue',"SELECT json_group_array(json_array(o.startNs,o.endNs,n.value)) FROM COMMUNICATION_OP o JOIN STRING_IDS n ON n.id=o.opType")
rs=Measure([[s,e] for s,e,n in allops if n=='hcom_reduceScatter_']);gather=Measure([[s,e] for s,e,n in allops if n=='hcom_allGather_'])
pending=[[he,s] for s,e,hs,he,count in ag if s>he];pm=Measure(pending)
queued=ns['category_measure']['next_compute_already_submitted']
no_task=base['difference'](queued.x,ns['all_main'].x)
v={'rank':0,'matched_allgathers':len(ag),'queries':base['queries'],
 'allgather_mean_submit_to_device_ms':statistics.mean(max(0,s-he) for s,e,hs,he,count in ag)/1e6,
 'mean_prior_rs_during_ag_queue_ms':statistics.mean(rs.inside(he,max(he,s)) for s,e,hs,he,count in ag)/1e6,
 'mean_prior_ag_during_ag_queue_ms':statistics.mean(gather.inside(he,max(he,s)) for s,e,hs,he,count in ag)/1e6,
 'rs_exposure_with_parameter_gather_already_pending_seconds':pm.union_inside(base['exposed'])/1e9,
 'queued_main_exposure_with_parameter_gather_pending_seconds':pm.union_inside(queued.x)/1e9,
 'queued_main_no_task_seconds':sum(e-s for s,e in no_task)/1e9,
 'queued_main_no_task_with_gather_pending_seconds':pm.union_inside(no_task)/1e9,
 'limits':'Pending gather plus queued compute is temporal evidence of queue contention; exact parameter-to-kernel dependency in the original capture has no parameter IDs.'}
(r/'pending-gathers.json').write_text(json.dumps(v,indent=2));print(json.dumps({k:x for k,x in v.items() if k!='queries'},indent=2))
