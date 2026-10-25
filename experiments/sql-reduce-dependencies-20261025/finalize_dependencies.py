"""Keep original timing and new dependency evidence separate."""
import json
from pathlib import Path
r=Path(__file__).resolve().parent;e=r.parents[1]/'evidence/2026-10-25'
load=lambda name:json.loads((e/name).read_text())
old=load('sql-reduce-dependencies.json');queue=load('sql-reduce-queue.json');ready=load('sql-reduce-compute-readiness.json');pending=load('sql-reduce-pending-gathers.json')
capture=load('sql-reduce-capture-verification.json');roles=load('sql-reduce-stream-roles.json');buckets=load('sql-reduce-bucket-chain.json')
assert capture['completed'] and capture['both_devices_released'] and buckets['matched_parameter_ids_and_count_sequence_both_ranks']
assert queue['allgather_reduce_device_overlap_seconds']==0
assert abs(sum(ready['exposed_seconds_by_next_main_compute_submission'].values())-old['exposed_reduction_seconds'])<1e-8
edges=[]
for rank in [0,1]:
 event=load(f'sql-reduce-events-rank{rank}.json');role=roles['ranks'][rank]
 edge=f"{role['main_compute_stream']}<-{role['parameter_prefetch_stream']}"
 assert any(x['edge']==edge for x in event['edges'])
 edges.append({'rank':rank,'main_waits_prefetch_edge':edge,'matched_device_wait_calls':next(x['wait_calls'] for x in event['edges'] if x['edge']==edge),
  'record_markers_matched':event['coverage']['record_matched_api'],'wait_markers_with_device':event['coverage']['wait_matched_api'],
  'wait_markers_total':event['coverage']['wait_markers'],'validation_exact_at_768':capture['ranks'][rank]['validation_exact']})
v={'original_profile_exposed_reduction_seconds':old['exposed_reduction_seconds'],
 'original_transport_breakdown':old['exposed_inside_transport'],
 'original_next_compute_submission_state_seconds':ready['exposed_seconds_by_next_main_compute_submission'],
 'original_largest_ffn_bucket_queue':next(x for x in queue['queue_lead_split'] if x['count']==44564480),
 'original_ag_pending_evidence':{k:v for k,v in pending.items() if k not in ['queries','limits']},
 'new_capture_parameter_and_collective_sequence_verified':True,'new_event_edges':edges,
 'confirmed_mechanisms':['Large gradient transfer/reduction occupies the shared HCCL execution stream.',
 'Parameter AllGather and gradient ReduceScatter are serialized on that stream in the selected profile.',
 'Many exposed intervals occur before the next main-stream compute has been submitted, within host autograd/ZeRO/event processing.',
 'Labeled event producers confirm main-stream waits on the parameter-prefetch processing stream in both ranks.'],
 'not_fully_attributed':'The original 4.161608807 seconds without a recorded main-stream task have not all been assigned to exact event or resource causes. 1.229202949 seconds coincide with pending parameter gathers; coincidence is not a complete parameter-to-kernel dependency map.',
 'next_optimization_targets':['Reduce per-parameter host dispatch/event/observer overhead without changing the update workload.',
 'Prioritize or separate parameter prefetch versus large gradient traffic only after verifying native HCCL support and measuring whole-step behavior.'],
 'default_configuration_changed':False,'speedup_claim':None,'devices_2_3_released':True,
 'limits':['Extra-labeled dual-rank capture changed timing and produced much larger rank0 Notify_Wait; never substitute it for the original profile or unprofiled throughput.',
 'Event pending intervals derived from stream semantics are not direct hardware wait-counter measurements.',
 'Background CPU-worker synchronize labels are not all present in framework API traces; do not treat main-thread marker counts as all-thread totals.']}
(e/'sql-reduce-dependency-conclusion.json').write_text(json.dumps(v,indent=2));print(json.dumps({'mechanisms':v['confirmed_mechanisms'],'unresolved':v['not_fully_attributed'],'default_configuration_changed':False},indent=2))
