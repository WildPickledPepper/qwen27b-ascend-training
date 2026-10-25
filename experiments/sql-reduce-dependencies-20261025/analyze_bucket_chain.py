"""Match exact parameter IDs, host gradient hooks, bucket and HCCL timelines."""
import bisect,collections,json,sqlite3,statistics
from pathlib import Path
r=Path(__file__).resolve().parent;report=[];sequences=[]
for rank in [0,1]:
 db=next((r/'capture-full/profile').rglob(f'ascend_pytorch_profiler_{rank}.db'));c=sqlite3.connect(f'file:{db}?mode=ro',uri=True)
 queries={
 'markers':"SELECT json_group_array(json_array(CAST(p.startNs AS INTEGER),CAST(p.endNs AS INTEGER),p.globalTid,n.value)) FROM PYTORCH_API p JOIN STRING_IDS n ON n.id=p.name WHERE n.value LIKE 'dep/bucket/%' OR n.value LIKE 'dep/host_grad_hook/%'",
 'ops':"SELECT json_group_array(json_array(o.startNs,o.endNs,o.count,h.startNs,h.endNs,h.globalTid)) FROM COMMUNICATION_OP o JOIN STRING_IDS n ON n.id=o.opType JOIN CANN_API h ON h.connectionId=o.connectionId JOIN STRING_IDS a ON a.id=h.name WHERE n.value='hcom_reduceScatter_' AND a.value='launch'"
 }
 markers=json.loads(c.execute(queries['markers']).fetchone()[0]);ops=json.loads(c.execute(queries['ops']).fetchone()[0])
 buckets=sorted([x for x in markers if x[3].startswith('dep/bucket/')]);starts=[x[0] for x in buckets]
 hooks=collections.defaultdict(list)
 for s,e,tid,label in markers:
  if label.startswith('dep/host_grad_hook/'):hooks[int(label.split('/')[-1])].append(s)
 for v in hooks.values():v.sort()
 groups=collections.defaultdict(list);sequence=[];seen=set()
 for ds,de,count,hs,he,tid in sorted(ops,key=lambda x:x[3]):
  index=bisect.bisect_right(starts,hs)-1;bs,be,bt,label=buckets[index]
  assert be>=he and bt==tid and index not in seen;seen.add(index)
  ids=tuple(int(x) for x in label.split('/')[-1].split(','));ready=[]
  for param in ids:
   i=bisect.bisect_right(hooks[param],bs)-1;assert i>=0
   ready.append(hooks[param][i])
  last_ready=max(ready);sequence.append((count,ids))
  groups[count].append({'host_hook_to_bucket_ms':(bs-last_ready)/1e6,'bucket_to_submit_ms':(hs-bs)/1e6,'submit_to_device_ms':(ds-he)/1e6,'communication_ms':(de-ds)/1e6})
 assert len(seen)==len(buckets)==len(ops)==4504
 assert sum(len(x) for x in hooks.values())==6808
 sequences.append(sequence)
 report.append({'rank':rank,'matched_buckets':4504,'host_gradient_hooks':6808,'queries':queries,
 'groups':[{'count':count,'calls':len(values),**{key:statistics.mean(x[key] for x in values) for key in values[0]}} for count,values in sorted(groups.items(),reverse=True)]})
assert sequences[0]==sequences[1]
v={'matched_parameter_ids_and_count_sequence_both_ranks':True,'ranks':report,
 'limits':['Gradient hook means host autograd callback reached; it does not mean asynchronous device gradient production finished.',
 'This is the extra-labeled dual-profile capture, not the original 13.36-second profile or unprofiled throughput.']}
(r/'bucket-chain.json').write_text(json.dumps(v,indent=2));print(json.dumps(v,indent=2))
