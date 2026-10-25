"""Expand only the measured CPU receive-queue bottleneck; preserve math/order."""
from pathlib import Path
r=Path(__file__).resolve().parent;b=r.parent/'sql-gradient-cpu-20261025'
for name in ['train_cpu.py','community_gdn.py','cpu_update.py','bounded_grad_copy.py','zero3-sql.json','gradient_scopes.py','gradient_cache.py','compare_probes.py','export_evidence.py','status.py','collect_result.py']:
 text=(b/name).read_text(encoding='utf-8')
 if name=='collect_result.py':text=text.replace("'sql-gradient-cpu-results.json'","'sql-gradient-pipeline-results.json'")
 (r/name).write_text(text,encoding='utf-8',newline='\n')
s=(b/'cpu_accumulation.py').read_text(encoding='utf-8')
s=s.replace('two bounded pinned incoming buffers','a bounded tiered pinned receive queue')
s=s.replace("slots=[torch.empty(maximum,dtype=torch.bfloat16,pin_memory=True) for _ in range(2)]\n pending=[None,None];counter=[0];stats=collections.defaultdict(float)",
 "capacities=[min(maximum,64*2**20)]*30+[maximum]*2\n slots=[torch.empty(n,dtype=torch.bfloat16,pin_memory=True) for n in capacities]\n pending=[None]*len(slots);issued=[-1]*len(slots);counter=[0];stats=collections.defaultdict(float)")
s=s.replace('slot=counter[0]%2;counter[0]+=1',"eligible=[i for i,n in enumerate(capacities) if n>=part.numel()]\n  minimum=min(capacities[i] for i in eligible)\n  eligible=[i for i in eligible if capacities[i]==minimum]\n  ready=[i for i in eligible if pending[i] is None or pending[i].done()]\n  slot=ready[0] if ready else min(eligible,key=lambda i:issued[i])\n  issued[slot]=counter[0];counter[0]+=1")
s=s.replace("'pinned_staging_bytes':2*maximum*2,'slots':2", "'pinned_staging_bytes':sum(capacities)*2,'slots':len(slots),'small_slot_elements':capacities[0]")
assert 'pending=[None,None]' not in s and 'capacities=' in s
(r/'cpu_accumulation.py').write_text(s,encoding='utf-8',newline='\n')
s=(b/'run_experiment.py').read_text(encoding='utf-8')
s=s.replace("try:\n for name,cache", "try:\n state['stage']='waiting-for-prior-cpu-run';save()\n prior=r.parent/'qwen-sql-gradient-cpu-20261025'/'job.json'\n deadline=time.time()+2400\n while json.loads(prior.read_text())['stage'] not in ['complete','failed']:\n  assert time.time()<deadline,'Prior job did not finish'\n  time.sleep(5)\n for name,cache")
(r/'run_experiment.py').write_text(s,encoding='utf-8',newline='\n')
s=(b/'launch.private.sh').read_text(encoding='utf-8').replace('qwen-sql-gradient-cpu-20261025','qwen-sql-gradient-pipeline-20261025')
(r/'launch.private.sh').write_text(s,encoding='utf-8',newline='\n')
print('Generated bounded 32-slot CPU pipeline entry')
