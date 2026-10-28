"""Audit the kernel-only run against the completed same-card replay."""
import argparse,hashlib,json,subprocess
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--reference',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
job=json.loads((a.root/'job.private.json').read_text());assert job['state']=='complete' and job['cleanup_verified']
kernel=json.loads((a.root/'kernel-check.json').read_text());tiny=json.loads((a.root/'tiny-check.json').read_text());assert kernel['differing_bits']==0 and tiny['all_exact']
ranks=[json.loads((a.root/f'full.private/rank{i}-result.json').read_text()) for i in [0,1]]
refs=[json.loads((a.reference/f'replay.private/rank{i}-result.json').read_text()) for i in [0,1]]
checks=[];metrics=[]
for rank,(cur,ref) in enumerate(zip(ranks,refs)):
    def events(folder):return [json.loads(x) for x in (folder/f'rank{rank}.jsonl').read_text().splitlines()]
    now=events(a.root/'full.private');old=events(a.reference/'replay.private')
    def event(xs,n):return next(x for x in xs if x['event']==n)
    for key in ['training_hashes','training_lengths','validation_hashes']:assert event(now,'data')[key]==event(old,'data')[key]
    nmodel=event(now,'model_ready');omodel=event(old,'model_ready')
    for key in ['manifest','checkpoint_layers','physical_card','cpu_cores','numa_node']:assert nmodel[key]==omodel[key],key
    for s,b in zip(cur['steps'],ref['steps']):
        assert not s['profiled'] and s['cpu_kernel_calls']>0
        for key in ['input_tokens','padded_tokens','gradient_d2h_bytes','resident_add_calls','resident_add_bytes']:assert s[key]==b[key],(rank,key)
    checks.append(dict(rank=rank,workload_exact=True,model_and_topology_exact=True,state_bytes_exact=cur['state_bytes']==ref['state_bytes'],validation_before_exact=cur['validation_before']==ref['validation_before'],validation_after_exact=cur['validation_after_768']==ref['validation_after_768'],grad_norms_exact=all(s['grad_norm']==b['grad_norm'] for s,b in zip(cur['steps'],ref['steps'])),norm_pairs=[[s['grad_norm'],b['grad_norm']] for s,b in zip(cur['steps'],ref['steps'])]))
    mm={}
    for name,x in [('reference',ref),('kernel_only',cur)]:
        ss=x['steps'][1:3]
        mm[name]={key:sum(s[key] for s in ss)/2 for key in ['seconds','forward_backward_and_drain_seconds','global_norm_seconds','cpu_update_return_seconds','worker_cpu_add_seconds','offload_slot_wait','worker_copy_wait_seconds']}
        mm[name]['tail_mean']={key:sum(s['tail_seconds'][key] for s in ss)/2 for key in ss[0]['tail_seconds']}
        mm[name]['allocated_peak_gib']=max(s['allocated_peak'] for s in x['steps'])/2**30
        mm[name]['reserved_peak_gib']=max(s['reserved_peak'] for s in x['steps'])/2**30
    metrics.append(dict(rank=rank,physical_card=[0,3][rank],metrics=mm))
seconds=[max(x['steps'][i]['seconds'] for x in ranks) for i in range(3)];baseline=[max(x['steps'][i]['seconds'] for x in refs) for i in range(3)]
warm=sum(seconds[1:])/2;refwarm=sum(baseline[1:])/2
smi=subprocess.check_output(['npu-smi','info'],text=True);idle={str(i):f'No running processes found in NPU {i}' in smi for i in [0,3]};assert all(idle.values())
def clean(x):return {**x,'steps':[{k:v for k,v in s.items() if k!='tail_chunk_times'} for s in x['steps']]}
report=dict(step_seconds=seconds,reference_step_seconds=baseline,warm_seconds=warm,reference_warm_seconds=refwarm,time_change_percent=(warm/refwarm-1)*100,checks=checks,warm_metrics=metrics,kernel_validation=kernel,tiny_validation=tiny,rank_results=[clean(x) for x in ranks],slots_per_rank=16,slot_bytes=128*2**20,kernel_calls_are_cumulative=True,source_sha256={f:hashlib.sha256((a.root/f).read_bytes()).hexdigest() for f in ['run_sequence.py','train_k1.py','slot_kernel.py','cpu_fused_accum.c','resident_gradients.py','chunked_loss_contiguous.py','community_gdn.py']},cleanup={'physical_card_no_processes':idle},limits=['Sequential candidate compared with preceding same-card replay, not an interleaved A/B; load may change.','Two timed updates only; first lr0 warmup excluded.','Tiny all-state equality and full-model norms/short validation are not a full-model state or long-term quality proof.','Host copy wait contains device dependency wait; not native DMA duration.'])
a.output.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({'warm_seconds':warm,'reference_warm_seconds':refwarm,'time_change_percent':report['time_change_percent'],'checks':checks,'warm_metrics':metrics},indent=2))
