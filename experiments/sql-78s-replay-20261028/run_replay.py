"""Replay the historical 77.8298s case, changing only physical topology.

One lr=0 warmup and two timed updates; no profiler or buffering changes.
Run under the original environment. Private runtime paths are CLI inputs.
"""
import argparse,hashlib,json,os,signal,subprocess,sys,time
from pathlib import Path

p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--expected',type=Path,required=True);a=p.parse_args()
root=Path(__file__).resolve().parent
state={'state':'preparing','physical_cards':[0,3],'runs':[]}
def save(): (root/'job.private.json').write_text(json.dumps(state,indent=2))
def idle():
    text=subprocess.check_output(['npu-smi','info'],text=True)
    assert all(f'No running processes found in NPU {i}' in text for i in [0,3]),text
    return text
expected=json.loads(a.expected.read_text())
manifest={}
for name in ['train_timed.py','resident_gradients.py','chunked_loss_contiguous.py','community_gdn.py']:
    raw=(a.source/name).read_bytes();digest=hashlib.sha256(raw).hexdigest()
    assert digest==expected[name],(name,digest,expected[name])
    text=raw.decode()
    if name=='train_timed.py':
        changes=[("== '3,4'","== '0,3'"),('list(range(96,120)) if rank == 0 else list(range(0,24))','list(range(144,168)) if rank == 0 else list(range(96,120))'),('[4,0][rank]','[6,4][rank]'),('[3,4][rank]','[0,3][rank]')]
        for old,new in changes:
            assert old in text,old
            text=text.replace(old,new)
        reverse=text
        for old,new in reversed(changes):reverse=reverse.replace(new,old)
        assert reverse==raw.decode(),'Unexpected training changes'
        (root/name).write_text(text)
    else:(root/name).write_bytes(raw)
    manifest[name]={'historical_sha256':digest,'replay_sha256':hashlib.sha256((root/name).read_bytes()).hexdigest()}
(root/'source-manifest.private.json').write_text(json.dumps(manifest,indent=2))
env=os.environ.copy();env.update(ASCEND_RT_VISIBLE_DEVICES='0,3',TASK_QUEUE_ENABLE='0',OMP_NUM_THREADS='24',MKL_NUM_THREADS='24',OPENBLAS_NUM_THREADS='24',TOKENIZERS_PARALLELISM='false',HCCL_EXEC_TIMEOUT='600')
out=root/'replay.private';assert not out.exists()
idle();save()
cmd=[sys.executable,'-m','torch.distributed.run','--nproc_per_node=2','--master_port=29859',str(root/'train_timed.py'),'--model',env['MODEL_DIR'],'--data',env['DATA_DIR'],'--output',str(out),'--norm-device','npu','--resident-gib-by-rank','4,8','--loss-chunk-tokens','512','--slots','16','--steps','3']
with (root/'train.private.log').open('x') as log:
    proc=subprocess.Popen(cmd,env=env,cwd=root,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    state.update(state='running',pid=proc.pid,started=time.time());save()
    try:code=proc.wait(timeout=2400)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid,signal.SIGTERM)
        try:proc.wait(timeout=20)
        except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
        code=124
state.update(returncode=code,finished=time.time(),state='complete' if code==0 else 'failed');save()
assert code==0,code
idle_text=idle();(root/'post-smi.private.txt').write_text(idle_text)
ranks=[json.loads((out/f'rank{i}-result.json').read_text()) for i in [0,1]]
refs=[json.loads((a.source/f'cache4-8.private/rank{i}-result.json').read_text()) for i in [0,1]]
checks=[]
for rank,(cur,old) in enumerate(zip(ranks,refs)):
    events=[json.loads(x) for x in (out/f'rank{rank}.jsonl').read_text().splitlines()]
    oldevents=[json.loads(x) for x in (a.source/f'cache4-8.private/rank{rank}.jsonl').read_text().splitlines()]
    def event(xs,n):return next(x for x in xs if x['event']==n)
    for key in ['training_hashes','training_lengths','validation_hashes']:assert event(events,'data')[key]==event(oldevents,'data')[key]
    model=event(events,'model_ready');refmodel=event(oldevents,'model_ready')
    assert model['manifest']==refmodel['manifest']
    assert model['checkpoint_layers']==refmodel['checkpoint_layers']
    for new,ref in zip(cur['steps'],old['steps']):
        for key in ['input_tokens','padded_tokens','gradient_d2h_bytes','resident_add_calls','resident_add_bytes']:assert new[key]==ref[key],(rank,key)
        assert not new['profiled']
    checks.append(dict(rank=rank,workload_exact=True,manifest_exact=True,checkpoint_layers=model['checkpoint_layers'],state_bytes_exact=cur['state_bytes']==old['state_bytes'],validation_before_exact=cur['validation_before']==old['validation_before'],validation_after_exact=cur['validation_after_768']==old['validation_after_768'],grad_norms_exact=all(x['grad_norm']==y['grad_norm'] for x,y in zip(cur['steps'],old['steps'])),norm_pairs=[[x['grad_norm'],y['grad_norm']] for x,y in zip(cur['steps'],old['steps'])]))
seconds=[max(r['steps'][i]['seconds'] for r in ranks) for i in range(3)]
report=dict(historical_warm_seconds=77.82983503991272,physical_cards=[0,3],historical_physical_cards=[3,4],step_seconds=seconds,warm_seconds=sum(seconds[1:])/2,checks=checks,rank_results=ranks,source_sha256=manifest,cleanup={'physical_cards_no_processes':{'0':True,'3':True}},limits=['Physical card/NUMA pair changed; historical timing is not a same-host-load paired control.','Warm mean uses two timed updates, excludes first lr0 warmup.','Reproduces historical chunked-loss/cache variant, not a change to the default entry.','Short validation and norms do not establish long-term training quality.'])
(root/'results.json').write_text(json.dumps(report,indent=2))
state.update(state='complete',report_written=True,cleanup_verified=True);save()
print(json.dumps({'warm_seconds':report['warm_seconds'],'step_seconds':seconds,'checks':checks}),flush=True)
