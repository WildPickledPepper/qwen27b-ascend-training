"""CPU-backpressure candidate selected from measured capacity, original copy path."""
import argparse,hashlib,json,os,signal,subprocess,time
from pathlib import Path
root=Path(__file__).resolve().parent;env=os.environ.copy()
p=argparse.ArgumentParser();p.add_argument('--entry',required=True);a=p.parse_args();entry=Path(a.entry).resolve()
sealed=json.loads((entry.parent/'selected-profile.json').read_text())
assert all(hashlib.sha256((entry.parent/name).read_bytes().replace(b'\r\n',b'\n')).hexdigest()==value for name,value in sealed['source_lf_sha256'].items())
env.update(ASCEND_RT_VISIBLE_DEVICES='2,3',TASK_QUEUE_ENABLE='0',OMP_NUM_THREADS='24',
    MKL_NUM_THREADS='24',OPENBLAS_NUM_THREADS='24',TOKENIZERS_PARALLELISM='false',HCCL_EXEC_TIMEOUT='600')
def run(name,cmd,timeout,gpu=False):
    if gpu:
        smi=subprocess.check_output(['npu-smi','info'],text=True)
        assert all(f'No running processes found in NPU {i}' in smi for i in [2,3]),smi
    with (root/(name+'.private.log')).open('w') as f:
        child=subprocess.Popen(cmd,cwd=root,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
        state=dict(state='running',phase=name,child_pid=child.pid,started=time.time())
        (root/'cache4-job.private.json').write_text(json.dumps(state))
        try:code=child.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(child.pid,signal.SIGTERM)
            try:child.wait(timeout=20)
            except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
            code=124
    state.update(state='complete' if code==0 else 'failed',code=code,finished=time.time())
    (root/'cache4-job.private.json').write_text(json.dumps(state));assert code==0,(name,code)
def train(name,tiny):
    output=root/(name+'.private');assert not output.exists()
    # Reuse the unchanged sealed training entry, changing only its existing cache argument.
    assert entry.exists()
    cmd=['python','-m','torch.distributed.run','--nproc_per_node=2','--master_port=29832',str(entry),
        '--model',env['MODEL_DIR'],'--data',env['DATA_DIR'],'--output',str(output),'--norm-device','npu',
        '--resident-gib',str(2**-12 if tiny else 4.),'--steps','3' if tiny else '4']
    cmd+=['--tiny'] if tiny else ['--profile']
    run(name,cmd,600 if tiny else 2400,gpu=True)
assert json.loads((root/'job.private.json').read_text())['state']=='complete','Wait for async job exit'
train('cache4-tiny',True)
run('cache4-verify-smoke',['python','verify_smoke.py','--root','cache4-tiny.private','--output','cache4-smoke.json'],120)
run('cache4-compare-states',['python','compare_states.py','--reference',env['PIPELINE_TINY_REFERENCE'],
    '--candidate','cache4-tiny.private','--output','cache4-state-comparison.json'],120)
train('cache4-full',False)
