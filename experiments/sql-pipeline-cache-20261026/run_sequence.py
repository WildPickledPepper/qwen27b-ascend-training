"""Tiny exact-state gate, then fixed-workload full run on idle physical 2/3."""
import hashlib,json,os,signal,subprocess,time
from pathlib import Path
root=Path(__file__).resolve().parent
env=os.environ.copy();env.update(ASCEND_RT_VISIBLE_DEVICES='2,3',TASK_QUEUE_ENABLE='0',
    OMP_NUM_THREADS='24',MKL_NUM_THREADS='24',OPENBLAS_NUM_THREADS='24',TOKENIZERS_PARALLELISM='false',HCCL_EXEC_TIMEOUT='600')
def run(phase,cmd,timeout,gpu=False):
    if gpu:
        smi=subprocess.check_output(['npu-smi','info'],text=True)
        assert all(f'No running processes found in NPU {i}' in smi for i in [2,3]),smi
    with (root/(phase+'.private.log')).open('w') as f:
        child=subprocess.Popen(cmd,cwd=root,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
        state=dict(state='running',phase=phase,pid=os.getpid(),child_pid=child.pid,started=time.time())
        (root/'job.private.json').write_text(json.dumps(state))
        try:code=child.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(child.pid,signal.SIGTERM)
            try:child.wait(timeout=20)
            except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
            code=124
    state.update(state='complete' if code==0 else 'failed',code=code,finished=time.time())
    (root/'job.private.json').write_text(json.dumps(state))
    assert code==0,(phase,code)
def train(name,resident,full):
    dest=root/(name+'.private');assert not dest.exists(),dest
    cmd=['python','-m','torch.distributed.run','--nproc_per_node=2','--master_port=29829',str(root/'train_cache.py'),
         '--model',env['MODEL_DIR'],'--data',env['DATA_DIR'],'--output',str(dest),'--norm-device','npu',
         '--resident-gib',str(resident),'--steps','4' if full else '3']+(['--profile'] if full else ['--tiny'])
    run(name,cmd,3000 if full else 600,gpu=True)
train('tiny',2**-13,False)
run('verify-smoke',['python','verify_smoke.py','--root','tiny.private','--output','smoke.json'],120)
run('compare-states',['python','compare_states.py','--reference',env['PIPELINE_TINY_REFERENCE'],
    '--candidate','tiny.private','--output','state-comparison.json'],120)
train('resident2-full',2.,True)
