"""Exact tiny gate, then same global256 full model, isolated on idle cards2/3."""
import json,os,signal,subprocess,time
from pathlib import Path
root=Path(__file__).resolve().parent;env=os.environ.copy()
env.update(ASCEND_RT_VISIBLE_DEVICES='2,3',TASK_QUEUE_ENABLE='0',OMP_NUM_THREADS='24',MKL_NUM_THREADS='24',
    OPENBLAS_NUM_THREADS='24',TOKENIZERS_PARALLELISM='false',HCCL_EXEC_TIMEOUT='600')
def run(name,command,timeout,gpu=False):
    if gpu:
        smi=subprocess.check_output(['npu-smi','info'],text=True)
        assert all(f'No running processes found in NPU {d}' in smi for d in [2,3]),'Cards occupied'
    with (root/(name+'.private.log')).open('w') as f:
        child=subprocess.Popen(command,cwd=root,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
        state=dict(phase=name,state='running',child_pid=child.pid,started=time.time())
        (root/'job.private.json').write_text(json.dumps(state))
        try:code=child.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(child.pid,signal.SIGTERM)
            try:child.wait(timeout=20)
            except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
            code=124
    state.update(state='complete' if code==0 else 'failed',code=code,finished=time.time())
    (root/'job.private.json').write_text(json.dumps(state));assert code==0,(name,code)
def train(name,tiny):
    dest=root/(name+'.private');assert not dest.exists()
    cmd=['python','-m','torch.distributed.run','--nproc_per_node=2','--master_port=29833',str(root/'train.py'),
        '--model',env['MODEL_DIR'],'--data',env['DATA_DIR'],'--output',str(dest),
        '--codec-library',str(root/'libonline_codec.so'),'--archive-root',env['BF16_ARCHIVE_ROOT'],
        '--resident-gib',str(2**-13 if tiny else 7.),'--norm-device','npu','--steps','3' if tiny else '4']
    cmd+=['--tiny'] if tiny else ['--profile']
    run(name,cmd,600 if tiny else 3000,True)
train('tiny',True)
run('verify-smoke',['python','verify_smoke.py','--root','tiny.private','--output','smoke.json'],120)
run('compare-states',['python','compare_states.py','--reference',env['PIPELINE_TINY_REFERENCE'],
    '--candidate','tiny.private','--output','state-comparison.json'],120)
train('full',False)
