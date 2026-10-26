"""Production-path tiny gate (no observing hooks), then matched 27B TP run."""
import json,os,signal,subprocess,time
from pathlib import Path
root=Path(__file__).resolve().parent
env=os.environ.copy();env.update(ASCEND_RT_VISIBLE_DEVICES='2,3',TASK_QUEUE_ENABLE='0',
    OMP_NUM_THREADS='24',MKL_NUM_THREADS='24',OPENBLAS_NUM_THREADS='24',TOKENIZERS_PARALLELISM='false',HCCL_EXEC_TIMEOUT='600')
env.pop('TP_OBSERVE_HOOKS',None)
def run(phase,cmd,timeout):
    smi=subprocess.check_output(['npu-smi','info'],text=True)
    assert all(f'No running processes found in NPU {i}' in smi for i in [2,3]),smi
    with (root/(phase+'.private.log')).open('w') as f:
        child=subprocess.Popen(cmd,cwd=root,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
        state=dict(state='running',phase=phase,child_pid=child.pid,started=time.time())
        (root/'job.private.json').write_text(json.dumps(state))
        try:code=child.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(child.pid,signal.SIGTERM)
            try:child.wait(timeout=20)
            except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
            code=124
    state.update(state='complete' if code==0 else 'failed',code=code,finished=time.time())
    (root/'job.private.json').write_text(json.dumps(state));assert code==0,(phase,code)
assert not (root/'probe.private').exists() and not (root/'full.private').exists()
run('probe',['python','-m','torch.distributed.run','--nproc_per_node=2','--master_port=29830',
    str(root/'probe_tp.py'),'--model',env['MODEL_DIR'],'--output',str(root/'probe.private')],600)
for rank in [0,1]:assert json.loads((root/'probe.private'/f'rank{rank}-result.json').read_text())['passed']
run('full',['python','-m','torch.distributed.run','--nproc_per_node=2','--master_port=29830',
    str(root/'train_tp.py'),'--model',env['MODEL_DIR'],'--data',env['DATA_DIR'],'--output',str(root/'full.private'),
    '--steps','4','--profile'],3000)
