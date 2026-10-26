"""Capacity fallback: same batch/micro/data, all layers recomputed; tiny gate first."""
import json,os,signal,subprocess,time
from pathlib import Path
root=Path(__file__).resolve().parent
env=os.environ.copy();env.update(ASCEND_RT_VISIBLE_DEVICES='2,3',TASK_QUEUE_ENABLE='0',OMP_NUM_THREADS='24',
    MKL_NUM_THREADS='24',OPENBLAS_NUM_THREADS='24',TOKENIZERS_PARALLELISM='false',HCCL_EXEC_TIMEOUT='600')
env.pop('TP_OBSERVE_HOOKS',None)
for phase,script,extra,timeout in [
    ('probe-all','probe_tp.py',['--checkpoint','all'],600),
    ('full-all','train_tp.py',['--data',env['DATA_DIR'],'--steps','4','--profile','--checkpoint','all'],3000)]:
    smi=subprocess.check_output(['npu-smi','info'],text=True)
    assert all(f'No running processes found in NPU {i}' in smi for i in [2,3]),smi
    dest=root/(phase+'.private');assert not dest.exists()
    cmd=['python','-m','torch.distributed.run','--nproc_per_node=2','--master_port=29830',str(root/script),
         '--model',env['MODEL_DIR'],'--output',str(dest)]+extra
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
