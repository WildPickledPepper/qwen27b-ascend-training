"""Launch one bounded two-NPU hybrid AutoTP probe after idle checks."""
import json,os,signal,subprocess,time
from pathlib import Path
root=Path(__file__).resolve().parent
env=os.environ.copy();env.update(ASCEND_RT_VISIBLE_DEVICES='2,3',TASK_QUEUE_ENABLE='0',
    OMP_NUM_THREADS='24',MKL_NUM_THREADS='24',OPENBLAS_NUM_THREADS='24',TOKENIZERS_PARALLELISM='false',HCCL_EXEC_TIMEOUT='240')
smi=subprocess.check_output(['npu-smi','info'],text=True)
assert all(f'No running processes found in NPU {i}' in smi for i in [2,3]),smi
dest=root/'probe.private';assert not dest.exists(),dest
cmd=['python','-m','torch.distributed.run','--nproc_per_node=2','--master_port=29830',str(root/'probe_tp.py'),
     '--model',env['MODEL_DIR'],'--output',str(dest)]
with (root/'probe.private.log').open('w') as f:
    child=subprocess.Popen(cmd,cwd=root,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
    (root/'job.private.json').write_text(json.dumps(dict(state='running',child_pid=child.pid,started=time.time())))
    try:code=child.wait(timeout=600)
    except subprocess.TimeoutExpired:
        os.killpg(child.pid,signal.SIGTERM)
        try:child.wait(timeout=20)
        except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
        code=124
(root/'job.private.json').write_text(json.dumps(dict(state='complete',code=code,finished=time.time())))
raise SystemExit(code)
