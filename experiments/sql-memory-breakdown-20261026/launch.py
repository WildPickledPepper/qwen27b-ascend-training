import json,os,subprocess,time
from pathlib import Path
root=Path(__file__).resolve().parent
smi=subprocess.check_output(['npu-smi','info'],text=True)
assert all(f'No running processes found in NPU {i}' in smi for i in [2,3]),'Cards occupied'
out=root/'full.private'; assert not out.exists()
env=os.environ.copy();env.update(ASCEND_RT_VISIBLE_DEVICES='2,3',TASK_QUEUE_ENABLE='0',OMP_NUM_THREADS='24',MKL_NUM_THREADS='24',OPENBLAS_NUM_THREADS='24',TOKENIZERS_PARALLELISM='false',HCCL_EXEC_TIMEOUT='600')
cmd=['python','-m','torch.distributed.run','--nproc_per_node=2','--master_port=29835',str(root/'train_memory.py'),'--model',env['MODEL_DIR'],'--data',env['DATA_DIR'],'--output',str(out),'--steps','1','--norm-device','npu','--resident-gib','4']
with (root/'full.private.log').open('w') as f:
 child=subprocess.Popen(cmd,env=env,cwd=root,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
 state=dict(state='running',pid=child.pid,started=time.time());(root/'job.private.json').write_text(json.dumps(state))
 try: code=child.wait(timeout=1800)
 except subprocess.TimeoutExpired:
  import signal
  os.killpg(child.pid,signal.SIGTERM);code=124
 state.update(state='complete' if code==0 else 'failed',code=code,finished=time.time());(root/'job.private.json').write_text(json.dumps(state))
assert code==0,code
