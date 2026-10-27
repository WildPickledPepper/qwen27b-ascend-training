import os,sys,time,json,subprocess,signal,hashlib
from pathlib import Path
r=Path(__file__).resolve().parent
env=os.environ.copy();env.update(ASCEND_RT_VISIBLE_DEVICES='3,4',TASK_QUEUE_ENABLE='0',OMP_NUM_THREADS='24',MKL_NUM_THREADS='24',OPENBLAS_NUM_THREADS='24',TOKENIZERS_PARALLELISM='false',HCCL_EXEC_TIMEOUT='600')
state={'state':'running','runs':[]}
def save():(r/'job.private.json').write_text(json.dumps(state,indent=2))
def idle():
 s=subprocess.check_output(['npu-smi','info'],text=True)
 assert all(f'No running processes found in NPU {i}' in s for i in [3,4])
(r/'source.private.json').write_text(json.dumps({p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in r.glob('*.py')},indent=2))
try:
 for name,budget in [('cache4-4','4,4'),('cache4-8','4,8')]:
  idle();out=r/(name+'.private');assert not out.exists()
  cmd=[sys.executable,'-m','torch.distributed.run','--nproc_per_node=2','--master_port=29848',str(r/'train_timed.py'),'--model',env['MODEL_DIR'],'--data',env['DATA_DIR'],'--output',str(out),'--norm-device','npu','--resident-gib-by-rank',budget,'--loss-chunk-tokens','512','--steps','3']
  with (r/(name+'.private.log')).open('x') as f:
   p=subprocess.Popen(cmd,env=env,cwd=r,stdout=f,stderr=subprocess.STDOUT,start_new_session=True);row=dict(name=name,pid=p.pid,started=time.time(),state='running');state['runs'].append(row);save()
   try:code=p.wait(timeout=1800)
   except subprocess.TimeoutExpired:
    os.killpg(p.pid,signal.SIGTERM)
    try:p.wait(timeout=20)
    except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
    code=124
   row.update(returncode=code,finished=time.time(),state='complete' if code==0 else 'failed');save();assert code==0
  idle()
 state['state']='complete';save()
except BaseException as e:state.update(state='failed',error=repr(e));save();raise
