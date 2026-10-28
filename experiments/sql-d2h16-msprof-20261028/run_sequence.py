"""Use only idle physical3/4; reuse the completed 4/4GiB reference."""
import hashlib,json,os,signal,subprocess,sys,time
from pathlib import Path
root=Path(__file__).resolve().parent;env=os.environ.copy()
env.update(ASCEND_RT_VISIBLE_DEVICES='3,4',TASK_QUEUE_ENABLE='0',OMP_NUM_THREADS='24',MKL_NUM_THREADS='24',OPENBLAS_NUM_THREADS='24',TOKENIZERS_PARALLELISM='false',HCCL_EXEC_TIMEOUT='600')
state=dict(state='starting',runs=[])
manifest={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in root.glob('*.py')}
(root/'source-manifest.private.json').write_text(json.dumps(manifest,indent=2))
def save():(root/'job.private.json').write_text(json.dumps(state,indent=2))
def idle():
 smi=subprocess.check_output(['npu-smi','info'],text=True)
 assert all(f'No running processes found in NPU {i}' in smi for i in [3,4]),smi

def run(name,cmd,timeout):
 idle()
 with (root/(name+'.private.log')).open('x') as f:
  child=subprocess.Popen(cmd,cwd=root,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
  row=dict(name=name,state='running',pid=child.pid,started=time.time());state['phase']=name;state['state']='running';state['runs'].append(row);save()
  try:code=child.wait(timeout=timeout)
  except subprocess.TimeoutExpired:
   os.killpg(child.pid,signal.SIGTERM)
   try:child.wait(timeout=20)
   except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
   code=124
  row.update(state='complete' if code==0 else 'failed',returncode=code,finished=time.time());save();assert code==0,(name,code)
 idle()
def train(name,tiny):
 out=root/(name+'.private');assert not out.exists()
 cmd=[sys.executable,'-m','torch.distributed.run','--nproc_per_node=2','--master_port=29849',str(root/'train_buffered.py'),'--model',env['MODEL_DIR'],'--data',env['DATA_DIR'],'--output',str(out),'--norm-device','npu','--resident-gib-by-rank','0.000244140625,0.00048828125' if tiny else '4,8','--loss-chunk-tokens','64' if tiny else '512','--steps','3','--slots','16','--profile']
 if tiny:cmd.append('--tiny')
 return cmd
try:
 run('tiny-buffered',train('tiny-buffered',True),600)
 run('tiny-exact-check',[sys.executable,str(root/'compare_exact.py'),'--reference',env['TINY_REFERENCE'],'--candidate',str(root/'tiny-buffered.private'),'--output',str(root/'tiny-exact.private.json')],120)
 run('profile16',train('profile16',False),2400)
 state.update(state='complete',finished=time.time());save()
except BaseException as exc:
 state.update(state='failed',error=repr(exc),finished=time.time());save();raise
