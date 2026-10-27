"""Isolated NPU3/4 sequence, tests followed by matched full-model runs."""
import json,os,signal,subprocess,sys,time
from pathlib import Path
root=Path(__file__).resolve().parent;env=os.environ.copy()
env.update(ASCEND_RT_VISIBLE_DEVICES='3,4',TASK_QUEUE_ENABLE='0',OMP_NUM_THREADS='24',MKL_NUM_THREADS='24',OPENBLAS_NUM_THREADS='24',TOKENIZERS_PARALLELISM='false',HCCL_EXEC_TIMEOUT='600')
state=dict(state='starting',runs=[])
def save(): (root/'job-contiguous.private.json').write_text(json.dumps(state,indent=2))
def idle():
 smi=subprocess.check_output(['npu-smi','info'],text=True)
 assert all(f'No running processes found in NPU {n}' in smi for n in [3,4]),smi

def run(name,cmd,timeout):
 idle()
 with (root/(name+'.private.log')).open('x') as f:
  child=subprocess.Popen(cmd,cwd=root,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
  row=dict(name=name,state='running',pid=child.pid,started=time.time());state['runs'].append(row);state['state']='running';state['phase']=name;save()
  try:code=child.wait(timeout=timeout)
  except subprocess.TimeoutExpired:
   os.killpg(child.pid,signal.SIGTERM)
   try:child.wait(timeout=20)
   except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
   code=124
  row.update(state='complete' if code==0 else 'failed',returncode=code,finished=time.time());save()
  assert code==0,(name,code)
 idle()

def command(tiny,chunk,name):
 out=root/(name+'.private');assert not out.exists()
 cmd=[sys.executable,'-m','torch.distributed.run','--nproc_per_node=2','--master_port=29846',str(root/'train_chunked_contiguous.py'),'--model',env['MODEL_DIR'],'--data',env['DATA_DIR'],'--output',str(out),'--norm-device','npu','--resident-gib',str(2**-12 if tiny else 4),'--steps','3','--loss-chunk-tokens',str(chunk)]
 if tiny:cmd.append('--tiny')
 return cmd
try:
 run('layout-contiguous',[sys.executable,str(root/'probe_layout_contiguous.py')],180)
 run('loss-contiguous-probe',[sys.executable,str(root/'probe_loss_contiguous.py')],300)
 run('tiny-contiguous',command(True,64,'tiny-contiguous'),600)
 run('tiny-contiguous-check',[sys.executable,str(root/'compare_tiny.py'),'--root',str(root),'--candidate','tiny-contiguous.private','--output',str(root/'tiny-contiguous-comparison.private.json')],120)
 run('chunk512-contiguous',command(False,512,'chunk512-contiguous'),2400)
 state.update(state='complete',finished=time.time());save()
except BaseException as exc:
 state.update(state='failed',error=repr(exc),finished=time.time());save();raise
