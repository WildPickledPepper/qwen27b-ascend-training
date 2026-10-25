"""Sequential, bounded throughput experiments; never touch other jobs."""
import hashlib,json,os,signal,subprocess,sys,time
from pathlib import Path
r=Path(__file__).resolve().parent;os.chdir(r)
sources=['train_throughput.py','bounded_grad_copy.py','zero3.json']
manifest={n:hashlib.sha256((r/n).read_bytes()).hexdigest() for n in sources}
(r/'source-manifest.json').write_text(json.dumps(manifest,indent=2))
state={'stage':'running','global_batch':32,'steps':3,'runs':[]}
def save():(r/'sweep-job.json').write_text(json.dumps(state,indent=2))
save()
for micro in [1,4,8,16]:
 name=f'mb{micro}-g32'
 assert not (r/name).exists()
 for _ in range(36):
  info=subprocess.check_output(['npu-smi','info'],text=True)
  if all('No running processes found in NPU '+str(i) in info for i in [2,3]):break
  time.sleep(5)
 else:raise RuntimeError('Authorized cards are occupied; no new experiment launched')
 env=dict(os.environ,OMP_NUM_THREADS='12',OPENBLAS_NUM_THREADS='12',MKL_NUM_THREADS='12')
 command=[sys.executable,'-m','torch.distributed.run','--standalone','--nproc_per_node=2',
  'train_throughput.py','--steps','3','--output',name,'--global-batch','32','--micro-batch',str(micro),
  '--model',os.environ['MODEL_DIR'],'--data',os.environ['DATA_DIR']]
 with (r/(name+'.log')).open('x') as log:
  proc=subprocess.Popen(command,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  row={'name':name,'micro_batch':micro,'pid':proc.pid,'started_unix':time.time()};state['runs'].append(row);save()
  try:code=proc.wait(timeout=2400)
  except subprocess.TimeoutExpired:
   os.killpg(proc.pid,signal.SIGTERM);proc.wait(timeout=60);code=-15
 row.update(returncode=code,finished_unix=time.time());save()
 if code:
  tail=(r/(name+'.log')).read_text()[-20000:]
  if 'out of memory' in tail.lower():
   row['failure']='oom';save();break
  state['stage']='failed';save();raise RuntimeError('Non-memory failure: retain evidence and investigate')
 assert all(json.loads((r/name/f'rank{i}-result.json').read_text())['completed'] for i in [0,1])
state['stage']='complete';save()
