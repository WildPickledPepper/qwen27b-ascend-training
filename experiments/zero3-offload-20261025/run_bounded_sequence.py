"""Run the matched baseline and dual case sequentially, with owned job limits."""
import hashlib,json,os,signal,subprocess,sys,time
from pathlib import Path
root=Path(__file__).resolve().parent;os.chdir(root)
assert json.loads((root/'bounded-copy-probe.json').read_text())['completed']
names=['zero3.json','train_zero3_bounded.py','bounded_grad_copy.py']
manifest={n:hashlib.sha256((root/n).read_bytes()).hexdigest() for n in names}
(root/'bounded-source-manifest.json').write_text(json.dumps(manifest,indent=2))
state={'stage':'running','runs':[]}
def save():(root/'bounded-job.json').write_text(json.dumps(state,indent=2))
save()
for name,world in [('single-run2',1),('dual-run2',2)]:
 assert not (root/name).exists()
 for _ in range(24):
  info=subprocess.check_output(['npu-smi','info'],text=True)
  if all('No running processes found in NPU '+str(i) in info for i in [2,3]):break
  time.sleep(5)
 else:raise RuntimeError('Authorized cards occupied; no new job launched')
 env=dict(os.environ,OMP_NUM_THREADS=str(24//world),OPENBLAS_NUM_THREADS=str(24//world),MKL_NUM_THREADS=str(24//world))
 command=[sys.executable,'-m','torch.distributed.run','--standalone',f'--nproc_per_node={world}',
  'train_zero3_bounded.py','--steps','4','--output',name,'--model',os.environ['MODEL_DIR'],'--data',os.environ['DATA_DIR']]
 with (root/(name+'.log')).open('x') as log:
  proc=subprocess.Popen(command,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  row={'name':name,'world_size':world,'pid':proc.pid,'started_unix':time.time()};state['runs'].append(row);save()
  try:code=proc.wait(timeout=1800)
  except subprocess.TimeoutExpired:
   os.killpg(proc.pid,signal.SIGTERM);proc.wait(timeout=60);raise
 row.update(returncode=code,finished_unix=time.time());save()
 if code:
  state['stage']='failed';save();raise RuntimeError('Keep failed case; do not run next configuration')
 assert all(json.loads((root/name/f'rank{i}-result.json').read_text())['completed'] for i in range(world))
state['stage']='complete';save()
