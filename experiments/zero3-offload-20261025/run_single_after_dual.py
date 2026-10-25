"""Wait for this experiment's dual run, then run a matched global-batch baseline."""
import json,os,subprocess,sys,time
from pathlib import Path
root=Path(__file__).resolve().parent;os.chdir(root)
deadline=time.monotonic()+1800
while True:
 done=all((root/'dual-run1'/f'rank{i}-result.json').exists() for i in [0,1])
 if done:break
 assert time.monotonic()<deadline,'Dual result not completed; do not launch baseline'
 time.sleep(5)
for i in [0,1]:
 result=json.loads((root/'dual-run1'/f'rank{i}-result.json').read_text())
 assert result['completed'] and len(result['steps'])==4
# Destruction follows result write; start only after both cards are released.
while True:
 info=subprocess.check_output(['npu-smi','info'],text=True)
 if all('No running processes found in NPU '+str(i) in info for i in [2,3]):break
 assert time.monotonic()<deadline,'Own dual process did not release devices'
 time.sleep(5)
env=dict(os.environ,OMP_NUM_THREADS='24',OPENBLAS_NUM_THREADS='24',MKL_NUM_THREADS='24')
command=[sys.executable,'-m','torch.distributed.run','--standalone','--nproc_per_node=1',
         'train_zero3_compare.py','--steps','4','--output','single-run1',
         '--model',os.environ['MODEL_DIR'],'--data',os.environ['DATA_DIR']]
assert not (root/'single-run1').exists()
state={'stage':'running'}
(root/'single-job.json').write_text(json.dumps(state))
with (root/'single-run1.log').open('x') as log:
 proc=subprocess.Popen(command,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
 state['pid']=proc.pid;(root/'single-job.json').write_text(json.dumps(state))
 try:code=proc.wait(timeout=1800)
 except subprocess.TimeoutExpired:
  proc.terminate();proc.wait(timeout=60);raise
state.update(stage='complete' if code==0 else 'failed',returncode=code)
(root/'single-job.json').write_text(json.dumps(state));assert code==0
