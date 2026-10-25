"""One bounded targeted reproduction; no shared changes or repeated baselines."""
import json,os,signal,subprocess,sys,time
from pathlib import Path
r=Path(__file__).resolve().parent;os.chdir(r);assert not (r/'job.json').exists()
assert os.environ['ASCEND_RT_VISIBLE_DEVICES']=='2,3'
info=subprocess.check_output(['npu-smi','info'],text=True)
assert all(f'No running processes found in NPU {i}' in info for i in [2,3])
state={'stage':'running','started_unix':time.time()}
cmd=[sys.executable,'-m','torch.distributed.run','--standalone','--nproc_per_node=2','train_diagnosis.py','--config','zero3-sql.json','--numa-spread','--cpu-threads','24','--cpu-mode','pipeline','--cpu-gradient-accum','--fused-cpu-update','--model',os.environ['MODEL_DIR'],'--data',os.environ['DATA_DIR'],'--output','cpu-full','--global-batch','256','--micro-batch','16','--steps','3','--checkpoint','on','--fuse-gdn','--pad-mode','dynamic64','--gradient-cache-gib','0','--profile']
with (r/'run.private.log').open('x') as log:
 p=subprocess.Popen(cmd,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 state['pid']=p.pid;(r/'job.json').write_text(json.dumps(state))
 try:code=p.wait(timeout=1800)
 except subprocess.TimeoutExpired:
  os.killpg(p.pid,signal.SIGTERM)
  try:p.wait(timeout=30)
  except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
  code=-15
state.update(stage='complete' if code==0 else 'failed',returncode=code,finished_unix=time.time())
(r/'job.json').write_text(json.dumps(state));assert code==0
