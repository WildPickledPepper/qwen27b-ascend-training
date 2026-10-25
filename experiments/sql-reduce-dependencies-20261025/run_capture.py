"""Bounded dependency capture, no optimization or throughput baseline rerun."""
import json,os,signal,subprocess,sys,time
from pathlib import Path
r=Path(__file__).resolve().parent;os.chdir(r);assert not (r/'capture-job.json').exists()
assert all((r/name).is_file() for name in ['zero3-sql.json','event_dependencies.py','train_capture.py','cpu_update.py']), 'Capture source/configuration incomplete'
state={'stage':'starting','runs':[]}
def save():(r/'capture-job.json').write_text(json.dumps(state,indent=2))
def run(name,args,timeout):
 info=subprocess.check_output(['npu-smi','info'],text=True)
 assert all(f'No running processes found in NPU {i}' in info for i in [2,3])
 state['stage']=name
 with (r/(name+'.private.log')).open('x') as f:
  p=subprocess.Popen(args,cwd=r,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
  row={'name':name,'pid':p.pid};state['runs'].append(row);save()
  try:code=p.wait(timeout=timeout)
  except subprocess.TimeoutExpired:
   os.killpg(p.pid,signal.SIGTERM)
   try:p.wait(timeout=30)
   except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
   code=-15
 row['returncode']=code;save();assert code==0,(name,code)
base=[sys.executable,'-m','torch.distributed.run','--standalone','--nproc_per_node=2','train_capture.py','--config','zero3-sql.json','--numa-spread','--cpu-threads','24','--cpu-mode','pipeline','--cpu-gradient-accum']
try:
 run('tiny-smoke',base+['--tiny','--steps','1','--global-batch','16','--micro-batch','1','--output','tiny-smoke','--gradient-cache-gib','0','--profile'],300)
 run('verify-smoke',[sys.executable,'verify_smoke.py'],90)
 run('capture-full',base+['--model',os.environ['MODEL_DIR'],'--data',os.environ['DATA_DIR'],'--output','capture-full','--global-batch','256','--micro-batch','16','--steps','4','--checkpoint','on','--fuse-gdn','--pad-mode','dynamic64','--gradient-cache-gib','0','--profile'],2400)
 state['stage']='complete'
except Exception as error:state['stage']='failed';state['error']=type(error).__name__;raise
finally:save()
