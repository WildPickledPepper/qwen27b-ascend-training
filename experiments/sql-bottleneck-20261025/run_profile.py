import json,os,signal,subprocess,sys,time
from pathlib import Path
r=Path(__file__).resolve().parent;os.chdir(r);assert not (r/'job.json').exists()
state={'stage':'starting','runs':[],'started_unix':time.time()}
def save():(r/'job.json').write_text(json.dumps(state,indent=2))
def idle():
 info=subprocess.check_output(['npu-smi','info'],text=True)
 assert all('No running processes found in NPU '+str(i) in info for i in [2,3])
def run(name,cmd,timeout):
 idle();state['stage']=name
 with (r/(name+'.log')).open('x') as log:
  proc=subprocess.Popen(cmd,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  entry={'name':name,'pid':proc.pid,'started_unix':time.time()};state['runs'].append(entry);save()
  try:code=proc.wait(timeout=timeout)
  except subprocess.TimeoutExpired:
   os.killpg(proc.pid,signal.SIGTERM)
   try:proc.wait(timeout=40)
   except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
   code=-15
 entry.update(returncode=code,finished_unix=time.time());save();assert code==0,(name,code)
try:
 run('observer-probe',[sys.executable,'probe_observer.py'],180)
 run('profile-g256-m16',[sys.executable,'-m','torch.distributed.run','--standalone','--nproc_per_node=2','train_profile.py',
 '--model',os.environ['MODEL_DIR'],'--data',os.environ['DATA_DIR'],'--output','profile-g256-m16','--config','zero3-sql.json',
 '--global-batch','256','--micro-batch','16','--steps','3','--checkpoint','on','--numa-spread','--cpu-threads','24',
 '--cpu-mode','pipeline','--fuse-gdn','--pad-mode','dynamic64','--profile'],2400)
 state['stage']='complete'
except Exception as error:
 state['stage']='failed';state['failure_type']=type(error).__name__;raise
finally:
 state['finished_unix']=time.time();save()
