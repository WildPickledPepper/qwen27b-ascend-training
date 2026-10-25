"""Numerical probes first, then fixed-workload model run and final profiling."""
import json,os,signal,subprocess,sys,time
from pathlib import Path
r=Path(__file__).resolve().parent;os.chdir(r);assert not (r/'job.json').exists()
state={'stage':'starting','runs':[],'started_unix':time.time()}
def save():(r/'job.json').write_text(json.dumps(state,indent=2))
def idle():
 text=subprocess.check_output(['npu-smi','info'],text=True)
 assert all(f'No running processes found in NPU {i}' in text for i in [2,3])
def run(name,cmd,timeout,needs_idle=True):
 if needs_idle:idle()
 state['stage']=name
 with (r/(name+'.log')).open('x') as log:
  proc=subprocess.Popen(cmd,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  row={'name':name,'pid':proc.pid,'started_unix':time.time()};state['runs'].append(row);save()
  try:code=proc.wait(timeout=timeout)
  except subprocess.TimeoutExpired:
   os.killpg(proc.pid,signal.SIGTERM)
   try:proc.wait(timeout=40)
   except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
   code=-15
 row.update(returncode=code,finished_unix=time.time());save();assert code==0,(name,code)
base=[sys.executable,'-m','torch.distributed.run','--standalone','--nproc_per_node=2','train_ready.py','--config','zero3-sql.json','--numa-spread','--cpu-threads','24','--cpu-mode','pipeline','--cpu-gradient-accum']
try:
 for name,flag in [('tiny-native',[]),('tiny-cache',['--ready-reduce'])]:
  run(name,base+['--tiny','--steps','3','--global-batch','16','--micro-batch','1','--output',name,'--gradient-cache-gib','0']+flag,300)
 run('compare-probes',[sys.executable,'compare_probes.py'],90)
 run('cpu-full',base+['--ready-reduce','--model',os.environ['MODEL_DIR'],'--data',os.environ['DATA_DIR'],'--output','cpu-full','--global-batch','256','--micro-batch','16','--steps','4','--checkpoint','on','--fuse-gdn','--pad-mode','dynamic64','--gradient-cache-gib','0','--profile'],2400)
 state['stage']='complete'
except Exception as error:
 state['stage']='failed';state['failure_type']=type(error).__name__;raise
finally:state['finished_unix']=time.time();save()
