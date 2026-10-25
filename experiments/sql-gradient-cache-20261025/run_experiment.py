import json,os,signal,subprocess,sys,time
from pathlib import Path
r=Path(__file__).resolve().parent;os.chdir(r);assert not (r/'job.json').exists()
state={'stage':'starting','runs':[],'started_unix':time.time()}
def save():(r/'job.json').write_text(json.dumps(state,indent=2))
def idle():
 info=subprocess.check_output(['npu-smi','info'],text=True)
 assert all(f'No running processes found in NPU {i}' in info for i in [2,3])
def run(name,cmd,timeout,needs_idle=True):
 if needs_idle:idle()
 state['stage']=name
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
base=[sys.executable,'-m','torch.distributed.run','--standalone','--nproc_per_node=2','train_cache.py','--config','zero3-sql.json','--numa-spread','--cpu-threads','24','--cpu-mode','pipeline']
try:
 for name,cache in [('tiny-native','0'),('tiny-cache','12')]:
  run(name,base+['--tiny','--steps','3','--global-batch','16','--micro-batch','1','--output',name,'--gradient-cache-gib',cache],300)
 run('compare-probes',[sys.executable,'compare_probes.py'],90)
 # First three updates reproduce the original 768 examples. Fourth update is
 # profiler-only, intentionally repeats the first 256 and is excluded from speed.
 run('cache12-full',base+['--model',os.environ['MODEL_DIR'],'--data',os.environ['DATA_DIR'],'--output','cache12-full','--global-batch','256','--micro-batch','16','--steps','4','--checkpoint','on','--fuse-gdn','--pad-mode','dynamic64','--gradient-cache-gib','12','--profile'],2400)
 state['stage']='complete'
except Exception as error:
 state['stage']='failed';state['failure_type']=type(error).__name__;raise
finally:
 state['finished_unix']=time.time();save()
