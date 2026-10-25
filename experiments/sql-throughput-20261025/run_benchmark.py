"""One bounded SQL benchmark on authorized idle cards 2/3, with retained failures."""
import hashlib,json,os,signal,subprocess,sys,time
from pathlib import Path
r=Path(__file__).resolve().parent;os.chdir(r)
state={'stage':'starting','runs':[],'started_unix':time.time()}
assert not (r/'job.json').exists()
def save():(r/'job.json').write_text(json.dumps(state,indent=2))
def idle():
 info=subprocess.check_output(['npu-smi','info'],text=True)
 assert all('No running processes found in NPU '+str(i) in info for i in [2,3]),'Authorized cards occupied'
def run(name,cmd,timeout):
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
 entry.update(returncode=code,finished_unix=time.time());save()
 assert code==0,('Benchmark stage failed; preserve logs and inspect',name,code)
try:
 run('prepare',[sys.executable,'prepare_data.py','--raw','raw','--model',os.environ['MODEL_DIR'],'--output','data','--samples','768'],900)
 idle();run('probe512',[sys.executable,'probe_gdn_512.py'],900)
 idle()
 state['source_sha256']={f:hashlib.sha256((r/f).read_bytes()).hexdigest() for f in ['train_sql.py','prepare_data.py','community_gdn.py','cpu_update.py','bounded_grad_copy.py','zero3-sql.json']};save()
 run('train-g256-m4',[sys.executable,'-m','torch.distributed.run','--standalone','--nproc_per_node=2',
  'train_sql.py','--model',os.environ['MODEL_DIR'],'--data','data','--output','train-g256-m4',
  '--config','zero3-sql.json','--global-batch','256','--micro-batch','4','--steps','1',
  '--checkpoint','on','--numa-spread','--cpu-threads','24','--cpu-mode','pipeline','--fuse-gdn','--pad-mode','dynamic64'],7200)
 state['stage']='complete'
except Exception as error:
 state['stage']='failed';state['failure_type']=type(error).__name__;raise
finally:
 state['finished_unix']=time.time();save()
