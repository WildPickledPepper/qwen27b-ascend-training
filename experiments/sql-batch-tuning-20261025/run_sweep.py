"""Three fixed-data full-model runs; only idle authorized physical cards 2/3."""
import hashlib,json,os,signal,subprocess,sys,time
from pathlib import Path
r=Path(__file__).resolve().parent;os.chdir(r)
assert not (r/'job.json').exists()
state={'stage':'starting','runs':[],'started_unix':time.time(),'samples_per_case':768,'warmup_samples':256}
def save():(r/'job.json').write_text(json.dumps(state,indent=2))
def idle():
 info=subprocess.check_output(['npu-smi','info'],text=True)
 assert all('No running processes found in NPU '+str(i) in info for i in [2,3]),'Authorized cards occupied'
def run(batch,micro):
 idle();name=f'g{batch}-m{micro}';state['stage']=name
 assert not (r/name).exists()
 cmd=[sys.executable,'-m','torch.distributed.run','--standalone','--nproc_per_node=2','train_batch.py',
 '--model',os.environ['MODEL_DIR'],'--data',os.environ['DATA_DIR'],'--output',name,
 '--config','zero3-sql.json','--global-batch',str(batch),'--micro-batch',str(micro),'--steps',str(768//batch),
 '--checkpoint','on','--numa-spread','--cpu-threads','24','--cpu-mode','pipeline','--fuse-gdn','--pad-mode','dynamic64']
 with (r/(name+'.log')).open('x') as log:
  proc=subprocess.Popen(cmd,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  row={'name':name,'pid':proc.pid,'batch':batch,'micro_batch':micro,'started_unix':time.time(),
       'source_sha256':{f:hashlib.sha256((r/f).read_bytes()).hexdigest() for f in ['train_batch.py','community_gdn.py','cpu_update.py','bounded_grad_copy.py','zero3-sql.json']}}
  state['runs'].append(row);save()
  try:code=proc.wait(timeout=3600)
  except subprocess.TimeoutExpired:
   os.killpg(proc.pid,signal.SIGTERM)
   try:proc.wait(timeout=40)
   except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
   code=-15
 row.update(returncode=code,finished_unix=time.time());save()
 if code:
  row['failure']='oom' if 'out of memory' in (r/(name+'.log')).read_text().lower() else 'unexpected';save()
  assert row['failure']=='oom',('Inspect non-OOM failure',name)
  return False
 return True
try:
 micro=16
 if not run(256,micro):
  micro=8;assert run(256,micro)
 # Hold microbatch fixed while comparing global batch. Each case starts from
 # the same original checkpoint and visits exactly the same 768 examples.
 assert run(128,micro)
 assert run(64,micro)
 state['stage']='complete';state['tested_micro_batch']=micro
except Exception as error:
 state['stage']='failed';state['failure_type']=type(error).__name__;raise
finally:
 state['finished_unix']=time.time();save()
