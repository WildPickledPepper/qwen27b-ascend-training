"""Keep the first conservative capacity step, then measure a larger safe microbatch."""
import hashlib,json,os,signal,subprocess,sys,time
from pathlib import Path
r=Path(__file__).resolve().parent;os.chdir(r)
assert not (r/'matched-job.json').exists()
state={'stage':'waiting_for_capacity_step','runs':[],'started_unix':time.time()}
def save():(r/'matched-job.json').write_text(json.dumps(state,indent=2))
save()
def events(rank):
 path=r/'train-g256-m4'/f'rank{rank}.jsonl'
 return [json.loads(s) for s in path.read_text().splitlines() if s.startswith('{')]
def idle():
 info=subprocess.check_output(['npu-smi','info'],text=True)
 return all('No running processes found in NPU '+str(i) in info for i in [2,3])
try:
 deadline=time.monotonic()+1200
 while True:
  records=[[e for e in events(rank) if e['event']=='step_complete'] for rank in [0,1]]
  if all(records):break
  assert time.monotonic()<deadline,'Conservative step did not finish'
  assert json.loads((r/'job.json').read_text())['stage']!='failed'
  time.sleep(2)
 capacity=[s[0] for s in records]
 assert all(s['peak_allocated']/2**30<52 and s['peak_reserved']/2**30<57 for s in capacity)
 state['capacity_step']=capacity;state['stage']='releasing_capacity_run';save()
 old=json.loads((r/'job.json').read_text());entry=old['runs'][-1]
 assert entry['name']=='train-g256-m4'
 if old['stage']!='complete':
  pid=entry['pid'];assert os.getpgid(pid)==pid
  os.killpg(pid,signal.SIGTERM)
 deadline=time.monotonic()+180
 while not idle():
  assert time.monotonic()<deadline,'Authorized cards not released'
  time.sleep(2)
 state['capacity_run_disposition']='One complete conservative capacity update; not a three-step performance result. The initial live run was intentionally stopped at this boundary.'
 # Micro8 doubles the conservative token batch, remaining within the token
 # volume of the previously successful micro16/length128 training workload.
 state['stage']='train-g256-m8'
 state['source_sha256']={f:hashlib.sha256((r/f).read_bytes()).hexdigest() for f in ['train_sql.py','prepare_data.py','community_gdn.py','cpu_update.py','bounded_grad_copy.py','zero3-sql.json']}
 cmd=[sys.executable,'-m','torch.distributed.run','--standalone','--nproc_per_node=2','train_sql.py',
 '--model',os.environ['MODEL_DIR'],'--data','data','--output','train-g256-m8','--config','zero3-sql.json',
 '--global-batch','256','--micro-batch','8','--steps','3','--checkpoint','on','--numa-spread',
 '--cpu-threads','24','--cpu-mode','pipeline','--fuse-gdn','--pad-mode','dynamic64']
 with (r/'train-g256-m8.log').open('x') as log:
  proc=subprocess.Popen(cmd,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  state['runs'].append({'name':'train-g256-m8','pid':proc.pid,'started_unix':time.time()});save()
  try:code=proc.wait(timeout=5400)
  except subprocess.TimeoutExpired:
   os.killpg(proc.pid,signal.SIGTERM)
   try:proc.wait(timeout=40)
   except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
   code=-15
 state['runs'][-1].update(returncode=code,finished_unix=time.time());assert code==0,code
 state['stage']='complete'
except Exception as error:
 state['stage']='failed';state['failure_type']=type(error).__name__;raise
finally:
 state['finished_unix']=time.time();save()
