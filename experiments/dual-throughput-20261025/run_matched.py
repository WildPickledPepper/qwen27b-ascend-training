"""After throughput tuning, retain the original batch-2 comparison protocol."""
import hashlib,json,os,statistics,subprocess,sys,time
from pathlib import Path
r=Path(__file__).resolve().parent;os.chdir(r)
deadline=time.monotonic()+10800
while True:
 if (r/'advanced-job.json').exists():
  previous=json.loads((r/'advanced-job.json').read_text())
  if previous['stage']=='complete':break
  assert previous['stage']!='failed','Inspect advanced failure before continuing'
 assert time.monotonic()<deadline;time.sleep(10)
best=previous['best'];name='matched-g2-optimized'
assert not (r/name).exists()
state={'stage':'running','best_throughput_configuration':best,'runs':[]}
def save():(r/'matched-job.json').write_text(json.dumps(state,indent=2))
def extra(name,threads,profile=False):
 assert not (r/name).exists()
 for _ in range(48):
  info=subprocess.check_output(['npu-smi','info'],text=True)
  if all('No running processes found in NPU '+str(i) in info for i in [2,3]):break
  time.sleep(5)
 else:raise RuntimeError('Authorized cards occupied')
 env=dict(os.environ,OMP_NUM_THREADS=str(threads),OPENBLAS_NUM_THREADS=str(threads),MKL_NUM_THREADS=str(threads))
 cmd=[sys.executable,'-m','torch.distributed.run','--standalone','--nproc_per_node=2',
  'train_advanced.py','--steps','3','--output',name,'--global-batch','32','--micro-batch',str(best['micro']),
  '--checkpoint',best['checkpoint'],'--config',best['config'],'--cpu-threads',str(threads),
  '--model',os.environ['MODEL_DIR'],'--data',os.environ['DATA_DIR']]
 if best['spread']:cmd.append('--numa-spread')
 if profile:cmd.append('--profile')
 with (r/(name+'.log')).open('x') as log:
  proc=subprocess.Popen(cmd,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
  row={'name':name,'pid':proc.pid,'cpu_threads':threads,'profile':profile}
  state['runs'].append(row);save();code=proc.wait();row['returncode']=code;save()
 assert code==0,'Inspect CPU-thread/profiling failure'
 results=[json.loads((r/name/f'rank{i}-result.json').read_text()) for i in [0,1]]
 assert all(x['completed'] and x['fp32_master'] and x['fp32_m_v'] for x in results)
 row['seconds']=statistics.mean(max(x['steps'][i]['seconds'] for x in results) for i in [1,2])
 row['validation_after']=results[0]['validation_after'];save()
 return row
threads=12
# With NUMA spreading, 24 cores per rank occupy disjoint CPU ranges.
if best['spread']:
 candidate=extra('cpu24-throughput',24)
 if candidate['seconds']<best['seconds'] and abs(candidate['validation_after']['loss']-best['loss'])<.01 and abs(candidate['validation_after']['correct']-best['correct'])<=4:
  threads=24;best=dict(best,name=candidate['name'],seconds=candidate['seconds'])
  state['best_throughput_configuration']=best;save()
  extra('profile-cpu24',24,True)
for _ in range(48):
 info=subprocess.check_output(['npu-smi','info'],text=True)
 if all('No running processes found in NPU '+str(i) in info for i in [2,3]):break
 time.sleep(5)
else:raise RuntimeError('Authorized cards occupied')
env=dict(os.environ,OMP_NUM_THREADS=str(threads),OPENBLAS_NUM_THREADS=str(threads),MKL_NUM_THREADS=str(threads))
cmd=[sys.executable,'-m','torch.distributed.run','--standalone','--nproc_per_node=2',
 'train_advanced.py','--steps','4','--output',name,'--global-batch','2','--micro-batch','1',
 '--checkpoint','off','--config',best['config'],'--cpu-threads',str(threads),
 '--model',os.environ['MODEL_DIR'],'--data',os.environ['DATA_DIR']]
if best['spread']:cmd.append('--numa-spread')
state.update(best_throughput_configuration=best,matched_checkpoint='off',cpu_threads=threads,
 source_sha256={n:hashlib.sha256((r/n).read_bytes()).hexdigest() for n in ['train_advanced.py',best['config']]})
with (r/(name+'.log')).open('x') as log:
 proc=subprocess.Popen(cmd,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
 state['pid']=proc.pid;(r/'matched-job.json').write_text(json.dumps(state,indent=2))
 code=proc.wait()
state.update(stage='complete' if code==0 else 'failed',returncode=code)
(r/'matched-job.json').write_text(json.dumps(state,indent=2))
