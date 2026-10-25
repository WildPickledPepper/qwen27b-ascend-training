"""Evidence-guided followups after the frozen batch sweep; one job at a time."""
import hashlib,json,os,signal,statistics,subprocess,sys,time
from pathlib import Path
r=Path(__file__).resolve().parent;os.chdir(r)
deadline=time.monotonic()+7200
while True:
 state=json.loads((r/'sweep2-job.json').read_text())
 if state['stage']=='complete':break
 assert state['stage']!='failed','Investigate unexpected sweep failure first'
 assert time.monotonic()<deadline;time.sleep(10)
state={'stage':'running','runs':[]}
def save():(r/'advanced-job.json').write_text(json.dumps(state,indent=2))
def score(name):
 results=[json.loads((r/name/f'rank{i}-result.json').read_text()) for i in [0,1]]
 assert all(x['completed'] and x['fp32_master'] and x['fp32_m_v'] and len(x['steps'])==3 for x in results)
 seconds=[max(x['steps'][i]['seconds'] for x in results) for i in [1,2]]
 return {'name':name,'seconds':statistics.mean(seconds),'peak_gib':max(s['peak_allocated'] for x in results for s in x['steps'])/2**30,
         'loss':results[0]['validation_after']['loss'],'correct':results[0]['validation_after']['correct'],
         'micro':results[0]['micro_batch'],'checkpoint':results[0]['checkpoint'],
         'spread':results[0].get('numa_spread',False),'config':'zero3.json'}
options=[score(x['name']) for x in json.loads((r/'sweep2-job.json').read_text())['runs'] if x['returncode']==0]
best=min(options,key=lambda x:x['seconds']);base_loss=best['loss'];base_correct=best['correct']
state['batch_best']=dict(best);save()
def run(name,checkpoint,spread=False,config='zero3.json',profile=False):
 assert not (r/name).exists()
 for _ in range(48):
  info=subprocess.check_output(['npu-smi','info'],text=True)
  if all('No running processes found in NPU '+str(i) in info for i in [2,3]):break
  time.sleep(5)
 else:raise RuntimeError('Authorized cards occupied; do not start')
 env=dict(os.environ,OMP_NUM_THREADS='12',OPENBLAS_NUM_THREADS='12',MKL_NUM_THREADS='12')
 cmd=[sys.executable,'-m','torch.distributed.run','--standalone','--nproc_per_node=2',
      'train_advanced.py','--steps','3','--output',name,'--global-batch','32','--micro-batch',str(best['micro']),
      '--checkpoint',checkpoint,'--config',config,'--model',os.environ['MODEL_DIR'],'--data',os.environ['DATA_DIR']]
 if spread:cmd.append('--numa-spread')
 if profile:cmd.append('--profile')
 manifest={n:hashlib.sha256((r/n).read_bytes()).hexdigest() for n in ['train_advanced.py','bounded_grad_copy.py',config]}
 with (r/(name+'.log')).open('x') as log:
  proc=subprocess.Popen(cmd,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  row={'name':name,'pid':proc.pid,'checkpoint':checkpoint,'spread':spread,'config':config,'profile':profile,'source_sha256':manifest}
  state['runs'].append(row);save()
  try:code=proc.wait(timeout=2400)
  except subprocess.TimeoutExpired:
   os.killpg(proc.pid,signal.SIGTERM);proc.wait(timeout=60);code=-15
 row['returncode']=code
 if code:
  text=(r/(name+'.log')).read_text()[-30000:]
  if 'out of memory' in text.lower():row['failure']='oom'
  elif 'NUMA bind unavailable' in text or 'set_mempolicy' in text:row['failure']='numa_policy_unavailable'
  else:row['failure']='unexpected';state['stage']='failed';save();raise RuntimeError('Investigate unexpected failure')
  save();return None
 result=score(name);result.update(spread=spread,config=config)
 result['validation_close']=abs(result['loss']-base_loss)<.01 and abs(result['correct']-base_correct)<=4
 row['result']=result;save();return result
def consider(candidate):
 global best
 if candidate and candidate['validation_close'] and candidate['peak_gib']<=58 and candidate['seconds']<best['seconds']:
  best=candidate
 state['best']=dict(best);save()
off=run('checkpoint-off', 'off');consider(off)
if off is None or off['peak_gib']>58 or off['seconds']>=state['batch_best']['seconds']:
 consider(run('checkpoint-selective','selective'))
consider(run('numa-spread',best['checkpoint'],spread=True))
cfg=json.loads((r/'zero3.json').read_text())
cfg['zero_optimization'].update(sub_group_size=268435456,stage3_prefetch_bucket_size=134217728,
                              stage3_max_live_parameters=3000000000,stage3_max_reuse_distance=3000000000)
(r/'prefetch-large-groups.json').write_text(json.dumps(cfg,indent=2))
consider(run('prefetch-large-groups',best['checkpoint'],spread=best['spread'],config='prefetch-large-groups.json'))
state['best']=dict(best);save()
profile=run('profile-best',best['checkpoint'],spread=best['spread'],config=best['config'],profile=True)
assert profile and profile['validation_close'],'Profile run failed numerical checks'
state['stage']='complete';save()
