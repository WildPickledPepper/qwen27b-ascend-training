"""Sequential evidence-gated CPU offload and community training-fusion study."""
import hashlib,json,os,signal,statistics,subprocess,sys,time
from pathlib import Path
r=Path(__file__).resolve().parent;os.chdir(r)
assert not (r/'research-job.json').exists()
state={'stage':'running','runs':[],'community_revision':'eba362dd4ca43b6cba53dc69c1d378cf3eda57b6'}
def save():(r/'research-job.json').write_text(json.dumps(state,indent=2))
def idle():
 for _ in range(60):
  info=subprocess.check_output(['npu-smi','info'],text=True)
  if all('No running processes found in NPU '+str(i) in info for i in [2,3]):return
  time.sleep(5)
 raise RuntimeError('Authorized cards occupied; do not start')
def run(name,mode,batch=2,fused=False,checkpoint='off',profile=False):
 idle();assert not (r/name).exists()
 cmd=[sys.executable,'-m','torch.distributed.run','--standalone','--nproc_per_node=2',
  'train_fusion.py','--steps',str(4 if batch==2 else 3),'--output',name,
  '--global-batch',str(batch),'--micro-batch',str(batch//2),'--checkpoint',checkpoint,
  '--cpu-threads','24','--numa-spread','--cpu-mode',mode,'--config','prefetch-large-groups.json',
  '--model',os.environ['MODEL_DIR'],'--data',os.environ['DATA_DIR']]
 if fused:cmd.append('--fuse-gdn')
 if profile:cmd.append('--profile')
 manifest={p:hashlib.sha256((r/p).read_bytes()).hexdigest() for p in ['train_fusion.py','cpu_update.py','community_gdn.py','bounded_grad_copy.py','prefetch-large-groups.json']}
 with (r/(name+'.log')).open('x') as log:
  proc=subprocess.Popen(cmd,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  row={'name':name,'pid':proc.pid,'cpu_mode':mode,'batch':batch,'fused':fused,'checkpoint':checkpoint,'profile':profile,'source_sha256':manifest}
  state['runs'].append(row);save()
  try:code=proc.wait(timeout=2400)
  except subprocess.TimeoutExpired:
   os.killpg(proc.pid,signal.SIGTERM);proc.wait(timeout=60);code=-15
 row['returncode']=code
 if code:
  log=(r/(name+'.log')).read_text()
  if 'out of memory' in log.lower():row['failure']='oom';save();return None
  row['failure']='unexpected';save();raise RuntimeError('Inspect failed case before continuing')
 results=[json.loads((r/name/f'rank{i}-result.json').read_text()) for i in [0,1]]
 assert all(x['completed'] and x['fp32_master'] and x['fp32_m_v'] for x in results)
 seconds=[max(x['steps'][i]['seconds'] for x in results) for i in range(1,len(results[0]['steps'])) if not results[0]['steps'][i]['profiled']]
 row['result']={'mean_seconds':statistics.mean(seconds),'peak_gib':max(s['peak_allocated'] for x in results for s in x['steps'])/2**30,
  'validation_after':results[0]['validation_after']}
 save();return row['result']
try:
 for filename in ['identity-probe.json','cast-probe.json','pipeline-probe.json','gdn-probe.json']:
  probe=json.loads((r/filename).read_text());assert probe['completed'],filename
 skip=run('cpu-skip-g2','skip');assert skip and abs(skip['validation_after']['loss']-1.5044201612472534)<1e-5 and skip['validation_after']['correct']==263
 pip=run('cpu-pipeline-g2','pipeline');assert pip and abs(pip['validation_after']['loss']-1.5044201612472534)<1e-5 and pip['validation_after']['correct']==263
 mode='pipeline' if pip['mean_seconds']<skip['mean_seconds'] else 'skip';state['cpu_best']=mode;save()
 idle()
 with (r/'gdn-actual-probe.log').open('x') as log:
  proc=subprocess.Popen([sys.executable,'warm_gdn.py'],stdout=log,stderr=subprocess.STDOUT)
  state['warmup_pid']=proc.pid;save();code=proc.wait(timeout=1800)
 assert code==0 and json.loads((r/'gdn-actual-probe.json').read_text())['completed']
 fused=run('fusion-g2',mode,fused=True);assert fused
 assert abs(fused['validation_after']['loss']-1.5044201612472534)<.02 and abs(fused['validation_after']['correct']-263)<=8
 throughput=run('fusion-g32-off',mode,batch=32,fused=True)
 checkpoint='off'
 if throughput is None or throughput['peak_gib']>58:
  throughput=run('fusion-g32-selective',mode,batch=32,fused=True,checkpoint='selective');checkpoint='selective'
 assert throughput and abs(throughput['validation_after']['loss']-1.4465612173080444)<.02 and abs(throughput['validation_after']['correct']-271)<=8
 profile=run('profile-fusion-g32',mode,batch=32,fused=True,checkpoint=checkpoint,profile=True)
 assert profile
 state['stage']='complete';save()
except Exception as error:
 state.update(stage='failed',error=str(error));save();raise
