"""Generate and seal predictions before a separate process opens observations."""
import argparse,hashlib,json,os,subprocess,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--cases',default='step1,step2,step3');a=p.parse_args()
root=Path(__file__).resolve().parent;cases=json.loads((root/'inputs.json').read_text());env=os.environ.copy()
env.update(OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',TOKENIZERS_PARALLELISM='false')
status={};jobs=[]
for rank in [0,1]:
 for case in a.cases.split(','):
  assert case in cases
  out=root/f'{case}-rank{rank}.prediction.private.json';assert not out.exists()
  cmd=['python',str(root/'predict_meta.py'),'--config',str(root/'config.json'),'--vendor-root',env['VEOMNI_SOURCE'],'--rank',str(rank),'--sequences',','.join(map(str,cases[case])),'--output',str(out)]
  jobs.append((case,rank,cmd,out))
# Two ranks in parallel per case; no cross-process timing model is claimed.
for case in a.cases.split(','):
 running=[]
 for job_case,rank,cmd,out in jobs:
  if job_case!=case:continue
  log=(root/f'{case}-rank{rank}.private.log').open('w')
  child=subprocess.Popen(cmd,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  status[f'{case}-{rank}']=dict(state='running',pid=child.pid,started=time.time())
  running.append((rank,child,log,out))
 (root/'jobs.private.json').write_text(json.dumps(status))
 for rank,child,log,out in running:
  try:code=child.wait(timeout=1800)
  except subprocess.TimeoutExpired:
   import signal
   os.killpg(child.pid,signal.SIGTERM);code=124
  log.close();entry=status[f'{case}-{rank}'];entry.update(state='complete' if code==0 else 'failed',code=code,finished=time.time())
  if code==0:entry['prediction_sha256']=hashlib.sha256(out.read_bytes()).hexdigest()
  (root/'jobs.private.json').write_text(json.dumps(status))
 if any(v['state']=='failed' for v in status.values()):raise SystemExit(1)
(root/'prediction-seal.json').write_text(json.dumps(dict(inputs_sha256=hashlib.sha256((root/'inputs.json').read_bytes()).hexdigest(),jobs=status,comparisons_not_yet_run=True),indent=2))
