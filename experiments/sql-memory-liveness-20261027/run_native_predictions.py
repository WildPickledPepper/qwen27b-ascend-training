import argparse,hashlib,json,os,subprocess,time
from pathlib import Path
root=Path(__file__).resolve().parent;cases=json.loads((root/'inputs.json').read_text());cases['cache2-step1']=cases['step1']
(root/'inputs.json').write_text(json.dumps(cases,indent=2))
env=os.environ.copy();env.update(OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',TOKENIZERS_PARALLELISM='false')
running=[];status={}
for case,sequences in cases.items():
 for rank in [0,1]:
  out=root/f'{case}-rank{rank}.prediction.private.json';assert not out.exists()
  cmd=['python',str(root/'predict_meta.py'),'--config',str(root/'config.json'),'--vendor-root',env['VEOMNI_SOURCE'],'--rank',str(rank),'--resident-gib','2' if case.startswith('cache2') else '4','--sequences',','.join(map(str,sequences)),'--output',str(out)]
  f=(root/f'{case}-rank{rank}.private.log').open('w');child=subprocess.Popen(cmd,env=env,cwd=root,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
  key=f'{case}-{rank}';status[key]=dict(state='running',pid=child.pid,started=time.time());running.append((key,child,f,out))
(root/'jobs.private.json').write_text(json.dumps(status))
for key,child,f,out in running:
 try:code=child.wait(timeout=1800)
 except subprocess.TimeoutExpired:
  import signal
  os.killpg(child.pid,signal.SIGTERM);code=124
 f.close();status[key].update(state='complete' if code==0 else 'failed',code=code,finished=time.time())
 if code==0:status[key]['prediction_sha256']=hashlib.sha256(out.read_bytes()).hexdigest()
 (root/'jobs.private.json').write_text(json.dumps(status))
assert all(v['state']=='complete' for v in status.values()),status
(root/'prediction-seal.json').write_text(json.dumps(dict(inputs_sha256=hashlib.sha256((root/'inputs.json').read_bytes()).hexdigest(),jobs=status,comparisons_not_yet_run=True,revision='pinned-NPU-linear-rule'),indent=2))
