"""Prospective CPU/meta predictions, baseline and chunk sizes sealed together."""
import hashlib,json,os,subprocess,time
from pathlib import Path
root=Path(__file__).resolve().parent
cases=json.loads((root/'inputs.json').read_text())
env=os.environ.copy()
env.update(OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',TOKENIZERS_PARALLELISM='false')
jobs={};running=[]
for chunk in [0,256,512,1024,2048]:
    key='baseline' if chunk==0 else f'chunk{chunk}'
    out=root/f'{key}.prediction.private.json'
    assert not out.exists()
    command=['python',str(root/'predict_meta.py'),'--config',str(root/'config.json'),'--vendor-root',env['VEOMNI_SOURCE'],'--rank','1','--loss-chunk-tokens',str(chunk),'--sequences',','.join(map(str,cases['step1'])),'--output',str(out)]
    log=(root/f'{key}.private.log').open('w')
    child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,env=env,cwd=root,start_new_session=True)
    jobs[key]=dict(state='running',pid=child.pid,started=time.time(),chunk_tokens=chunk)
    running.append((key,child,log,out))
(root/'jobs.private.json').write_text(json.dumps(jobs))
for key,child,log,out in running:
    code=child.wait(timeout=2400);log.close()
    jobs[key].update(state='complete' if code==0 else 'failed',code=code,finished=time.time())
    if code==0:jobs[key]['prediction_sha256']=hashlib.sha256(out.read_bytes()).hexdigest()
    (root/'jobs.private.json').write_text(json.dumps(jobs))
assert all(x['state']=='complete' for x in jobs.values()),jobs
seal=dict(jobs=jobs,inputs_sha256=hashlib.sha256((root/'inputs.json').read_bytes()).hexdigest(),observations_read=False)
(root/'prediction-seal.private.json').write_text(json.dumps(seal,indent=2))
