"""Wait for first capture to release cards, verify NPU norm tiny, then full run."""
import json,os,shutil,subprocess,time
from pathlib import Path
root=Path(__file__).resolve().parent
deadline=time.monotonic()+1200
while True:
    job=json.loads((root/'job.private.json').read_text())
    if job['state']=='complete':
        assert job['mode']=='full' and job['code']==0,job
        break
    assert time.monotonic()<deadline,'Previous capture did not finish'
    time.sleep(10)
shutil.copy2(root/'train_pipeline.py',root/'train_pipeline-cpu-v1.private.py')
for name in ['train_pipeline','run_remote']:
    shutil.copy2(root/(name+'.next.private.py'),root/(name+'.py'))
def run(name,cmd,timeout):
    (root/'followup.private.json').write_text(json.dumps(dict(state='running',phase=name)))
    with (root/(name+'.private.log')).open('w') as f:
        result=subprocess.run(cmd,cwd=root,stdout=f,stderr=subprocess.STDOUT,timeout=timeout)
    assert result.returncode==0,(name,result.returncode)
run('tiny-npu-controller',['python','run_remote.py','--norm-device','npu'],700)
run('verify-tiny-npu',['python','verify_smoke.py','--root','npu-norm-tiny.private','--output','smoke-npu.json'],120)
run('full-npu-controller',['python','run_remote.py','--norm-device','npu','--full'],3100)
(root/'followup.private.json').write_text(json.dumps(dict(state='complete',phase='full-npu',code=0)))
