"""Sequential tiny/full run with state verification; no simultaneous card jobs."""
import argparse,json,subprocess
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--norm-device',choices=['cpu','npu'],default='npu');a=p.parse_args()
root=Path(__file__).resolve().parent
def run(phase,args,timeout):
    (root/'sequence.private.json').write_text(json.dumps(dict(state='running',phase=phase)))
    with (root/(phase+'.controller.private.log')).open('w') as f:
        result=subprocess.run(args,cwd=root,stdout=f,stderr=subprocess.STDOUT,timeout=timeout)
    if result.returncode:
        (root/'sequence.private.json').write_text(json.dumps(dict(state='failed',phase=phase,code=result.returncode)))
        raise SystemExit(result.returncode)
run('tiny',['python','run_remote.py','--norm-device',a.norm_device],700)
name='npu-norm-tiny.private' if a.norm_device=='npu' else 'tiny.private'
run('verify-tiny',['python','verify_smoke.py','--root',name,'--output','smoke-fixed-rope.json'],120)
run('full',['python','run_remote.py','--norm-device',a.norm_device,'--full'],3100)
(root/'sequence.private.json').write_text(json.dumps(dict(state='complete',code=0)))
