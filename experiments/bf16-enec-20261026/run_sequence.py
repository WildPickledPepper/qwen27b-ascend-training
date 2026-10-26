"""Probe, two disjoint full partitions, independent archive reload; no training."""
import json,os,signal,subprocess,time
from pathlib import Path
root=Path(__file__).resolve().parent;env=os.environ.copy();env.update(TORCH_DEVICE_BACKEND_AUTOLOAD='0',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4')
def phase(name,commands,timeout):
    children=[];logs=[];started=time.time()
    smi=subprocess.check_output(['npu-smi','info'],text=True)
    assert all(f'No running processes found in NPU {d}' in smi for d in [2,3]),'Card2/3 occupied'
    for i,command in enumerate(commands):
        log=(root/f'{name}-{i}.private.log').open('w');logs.append(log)
        children.append(subprocess.Popen(command,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True))
    state=dict(phase=name,state='running',child_pids=[c.pid for c in children],started=started)
    (root/'job.private.json').write_text(json.dumps(state))
    try:
        while any(c.poll() is None for c in children):
            assert time.time()-started<timeout,'Phase timeout'
            assert not any(c.poll() not in [None,0] for c in children),'Child failed; see phase log'
            time.sleep(2)
        assert all(c.returncode==0 for c in children),'Phase failed'
        state.update(state='complete',finished=time.time())
    except BaseException:
        for c in children:
            if c.poll() is None:os.killpg(c.pid,signal.SIGTERM)
        for c in children:
            try:c.wait(timeout=20)
            except subprocess.TimeoutExpired:os.killpg(c.pid,signal.SIGKILL);c.wait()
        state.update(state='failed',finished=time.time());raise
    finally:
        for log in logs:log.close()
        (root/'job.private.json').write_text(json.dumps(state))
def command(device,mode,name):
    return ['python',str(root/'run.py'),'--model',env['MODEL_DIR'],'--reference',env['PIPELINE_REFERENCE'],
        '--device',str(device),'--mode',mode,'--output',str(root/(name+'.private'))]
phase('probe',[command(2,'probe','probe-card2')],300)
probe=json.loads((root/'probe-card2.private/report.json').read_text())
assert probe['complete'] and any(r['storage_kind']=='enec' for r in probe['rows'] if not r['diagnostic'])
phase('full',[command(2,'all','full-card2'),command(3,'all','full-card3')],1500)
phase('verify',[[ 'python',str(root/'verify_archive.py'),'--archive',str(root/f'full-card{d}.private'),'--device',str(d)] for d in [2,3]],1200)
phase('summarize',[['python',str(root/'summarize.py'),'--root',str(root),'--output',str(root/'summary.json')]],120)
