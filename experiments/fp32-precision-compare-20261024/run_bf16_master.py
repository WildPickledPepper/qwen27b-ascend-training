"""Launch and supervise the explicitly authorized 20-step BF16/FP32 run."""
import argparse, hashlib, json, os, subprocess, sys, time
from pathlib import Path
p = argparse.ArgumentParser()
for key in ['model', 'archives', 'data']: p.add_argument('--'+key, required=True)
p.add_argument('--foreground', action='store_true')
p.add_argument('--name', default='bf16-master-run1')
p.add_argument('--entry', choices=['train_bf16_master.py','train_bf16_native.py','train_bf16_single.py','train_bf16_dual.py'],default='train_bf16_master.py')
a = p.parse_args(); root = Path(__file__).resolve().parent; os.chdir(root)
name = a.name; assert name.replace('-','').isalnum()
if not a.foreground:
    assert not (root/name).exists(), 'Existing output must not be overwritten'
    with (root/(name+'-supervisor.log')).open('x') as log:
        proc = subprocess.Popen([sys.executable, '-u', __file__, *sys.argv[1:], '--foreground'],
                                stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                start_new_session=True)
    print(json.dumps({'supervisor_pid': proc.pid}), flush=True)
    sys.exit(0)
state = {'started_unix':time.time(),'physical_devices':[2] if a.entry=='train_bf16_single.py' else [2,3],'steps':20,'stage':'checking'}
def save():
    (root/(name+'-job.json')).write_text(json.dumps(state,indent=2))
try:
    text = subprocess.check_output(['npu-smi','info'],text=True)
    assert all('No running processes found in NPU '+str(d) in text for d in [2,3]), 'Authorized cards occupied'
    probe = json.loads((root/'bf16-master-probe.json').read_text()); assert probe['complete']
    if a.entry in ['train_bf16_native.py','train_bf16_single.py','train_bf16_dual.py']:
        native_probe=json.loads((root/'bf16-ffn-native-persistent-probe.json').read_text())
        assert all(r['input_gradient']['relative_l2']<.02 and all(g['relative_l2']<.02 for g in r['weight_gradients'].values()) for r in native_probe['rows'])
    names = ['bench_bf16_master.py',a.entry,'bench_cpu_pipeline.py','bench_cpu_master.py','probe_bf16_master.py']
    if a.entry in ['train_bf16_native.py','train_bf16_single.py','train_bf16_dual.py']: names.append('bench_bf16_native.py')
    if a.entry=='train_bf16_dual.py':
        assert os.environ.get('TASK_QUEUE_ENABLE')=='0'
        assert json.loads((root/'bf16-checkpoint-noqueue-probe.json').read_text())['completed']
    state['source_sha256'] = {n:hashlib.sha256((root/n).read_bytes()).hexdigest() for n in names}
    with (root/(name+'.log')).open('x') as log:
        command = [sys.executable,'-u',a.entry,'--model',a.model,'--archives',a.archives,
                   '--data',a.data,'--output',name,'--mode','bf16','--steps','20','--eval-count','8','--window','16777216']
        proc = subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
        state.update(pid=proc.pid,stage='running'); save()
        try: code = proc.wait(timeout=2400)
        except subprocess.TimeoutExpired:
            proc.terminate(); proc.wait(timeout=60); raise
    state['returncode'] = code
    assert code == 0, 'Training failed; retain log and diagnostic'
    result = json.loads((root/name/'status.json').read_text())
    assert result['completed'] and result['full_parameter_steps'] == 20
    state.update(stage='complete',finished_unix=time.time()); save()
except Exception:
    import traceback
    state.update(stage='failed',error=traceback.format_exc(),finished_unix=time.time()); save(); raise
