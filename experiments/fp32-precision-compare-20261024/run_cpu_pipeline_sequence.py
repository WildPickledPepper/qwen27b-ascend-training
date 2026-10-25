"""Serialize controlled experiments; never start on occupied physical cards."""
import argparse, hashlib, json, os, subprocess, sys, time
from pathlib import Path
p = argparse.ArgumentParser()
for name in ['root', 'model', 'archives', 'data']: p.add_argument('--'+name, required=True)
p.add_argument('--wait-pid', type=int, required=True)
a = p.parse_args(); root = Path(a.root).resolve(); os.chdir(root)
state = {'stage': 'waiting_for_synchronous_run', 'started_unix': time.time(), 'physical_devices': [2, 3]}

def save():
    (root/'cpu-pipeline-sequence.json').write_text(json.dumps(state, indent=2))
    print(json.dumps(state), flush=True)

def free_cards():
    text = subprocess.check_output(['npu-smi', 'info'], text=True)
    assert all('No running processes found in NPU '+str(d) in text for d in [2, 3]), 'Physical 2/3 not idle; no task started'

def run(name, command):
    free_cards(); state['stage'] = name; save()
    hashes = {f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in root.glob('*.py')
              if f.name in ['bench_cpu_master.py', 'bench_cpu_pipeline.py', 'train_cpu_master.py',
                            'train_cpu_pipeline.py', 'probe_cpu_pipeline.py', 'run_cpu_pipeline_sequence.py']}
    with (root/(name+'.log')).open('w') as log:
        proc = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
        (root/(name+'-job.json')).write_text(json.dumps({'pid': proc.pid, 'started_unix': time.time(),
                                                       'source_sha256': hashes, 'physical_devices': [2, 3]}, indent=2))
        try:
            code = proc.wait(timeout=2400)
        except subprocess.TimeoutExpired:
            proc.terminate(); proc.wait(timeout=60)
            raise RuntimeError('Own experiment exceeded deadline: '+name)
    assert code == 0, (name, code)

try:
    save(); deadline = time.monotonic()+2400
    while Path('/proc/'+str(a.wait_pid)).exists():
        if time.monotonic() > deadline: raise TimeoutError('Previous run did not exit')
        time.sleep(5)
    previous = json.loads((root/'cpu-master-run1/status.json').read_text())
    assert previous['completed'] and previous['full_parameter_steps'] == 4
    run('cpu-pipeline-probe', [sys.executable, '-u', 'probe_cpu_pipeline.py', '--output', 'cpu-pipeline-probe.json'])
    probe = json.loads((root/'cpu-pipeline-probe.json').read_text()); assert probe['complete']
    assert all(all(row['checks'].values()) for row in probe['rows'])
    run('cpu-pipeline-run1', [sys.executable, '-u', 'train_cpu_pipeline.py', '--model', a.model,
                            '--archives', a.archives, '--data', a.data, '--output', 'cpu-pipeline-run1',
                            '--mode', 'fp32', '--steps', '4', '--eval-count', '8', '--window', '16777216'])
    result = json.loads((root/'cpu-pipeline-run1/status.json').read_text())
    assert result['completed'] and result['full_parameter_steps'] == 4
    free_cards(); state['stage'] = 'complete'; state['finished_unix'] = time.time(); save()
except Exception:
    import traceback
    state['stage'] = 'failed'; state['error'] = traceback.format_exc(); save(); raise
