"""Sequential controller: tiny correctness for k1/k2, then matched 3-step comparisons.

train_resident's own argparse only accepts:
  --model --data --output --steps --tiny --profile --slots --resident-gib
  --norm-device --cpu-accum
so microbatch count and accumulation steps are fixed by the entry itself (16 x 16).

Only physical NPU 0 and 3 are used; both must be idle before every run.
"""
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
CARDS = ['0', '3']
MODEL = os.environ.get('MODEL_DIR') or str(Path.home() / 'mindstudio/qwen35-27b/models/Qwen3.5-27B')
DATA = os.environ.get('DATA_DIR') or str(Path.home() / 'mindstudio/projects/qwen-sql-throughput-20261025/data')


def idle():
    info = subprocess.check_output(['npu-smi', 'info'], text=True)
    for card in CARDS:
        assert f'No running processes found in NPU {card}' in info, f'NPU {card} busy'


def run(name, args, timeout):
    idle()
    if (HERE / f'{name}.log').exists():
        raise SystemExit(f'{name}.log exists')
    log = (HERE / f'{name}.log').open('x')
    env = dict(os.environ, ASCEND_RT_VISIBLE_DEVICES=','.join(CARDS),
               HCCL_EXEC_TIMEOUT='600', TASK_QUEUE_ENABLE='0',
               OMP_NUM_THREADS='24', MKL_NUM_THREADS='24', OPENBLAS_NUM_THREADS='24',
               TOKENIZERS_PARALLELISM='false',
               CPU_ADAM_EXTENSION=os.environ.get('CPU_ADAM_EXTENSION') or str(Path.home() / 'mindstudio/projects/qwen27b-full-sft-20261023/cache/torch_extensions/cpu_adam/cpu_adam.so'),
               VEOMNI_SOURCE=os.environ.get('VEOMNI_SOURCE') or str(Path.home() / 'mindstudio/projects/qwen-training-fusion-20261025/VeOmni'))
    cmd = [sys.executable, '-m', 'torch.distributed.run', '--standalone',
           '--nproc_per_node=2', 'train_accum.py'] + args
    print('RUN', name, flush=True)
    t0 = time.monotonic()
    proc = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=log,
                            stderr=subprocess.STDOUT, start_new_session=True, env=env)
    try:
        code = proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGTERM)
        try:
            proc.wait(timeout=40)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()
        code = -15
    print(f'DONE {name} code={code} {time.monotonic()-t0:.0f}s', flush=True)
    return code


def train_args(output, mode, tiny=False, steps=3, resident=None):
    args = ['--model', MODEL, '--data', DATA, '--output', output,
            '--steps', str(steps), '--cpu-accum', mode]
    if tiny:
        args.append('--tiny')
    if resident:
        args += ['--resident-gib', str(resident)]
    return args


plan = sys.argv[1] if len(sys.argv) > 1 else 'tiny'
runs = []
if plan == 'tiny':
    for mode in ('torch', 'k1', 'k2'):
        runs.append((f'tiny-{mode}', train_args(f'tiny-{mode}', mode, tiny=True), 900))
elif plan == 'match':
    for mode in ('torch', 'k2', 'k1'):
        runs.append((f'full-{mode}', train_args(f'full-{mode}', mode, resident=4), 2400))

for name, args, timeout in runs:
    code = run(name, args, timeout)
    if code != 0:
        print(f'ABORT at {name} code={code}', flush=True)
        raise SystemExit(1)
print('sequence complete', flush=True)
