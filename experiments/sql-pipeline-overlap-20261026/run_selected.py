"""Replay the selected 4GiB pipeline; verify all source hashes before NPU work."""
import argparse,hashlib,json,os,subprocess
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--data',required=True)
p.add_argument('--output',required=True);p.add_argument('--steps',type=int,choices=[3,4],default=3)
p.add_argument('--dry-run',action='store_true');a=p.parse_args()
root=Path(__file__).resolve().parent;cfg=json.loads((root/'selected-profile.json').read_text())
checks={name:hashlib.sha256((root/name).read_bytes().replace(b'\r\n',b'\n')).hexdigest()==value for name,value in cfg['source_lf_sha256'].items()}
assert all(checks.values()),checks
if a.dry_run:print(json.dumps(dict(valid=True,source_checks=checks,profile=cfg)));raise SystemExit(0)
assert not Path(a.output).exists(),'Choose a fresh output directory'
assert all(os.environ.get(k) for k in ['CPU_ADAM_EXTENSION','VEOMNI_SOURCE'])
smi=subprocess.check_output(['npu-smi','info'],text=True)
assert all(f'No running processes found in NPU {i}' in smi for i in [2,3]),smi
env=os.environ.copy();env.update(ASCEND_RT_VISIBLE_DEVICES='2,3',TASK_QUEUE_ENABLE='0',OMP_NUM_THREADS='24',
    MKL_NUM_THREADS='24',OPENBLAS_NUM_THREADS='24',TOKENIZERS_PARALLELISM='false',HCCL_EXEC_TIMEOUT='600')
cmd=['python','-m','torch.distributed.run','--nproc_per_node=2','--master_port=29832',str(root/'train_resident.py'),
    '--model',a.model,'--data',a.data,'--output',a.output,'--steps',str(a.steps),'--norm-device','npu','--resident-gib','4']
if a.steps==4:cmd.append('--profile')
raise SystemExit(subprocess.call(cmd,cwd=root,env=env))
