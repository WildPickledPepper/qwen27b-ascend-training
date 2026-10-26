"""Replay the verified faster pipeline in a fresh output; original baseline kept."""
import argparse,hashlib,json,os,subprocess
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--data',required=True)
p.add_argument('--output',required=True);p.add_argument('--steps',type=int,choices=[3,4],default=3)
p.add_argument('--dry-run',action='store_true');a=p.parse_args()
root=Path(__file__).resolve().parent;config=json.loads((root/'selected-profile.json').read_text())
checks={name:hashlib.sha256((root/name).read_bytes().replace(b'\r\n',b'\n')).hexdigest()==expected for name,expected in config['source_lf_sha256'].items()}
assert all(checks.values()),checks
if a.dry_run:print(json.dumps(dict(valid=True,source_checks=checks,profile=config)));raise SystemExit(0)
assert not Path(a.output).exists(),'Choose a fresh output directory'
assert all(os.environ.get(key) for key in ['CPU_ADAM_EXTENSION','VEOMNI_SOURCE'])
smi=subprocess.check_output(['npu-smi','info'],text=True)
assert all(f'No running processes found in NPU {i}' in smi for i in [2,3]),smi
env=os.environ.copy();env.update(ASCEND_RT_VISIBLE_DEVICES='2,3',TASK_QUEUE_ENABLE='0',OMP_NUM_THREADS='24',
    MKL_NUM_THREADS='24',OPENBLAS_NUM_THREADS='24',TOKENIZERS_PARALLELISM='false',HCCL_EXEC_TIMEOUT='600')
cmd=['python','-m','torch.distributed.run','--nproc_per_node=2','--master_port=29829',str(root/'train_cache.py'),
     '--model',a.model,'--data',a.data,'--output',a.output,'--norm-device','npu','--resident-gib','2','--steps',str(a.steps)]
if a.steps==4:cmd.append('--profile')
raise SystemExit(subprocess.call(cmd,env=env,cwd=root))
