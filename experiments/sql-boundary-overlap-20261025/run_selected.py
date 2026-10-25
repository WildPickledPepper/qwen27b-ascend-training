"""Replay only the selected setup. No search, baseline rerun or automatic service."""
import argparse,hashlib,json,os,subprocess,sys
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--data',required=True);p.add_argument('--output',required=True);p.add_argument('--steps',type=int,default=3);p.add_argument('--profile',action='store_true');p.add_argument('--dry-run',action='store_true');a=p.parse_args()
r=Path(__file__).resolve().parent;selected=json.loads((r/'selected-profile.json').read_text())
assert a.steps>=1
for name,digest in selected['files_sha256'].items():
 assert hashlib.sha256((r/name).read_bytes().replace(b'\r\n',b'\n')).hexdigest()==digest,('Tested source/config differs',name)
cmd=[sys.executable,'-m','torch.distributed.run','--standalone','--nproc_per_node=2',str(r/'train_boundary.py'),
 '--config',str(r/selected['config']),'--model',a.model,'--data',a.data,'--output',a.output,'--steps',str(a.steps),
 '--global-batch','256','--micro-batch','16','--checkpoint','on','--numa-spread','--cpu-threads','24',
 '--cpu-mode','pipeline','--fuse-gdn','--pad-mode','dynamic64','--gradient-cache-gib','0','--cpu-gradient-accum']
if a.profile:cmd.append('--profile')
if a.dry_run:
 print(json.dumps({'source_and_config_hashes_verified':True,'arguments':cmd,'starts_training':False}));raise SystemExit(0)
assert os.environ.get('ASCEND_RT_VISIBLE_DEVICES')=='2,3','Activate the validated server environment for physical cards2,3 first'
for name in ['CPU_ADAM_EXTENSION','VEOMNI_SOURCE']:
 assert os.environ.get(name) and Path(os.environ[name]).exists(),('Required tested runtime missing',name)
info=subprocess.check_output(['npu-smi','info'],text=True)
assert all(f'No running processes found in NPU {i}' in info for i in [2,3]),'Selected cards are occupied'
assert not Path(a.output).exists(),'Use a fresh output directory'
env=dict(os.environ);env.update(TASK_QUEUE_ENABLE='0',OMP_NUM_THREADS='24',OPENBLAS_NUM_THREADS='24',MKL_NUM_THREADS='24',TOKENIZERS_PARALLELISM='false')
raise SystemExit(subprocess.call(cmd,cwd=r,env=env))
