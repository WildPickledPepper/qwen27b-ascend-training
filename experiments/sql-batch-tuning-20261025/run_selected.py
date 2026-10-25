"""Run the measured default; full original checkpoint and data are supplied via environment."""
import argparse,json,os,subprocess,sys
from pathlib import Path
r=Path(__file__).resolve().parent
p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--steps',type=int);a=p.parse_args()
profile=json.loads((r/'selected-profile.json').read_text())
batch=profile['global_batch'];micro=profile['micro_batch_per_rank'];steps=a.steps if a.steps is not None else 768//batch
assert steps>0 and batch==micro*2*profile['gradient_accumulation_steps']
assert not Path(a.output).exists(),'Use a new output directory'
for key in ['MODEL_DIR','DATA_DIR','CPU_ADAM_EXTENSION','VEOMNI_SOURCE']:assert os.environ.get(key),f'Set {key}'
assert subprocess.check_output(['git','-C',os.environ['VEOMNI_SOURCE'],'rev-parse','HEAD'],text=True).strip()=='eba362dd4ca43b6cba53dc69c1d378cf3eda57b6'
rows=(Path(os.environ['DATA_DIR'])/'train.jsonl').read_text().splitlines()
assert len(rows)>=steps*batch,'Provide enough distinct training rows; do not silently cycle a small benchmark dataset'
training_hashes={json.loads(row)['hash'] for row in rows[:steps*batch]}
validation_hashes={json.loads(row)['hash'] for row in (Path(os.environ['DATA_DIR'])/'validation.jsonl').read_text().splitlines()}
assert len(training_hashes)==steps*batch and not training_hashes&validation_hashes
info=subprocess.check_output(['npu-smi','info'],text=True)
assert all('No running processes found in NPU '+str(i) in info for i in [2,3]),'Authorized cards occupied'
env=dict(os.environ,ASCEND_RT_VISIBLE_DEVICES='2,3',DS_ACCELERATOR='npu',DS_BUILD_OPS='0',TASK_QUEUE_ENABLE='0',
 OMP_NUM_THREADS='24',OPENBLAS_NUM_THREADS='24',MKL_NUM_THREADS='24',TOKENIZERS_PARALLELISM='false',HCCL_CONNECT_TIMEOUT='120',HCCL_EXEC_TIMEOUT='120')
cmd=[sys.executable,'-m','torch.distributed.run','--standalone','--nproc_per_node=2',str(r/'train_batch.py'),
 '--model',env['MODEL_DIR'],'--data',env['DATA_DIR'],'--output',str(Path(a.output).resolve()),'--config',str(r/profile['config']),
 '--micro-batch',str(micro),'--global-batch',str(batch),'--steps',str(steps),'--checkpoint',profile['checkpoint'],
 '--numa-spread','--cpu-threads','24','--cpu-mode',profile['cpu_mode'],'--fuse-gdn','--pad-mode',profile['pad_mode']]
raise SystemExit(subprocess.call(cmd,env=env,cwd=r))
