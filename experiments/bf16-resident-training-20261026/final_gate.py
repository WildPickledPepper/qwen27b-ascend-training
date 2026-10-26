"""Recheck exact states with the final graph-reference fix; no baseline rerun."""
import json,os,subprocess
from pathlib import Path
root=Path(__file__).resolve().parent
state=json.loads((root/'job.private.json').read_text());assert state['phase']=='full' and state['state']=='complete'
smi=subprocess.check_output(['npu-smi','info'],text=True)
assert all(f'No running processes found in NPU {d}' in smi for d in [2,3])
env=os.environ.copy();env.update(ASCEND_RT_VISIBLE_DEVICES='2,3',TASK_QUEUE_ENABLE='0',OMP_NUM_THREADS='24',
    MKL_NUM_THREADS='24',OPENBLAS_NUM_THREADS='24',TOKENIZERS_PARALLELISM='false')
output=root/'tiny-final.private';assert not output.exists()
command=['python','-m','torch.distributed.run','--nproc_per_node=2','--master_port=29833',str(root/'train.py'),
    '--model',env['MODEL_DIR'],'--data',env['DATA_DIR'],'--output',str(output),'--codec-library',str(root/'libonline_codec.so'),
    '--archive-root',env['BF16_ARCHIVE_ROOT'],'--resident-gib',str(2**-13),'--norm-device','npu','--steps','3','--tiny']
with (root/'tiny-final.private.log').open('w') as f:subprocess.run(command,check=True,env=env,stdout=f,stderr=subprocess.STDOUT)
subprocess.run(['python',str(root/'verify_smoke.py'),'--root',str(output),'--output',str(root/'final-smoke.json')],check=True,env=env,stdout=subprocess.DEVNULL)
subprocess.run(['python',str(root/'compare_states.py'),'--reference',env['PIPELINE_TINY_REFERENCE'],'--candidate',str(output),
    '--output',str(root/'final-state-comparison.json')],check=True,env=env,stdout=subprocess.DEVNULL)
print('Final production-hook three-update exact-state gate passed')
