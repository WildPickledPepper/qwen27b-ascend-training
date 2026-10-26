"""Run only on the authorized host; environment/model/data paths supplied via env."""
import argparse,json,os,signal,subprocess,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--full',action='store_true')
p.add_argument('--norm-device',choices=['cpu','npu'],default='cpu');a=p.parse_args()
root=Path(__file__).resolve().parent
env=os.environ.copy();env.update(ASCEND_RT_VISIBLE_DEVICES='2,3',TASK_QUEUE_ENABLE='0',
    OMP_NUM_THREADS='24',MKL_NUM_THREADS='24',OPENBLAS_NUM_THREADS='24',TOKENIZERS_PARALLELISM='false',
    HCCL_EXEC_TIMEOUT='600',HCCL_CONNECT_TIMEOUT='120')
def idle():
    text=subprocess.check_output(['npu-smi','info'],text=True)
    for card in [2,3]: assert f'No running processes found in NPU {card}' in text,text
idle()
name='full' if a.full else 'tiny'
if a.norm_device=='npu':name='npu-norm-'+name
dest=root/(name+'.private');assert not dest.exists(),dest
cmd=['python','-m','torch.distributed.run','--nproc_per_node=2','--master_port=29826',
     str(root/'train_pipeline.py'),'--model',env['MODEL_DIR'],'--data',env['DATA_DIR'],
     '--output',str(dest),'--steps','4' if a.full else '3','--norm-device',a.norm_device]
cmd+=['--profile'] if a.full else ['--tiny']
(root/'job.private.json').write_text(json.dumps(dict(state='running',mode=name,pid=os.getpid(),started=time.time())))
with (root/(name+'.private.log')).open('w') as f:
    process=subprocess.Popen(cmd,env=env,cwd=root,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
    try:code=process.wait(timeout=3000 if a.full else 600)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid,signal.SIGTERM)
        try:process.wait(timeout=20)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid,signal.SIGKILL);process.wait()
        code=124
(root/'job.private.json').write_text(json.dumps(dict(state='complete',mode=name,code=code,finished=time.time())))
raise SystemExit(code)
