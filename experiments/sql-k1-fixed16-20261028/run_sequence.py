"""Fixed16 CPU-kernel-only candidate against the completed 86.79s replay."""
import argparse,hashlib,json,os,signal,subprocess,sys,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--reference',type=Path,required=True);p.add_argument('--tiny-reference',type=Path,required=True);a=p.parse_args()
root=Path(__file__).resolve().parent;state=dict(state='preparing',runs=[])
def save():(root/'job.private.json').write_text(json.dumps(state,indent=2))
def idle():
    smi=subprocess.check_output(['npu-smi','info'],text=True)
    assert all(f'No running processes found in NPU {i}' in smi for i in [0,3]),smi
    return smi
manifest=json.loads((a.reference/'source-manifest.private.json').read_text());source_info={}
for name in ['train_timed.py','resident_gradients.py','chunked_loss_contiguous.py','community_gdn.py']:
    raw=(a.reference/name).read_bytes();sha=hashlib.sha256(raw).hexdigest();assert sha==manifest[name]['replay_sha256']
    if name!='train_timed.py':(root/name).write_bytes(raw)
    else:
        text=raw.decode();changes=[('        self.thread=threading.Thread(target=self.worker,daemon=True); self.thread.start()', '        from slot_kernel import SlotAdd\n        self.kernel=SlotAdd(threads=24)\n        self.thread=threading.Thread(target=self.worker,daemon=True); self.thread.start()'),('self.accum[idx][start:start+n].add_(self.slots[slot][:n])','self.kernel.add(self.accum[idx][start:start+n],self.slots[slot][:n])'),('worker_cpu_add_seconds=off.cpu_add,','worker_cpu_add_seconds=off.cpu_add,cpu_kernel_calls=off.kernel.calls,')]
        for old,new in changes:
            assert text.count(old)==1,(old,text.count(old));text=text.replace(old,new,1)
        reverse=text
        for old,new in reversed(changes):reverse=reverse.replace(new,old,1)
        assert reverse==raw.decode()
        (root/'train_k1.py').write_text(text)
    source_info[name]=sha
source_info['kernel_c_sha256']=hashlib.sha256((root/'cpu_fused_accum.c').read_bytes()).hexdigest()
(root/'source-manifest.private.json').write_text(json.dumps(source_info,indent=2))
from slot_kernel import validate
(root/'kernel-check.json').write_text(json.dumps(validate(),indent=2))
env=os.environ.copy();env.update(ASCEND_RT_VISIBLE_DEVICES='0,3',TASK_QUEUE_ENABLE='0',OMP_NUM_THREADS='24',MKL_NUM_THREADS='24',OPENBLAS_NUM_THREADS='24',TOKENIZERS_PARALLELISM='false',HCCL_EXEC_TIMEOUT='600')
def run(name,cmd,timeout):
    idle();save()
    with (root/(name+'.private.log')).open('x') as f:
        child=subprocess.Popen(cmd,cwd=root,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
        row=dict(name=name,pid=child.pid,started=time.time(),state='running');state['runs'].append(row);state.update(state='running',phase=name);save()
        try:code=child.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(child.pid,signal.SIGTERM)
            try:child.wait(timeout=20)
            except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
            code=124
        row.update(returncode=code,finished=time.time(),state='complete' if code==0 else 'failed');save();assert code==0,(name,code)
    idle()
def train(name,tiny):
    cmd=[sys.executable,'-m','torch.distributed.run','--nproc_per_node=2','--master_port=29860',str(root/'train_k1.py'),'--model',env['MODEL_DIR'],'--data',env['DATA_DIR'],'--output',str(root/(name+'.private')),'--norm-device','npu','--resident-gib-by-rank','0.000244140625,0.00048828125' if tiny else '4,8','--loss-chunk-tokens','64' if tiny else '512','--steps','3','--slots','16']
    if tiny:cmd.append('--tiny')
    return cmd
try:
    run('tiny',train('tiny',True),600)
    run('tiny-check',[sys.executable,str(root/'compare_exact.py'),'--reference',str(a.tiny_reference),'--candidate',str(root/'tiny.private'),'--output',str(root/'tiny-check.json')],120)
    run('full',train('full',False),2400)
    (root/'post-smi.private.txt').write_text(idle());state.update(state='complete',cleanup_verified=True);save()
except BaseException as exc:
    state.update(state='failed',error=repr(exc));save();raise
