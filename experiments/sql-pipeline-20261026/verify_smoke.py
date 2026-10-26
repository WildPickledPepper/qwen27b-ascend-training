"""CPU-only full-state comparison for the tiny split pipeline's first update."""
import argparse,hashlib,importlib.util,json,math,os
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--output',required=True);a=p.parse_args()
import torch
torch.set_num_threads(2)
spec=importlib.util.spec_from_file_location('cpu_adam',os.environ['CPU_ADAM_EXTENSION'])
ext=importlib.util.module_from_spec(spec);spec.loader.exec_module(ext)
from deepspeed.ops.adam.cpu_adam import CPUAdamBuilder,DeepSpeedCPUAdam
CPUAdamBuilder.load=lambda self,verbose=True:ext
root=Path(a.root)
ref=torch.load(root/'tiny-reference.private.pt',weights_only=True)
reference_norm=math.sqrt(sum(float(g.float().double().square().sum()) for g in ref.values()))
rows=[]
for rank in [0,1]:
    logs=[json.loads(x) for x in (root/f'rank{rank}.jsonl').read_text().splitlines()]
    manifest=next(x['manifest'] for x in logs if x['event']=='model_ready')
    checks=next(x['checks'] for x in logs if x['event']=='tiny_gradient_comparison')
    assert all(x['exact'] for x in checks),checks
    norm=next(x['grad_norm'] for x in logs if x['event']=='step' and x['step']==1)
    assert math.isclose(norm,reference_norm,rel_tol=1e-5),(norm,reference_norm)
    actual=torch.load(root/f'probe-rank{rank}-step1.private.pt',weights_only=False)
    masters=[];grads=[]
    for info in manifest:
        gen=torch.Generator().manual_seed(int(hashlib.sha256(info['name'].encode()).hexdigest()[:8],16))
        value=(torch.randn(info['shape'],generator=gen)*.02).bfloat16().float().view(-1)
        grad=ref[info['name']].view(-1).float()/max(1.,norm+1e-6)
        for start in range(0,value.numel(),65536):
            m=torch.nn.Parameter(value[start:start+65536].clone());m.grad=grad[start:start+65536].clone()
            masters.append(m);grads.append(m.grad)
    opt=DeepSpeedCPUAdam(masters,lr=0.,betas=(.9,.999),eps=1e-8,weight_decay=.01)
    opt.step()
    for i,(expected,got) in enumerate(zip(masters,actual['master'])):
        assert torch.equal(expected,got),(rank,i,'master')
        for key in ['exp_avg','exp_avg_sq']:
            assert torch.equal(opt.state[expected][key],actual['states']['state'][i][key]),(rank,i,key)
    rows.append(dict(rank=rank,parameter_count=len(manifest),gradient_all_exact=True,
                     master_m_v_all_exact=True,optimizer_chunks=len(masters),normal_updates=3,
                     global_norm=norm,independent_reference_norm=reference_norm))
result=dict(passed=True,rows=rows,scope='first tiny update full gradient/master/m/v; three pipeline updates completed; not whole-model bitwise proof')
Path(a.output).write_text(json.dumps(result,indent=2));print(json.dumps(result))
