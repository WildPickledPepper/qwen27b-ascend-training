"""Aggregate first-step mixed vs FP32 probes, never export raw parameters."""
import argparse,collections, json, math, re
from pathlib import Path
root=Path(__file__).resolve().parent
p=argparse.ArgumentParser();p.add_argument('--candidate',default='bf16-master-run1')
p.add_argument('--output',default='bf16-master-gradient-diagnostic.json');options=p.parse_args()
def load(name):
    result={}
    for r in json.loads((root/name/'parameter-probes.json').read_text()):
        if r['step']!=1:continue
        for index,w,g in zip(r['indices'],r['weight_before'],r['gradient']):
            key=(r['name'],index);assert key not in result
            result[key]=(w,g)
    return result
old=load('cpu-pipeline-run1'); new=load(options.candidate); groups=collections.defaultdict(lambda:[0.,0.,0.,0,0.,0.])
assert old.keys()==new.keys()
for key,b in new.items():
    a=old[key]
    name=key[0]
    category='linear_attention' if 'linear_attn' in name else 'ffn' if '.mlp.' in name else 'attention' if 'self_attn' in name else 'other'
    layer=re.search(r'\.layers\.(\d+)\.',name)
    categories=['all',category]
    if layer:categories.append('layer_'+layer.group(1))
    if not layer or layer.group(1)=='63':categories.append('tensor:'+name)
    for category in categories:
        v=groups[category]
        w,x=a;z,y=b
        v[0]+=x*x;v[1]+=y*y;v[2]+=(x-y)**2;v[3]+=1;v[4]=max(v[4],abs(x-y));v[5]=max(v[5],abs(w-z))
result={k:{'coordinates':v[3],'reference_grad_l2':math.sqrt(v[0]),'bf16_grad_l2':math.sqrt(v[1]),
           'gradient_relative_l2':math.sqrt(v[2]/max(v[0],1e-300)),
           'gradient_max_abs_difference':v[4],'weight_before_max_abs_difference':v[5]} for k,v in groups.items()}
(root/options.output).write_text(json.dumps(result,indent=2))
print(json.dumps(result),flush=True)
