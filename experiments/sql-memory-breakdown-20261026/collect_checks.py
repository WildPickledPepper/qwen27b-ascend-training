import argparse,hashlib,json,subprocess
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--original',required=True);p.add_argument('--root',required=True);p.add_argument('--output',required=True);a=p.parse_args()
original=Path(a.original);root=Path(a.root);old=original/'cache4-full.private';cfg=json.loads((original/'selected-profile.json').read_text());checks={}
checks['source_hashes']={f:hashlib.sha256((original/f).read_bytes().replace(b'\r\n',b'\n')).hexdigest()==h for f,h in cfg['source_lf_sha256'].items()};assert all(checks['source_hashes'].values())
checks['ranks']=[]
for rank in [0,1]:
 def rows(folder):return [json.loads(x) for x in (folder/f'rank{rank}.jsonl').read_text().splitlines()]
 before=rows(old);now=rows(root/'full.private')
 def event(rr,name):return next(x for x in rr if x['event']==name)
 ob=event(before,'step');nb=event(now,'step');od=event(before,'data');nd=event(now,'data')
 c=dict(rank=rank,first_256_hashes_and_lengths_same=nd['training_hashes']==od['training_hashes'][:256] and nd['training_lengths']==od['training_lengths'][:256],model_manifest_same=event(before,'model_ready')['manifest']==event(now,'model_ready')['manifest'],resident_plan_same=all(event(before,'resident_plan')[k]==event(now,'resident_plan')[k] for k in ['allocated_bytes','parameter_indices']),first_step_norm_exact=ob['grad_norm']==nb['grad_norm'],initial_validation_exact=event(before,'validation')['metrics']==event(now,'validation')['metrics'],input_and_padded_tokens_same=all(ob[k]==nb[k] for k in ['input_tokens','padded_tokens']),old_first_peak=ob['allocated_peak'],new_peak=nb['allocated_peak'],gradient_d2h_bytes_same=ob['gradient_d2h_bytes']==nb['gradient_d2h_bytes'])
 assert all(v for k,v in c.items() if k.endswith(('same','exact'))),c
 checks['ranks'].append(c)
smi=subprocess.check_output(['npu-smi','info'],text=True);checks['resource_release']={str(i):f'No running processes found in NPU {i}' in smi for i in [2,3]};assert all(checks['resource_release'].values())
Path(a.output).write_text(json.dumps(checks,indent=2));print(json.dumps(checks))
