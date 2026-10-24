"""Select existing CPU-trained states only; does not train or rerun Stage5/10."""
from pathlib import Path
import hashlib,json,shutil
import numpy as np
ROOT=Path(__file__).resolve().parent;BASE=ROOT.parent;OUT=ROOT/'inputs';PROV=ROOT/'provenance';OUT.mkdir(exist_ok=True);PROV.mkdir(exist_ok=True)
sha=lambda b:hashlib.sha256(b).hexdigest()
rows=[]
index=json.loads((BASE/'stage5/snapshots/index.json').read_text())
for src in index:
 if src['step'] not in [1,128,4096]:continue
 raw=(BASE/'stage5/snapshots'/src['filename']).read_bytes();assert sha(raw)==src['sha256']
 name='stage5_'+src['filename'];(OUT/name).write_bytes(raw)
 rows.append({'case':name[:-4],'family':'stage5_trained','step':src['step'],'seed':src['seed'],'filename':name,'sha256':sha(raw),'shapes':src['shapes'],'raw_state_bytes':len(raw),'origin':'Stage5 stored actual nonlinear MLP FP32 AdamW moments; no padding'})
source=BASE/'stage10/formal_results/s42_t16_m0_b128_c16.states';raw=source.read_bytes();states=np.frombuffer(raw,dtype='<f4').reshape(16,6288,3)
for step in [1,8,16]:
 mv=states[step-1,:,1:].T.copy().tobytes();name=f'stage10_seed42_step{step:04d}.bin';(OUT/name).write_bytes(mv)
 rows.append({'case':name[:-4],'family':'stage10_trained','step':step,'seed':42,'filename':name,'sha256':sha(mv),'shapes':[[32,128],[128],[128,16],[16]],'raw_state_bytes':len(mv),'origin':'Stage10 actual coupled MLP reference mode0 saved states; extract m/v, exclude current weights','source_states_sha256':sha(raw),'source_step_offset_bytes':(step-1)*6288*12})
shutil.copy2(source,PROV/source.name)
for stage,path in [('stage5','snapshots/run.json'),('stage5','snapshots/index.json'),('stage5','collect_snapshots.py'),('stage5','lossless_codecs.py'),('stage5','results/compression.csv'),('stage5','results/run.json'),('stage10','factor_replay.c'),('stage10','formal_results/s42_t16_m0_b128_c16.json')]:
 dest=PROV/(stage+'_'+path.replace('/','_'));shutil.copy2(BASE/stage/path,dest)
(OUT/'index.json').write_text(json.dumps(rows,indent=2)+'\n')
print(json.dumps({'cases':len(rows),'raw_state_bytes':sum(x['raw_state_bytes'] for x in rows),'source_hashes_checked':True},indent=2))
