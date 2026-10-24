"""Collect genuine FP32 Adam states from a wider nonlinear student, CPU only."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1');os.environ.setdefault('OMP_NUM_THREADS','1')
from pathlib import Path
import sys,hashlib,json,time,math
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'stage4'))
import mlp_teacher as m
s=m.s
OUT=Path(__file__).resolve().parent/'snapshots';OUT.mkdir(exist_ok=True)
# 21,508 active parameters, 84+ partial blocks at B256; no padding is stored.
m.H=1024;m.SHAPES=[(m.DIN,m.H),(m.H,),(m.H,m.K),(m.K,)];m.SIZES=[int(np.prod(x)) for x in m.SHAPES];m.P=sum(m.SIZES);m.PAD=((m.P+127)//128)*128
SNAPSHOTS=[1,2,128,1024,2048,4095,4096]
rows=[];inputs={};started=time.perf_counter()
for seed in [101,202,303]:
 parts,w,batches,hashes=m.make(seed);inputs[str(seed)]=hashes;train,val,test=parts
 mm=vv=np.zeros(m.PAD,dtype=np.float32)
 for t,indices in enumerate(batches,1):
  loss,acc,g=m.lossgrad(w,train[0][indices],train[1][indices]);mm,vv=s.moments(mm,vv,g);s.LR=np.float32(m.lr_at(t,1e-3));d,w=s.update(mm,vv,w,t)
  assert np.all(np.isfinite(w)) and np.all(np.isfinite(mm)) and np.all(np.isfinite(vv))
  if t in SNAPSHOTS:
   raw=mm[:m.P].astype('<f4').tobytes()+vv[:m.P].astype('<f4').tobytes();name=f'seed{seed}_step{t:04d}.bin';(OUT/name).write_bytes(raw)
   vl,va=m.lossgrad(w,*val,False)
   rows.append(dict(seed=seed,step=t,active_parameters=m.P,raw_state_bytes=len(raw),filename=name,sha256=hashlib.sha256(raw).hexdigest(),validation_loss=vl,validation_accuracy=va,lr=float(s.LR),shapes=m.SHAPES))
   print(seed,t,'val',round(vl,6),'statebytes',len(raw),flush=True)
   (OUT/'index.json').write_text(json.dumps(rows,indent=2))
run=dict(seconds=time.perf_counter()-started,hidden=m.H,active_parameters=m.P,padded_working_coordinates=m.PAD,seeds=[101,202,303],steps=m.STEPS,batch=m.BATCH,snapshot_steps=SNAPSHOTS,protocol='FP32 AdamW same explicit operation order; peak LR=.001, warmup64/cosine to1%; independent train4096/val2048/test2048; test not inspected for collection',source_hashes={'collector':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'mlp':hashlib.sha256(Path(m.__file__).read_bytes()).hexdigest(),'adam':hashlib.sha256(Path(s.__file__).read_bytes()).hexdigest()},input_hashes=inputs,python=sys.version,numpy=np.__version__)
(OUT/'run.json').write_text(json.dumps(run,indent=2));print('completed',run['seconds'])
