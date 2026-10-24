"""All-coordinate finite-difference gradient and input split audit."""
import json,hashlib,sys
from pathlib import Path
import numpy as np
import mlp_teacher as m
out=Path(__file__).resolve().parent/'results';out.mkdir(exist_ok=True)
parts,w,_,_=m.make(101);x,y=parts[0];x=x[:8].astype(np.float64);y=y[:8];w=w.astype(np.float64)
_,_,g=m.lossgrad(w,x,y);errs=[]
for i in range(m.P):
 old=w[i];eps=1e-5;w[i]=old+eps;plus=m.lossgrad(w,x,y,False)[0];w[i]=old-eps;minus=m.lossgrad(w,x,y,False)[0];w[i]=old
 errs.append(abs((plus-minus)/(2*eps)-g[i]))
assert max(errs)<1e-7
splits={}
for seed in [101,202,303]:
 parts,w,batches,hashes=m.make(seed)
 splits[str(seed)]={label:dict(n=len(y),class_counts=np.bincount(y,minlength=m.K).tolist(),sha256=hashes[label]) for label,(x,y) in zip(['train','validation','test'],parts)}
 # Every epoch visits each training row exactly once; no held-out index exists.
 for epoch in range(m.STEPS//(m.NTRAIN//m.BATCH)):
  a=batches[epoch*(m.NTRAIN//m.BATCH):(epoch+1)*(m.NTRAIN//m.BATCH)].ravel()
  assert np.array_equal(np.sort(a),np.arange(m.NTRAIN))
result=dict(all_parameter_coordinates_checked=m.P,finite_difference_max_absolute_error=max(errs),passed=True,padded_coordinates_gradient_zero=bool(np.all(g[m.P:]==0)),every_epoch_training_permutation_verified=True,splits=splits,script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
(out/'full_gradient_audit.json').write_text(json.dumps(result,indent=2));print('full gradient max error',max(errs))
