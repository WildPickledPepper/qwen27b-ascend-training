"""Check scalar transition against earlier independently validated FP32 vector path."""
from pathlib import Path
import sys,json,hashlib
import numpy as np
import exact_replay as e
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'stage3'))
import run_stage3 as s
s.LR=e.LR;r=np.random.default_rng(1701);checks=[]
for t in [1,2,3,6,32,128,4096]:
 w=r.normal(0,.1,128).astype(np.float32);m=r.normal(0,.01,128).astype(np.float32);v=r.uniform(0,.01,128).astype(np.float32);g=r.normal(0,.2,128).astype(np.float32)
 mm,vv=s.moments(m,v,g);_,ww=s.update(mm,vv,w,t);actual=np.array([e.transition(w[i],m[i],v[i],g[i],t) for i in range(128)],dtype=np.float32);expected=np.stack([ww,mm,vv],axis=1);assert actual.tobytes()==expected.tobytes();checks.append(dict(t=t,coordinates=128,u32_compared=384,exact=True))
out=dict(checks=checks,total_u32_compared=2688,selected_reference='stage3.run_stage3 moments/update, itself validated against original scalar reference; not every framework kernel',bias_correction='Python binary64 exponentiation followed by float32 cast, identical in both paths',scalar_sha256=hashlib.sha256(Path(e.__file__).read_bytes()).hexdigest(),vector_sha256=hashlib.sha256(Path(s.__file__).read_bytes()).hexdigest())
(Path(__file__).resolve().parent/'results'/'transition_audit.json').write_text(json.dumps(out,indent=2));print('2688 exact uint32 transition checks passed')
