"""Predeclared horizon, orientation, noise and learning-rate stress checks."""
from pathlib import Path
import argparse,csv,hashlib,json,math,time
import numpy as np
import run_stage3 as s
OUT=Path(__file__).resolve().parent/'stress_results'

def main():
 OUT.mkdir(exist_ok=True);start=time.perf_counter();rows=[];cp=[]
 configs=[s.config('fp32','fp32'),s.config('mv_nf4_sr_b64','mv','nf4','sr',64),s.config('qv_nf4_rtn_b128','qv','nf4','rtn',128),s.config('qv_u15_rtn_b128','qv','u15','rtn',128),s.config('qv_nf4_sr_b64','qv','nf4','sr',64)]
 # 1024-step learning-rate sweep; 8192-step default-LR horizon check.
 schedule=[(1024,lr) for lr in [5e-5,1e-4,2e-4,4e-4]]+[(8192,1e-4)]
 for seed in [101,202,303]:
  diag,target,initial=s.inputs(seed,256);rng=np.random.default_rng(seed+5000)
  h=rng.normal(size=256).astype(np.float32);h=h/np.linalg.norm(h)
  reflect=lambda x:x-np.float32(2)*h*np.dot(h,x)
  for task in ['axis','rotated','rotated_noisy']:
   transform=(lambda x:x) if task=='axis' else reflect
   grad=lambda w:transform(diag*transform(w-target))/np.float32(256)
   objective=lambda w:float(np.sum(diag.astype(np.float64)*transform(w-target).astype(np.float64)**2)/(2*256))
   # Same additive stochastic gradient perturbation for every paired optimizer/LR.
   noise_scale=float(np.sqrt(np.mean(grad(initial).astype(np.float64)**2)))*.05
   noise=np.random.default_rng(seed+9000).normal(0,noise_scale,(8192,256)).astype(np.float32) if task=='rotated_noisy' else None
   init_loss=objective(initial)
   for steps,lr in schedule:
    s.LR=np.float32(lr);ref_loss=None;ref_w=None
    for c in configs:
     w=initial.copy();z=np.zeros(256,dtype=np.float32);blob=s.encode(z,z,0,seed,c)
     for t in range(1,steps+1):
      m,v,_,_=s.decode(blob,c);g=grad(w)
      if noise is not None:g=g+noise[t-1]
      m,v=s.moments(m,v,g);d,w=s.update(m,v,w,t);blob=s.encode(m,v,t,seed,c)
      if t in [1024,2048,4096,8192]:cp.append(dict(seed=seed,task=task,steps=steps,lr=lr,configuration=c['name'],step=t,objective=objective(w),gradient_norm=float(np.linalg.norm(grad(w).astype(np.float64))),distance_to_known_optimum=float(np.linalg.norm((w-target).astype(np.float64)))))
     cur=objective(w)
     if c['form']=='fp32':ref_loss=cur;ref_w=w.copy()
     rows.append(dict(seed=seed,task=task,steps=steps,lr=lr,configuration=c['name'],initial_objective=init_loss,final_objective=cur,reference_final_objective=ref_loss,objective_ratio=cur/ref_loss,objective_fraction_initial=cur/init_loss,excess_objective_fraction_initial=(cur-ref_loss)/init_loss,gradient_norm=float(np.linalg.norm(grad(w).astype(np.float64))),distance_to_known_optimum=float(np.linalg.norm((w-target).astype(np.float64))),weight_RMS_vs_FP32=float(np.sqrt(np.mean((w.astype(np.float64)-ref_w)**2))),noise_std=noise_scale if noise is not None else 0,**s.memory(c)))
    s.csvwrite(OUT/'summary.csv',rows);s.csvwrite(OUT/'checkpoints.csv',cp)
    print(seed,task,steps,lr,flush=True)
 run=dict(seconds=time.perf_counter()-start,script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),imported_script_sha256=hashlib.sha256(Path(s.__file__).read_bytes()).hexdigest(),schedule=schedule,configs=configs,meaning='Known data objective optimum w=target, L=0, but decoupled weight decay and finite horizon mean Adam need not attain it. Reflected quadratic uses one Householder transform. FP32 transform orthogonality is approximate. Noise is fixed across paired configurations and LR. No neural network generalization claim.')
 (OUT/'run.json').write_text(json.dumps(run,indent=2));print('completed seconds',run['seconds'])
if __name__=='__main__':main()
