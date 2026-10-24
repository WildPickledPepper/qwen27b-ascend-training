"""Absolute diagnostics so near-zero reference updates do not distort relative-error claims."""
import csv,json,hashlib,time
from pathlib import Path
import numpy as np
import run_stage3 as s
OUT=Path(__file__).resolve().parent/'results'
cs=[s.config('qv_nf4_rtn_b128','qv','nf4','rtn',128),s.config('qv_nf4_sr_b64','qv','nf4','sr',64),s.config('mv_nf4_sr_b64','mv','nf4','sr',64)]
rows=[]
for case in ['random','sparse_spikes','scale_jump','near_cancel','constant','alternating']:
 gs=[np.array(g,dtype=np.float32) for g in s.old.gradients(case,256,256,20261024)]
 z=np.zeros(256,dtype=np.float32);rm=rv=z;rw=np.array([s.old.F(.1+i/256) for i in range(256)],dtype=np.float32);refs=[]
 for t,g in enumerate(gs,1):rm,rv=s.moments(rm,rv,g);rd,rw=s.update(rm,rv,rw,t);refs.append((rd.copy(),rw.copy()))
 for c in cs:
  for seed in [17,29,43]:
   w=np.array([s.old.F(.1+i/256) for i in range(256)],dtype=np.float32);blob=s.encode(z,z,0,seed,c);maxrel=maxabs=maxrms=0.;minref=np.inf;sum_err2=sum_ref2=0;worst=None
   for t,(g,(rd,rw)) in enumerate(zip(gs,refs),1):
    m,v,_,_=s.decode(blob,c);m,v=s.moments(m,v,g);d,w=s.update(m,v,w,t);blob=s.encode(m,v,t,seed,c)
    df=d.astype(np.float64);rf=rd.astype(np.float64);error=df-rf;refnorm=np.linalg.norm(rf);rel=np.linalg.norm(error)/max(refnorm,1e-30)
    maxabs=max(maxabs,np.max(abs(error)));maxrms=max(maxrms,np.sqrt(np.mean(error**2)));minref=min(minref,refnorm);sum_err2+=float(np.dot(error,error));sum_ref2+=float(np.dot(rf,rf))
    if rel>maxrel:maxrel=rel;worst=(t,refnorm,float(np.linalg.norm(error)),float(np.max(abs(error))))
   rows.append(dict(case=case,configuration=c['name'],seed=seed,max_update_rel_L2=maxrel,max_update_absolute_error=maxabs,max_update_RMS_error=maxrms,min_reference_update_L2=minref,whole_trajectory_update_rel_L2=np.sqrt(sum_err2/sum_ref2),worst_relative_step=worst[0],worst_relative_reference_norm=worst[1],worst_relative_absolute_error_norm=worst[2],worst_relative_max_coordinate_error=worst[3],max_per_coordinate_parameter_step_difference=float(s.LR)*maxabs))
s.csvwrite(OUT/'replay_diagnostics.csv',rows)
print('wrote',len(rows))
