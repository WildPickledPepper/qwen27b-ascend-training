"""Own strict FP32 replay demonstrations, CPU/NumPy only.
No history/memoization in recursive State; no state backing/offload.
Reference, replay benchmark, and validation are separate CLI phases.
"""
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
from pathlib import Path
import sys,math,time,json,csv,hashlib,gc,tracemalloc,struct,argparse,statistics
import numpy as np
OUT=Path(__file__).resolve().parent/'results';F=np.float32
P=4;B1=F(.9);B2=F(.999);O1=F(.1);O2=F(.001);EPS=F(1e-8);LR=F(.01);WD=F(.01)

def init(i):return F(F(i+1)*F(.125)-F(.5))
def hij(i,j):return F(1.25 if i==j else .25)
def transition(w,m,v,g,t):
 m=F(F(B1*m)+F(O1*g));v=F(F(B2*v)+F(O2*F(g*g)))
 denom=F(np.sqrt(F(v/F(1-.999**t)))+EPS);d=F(F(m/F(1-.9**t))/denom)
 w=F(F(w*F(1-F(LR*WD)))-F(LR*d));return w,m,v

def bits(st):return [int(np.asarray(x).view(np.uint32)) for x in st]
def packet(st):return struct.pack('<III',*bits(st))
def objective(w):
 # Reporting only; not used to update or reconstruct gradients.
 return .5*sum(float(w[i])*(float(w[i])+.25*sum(map(float,w))) for i in range(P))

def reference(T,capture=False):
 w=np.array([init(i) for i in range(P)],dtype=np.float32);m=np.zeros(P,dtype=np.float32);v=np.zeros(P,dtype=np.float32);hist=[]
 for t in range(1,T+1):
  g=np.zeros(P,dtype=np.float32)
  for i in range(P):
   for j in range(P):g[i]=F(g[i]+F(hij(i,j)*w[j]))
  for i in range(P):w[i],m[i],v[i]=transition(w[i],m[i],v[i],g[i],t)
  if capture:hist.append(dict(step=t,raw_u32=[bits((w[i],m[i],v[i])) for i in range(P)],objective=objective(w)))
 return w,m,v,hist

class Stats:
 __slots__=('calls','depth','maxdepth','updates','profile','max_fields','max_float_bytes','max_object_bytes','max_frame_bytes','max_localdict_bytes')
 def __init__(self,profile=False):
  self.calls=self.depth=self.maxdepth=self.updates=self.max_fields=self.max_float_bytes=self.max_object_bytes=self.max_frame_bytes=self.max_localdict_bytes=0;self.profile=profile

def inspect_stack(stats):
 # Diagnostic only: materializing frames/locals changes allocation behavior.
 # Report separately from the uninstrumented timing and tracemalloc run.
 f=sys._getframe(1);seen=set();floats=objects=frames=dicts=0
 def add(x):
  nonlocal floats,objects
  if id(x) in seen:return
  seen.add(id(x));objects+=sys.getsizeof(x)
  if isinstance(x,np.float32):floats+=1
  elif isinstance(x,tuple):
   for y in x:add(y)
 while f:
  if f.f_code.co_name=='state':
   frames+=sys.getsizeof(f);loc=f.f_locals;dicts+=sys.getsizeof(loc)
   for k in ('t','i','prior','g','j','wj'):
    if k in loc:add(loc[k])
  f=f.f_back
 stats.max_fields=max(stats.max_fields,floats);stats.max_float_bytes=max(stats.max_float_bytes,4*floats);stats.max_object_bytes=max(stats.max_object_bytes,objects);stats.max_frame_bytes=max(stats.max_frame_bytes,frames);stats.max_localdict_bytes=max(stats.max_localdict_bytes,dicts)

def state(t,i,stats):
 stats.calls+=1;stats.depth+=1;stats.maxdepth=max(stats.maxdepth,stats.depth)
 try:
  if t==0:
   if stats.profile:inspect_stack(stats)
   return init(i),F(0),F(0)
  prior=state(t-1,i,stats);g=F(0)
  for j in range(P):
   wj=prior[0] if j==i else state(t-1,j,stats)[0]
   g=F(g+F(hij(i,j)*wj))
  stats.updates+=1
  return transition(prior[0],prior[1],prior[2],g,t)
 finally:stats.depth-=1

def dfs_digest(T,profile=False,capture=False):
 stats=Stats(profile);h=hashlib.sha256();outputs=[] if capture else None
 for i in range(P):
  st=state(T,i,stats);h.update(packet(st))
  if capture:outputs.append(bits(st))
  del st
 return h.hexdigest(),stats,outputs

def ref_digest(T):
 w,m,v,_=reference(T);h=hashlib.sha256()
 for i in range(P):h.update(packet((w[i],m[i],v[i])))
 return h.hexdigest()

def bench(fn):
 secs=[]
 for _ in range(5):
  tick=time.perf_counter();value=fn();secs.append(time.perf_counter()-tick)
 del value
 gc.collect();tracemalloc.start();value=fn();live,peak=tracemalloc.get_traced_memory();tracemalloc.stop()
 return dict(seconds_median=statistics.median(secs),seconds_min=min(secs),seconds_max=max(secs),tracemalloc_peak_bytes=peak,tracemalloc_live_return_bytes=live),value

def fake_replay(T):
 rows=[];fw=np.array([init(i) for i in range(P)],dtype=np.float32)
 for t in range(1,T+1):
  trueprev,_,_,_=reference(t-1);rw,rm,rv,_=reference(t);isolated=[];closed=[]
  for current,output in [(trueprev,isolated),(fw,closed)]:
   for i in range(P):
    g=F(0)
    for j in range(P):g=F(g+F(hij(i,j)*current[j]))
    # Incorrect assumption: g_s = gradient evaluated at w_(t-1), for every s.
    m=v=F(0)
    for s in range(1,t+1):m=F(F(B1*m)+F(O1*g));v=F(F(B2*v)+F(O2*F(g*g)))
    # Last recurrence already applied; only perform the ordinary parameter step.
    denom=F(np.sqrt(F(v/F(1-.999**t)))+EPS);d=F(F(m/F(1-.9**t))/denom);w=F(F(current[i]*F(1-F(LR*WD)))-F(LR*d));output.append((w,m,v))
  newfw=np.array([x[0] for x in closed],dtype=np.float32);row=dict(step=t,reference_objective=objective(rw),fake_closed_objective=objective(newfw))
  for name,out in [('given_true_current_weights',isolated),('closed_loop',closed)]:
   a=np.array(out,dtype=np.float32);r=np.stack([rw,rm,rv],axis=1)
   row[name+'_raw_u32']=a.view(np.uint32).tolist();row[name+'_different_u32']=int(np.count_nonzero(a.view(np.uint32)!=r.view(np.uint32)))
   for k,idx in [('w',0),('m',1),('v',2)]:row[name+'_max_abs_'+k+'_error']=float(np.max(np.abs(a[:,idx].astype(np.float64)-r[:,idx].astype(np.float64))))
  fw=newfw;rows.append(row)
 return rows

def writecsv(path,rows):
 with path.open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def main():
 ap=argparse.ArgumentParser();ap.add_argument('phase',choices=['reference','replay','validate']);args=ap.parse_args();OUT.mkdir(exist_ok=True)
 if args.phase=='reference':
  _,_,_,hist=reference(6,True);rows=[]
  for t in range(1,7):
   b,_=bench(lambda:ref_digest(t));rows.append(dict(step=t,hash=ref_digest(t),moment_transitions=P*t,reference_w_m_v_array_numeric_bytes=12*P,**b))
  (OUT/'coupled_reference.json').write_text(json.dumps(dict(history=hist,benchmarks=rows),indent=2));print('reference completed')
 elif args.phase=='replay':
  rows=[]
  for t in range(1,7):
   b,val=bench(lambda:dfs_digest(t));h,st,_=val;ph,pst,_=dfs_digest(t,True)
   assert h==ph and st.calls==P*(P**(t+1)-1)//(P-1) and st.updates==P*(P**t-1)//(P-1) and st.maxdepth==t+1
   rows.append(dict(step=t,hash=h,calls=st.calls,moment_transitions=st.updates,max_recursion_depth=st.maxdepth,retained_history_vectors=0,memo_entries=0,stored_initial_weight_vectors=0,stored_H_matrices=0,reference_state_reads=0,profile_max_unique_fp32_objects_in_selected_State_locals=pst.max_fields,profile_selected_local_numeric_fp32_bytes=pst.max_float_bytes,profile_named_object_bytes=pst.max_object_bytes,profile_frame_shallow_bytes=pst.max_frame_bytes,profile_materialized_locals_dict_bytes=pst.max_localdict_bytes,stats_object_shallow_bytes=sys.getsizeof(st),stats_unique_field_python_bytes=sum(sys.getsizeof(v) for v in {id(getattr(st,k)):getattr(st,k) for k in Stats.__slots__}.values()),**b));print(t,st.calls,st.updates,b,flush=True)
  writecsv(OUT/'coupled_replay.csv',rows);(OUT/'coupled_replay.json').write_text(json.dumps(rows,indent=2))
 else:
  # Validation-only retained outputs; never used in DFS replay memory benchmark.
  ref=json.loads((OUT/'coupled_reference.json').read_text());rr=json.loads((OUT/'coupled_replay.json').read_text());checks=[]
  for t in range(1,7):
   h,st,out=dfs_digest(t,capture=True);expected=ref['history'][t-1]['raw_u32'];assert out==expected and h==rr[t-1]['hash']==ref['benchmarks'][t-1]['hash']
   checks.append(dict(step=t,all_12_uint32_equal=True,raw_u32=out,digest=h))
  fake=fake_replay(6);assert fake[0]['given_true_current_weights_different_u32']==0 and fake[1]['given_true_current_weights_different_u32']>0
  (OUT/'coupled_validation.json').write_text(json.dumps(dict(checks=checks,validation_packed_logical_payload_bytes_per_endpoint=12*P,validation_output_scope='Validation only, not read/retained by benchmark DFS; Python nested-list/int resident size exceeds logical packed payload',all_raw_bits_equal=True,fake_replay=fake,script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),indent=2));writecsv(OUT/'fake_replay.csv',fake);print('all72 uint32 comparisons passed; fake begins divergence step2')
if __name__=='__main__':main()
