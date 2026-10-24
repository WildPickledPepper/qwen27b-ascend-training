"""Easier comparison only: externally regenerable gradients independent of weights.
No gradient history. Persistent full model weights are COUNTED, block m/v only.
Reference and replay run in separate phases/processes; validation runs afterward.
"""
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
from pathlib import Path
import sys,time,json,gc,tracemalloc,hashlib,argparse,math
import numpy as np
from exact_replay import F,B1,B2,O1,O2,EPS,LR,WD,writecsv
OUT=Path(__file__).resolve().parent/'results';N=2048;SEED=173

def gradient(lo,hi,t):
 # Integer counter formula; coordinates are exact-representable integers here.
 q=np.arange(lo,hi,dtype=np.uint64)*np.uint64(1664525)+np.uint64(t)*np.uint64(1013904223)+np.uint64(SEED)
 q=(q^(q>>np.uint64(11)))&np.uint64(65535)
 return (q.astype(np.float32)-F(32768))/F(32768)

def initial(n):return (np.arange(n,dtype=np.float32)%F(17)-F(8))*F(.01)
def moments(m,v,g):return B1*m+O1*g,B2*v+O2*(g*g)
def update(w,m,v,t):
 d=(m/F(1-.9**t))/(np.sqrt(v/F(1-.999**t))+EPS)
 return w*F(1-F(LR*WD))-LR*d

def solve(n,T,b,replay,capture=False):
 w=initial(n);digest=hashlib.sha256();raw=[] if capture else None;calls=coords=0
 if not replay:m=np.zeros(n,dtype=np.float32);v=np.zeros(n,dtype=np.float32)
 for t in range(1,T+1):
  step=hashlib.sha256();blocks=[] if capture else None
  if not replay:g=gradient(0,n,t);calls+=1;coords+=n;m,v=moments(m,v,g);w=update(w,m,v,t)
  for lo in range(0,n,b):
   hi=min(lo+b,n)
   if replay:
    m=np.zeros(hi-lo,dtype=np.float32);v=np.zeros(hi-lo,dtype=np.float32)
    for k in range(1,t+1):g=gradient(lo,hi,k);calls+=1;coords+=hi-lo;m,v=moments(m,v,g)
    w[lo:hi]=update(w[lo:hi],m,v,t);mm,vv=m,v
   else:mm,vv=m[lo:hi],v[lo:hi]
   # Ordered block w,m,v checksum; O(block) bytes only, no state-history list.
   step.update(w[lo:hi].tobytes());step.update(mm.tobytes());step.update(vv.tobytes())
   if capture:blocks.append(dict(lo=lo,hi=hi,w=w[lo:hi].view(np.uint32).tolist(),m=mm.view(np.uint32).tolist(),v=vv.view(np.uint32).tolist()))
  digest.update(step.digest())
  if capture:raw.append(blocks)
 return dict(digest=digest.hexdigest(),gradient_calls=calls,moment_coordinate_transitions=coords,weights_digest=hashlib.sha256(w.tobytes()).hexdigest(),raw=raw)

def measure(n,T,b,replay):
 tick=time.perf_counter();r=solve(n,T,b,replay);seconds=time.perf_counter()-tick
 # Prior result has only digests/counts, no state arrays; excluded instrument output.
 gc.collect();tracemalloc.start();r2=solve(n,T,b,replay);live,peak=tracemalloc.get_traced_memory();tracemalloc.stop();assert r2==r
 row=dict(mode='replay' if replay else 'reference',n=n,steps=T,block=b,seconds=seconds,tracemalloc_peak_bytes=peak,tracemalloc_live_return_bytes=live,full_weight_numeric_bytes=4*n,persistent_full_m_v_bytes=0 if replay else 8*n,nominal_current_block_m_v_numeric_bytes=8*min(n,b) if replay else 0,gradient_history_bytes=0,metadata_scalar_fields='N,T,B,seed,lo,hi,t,k,calls,coords and fixed AdamW hyperparameters; new allocations included in trace; pre-existing globals/arguments/prior reports excluded; not whole-runtime overhead',**{k:v for k,v in r.items() if k!='raw'})
 return row

def main():
 ap=argparse.ArgumentParser();ap.add_argument('phase',choices=['reference','replay','validate']);args=ap.parse_args();OUT.mkdir(exist_ok=True)
 if args.phase in ['reference','replay']:
  rows=[]
  for T in [16,32,64]:
   for b in [32,128,512]:
    row=measure(N,T,b,args.phase=='replay');rows.append(row);print(T,b,row['seconds'],row['tracemalloc_peak_bytes'],flush=True)
  writecsv(OUT/f'oracle_{args.phase}.csv',rows);(OUT/f'oracle_{args.phase}.json').write_text(json.dumps(rows,indent=2))
 else:
  ref=json.loads((OUT/'oracle_reference.json').read_text());rep=json.loads((OUT/'oracle_replay.json').read_text());checks=[]
  for a,b in zip(ref,rep):
   assert a['digest']==b['digest'] and a['weights_digest']==b['weights_digest'];assert b['moment_coordinate_transitions']==N*b['steps']*(b['steps']+1)//2
   checks.append(dict(steps=a['steps'],block=a['block'],all_steps_state_digest_equal=True,final_weights_digest_equal=True,coordinate_transition_factor=b['moment_coordinate_transitions']/a['moment_coordinate_transitions'],wall_time_factor=b['seconds']/a['seconds'],traced_peak_ratio=b['tracemalloc_peak_bytes']/a['tracemalloc_peak_bytes']))
  a=solve(31,8,7,False,True);b=solve(31,8,7,True,True);assert a['raw']==b['raw']
  (OUT/'oracle_validation.json').write_text(json.dumps(dict(grid_checks=checks,small_all_raw_bits_equal=True,small_validation_uint32_values=31*3*8,small_validation_shape={'N':31,'T':8,'B':7},script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),assumption='gradient(seed,time,coordinate) does not depend on historical model weights; synthetic oracle only',scope='Persistent full weights included in traces; no full m/v history/backing; validation raw arrays outside benchmark'),indent=2));print('all9 grids exact hashes and744 rawuint32 values matched')
if __name__=='__main__':main()
