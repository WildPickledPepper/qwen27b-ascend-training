"""Post-run independent audit and actual retained-buffer optimizer-loop trace.
The trace includes all newly allocated Python/NumPy optimizer arrays from state
initialization through 16 steps of the same decode/moments/update/encode pattern.
Model input weights/gradient are prepared before tracing, so this is not a full
training RSS or accelerator measurement. Persisted zero/m/v/d buffers ARE counted.
"""
from pathlib import Path
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
import sys,json,csv,gc,tracemalloc,hashlib,time
import numpy as np
import attention_retrieval as a
s=a.s;OUT=Path(__file__).resolve().parent/'results'

def memory_loop(c,initial,g):
 w=initial.copy();gc.collect();tracemalloc.start();zero=np.zeros(a.PAD,dtype=np.float32);blob=s.encode(zero,zero,0,101,c);peaks=[]
 for t in range(1,17):
  tracemalloc.reset_peak();m,v,_,_=s.decode(blob,c);m,v=s.moments(m,v,g);s.LR=np.float32(.001);d,w=s.update(m,v,w,t);blob=s.encode(m,v,t,101,c);cur,pk=tracemalloc.get_traced_memory();peaks.append(pk)
 cur,pk=tracemalloc.get_traced_memory();tracemalloc.stop();cb=0 if c['form']=='fp32' else 64
 return dict(configuration=c['name'],serialized_bytes=len(blob),codebook_numeric_bytes=cb,full_retained_loop_trace_peak=max(peaks),full_retained_loop_peak_plus_codebook=max(peaks)+cb,trace_live_after_step16=cur,steps=16,window_coordinates=a.PAD,includes='zero initialization retained, old/new decoded m/v, update d, newly allocated returned weights, encoding temporaries, packed state; Python/NumPy traced allocations',excludes='input initial model weights and input gradient allocated before trace; teacher/model/data/forward-backward; native allocations not visible to tracemalloc; shared codebook separately charged')

def main():
 rows=list(csv.DictReader((OUT/'summary.csv').open()));assert len(rows)==30
 run=json.loads((OUT/'run.json').read_text());assert run['script_sha256']==hashlib.sha256(Path(a.__file__).read_bytes()).hexdigest()
 out={};ms=[];dead=[];permutation_checks=[]
 for seed in [101,202,303]:
  parts,w,batches,hashes,meta=a.make(seed);assert hashes==run['input_hashes'][str(seed)]
  assert len(set(map(bytes,parts[0][0].reshape(a.NTRAIN,-1))))==a.NTRAIN
  for start in range(0,a.STEPS,a.NTRAIN//a.BATCH):assert np.array_equal(np.sort(batches[start:start+a.NTRAIN//a.BATCH].ravel()),np.arange(a.NTRAIN))
  _,_,g=a.lossgrad(w,parts[0][0][batches[0]],parts[0][1][batches[0]])
  observed=np.zeros(a.PAD,dtype=bool)
  for idx in batches[:64]:observed|=a.lossgrad(w,parts[0][0][idx],parts[0][1][idx])[2]!=0
  assert int(observed[:a.P].sum())==1940 and not observed[a.P:].any()
  dead.append(dict(seed=seed,declared_parameters=a.P,structurally_data_active_parameters=1940,structurally_zero_gradient_parameters=240,extra_padding=a.PAD-a.P,first_epoch_observed_nonzero_gradient_coordinates=int(observed.sum())))
  for row in [r for r in rows if int(r['seed'])==seed]:
   peak=float(row['peak_lr']);name=row['configuration'];config=next(x for x in a.CONFIGS if x['name']==name)
   data=np.load(OUT/f'final_s{seed}_lr{peak}_{name}.npz');finalw=data['weights'];blob=data['state'].tobytes();m,v,t,rs=s.decode(blob,config)
   assert t==a.STEPS and rs==seed and len(blob)==int(row['serialized_state_bytes'])
   assert np.isfinite(finalw).all() and np.isfinite(m).all() and np.isfinite(v).all();assert np.all(m[~observed]==0) and np.all(v[~observed]==0)
   for part,label in [(parts[0],'train'),(parts[1],'validation'),(parts[2],'test')]:
    loss,acc=a.evaluate(finalw,part);assert loss==float(row['final_'+label+'_loss']) and acc==float(row['final_'+label+'_accuracy'])
   assert row['nonfinite_detected']=='False' and int(row['steps_completed'])==a.STEPS
   xperm=parts[2][0][:128].copy();order=np.array([6,0,5,1,4,2,3]);xperm[:,:a.T]=xperm[:,order];base=a.forward(finalw,parts[2][0][:128])[0];changed=a.forward(finalw,xperm)[0];assert np.array_equal(base.argmax(1),changed.argmax(1));permutation_checks.append(dict(seed=seed,peak_lr=peak,configuration=name,examples=128,max_logit_absolute_drift=float(np.max(np.abs(base-changed))),predictions_identical=True))
  if seed==101:
   for config in a.CONFIGS:ms.append(memory_loop(config,w,g))
 out.update(context_permutation_checks=permutation_checks,artifact_rows_verified=30,all_saved_final_losses_and_accuracies_recomputed=True,all_seed_data_init_batch_hashes_reproduced=True,all_epoch_batches_permutations=True,structural_zero_audit=dead,all_zero_gradient_moments_remain_zero=True,all_runs_completed_finite=True,gradient_checks=run['checks'],memory_scope='Same retained-buffer optimizer pattern, not training-process peak; see memory_loop.csv',full_loop_memory=ms)
 s.csvwrite(OUT/'memory_loop.csv',ms);(OUT/'audit.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
if __name__=='__main__':main()
