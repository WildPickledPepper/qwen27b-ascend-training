"""CPU NumPy-only nonlinear teacher/student classification experiment.
Generated independent train/validation/test draws. No external data or code.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1');os.environ.setdefault('OMP_NUM_THREADS','1')
from pathlib import Path
import sys,math,time,json,hashlib,csv,tracemalloc,platform
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'stage3'))
import run_stage3 as s
OUT=Path(__file__).resolve().parent/'results'
DIN,H,K=16,64,4
SHAPES=[(DIN,H),(H,),(H,K),(K,)];SIZES=[int(np.prod(x)) for x in SHAPES]
P=sum(SIZES);PAD=((P+127)//128)*128
NTRAIN,NVAL,NTEST,BATCH,STEPS=4096,2048,2048,64,4096

def unpack(w):
 off=0;ans=[]
 for shape,size in zip(SHAPES,SIZES):ans.append(w[off:off+size].reshape(shape));off+=size
 return ans

def forward(w,x):
 w1,b1,w2,b2=unpack(w);h=np.tanh(x@w1+b1);logits=h@w2+b2
 return h,logits

def lossgrad(w,x,y,gradient=True):
 h,logits=forward(w,x);shifted=logits-np.max(logits,axis=1,keepdims=True)
 exp=np.exp(shifted);den=exp.sum(axis=1,keepdims=True)
 losses=np.log(den[:,0])-shifted[np.arange(len(y)),y]
 loss=float(np.mean(losses,dtype=np.float64));accuracy=float(np.mean(np.argmax(logits,axis=1)==y))
 if not gradient:return loss,accuracy
 probs=exp/den;probs[np.arange(len(y)),y]-=1;probs/=np.array(len(y),dtype=w.dtype)
 w1,b1,w2,b2=unpack(w);dw2=h.T@probs;db2=probs.sum(axis=0)
 dh=(probs@w2.T)*(1-h*h);dw1=x.T@dh;db1=dh.sum(axis=0)
 g=np.zeros_like(w);g[:P]=np.concatenate([dw1.ravel(),db1,dw2.ravel(),db2])
 return loss,accuracy,g

def make(seed):
 r=np.random.default_rng(seed+1000)
 tw1=r.normal(0,1/math.sqrt(DIN),(DIN,48)).astype(np.float32);tb1=r.normal(0,.15,48).astype(np.float32)
 tw2=r.normal(0,1/math.sqrt(48),(48,K)).astype(np.float32);tb2=r.normal(0,.1,K).astype(np.float32)
 parts=[];hashes={};rowsets=[]
 for name,n in [('train',NTRAIN),('validation',NVAL),('test',NTEST)]:
  x=r.normal(size=(n,DIN)).astype(np.float32);logits=np.tanh(x@tw1+tb1)@tw2+tb2;y=np.argmax(logits,axis=1)
  parts.append((x,y));hashes[name]=hashlib.sha256(x.tobytes()+y.astype('<i8').tobytes()).hexdigest();rowsets.append(set(map(bytes,x)))
 assert not rowsets[0]&rowsets[1] and not rowsets[0]&rowsets[2] and not rowsets[1]&rowsets[2]
 r=np.random.default_rng(seed+2000)
 w=np.zeros(PAD,dtype=np.float32);arrays=unpack(w);arrays[0][:]=r.normal(0,1/math.sqrt(DIN),arrays[0].shape);arrays[2][:]=r.normal(0,1/math.sqrt(H),arrays[2].shape)
 batches=[];r=np.random.default_rng(seed+3000)
 for _ in range(math.ceil(STEPS/(NTRAIN//BATCH))):batches.extend(r.permutation(NTRAIN).reshape(-1,BATCH))
 batches=np.array(batches[:STEPS]);hashes['initial_weights']=hashlib.sha256(w.tobytes()).hexdigest();hashes['batches']=hashlib.sha256(batches.astype('<i8').tobytes()).hexdigest()
 return parts,w,batches,hashes

def lr_at(t,peak):
 warm=64
 if t<=warm:return peak*t/warm
 phase=(t-warm)/(STEPS-warm)
 return peak*(.01+.99*.5*(1+math.cos(math.pi*phase)))

def audit():
 parts,w,_,_=make(101);x,y=parts[0];w=w.astype(np.float64);x=x[:8].astype(np.float64);y=y[:8]
 _,_,g=lossgrad(w,x,y);errors=[]
 for i in np.random.default_rng(42).choice(P,32,replace=False):
  old=w[i];delta=1e-5;w[i]=old+delta;lp=lossgrad(w,x,y,False)[0];w[i]=old-delta;lm=lossgrad(w,x,y,False)[0];w[i]=old
  fd=(lp-lm)/(2*delta);err=abs(fd-g[i]);errors.append(err);assert err<1e-7,(i,fd,g[i])
 assert np.all(g[P:]==0)
 return dict(finite_difference_coordinates=32,finite_difference_max_abs_error=max(errors),gradient_check_passed=True,train_val_test_rows_disjoint=True,teacher='Fixed random 16-48-4 tanh network with hard argmax labels per seed',student='16-64-4 tanh network with biases; 1348 active parameters, 1408 stored padded coordinates')

def main():
 OUT.mkdir(exist_ok=True);started=time.perf_counter();checks=audit();schecks=s.audit();checkpoints=[];summary=[];inputs={};memory=[]
 configs=[s.config('fp32','fp32'),s.config('mv_nf4_sr_b64','mv','nf4','sr',64),s.config('mv_nf4_sr_b128','mv','nf4','sr',128),s.config('qv_nf4_sr_b64','qv','nf4','sr',64),s.config('qv_nf4_sr_b128','qv','nf4','sr',128)]
 for seed in [101,202,303]:
  parts,initial,batches,hashes=make(seed);inputs[str(seed)]=hashes
  train,val,test=parts
  for peak in [1e-3,3e-3]:
   refs={};reference_final=None;reference_test=None
   for c in configs:
    started_config=time.perf_counter();w=initial.copy();z=np.zeros(PAD,dtype=np.float32);blob=s.encode(z,z,0,seed,c)
    init_loss,init_acc=lossgrad(w,*val,False);opt_seconds=0.;grad_seconds=0.;maxgrad=0.;maxupdate=0.;nonfinite=False
    for t,indices in enumerate(batches,1):
     tick=time.perf_counter();bl,ba,g=lossgrad(w,train[0][indices],train[1][indices]);grad_seconds+=time.perf_counter()-tick
     maxgrad=max(maxgrad,float(np.linalg.norm(g.astype(np.float64))))
     tick=time.perf_counter();m,v,_,_=s.decode(blob,c);m,v=s.moments(m,v,g);s.LR=np.float32(lr_at(t,peak));d,w=s.update(m,v,w,t);blob=s.encode(m,v,t,seed,c);opt_seconds+=time.perf_counter()-tick
     maxupdate=max(maxupdate,float(np.max(np.abs(np.float64(s.LR)*d))))
     if not np.all(np.isfinite(w)) or not np.all(np.isfinite(m)) or not np.all(np.isfinite(v)):nonfinite=True;break
     if t%256==0 or t==STEPS:
      tl,ta=lossgrad(w,*train,False);vl,va=lossgrad(w,*val,False)
      if c['form']=='fp32':refs[t]=(vl,w.copy())
      checkpoints.append(dict(seed=seed,peak_lr=peak,configuration=c['name'],step=t,lr=float(s.LR),train_loss=tl,train_accuracy=ta,validation_loss=vl,validation_accuracy=va,reference_validation_loss=refs[t][0],validation_loss_ratio=vl/refs[t][0],weight_RMS_vs_FP32=float(np.sqrt(np.mean((w[:P].astype(np.float64)-refs[t][1][:P])**2))),last_batch_gradient_norm=float(np.linalg.norm(g.astype(np.float64))),last_update_RMS=float(np.sqrt(np.mean(d.astype(np.float64)**2)))))
    # Test is evaluated only at the predeclared final checkpoint, never for selection.
    train_loss,train_acc=lossgrad(w,*train,False);val_loss,val_acc=lossgrad(w,*val,False);test_loss,test_acc=lossgrad(w,*test,False)
    if c['form']=='fp32':reference_final=val_loss;reference_test=test_loss
    row=dict(seed=seed,peak_lr=peak,configuration=c['name'],steps_completed=t,active_parameters=P,padded_state_coordinates=PAD,batch_size=BATCH,initial_validation_loss=init_loss,final_train_loss=train_loss,final_train_accuracy=train_acc,final_validation_loss=val_loss,final_validation_accuracy=val_acc,reference_validation_loss=reference_final,validation_loss_ratio=val_loss/reference_final,final_test_loss=test_loss,final_test_accuracy=test_acc,reference_test_loss=reference_test,test_loss_ratio=test_loss/reference_test,nonfinite_detected=nonfinite,max_gradient_L2=maxgrad,max_parameter_step_absolute=maxupdate,serialized_state_bytes=len(blob),python_bytes_object_allocation=sys.getsizeof(blob),shared_codebook_bytes=0 if c['form']=='fp32' else 64,optimizer_seconds=opt_seconds,forward_backward_seconds=grad_seconds,total_seconds=time.perf_counter()-started_config,**s.memory(c))
    summary.append(row);s.csvwrite(OUT/'summary.csv',summary);s.csvwrite(OUT/'checkpoints.csv',checkpoints)
    print(seed,peak,c['name'],'val',round(val_loss,6),'ratio',round(val_loss/reference_final,4),'acc',round(val_acc,4),flush=True)
    # Workspace audit excludes existing packed state, parameters, input gradient,
    # and data/model activations, which are not the m/v replacement target.
    tracemalloc.start();mm,vv,_,_=s.decode(blob,c);mm,vv=s.moments(mm,vv,g);dd,ww=s.update(mm,vv,w,t+1);new=s.encode(mm,vv,t+1,seed,c);_,pk=tracemalloc.get_traced_memory();tracemalloc.stop()
    memory.append(dict(seed=seed,peak_lr=peak,configuration=c['name'],serialized_state_bytes=len(blob),tracemalloc_optimizer_step_peak_bytes=pk,window_coordinates=PAD))
 s.csvwrite(OUT/'memory.csv',memory)
 manifest=dict(checks=checks,codec_checks=schecks,input_hashes=inputs,python=sys.version,numpy=np.__version__,platform=platform.platform(),script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),codec_sha256=hashlib.sha256(Path(s.__file__).read_bytes()).hexdigest(),elapsed_seconds=time.perf_counter()-started,configs=configs,protocol=dict(train=NTRAIN,validation=NVAL,test=NTEST,batch=BATCH,steps=STEPS,peaks=[1e-3,3e-3],schedule='64-step linear warmup then cosine decay to 1% peak',seeds=[101,202,303],optimizer='same beta=.9/.999 epsilon=1e-8 decoupled weight decay=.01 and FP32 master weights',test_policy='final checkpoint only; no selection or hyperparameter tuning based on test',scope='Small CPU supervised synthetic teacher/student, not 27B evidence. Metadata/workspace design projections are unchanged stage3 assumptions; measured CPU working buffers reported separately.'))
 (OUT/'run.json').write_text(json.dumps(manifest,indent=2));print('completed',manifest['elapsed_seconds'],flush=True)
if __name__=='__main__':main()
