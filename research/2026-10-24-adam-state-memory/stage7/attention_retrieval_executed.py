"""Own NumPy attention/backprop and fresh-key synthetic retrieval, no external code.
Trainable last-query cross-attention, residual tanh FFN, classification head.
CPU algorithm experiment, not a transformer/27B training claim.
"""
import os
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
from pathlib import Path
import sys,math,time,json,hashlib,tracemalloc,argparse,platform
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'stage3'))
import run_stage3 as s
OUT=Path(__file__).resolve().parent/'results'
D,H,DH,F,K,T=16,2,8,32,4,7
SHAPES=[(D,D)]*4+[(D,),(D,F),(F,),(F,D),(D,),(D,K),(K,)]
SIZES=[int(np.prod(x)) for x in SHAPES];P=sum(SIZES);PAD=((P+127)//128)*128
NTRAIN,NVAL,NTEST,BATCH,STEPS=4096,2048,2048,64,4096
CONFIGS=[s.config('fp32','fp32'),s.config('mv_nf4_sr_b64','mv','nf4','sr',64),s.config('mv_nf4_sr_b128','mv','nf4','sr',128),s.config('qv_nf4_sr_b64','qv','nf4','sr',64),s.config('qv_nf4_sr_b128','qv','nf4','sr',128)]

def unpack(w):
 off=0;ans=[]
 for shape,size in zip(SHAPES,SIZES):ans.append(w[off:off+size].reshape(shape));off+=size
 return ans

def forward(w,x,uniform=False):
 wq,wk,wv,wo,bo,w1,b1,w2,b2,wc,bc=unpack(w);ctx=x[:,:T];query=x[:,T]
 q=(query@wq).reshape(-1,H,DH);k=(ctx@wk).reshape(-1,T,H,DH);v=(ctx@wv).reshape(-1,T,H,DH)
 scores=np.einsum('bhd,bthd->bht',q,k)/np.array(math.sqrt(DH),dtype=w.dtype)
 scores-=np.max(scores,axis=-1,keepdims=True);a=np.exp(scores);a/=a.sum(axis=-1,keepdims=True)
 if uniform:a[:]=np.array(1/T,dtype=w.dtype)
 cat=np.einsum('bht,bthd->bhd',a,v).reshape(-1,D)
 z=cat@wo+bo+query;f=np.tanh(z@w1+b1);r=z+f@w2+b2;logits=r@wc+bc
 return logits,(ctx,query,q,k,v,a,cat,z,f,r)

def lossgrad(w,x,y,gradient=True,uniform=False):
 logits,c=forward(w,x,uniform);shifted=logits-np.max(logits,axis=1,keepdims=True);ex=np.exp(shifted);den=ex.sum(axis=1,keepdims=True)
 loss=float(np.mean(np.log(den[:,0])-shifted[np.arange(len(y)),y],dtype=np.float64));acc=float(np.mean(logits.argmax(axis=1)==y))
 if not gradient:return loss,acc
 wq,wk,wv,wo,bo,w1,b1,w2,b2,wc,bc=unpack(w);ctx,query,q,k,v,a,cat,z,f,r=c
 dl=ex/den;dl[np.arange(len(y)),y]-=1;dl/=np.array(len(y),dtype=w.dtype)
 dwc=r.T@dl;dbc=dl.sum(axis=0);dr=dl@wc.T;dw2=f.T@dr;db2=dr.sum(axis=0)
 df=(dr@w2.T)*(1-f*f);dw1=z.T@df;db1=df.sum(axis=0);dz=dr+df@w1.T
 dwo=cat.T@dz;dbo=dz.sum(axis=0);dc=(dz@wo.T).reshape(-1,H,DH)
 da=np.einsum('bhd,bthd->bht',dc,v);dv=np.einsum('bht,bhd->bthd',a,dc)
 ds=a*(da-(da*a).sum(axis=-1,keepdims=True))/np.array(math.sqrt(DH),dtype=w.dtype)
 if uniform:ds[:]=0
 dq=np.einsum('bht,bthd->bhd',ds,k).reshape(-1,D);dk=np.einsum('bht,bhd->bthd',ds,q).reshape(-1,D)
 dwq=query.T@dq;dwk=ctx.reshape(-1,D).T@dk;dwv=ctx.reshape(-1,D).T@dv.reshape(-1,D)
 g=np.zeros_like(w);g[:P]=np.concatenate([dwq.ravel(),dwk.ravel(),dwv.ravel(),dwo.ravel(),dbo,dw1.ravel(),db1,dw2.ravel(),db2,dwc.ravel(),dbc])
 return loss,acc,g

def make(seed):
 rng=np.random.default_rng(seed+7000);parts=[];hashes={};sets=[];meta={}
 for split_id,(name,n) in enumerate([('train',NTRAIN),('validation',NVAL),('test',NTEST)]):
  rng=np.random.default_rng(seed+17000+1000*split_id)
  keys=rng.normal(size=(n,T,8)).astype(np.float32);keys/=np.linalg.norm(keys,axis=-1,keepdims=True)
  values=rng.integers(K,size=(n,T));target=rng.integers(T,size=n)
  x=np.zeros((n,T+1,D),dtype=np.float32);x[:,:T,:8]=keys;x[:,:T,8:12]=np.eye(K,dtype=np.float32)[values];x[:,T,:8]=keys[np.arange(n),target];x[:,T,12]=1
  # Fixed hard-attention teacher: nearest key by dot product, copy its new value.
  dots=np.einsum('btd,bd->bt',keys,x[:,T,:8]);picked=dots.argmax(axis=1);assert np.array_equal(picked,target)
  y=values[np.arange(n),picked];parts.append((x,y,target,values))
  hashes[name]=hashlib.sha256(x.tobytes()+y.astype('<i8').tobytes()).hexdigest();sets.append(set(map(bytes,keys.reshape(n,-1))))
  meta[name]={'class_counts':np.bincount(y,minlength=K).tolist(),'teacher_accuracy':1.,'context_majority_accuracy':float(np.mean(np.array([np.bincount(v,minlength=K).argmax() for v in values])==y))}
 assert not sets[0]&sets[1] and not sets[0]&sets[2] and not sets[1]&sets[2]
 rng=np.random.default_rng(seed+8000);w=np.zeros(PAD,dtype=np.float32)
 for a in unpack(w):
  if a.ndim==2:a[:]=rng.normal(0,1/math.sqrt(a.shape[0]),a.shape)
 rng=np.random.default_rng(seed+9000);batches=[]
 for _ in range(math.ceil(STEPS/(NTRAIN//BATCH))):batches.extend(rng.permutation(NTRAIN).reshape(-1,BATCH))
 batches=np.array(batches[:STEPS]);hashes['initial_weights']=hashlib.sha256(w.tobytes()).hexdigest();hashes['batches']=hashlib.sha256(batches.astype('<i8').tobytes()).hexdigest()
 return parts,w,batches,hashes,meta

def evaluate(w,part,uniform=False):
 x,y=part[:2];loss=acc=0.
 for lo in range(0,len(y),256):
  hi=min(lo+256,len(y));a,b=lossgrad(w,x[lo:hi],y[lo:hi],False,uniform);loss+=a*(hi-lo);acc+=b*(hi-lo)
 return loss/len(y),acc/len(y)

def diagnostics(w,part):
 x,y,target,values=part;eligible=[];newpos=[]
 for i in range(len(y)):
  choices=[(target[i]+j)%T for j in range(1,T) if values[i,(target[i]+j)%T]!=y[i]]
  if choices:eligible.append(i);newpos.append(choices[0])
 ix=np.array(eligible);j=np.array(newpos);changed=x[ix].copy();changed[:,T,:8]=changed[np.arange(len(ix)),j,:8];newy=values[ix,j]
 orig=evaluate(w,(changed,y[ix]));swapped=evaluate(w,(changed,newy));ul,ua=evaluate(w,part,True)
 _,cache=forward(w,x);a=cache[5];chosen=np.mean(a,axis=1).argmax(axis=-1)
 return dict(uniform_attention_loss=ul,uniform_attention_accuracy=ua,query_swap_eligible_examples=len(ix),query_swap_old_label_loss=orig[0],query_swap_old_label_accuracy=orig[1],query_swap_new_target_loss=swapped[0],query_swap_new_target_accuracy=swapped[1],mean_attention_top1_retrieval_accuracy=float(np.mean(chosen==target)),mean_attention_target_mass=float(np.mean(np.mean(a,axis=1)[np.arange(len(y)),target])))

def lr_at(t,peak):
 if t<=64:return peak*t/64
 return peak*(.01+.99*.5*(1+math.cos(math.pi*(t-64)/(STEPS-64))))

def audit():
 parts,w,_,_,_=make(101);w=w.astype(np.float64);x=parts[0][0][:5].astype(np.float64);y=parts[0][1][:5];_,_,g=lossgrad(w,x,y);errs=[]
 for i in range(P):
  old=w[i];delta=1e-5;w[i]=old+delta;lp=lossgrad(w,x,y,False)[0];w[i]=old-delta;lm=lossgrad(w,x,y,False)[0];w[i]=old;fd=(lp-lm)/(2*delta);err=abs(fd-g[i]);errs.append(err);assert err<1e-7,(i,fd,g[i])
 assert np.all(g[P:]==0)
 # Independent perturbed point: full-vector directional derivatives check every tensor.
 rng=np.random.default_rng(88);wp=w.copy();wp[:P]+=rng.normal(0,.02,P);_,_,gp=lossgrad(wp,x,y);direction_errors=[]
 for _ in range(12):
  direction=np.zeros_like(wp);direction[:P]=rng.normal(size=P);direction/=np.linalg.norm(direction);delta=1e-5;fd=(lossgrad(wp+delta*direction,x,y,False)[0]-lossgrad(wp-delta*direction,x,y,False)[0])/(2*delta);error=abs(fd-np.dot(gp,direction));direction_errors.append(error);assert error<1e-7
 wf=w.astype(np.float32);_,_,gf=lossgrad(wf,x.astype(np.float32),y);assert gf.dtype==np.float32
 return dict(finite_difference_coordinates=P,finite_difference_max_abs_error=max(errs),finite_difference_mean_abs_error=float(np.mean(errs)),perturbed_point_directional_checks=12,perturbed_directional_max_abs_error=max(direction_errors),gradient_check_passed=True,padding_gradient_exact_zero=True,all_training_arithmetic_dtype='float32, loss reporting mean float64',disjoint_splits=True,teacher='Fixed nearest-key hard attention with per-sequence independent random key-value assignment; no learned teacher weights')

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--pilot',action='store_true');args=ap.parse_args();OUT.mkdir(exist_ok=True);started=time.perf_counter();checks=audit();codec_checks=s.audit();summary=[];checkpoints=[];memory=[];inputs={};data_meta={}
 configs=CONFIGS[:1] if args.pilot else CONFIGS;seeds=[909] if args.pilot else [101,202,303];peaks=[.003] if args.pilot else [.001,.003];limit=512 if args.pilot else STEPS
 prefix='pilot_' if args.pilot else ''
 for seed in seeds:
  parts,initial,batches,hashes,met=make(seed);inputs[str(seed)]=hashes;data_meta[str(seed)]=met;train,val,test=parts
  for peak in peaks:
   ref_losses={};ref_final=ref_test=None
   for c in configs:
    tick_run=time.perf_counter();w=initial.copy();zero=np.zeros(PAD,dtype=np.float32);blob=s.encode(zero,zero,0,seed,c);init_loss,init_acc=evaluate(w,val);opt_time=grad_time=0.;maxgrad=maxupdate=0.;nonfinite=False
    for t,indices in enumerate(batches[:limit],1):
     tick=time.perf_counter();bl,ba,g=lossgrad(w,train[0][indices],train[1][indices]);grad_time+=time.perf_counter()-tick;maxgrad=max(maxgrad,float(np.linalg.norm(g.astype(np.float64))))
     tick=time.perf_counter();m,v,_,_=s.decode(blob,c);m,v=s.moments(m,v,g);s.LR=np.float32(lr_at(t,peak));d,w=s.update(m,v,w,t);blob=s.encode(m,v,t,seed,c);opt_time+=time.perf_counter()-tick;maxupdate=max(maxupdate,float(np.max(np.abs(np.float64(s.LR)*d))))
     if not np.isfinite(bl) or not np.all(np.isfinite(g)) or not np.all(np.isfinite(d)) or not np.all(np.isfinite(w)) or not np.all(np.isfinite(m)) or not np.all(np.isfinite(v)):nonfinite=True;break
     if t%256==0 or t==limit:
      tl,ta=evaluate(w,train);vl,va=evaluate(w,val)
      if c['form']=='fp32':ref_losses[t]=(vl,w.copy())
      checkpoints.append(dict(seed=seed,peak_lr=peak,configuration=c['name'],step=t,lr=float(s.LR),train_loss=tl,train_accuracy=ta,validation_loss=vl,validation_accuracy=va,reference_validation_loss=ref_losses[t][0],validation_loss_ratio=vl/ref_losses[t][0],weight_RMS_vs_FP32=float(np.sqrt(np.mean((w[:P].astype(np.float64)-ref_losses[t][1][:P])**2))),last_batch_gradient_L2=float(np.linalg.norm(g.astype(np.float64))),last_update_RMS=float(np.sqrt(np.mean(d.astype(np.float64)**2)))))
    tl,ta=evaluate(w,train);vl,va=evaluate(w,val);xl,xa=evaluate(w,test)
    if c['form']=='fp32':ref_final=vl;ref_test=xl
    row=dict(seed=seed,peak_lr=peak,configuration=c['name'],steps_completed=t,active_parameters=P,padded_state_coordinates=PAD,initial_validation_loss=init_loss,initial_validation_accuracy=init_acc,final_train_loss=tl,final_train_accuracy=ta,final_validation_loss=vl,final_validation_accuracy=va,reference_validation_loss=ref_final,validation_loss_ratio=vl/ref_final,final_test_loss=xl,final_test_accuracy=xa,reference_test_loss=ref_test,test_loss_ratio=xl/ref_test,nonfinite_detected=nonfinite,max_gradient_L2=maxgrad,max_parameter_step_absolute=maxupdate,serialized_state_bytes=len(blob),shared_codebook_bytes=0 if c['form']=='fp32' else 64,optimizer_seconds=opt_time,forward_backward_seconds=grad_time,total_seconds=time.perf_counter()-tick_run,**diagnostics(w,test))
    summary.append(row);s.csvwrite(OUT/(prefix+'summary.csv'),summary);s.csvwrite(OUT/(prefix+'checkpoints.csv'),checkpoints)
    np.savez(OUT/(prefix+f'final_s{seed}_lr{peak}_{c["name"]}.npz'),weights=w,state=np.frombuffer(blob,dtype=np.uint8))
    tracemalloc.start();mm,vv,_,_=s.decode(blob,c);mm,vv=s.moments(mm,vv,g);dd,ww=s.update(mm,vv,w,t+1);new=s.encode(mm,vv,t+1,seed,c);_,pk=tracemalloc.get_traced_memory();tracemalloc.stop();cb=0 if c['form']=='fp32' else 64
    memory.append(dict(seed=seed,peak_lr=peak,configuration=c['name'],serialized_state_bytes=len(blob),codebook_bytes=cb,old_state_python_allocation=sys.getsizeof(blob),traced_step_new_allocations_peak=pk,old_state_plus_codebook_plus_traced_step_peak=sys.getsizeof(blob)+cb+pk,window_coordinates=PAD))
    print(seed,peak,c['name'],'val',round(vl,6),'ratio',round(vl/ref_final,4),'test_acc',round(xa,4),'uniform_acc',round(row['uniform_attention_accuracy'],4),'swap_old/new',round(row['query_swap_old_label_accuracy'],4),round(row['query_swap_new_target_accuracy'],4),'secs',round(row['total_seconds'],2),flush=True)
 s.csvwrite(OUT/(prefix+'memory.csv'),memory)
 manifest=dict(checks=checks,codec_checks=codec_checks,input_hashes=inputs,data_metadata=data_meta,python=sys.version,numpy=np.__version__,platform=platform.platform(),script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),codec_sha256=hashlib.sha256(Path(s.__file__).read_bytes()).hexdigest(),elapsed_seconds=time.perf_counter()-started,configs=configs,protocol=dict(seeds=seeds,peaks=peaks,steps=limit,batch=BATCH,train=NTRAIN,validation=NVAL,test=NTEST,student='Last-query 2-head attention d16, residual tanh FFN16-32-16,4classes,2180active/2304padded',teacher=checks['teacher'],task='7 fresh unit 8D keys with independent uniform class values, plus query equal to uniformly selected key; context/query input16features',optimizer='beta1=.9,beta2=.999,eps=1e-8,WD=.01;64-step warmup, cosine to1% peak',test_policy='Test once at predeclared final checkpoint; paired seed/input/init/batch; no best-method or best-step selection; initial pilot used different arrays preserved separately, final protocol data regenerated with independent split RNGs before full run',diagnostics='Uniform attention weights at eval; replace query by a different-class context key and report old/new labels; no retraining controls',scope='Small CPU synthetic attention task, not 27B or NPU evidence; lossy state encoding; actual CPU prototype working set separate from theoretical native packed design'))
 (OUT/(prefix+'run.json')).write_text(json.dumps(manifest,indent=2));print('completed',manifest['elapsed_seconds'],flush=True)
if __name__=='__main__':main()
