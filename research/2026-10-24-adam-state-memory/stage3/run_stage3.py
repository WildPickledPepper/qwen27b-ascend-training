"""Independent CPU experiment; no third-party implementation execution.
All persistent optimizer data are packed bytes. No residual/error-feedback cache.
Run: python run_stage3.py --out results
"""
from pathlib import Path
import argparse, csv, hashlib, json, math, platform, random, struct, sys, time, tracemalloc
import numpy as np

BASE=Path(__file__).resolve().parents[1]/'adam-state-cpu-20261024'
sys.path.insert(0,str(BASE))
import experiment as old
P=old.P
B1,B2,OM1,OM2,LR,EPS,WD=map(np.float32,(.9,.999,.1,.001,.0001,1e-8,.01))
NF4=np.array([-1.,-.6961928009986877,-.5250730514526367,-.39491748809814453,-.28444138169288635,-.18477343022823334,-.09105003625154495,0.,.07958029955625534,.16093020141124725,.24611230194568634,.33791524171829224,.44070982933044434,.5626170039176941,.7229568362236023,1.],dtype=np.float32)
U15=np.linspace(-1,1,15,dtype=np.float32)
HEADER=struct.Struct('<QQIQI') # step, n, block, seed, format-id: 32 bytes
FORMATS={'mv':1,'qv':2,'fp32':0}

def moments(m,v,g):
    return B1*m+OM1*g, B2*v+OM2*(g*g)

def denom(v,t):
    return np.sqrt(v/np.float32(1-.999**max(1,t)))+EPS

def update(m,v,w,t):
    d=(m/np.float32(1-.9**t))/denom(v,t)
    return d, w*np.float32(1-np.float32(LR*WD))-LR*d

def uniform(seed,t,n,stream):
    # Stateless SplitMix64; header seed + time + coordinate reproduces every draw.
    with np.errstate(over='ignore'):
        z=np.arange(n,dtype=np.uint64)+np.uint64(seed)+np.uint64(t)*np.uint64(0x9E3779B97F4A7C15)+np.uint64(stream)*np.uint64(0xD1B54A32D192ED03)
        z=(z^(z>>30))*np.uint64(0xBF58476D1CE4E5B9)
        z=(z^(z>>27))*np.uint64(0x94D049BB133111EB)
        z=z^(z>>31)
    return (z>>11).astype(np.float64)*(1./2**53)

def pack4(c):
    a=c.ravel().astype(np.uint8)
    if len(a)%2:a=np.pad(a,(0,1))
    return (a[::2]|(a[1::2]<<4)).tobytes()

def unpack4(buf,n):
    a=np.frombuffer(buf,dtype=np.uint8);out=np.empty(2*len(a),dtype=np.uint8)
    out[::2]=a&15;out[1::2]=a>>4
    return out[:n]

def config(name,form='mv',book='nf4',rounding='sr',block=128):
    return dict(name=name,form=form,book=book,rounding=rounding,block=block)

def encode(m,v,t,seed,c):
    n=len(m);b=c['block'];h=HEADER.pack(t,n,b,seed,FORMATS[c['form']])
    if c['form']=='fp32':return h+m.astype('<f4').tobytes()+v.astype('<f4').tobytes()
    if n%b:raise ValueError('Experiment deliberately requires full blocks; tail overhead separate in budget.')
    signed=m if c['form']=='mv' else m/denom(v,t)
    a=signed.reshape(-1,b);scale=np.max(np.abs(a),axis=1).astype(np.float32)
    x=np.divide(a,scale[:,None],out=np.zeros_like(a),where=scale[:,None]!=0).astype(np.float64)
    book=NF4 if c['book']=='nf4' else U15
    r=np.searchsorted(book,x).clip(0,len(book)-1);l=(r-1).clip(0)
    lo=book[l].astype(np.float64);hi=book[r].astype(np.float64)
    if c['rounding']=='sr':
        p=np.divide(x-lo,hi-lo,out=np.zeros_like(x),where=hi!=lo).clip(0,1)
        code=np.where(uniform(seed,t,n,1).reshape(x.shape)<p,r,l)
    else:code=np.where(abs(x-lo)<=abs(hi-x),l,r)
    # Exact zero code, and positive-only log grid: no positive-to-zero rounding.
    a=v.reshape(-1,b);pos=a>0
    vmin=np.min(np.where(pos,a,np.inf),axis=1).astype(np.float32)
    vmax=np.max(a,axis=1).astype(np.float32)
    vmin=np.where(np.isfinite(vmin),vmin,0).astype(np.float32)
    # Store the positive endpoints themselves (FP32); compute log interpolation
    # at encode and decode identically. Constant/zero blocks never use log(0).
    safe_min=np.where(vmin>0,vmin,1).astype(np.float64)
    safe_max=np.where(vmax>0,vmax,1).astype(np.float64)
    logs=np.log(safe_min);span=np.log(safe_max)-logs
    levels=np.exp(logs[:,None]+span[:,None]*np.arange(15)[None,:]/14).astype(np.float32)
    levels[:,0]=vmin;levels[:,-1]=vmax
    idx=np.sum(a[:,:,None]>levels[:,None,:],axis=2).clip(0,14)
    left=(idx-1).clip(0)
    lower=np.take_along_axis(levels,left,axis=1).astype(np.float64)
    upper=np.take_along_axis(levels,idx,axis=1).astype(np.float64)
    if c['rounding']=='sr':
        # Conditioned on the present v, real-arithmetic reciprocal-denominator SR.
        c2=float(np.float32(1-.999**max(1,t)))
        f=lambda z:1/(np.sqrt(z/c2)+float(EPS))
        fl,fu,fx=f(lower),f(upper),f(a.astype(np.float64))
        pl=np.divide(fx-fu,fl-fu,out=np.zeros_like(fx),where=fl!=fu).clip(0,1)
        vi=np.where(uniform(seed,t,n,2).reshape(a.shape)<pl,left,idx)
    else:
        # Nearest in log space, avoiding dominance by large coordinates.
        loga=np.log(np.where(pos,a,1).astype(np.float64))
        dl=np.abs(loga-np.log(np.where(lower>0,lower,1)))
        du=np.abs(loga-np.log(np.where(upper>0,upper,1)))
        vi=np.where(dl<=du,left,idx)
    vcode=np.where(pos,vi+1,0)
    return h+scale.astype('<f4').tobytes()+vmin.astype('<f4').tobytes()+vmax.astype('<f4').tobytes()+pack4(code)+pack4(vcode)

def decode(blob,c):
    t,n,b,seed,kind=HEADER.unpack_from(blob);off=HEADER.size
    if kind==0:
        vals=np.frombuffer(blob,dtype='<f4',offset=off).copy();return vals[:n],vals[n:],t,seed
    nb=n//b;meta=np.frombuffer(blob,dtype='<f4',count=3*nb,offset=off);off+=12*nb
    scale,lo,hi=meta.reshape(3,nb)
    signed_code=unpack4(blob[off:off+(n+1)//2],n).reshape(nb,b);off+=(n+1)//2
    vc=unpack4(blob[off:],n).reshape(nb,b)
    book=NF4 if c['book']=='nf4' else U15
    signed=(book[signed_code]*scale[:,None]).astype(np.float32)
    sl=np.where(lo>0,lo,1).astype(np.float64);sh=np.where(hi>0,hi,1).astype(np.float64)
    logs=np.log(sl);span=np.log(sh)-logs
    levels=np.exp(logs[:,None]+span[:,None]*np.arange(15)[None,:]/14).astype(np.float32)
    levels[:,0]=lo;levels[:,-1]=hi
    v=np.where(vc==0,0,np.take_along_axis(levels,(vc.astype(np.int16)-1).clip(0),axis=1)).astype(np.float32).ravel()
    m=signed.ravel() if kind==1 else signed.ravel()*denom(v,t)
    assert len(blob)==HEADER.size+12*nb+n
    return m,v,t,seed

def memory(c):
    b=c['block'];codebook=64 if c['book']=='nf4' else 60
    # Flat packed stream; one MiB separately reserved for tensor descriptors,
    # nonuniform-tensor tail packing and 64-byte per-allocation alignment.
    persistent=8*P if c['form']=='fp32' else P+12*math.ceil(P/b)
    workspace=8*2**20;layout_reserve=2**20
    total=persistent+HEADER.size+codebook+workspace+layout_reserve
    return dict(bits_per_parameter=(8*persistent/P),projected_state_GiB=persistent/2**30,projected_peak_GiB=total/2**30,within_30GiB=total<=30*2**30,workspace_reserve_bytes=workspace,tensor_layout_reserve_bytes=layout_reserve)

def audit():
    checks={}
    # Exact operation-order parity, not merely matching dtype.
    for case in ['random','constant','scale_jump','near_cancel']:
        m=v=np.zeros(256,dtype=np.float32);w=np.array([old.F(.1+i/256) for i in range(256)],dtype=np.float32)
        mm=vv=[0.]*256;ww=w.tolist()
        for t,g in enumerate(old.gradients(case,256,32,20261024),1):
            ga=np.array(g,dtype=np.float32);m,v=moments(m,v,ga);d,w=update(m,v,w,t)
            mm,vv=old.moments(mm,vv,g);dd,ww=old.update(mm,vv,ww,t)
            for a,b in [(m,mm),(v,vv),(d,dd),(w,ww)]:assert a.tobytes()==np.array(b,dtype=np.float32).tobytes(),(case,t)
    checks['reference_bit_parity_4_cases_32_steps']=True
    edge={}
    for form in ['mv','qv']:
        for book in ['nf4','u15']:
            c=config('edge',form,book);n=256
            for label,v in [('zeros',np.zeros(n)),('constant',np.ones(n)),('mixed_zero',np.r_[np.zeros(n-1),1]),('subnormal',np.full(n,np.nextafter(np.float32(0),np.float32(1))))]:
                v=v.astype(np.float32);m=np.sqrt(v).astype(np.float32)
                blob=encode(m,v,1,17,c);mm,vv,_,_=decode(blob,c)
                assert np.all(np.isfinite(mm)) and np.all(np.isfinite(vv))
                assert np.array_equal(v==0,vv==0)
                edge[form+'_'+book+'_'+label]=dict(finite=True,zeros_preserved=True,serialized_bytes=len(blob))
    checks['edge_cases']=edge
    checks['nf4_source']='https://raw.githubusercontent.com/bitsandbytes-foundation/bitsandbytes/main/bitsandbytes/functional.py (lookup values only, no downloaded code executed)'
    return checks

def inputs(seed,n):
    r=random.Random(seed)
    diag=np.array([old.F(10**r.uniform(-2,2)) for _ in range(n)],dtype=np.float32)
    target=np.array([old.F(r.uniform(-.05,.05)) for _ in range(n)],dtype=np.float32)
    initial=np.array([old.F(r.uniform(-.05,.05)) for _ in range(n)],dtype=np.float32)
    return diag,target,initial

def loss(w,diag,target):
    return float(np.sum(diag.astype(np.float64)*(w.astype(np.float64)-target.astype(np.float64))**2)/(2*len(w)))

def csvwrite(path,rows):
    with path.open('w',newline='') as f:
        wr=csv.DictWriter(f,fieldnames=list(rows[0]));wr.writeheader();wr.writerows(rows)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',default='results');ap.add_argument('--steps',type=int,default=1024);ap.add_argument('--n',type=int,default=256);ap.add_argument('--replay',action='store_true');args=ap.parse_args()
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True);start=time.perf_counter();checks=audit();(out/'audit.json').write_text(json.dumps(checks,indent=2))
    configs=[config('fp32','fp32')]
    for form in ['mv','qv']:
        for book in ['nf4','u15']:
            for rounding in ['rtn','sr']:
                configs.append(config(f'{form}_{book}_{rounding}_b128',form,book,rounding))
    for b in [32,64,256]:
        for form in ['mv','qv']:
            configs.append(config(f'{form}_nf4_sr_b{b}',form,'nf4','sr',b))
    allrows=[];checkpoints=[];hashes={}
    for seed in [101,202,303]:
        diag,target,initial=inputs(seed,args.n);hashes[str(seed)]=hashlib.sha256(diag.tobytes()+target.tobytes()+initial.tobytes()).hexdigest()
        ref_final=None;ref_w=None;ref_losses={}
        for c in configs:
            w=initial.copy();z=np.zeros(args.n,dtype=np.float32);blob=encode(z,z,0,seed,c)
            initial_loss=loss(w,diag,target);maxerr=0.;min_cos=1.;zerocount=0;first_1pct=None
            for t in range(1,args.steps+1):
                m,v,_,_=decode(blob,c)
                g=diag*(w-target)/np.float32(args.n)
                m,v=moments(m,v,g);d,w=update(m,v,w,t)
                blob=encode(m,v,t,seed,c)
                if t%64==0 or t==args.steps:
                    cur=loss(w,diag,target)
                    if first_1pct is None and cur<=initial_loss*.01:first_1pct=t
                    if c['form']=='fp32':ref_losses[t]=cur
                    checkpoints.append(dict(seed=seed,configuration=c['name'],step=t,loss=cur,loss_fraction_initial=cur/initial_loss,reference_loss=ref_losses[t],loss_ratio=cur/ref_losses[t]))
            cur=loss(w,diag,target)
            if c['form']=='fp32':ref_final=cur;ref_w=w.copy()
            allrows.append(dict(seed=seed,configuration=c['name'],n=args.n,steps=args.steps,**{k:v for k,v in c.items() if k!='name'},initial_loss=initial_loss,final_loss=cur,reference_final_loss=ref_final,final_loss_ratio=cur/ref_final,final_loss_fraction_initial=cur/initial_loss,excess_loss_fraction_initial=(cur-ref_final)/initial_loss,weight_RMS_vs_reference=float(np.sqrt(np.mean((w.astype(np.float64)-ref_w)**2))),checkpoint_to_1pct=first_1pct,serialized_bytes=len(blob),**memory(c)))
            print(seed,c['name'],'ratio',round(cur/ref_final,5),'initial_fraction',round(cur/initial_loss,6),flush=True)
            csvwrite(out/'toy_summary.csv',allrows);csvwrite(out/'toy_checkpoints.csv',checkpoints)
    if args.replay:
        rr=[]
        selected=[c for c in configs if c['name'] in ['fp32','mv_nf4_sr_b64','qv_nf4_sr_b64','mv_nf4_sr_b128','qv_nf4_sr_b128','mv_nf4_rtn_b128','qv_nf4_rtn_b128']]
        for case in ['random','sparse_spikes','scale_jump','near_cancel','constant','alternating']:
            gs=[np.array(g,dtype=np.float32) for g in old.gradients(case,args.n,256,20261024)]
            z=np.zeros(args.n,dtype=np.float32);rm=rv=z;rw=np.array([old.F(.1+i/args.n) for i in range(args.n)],dtype=np.float32);refs=[]
            for t,g in enumerate(gs,1):rm,rv=moments(rm,rv,g);rd,rw=update(rm,rv,rw,t);refs.append((rd.copy(),rw.copy()))
            for c in selected:
                for seed in [17,29,43]:
                    w=np.array([old.F(.1+i/args.n) for i in range(args.n)],dtype=np.float32);blob=encode(z,z,0,seed,c);maxrel=0;mincos=1;zeros=0
                    for t,(g,(rd,rw)) in enumerate(zip(gs,refs),1):
                        m,v,_,_=decode(blob,c);m,v=moments(m,v,g);d,w=update(m,v,w,t);blob=encode(m,v,t,seed,c)
                        df=d.astype(np.float64);rf=rd.astype(np.float64);maxrel=max(maxrel,float(np.linalg.norm(df-rf)/max(np.linalg.norm(rf),1e-30)))
                        mincos=min(mincos,float(np.dot(df,rf)/max(np.linalg.norm(df)*np.linalg.norm(rf),1e-30)))
                    rr.append(dict(case=case,configuration=c['name'],seed=seed,max_update_rel_L2=maxrel,min_update_cosine=mincos,final_weight_RMS=float(np.sqrt(np.mean((w.astype(np.float64)-rw)**2))),**memory(c)))
            print('replay',case,flush=True);csvwrite(out/'replay_summary.csv',rr)
    mem=[]
    for c in configs:
        n=16384;rg=np.random.default_rng(1);m=rg.normal(size=n).astype(np.float32);v=(rg.normal(size=n)**2).astype(np.float32)
        blob=encode(m,v,20,17,c);tracemalloc.start();mm,vv,_,_=decode(blob,c);mm,vv=moments(mm,vv,m);dd,ww=update(mm,vv,m,21);new=encode(mm,vv,21,17,c);_,peak=tracemalloc.get_traced_memory();tracemalloc.stop()
        mem.append(dict(configuration=c['name'],window_parameters=n,serialized_bytes=len(new),tracemalloc_decode_update_encode_peak_bytes=peak,**memory(c)))
    csvwrite(out/'memory.csv',mem)
    run=dict(seconds=time.perf_counter()-start,python=sys.version,numpy=np.__version__,platform=platform.platform(),script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),input_hashes=hashes,steps=args.steps,n=args.n,configs=configs,reference_hyperparameters={'beta1':.9,'beta2':.999,'lr':1e-4,'eps':1e-8,'wd':.01},budget='m/v replacement only, not full-model peak; all global metadata and 8 MiB workspace plus 1 MiB layout reserve included. Flat stream format; actual production tensor tails, allocator, and kernel workspaces unverified.',limitations='No neural-network or NPU training. 3 paired data/init seeds, one stateless-rounding seed per data seed. Ratios describe fixed time horizon and reference objective is not an optimum. All quantized representations are lossy.')
    (out/'run.json').write_text(json.dumps(run,indent=2));print('completed seconds',run['seconds'])
if __name__=='__main__':main()
