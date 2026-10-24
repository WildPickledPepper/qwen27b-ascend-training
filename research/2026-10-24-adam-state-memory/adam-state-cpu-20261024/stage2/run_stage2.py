"""Stage 2: serialized m/v ablations, closed-loop convex optimization, capped protection.
Run from parent research directory: python stage2/run_stage2.py
CPU stdlib only. No third-party execution, no model or remote access.
"""
import argparse,csv,hashlib,heapq,json,math,random,struct,sys,time,tracemalloc
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import experiment as e
import candidates as c
F=e.F
OUT=Path(__file__).resolve().parent

def store(m,v,kind,mode,rng,t):
    if kind=='fp32':return struct.pack('<QIIQ',t,len(m),128,rng)+e.pack(m+v)
    blob=c.encode(m,v,kind,rng,t);n=len(m);msize=n//2+4*((n+127)//128)
    a=blob[24:24+msize] if mode in ('m','both') else e.pack(m)
    b=blob[24+msize:] if mode in ('v','both') else e.pack(v)
    return blob[:24]+a+b

def restore(blob,kind,mode):
    t,n,block,rng=struct.unpack('<QIIQ',blob[:24]);off=24;vals=[]
    if kind=='fp32':a=e.unpack(blob[24:]);return a[:n],a[n:],rng
    for name in ('m','v'):
        if mode not in (name,'both'):vals+=e.unpack(blob[off:off+4*n]);off+=4*n;continue
        for st in range(0,n,block):
            count=min(block,n-st);scale=e.unpack(blob[off:off+4])[0];off+=4;bits=4
            if name=='v' and kind=='solo42':alpha=e.unpack(blob[off:off+4])[0];off+=4;bits=2
            per=8//bits;codes=[(blob[off+i//per]>>((i%per)*bits))&((1<<bits)-1) for i in range(count)];off+=(count+per-1)//per
            vals += [F(scale*(c.M[k] if name=='m' else alpha**k if kind=='solo42' else c.Q[k])) for k in codes]
    assert off==len(blob);return vals[:n],vals[n:],rng

def pct(a,p):
    s=sorted(a);return s[min(len(s)-1,int((len(s)-1)*p))] if s else 0.0

def metrics(d,rd,w,rw,sv,rawv):
    abses=[abs(x-y) for x,y in zip(d,rd)];na=sum(x*x for x in d);nb=sum(x*x for x in rd)
    cosine=sum(x*y for x,y in zip(d,rd))/math.sqrt(na*nb) if na*nb>0 else (1.0 if na==nb else 0.0)
    wd=[abs(x-y) for x,y in zip(w,rw)]
    return {'update_rel_L2':math.sqrt(sum(x*x for x in abses)/max(nb,1e-60)),
        'update_cosine':max(-1,min(1,cosine)), 'update_abs_max':max(abses),
        'update_abs_p50':pct(abses,.5),'update_abs_p95':pct(abses,.95),'update_abs_p99':pct(abses,.99),
        'coordinate_relative_error_p99':pct([a/max(abs(b),1e-8) for a,b in zip(abses,rd)],.99),
        'v_zeroized_coordinates':sum(x==0 and y>0 for x,y in zip(sv,rawv)),
        'v_zero_coordinates':sum(x==0 for x in sv),
        'weight_abs_max':max(wd),'weight_RMS':math.sqrt(sum(x*x for x in wd)/len(wd))}

def agg(rows):
    ans={}
    for k in rows[0]:
        if k.startswith(('update_','coordinate_','weight_')):
            ans[k+('_min_over_steps' if k=='update_cosine' else '_max_over_steps')]=(min if k=='update_cosine' else max)(r[k] for r in rows)
        if k.startswith('v_'):ans[k+'_sum_coordinate_steps']=sum(r[k] for r in rows)
    ans['final_weight_RMS']=rows[-1]['weight_RMS'];ans['final_weight_abs_max']=rows[-1]['weight_abs_max']
    return ans

def csvsave(name,rows):
    with (OUT/name).open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def replay():
    rows=[];summary=[];n=256;steps=256
    for case in ('constant','alternating','scale_jump','near_cancel','sparse_spikes','random'):
        gs=list(e.gradients(case,n,steps,20261024));rm=rv=[0.]*n;rw=[F(.1+i/n) for i in range(n)];refs=[]
        for t,g in enumerate(gs,1):rm,rv=e.moments(rm,rv,g);rd,rw=e.update(rm,rv,rw,t);refs.append((rd,rw))
        for kind in ('solo42','zip_sr','ze_eden'):
            for mode in ('m','v','both'):
                for seed in (17,29,43):
                    blob=store([0.]*n,[0.]*n,kind,mode,seed,0);w=[F(.1+i/n) for i in range(n)];rr=[]
                    for t,(g,(rd,rw)) in enumerate(zip(gs,refs),1):
                        m,v,rng=restore(blob,kind,mode);m,v=e.moments(m,v,g);d,w=e.update(m,v,w,t);blob=store(m,v,kind,mode,rng,t);sm,sv,_=restore(blob,kind,mode)
                        x={'case':case,'kind':kind,'mode':mode,'rounding_seed':seed,'step':t,**metrics(d,rd,w,rw,sv,v)};rows.append(x);rr.append(x)
                    summary.append({'case':case,'kind':kind,'mode':mode,'rounding_seed':seed,'n':n,'steps':steps,'actual_state_bytes':len(blob),'header_rng_bytes':24,'indices_bytes':0,'codebook_bytes':64 if kind=='solo42' else 128,**agg(rr)})
        print('replay',case,flush=True)
    csvsave('replay_per_step.csv',rows);csvsave('replay_summary.csv',summary)

def toy():
    rows=[];summary=[];n=256;steps=1024;fingerprints={}
    for seed in (101,202,303):
        r=random.Random(seed);diag=[F(10**r.uniform(-2,2)) for _ in range(n)];target=[F(r.uniform(-.05,.05)) for _ in range(n)];initial=[F(r.uniform(-.05,.05)) for _ in range(n)]
        fingerprints[str(seed)]=hashlib.sha256(e.pack(diag+target+initial)).hexdigest()
        # Objective 1/(2N) sum a_i (w_i-target_i)^2. Exact gradient then F32.
        grad=lambda w:[F(a*F(x-y)/n) for a,x,y in zip(diag,w,target)]
        loss=lambda w:sum(a*(x-y)**2 for a,x,y in zip(diag,w,target))/(2*n)
        rm=rv=[0.]*n;rw=initial[:];refs=[]
        for t in range(1,steps+1):rm,rv=e.moments(rm,rv,grad(rw));rd,rw=e.update(rm,rv,rw,t);refs.append((rd,rw,loss(rw)))
        for kind,mode in [('fp32','both')]+[(k,m) for k in ('solo42','zip_sr','ze_eden') for m in ('m','v','both')]:
            blob=store([0.]*n,[0.]*n,kind,mode,seed,0);w=initial[:];rr=[];first_1pct=None
            for t,(rd,rw,ref_loss) in enumerate(refs,1):
                g=grad(w);m,v,rng=restore(blob,kind,mode);m,v=e.moments(m,v,g);d,w=e.update(m,v,w,t);blob=store(m,v,kind,mode,rng,t);sm,sv,_=restore(blob,kind,mode)
                cur=loss(w)
                if first_1pct is None and cur<=.01*loss(initial):first_1pct=t
                x={'data_init_seed':seed,'kind':kind,'mode':mode,'step':t,'loss':cur,'reference_loss':ref_loss,**metrics(d,rd,w,rw,sv,v)};rows.append(x);rr.append(x)
            summary.append({'data_init_seed':seed,'kind':kind,'mode':mode,'n':n,'steps':steps,'initial_loss':loss(initial),'final_loss':cur,'reference_final_loss':ref_loss,'final_loss_ratio':cur/ref_loss,'step_to_1pct_initial_loss':first_1pct,'actual_state_bytes':len(blob),**agg(rr)})
        print('toy',seed,flush=True)
    csvsave('toy_per_step.csv',rows);csvsave('toy_summary.csv',summary)
    (OUT/'toy_inputs_sha256.json').write_text(json.dumps(fingerprints,indent=2),encoding='utf-8')

def protect():
    # Store the entire compact baseline plus explicit uint32 block index +
    # two FP32 moment blocks for selected exceptions. No uncharged cache.
    # Repeat observed block density over P parameters, add global metadata,
    # shared codebooks and conservative fixed 1 MiB implementation allowance.
    n=16384;steps=48;window=128;blocks=n//128;rows=[];summary=[]
    for kind in ('zip_sr','solo42'):
        base=store([0.]*n,[0.]*n,kind,'both',17,0);payload=len(base)-24;codebook=64 if kind=='solo42' else 128
        # Peak includes BOTH old/new exception sets, indices in each set,
        # 128 bytes per selected block for bounded top-k heap accounting,
        # and an 8 MiB fixed window/codec allowance. Actual CPU heap is audited
        # separately; this remains a storage design budget, not NPU measurement.
        fixed=24+codebook+16+8*2**20
        projected=lambda k:(payload/n*e.P)+math.ceil((k/blocks)*math.ceil(e.P/window))*(2*(4+8*window)+128)+fixed
        cap=max(k for k in range(blocks+1) if projected(k)<=30*2**30)
        for case in ('sparse_spikes','random'):
            for protected in (False,True):
                blob=base;exceptions=struct.pack('<Q',0);w=[F(.1+i/n) for i in range(n)];rm=rv=[0.]*n;rw=w[:];rr=[];peak=0
                for t,g in enumerate(e.gradients(case,n,steps,20261024),1):
                    m,v,rng=restore(blob,kind,'both')
                    for off in range(8,len(exceptions),1028):
                        bi=struct.unpack('<I',exceptions[off:off+4])[0];data=e.unpack(exceptions[off+4:off+1028]);m[bi*128:(bi+1)*128]=data[:128];v[bi*128:(bi+1)*128]=data[128:]
                    m,v=e.moments(m,v,g);d,w=e.update(m,v,w,t);rm,rv=e.moments(rm,rv,g);rd,rw=e.update(rm,rv,rw,t)
                    blob=store(m,v,kind,'both',rng,t);sm,sv,_=restore(blob,kind,'both')
                    def scores():
                        for bi in range(blocks):
                            sl=slice(bi*128,(bi+1)*128)
                            dd,_=e.update(sm[sl],sv[sl],[0.]*128,t);truth=d[sl]
                            yield (e.err(dd,truth)[1],bi)
                    chosen=[bi for score,bi in heapq.nlargest(cap,scores())] if protected else []
                    exceptions=struct.pack('<Q',len(chosen))+b''.join(struct.pack('<I',bi)+e.pack(m[bi*128:(bi+1)*128]+v[bi*128:(bi+1)*128]) for bi in sorted(chosen))
                    assert len(exceptions)==8+len(chosen)*1028
                    for bi in chosen:sm[bi*128:(bi+1)*128]=m[bi*128:(bi+1)*128];sv[bi*128:(bi+1)*128]=v[bi*128:(bi+1)*128]
                    persistent=len(blob)+len(exceptions)+codebook;peak=max(peak,persistent)
                    x={'case':case,'kind':kind,'protected':protected,'step':t,'n':n,'protected_blocks':len(chosen),'protected_parameter_fraction':len(chosen)/blocks,'index_bytes':4*len(chosen),'fp32_exception_bytes':1024*len(chosen),'persistent_bytes_including_codebook':persistent,'projected_peak_GiB_including_workspaces':projected(len(chosen))/2**30,**metrics(d,rd,w,rw,sv,v)};rr.append(x);rows.append(x)
                summary.append({'case':case,'kind':kind,'protected':protected,'n':n,'steps':steps,'max_protected_blocks':cap if protected else 0,'max_protected_fraction':cap/blocks if protected else 0,'peak_persistent_bytes':peak,'projected_peak_GiB_including_workspaces':projected(cap if protected else 0)/2**30,**agg(rr)})
                print('protection',kind,case,protected,flush=True)
    csvsave('protection_per_step.csv',rows);csvsave('protection_summary.csv',summary)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--part',choices=['all','replay','toy','protect'],default='all');a=ap.parse_args();t=time.perf_counter()
    for name,fn in [('replay',replay),('toy',toy),('protect',protect)]:
        if a.part in ('all',name):fn()
    config={'part':a.part,'elapsed_seconds':time.perf_counter()-t,'reference':'experiment.py fixed lr=1e-4, betas=.9/.999, eps=1e-8, wd=.01; FP32 master weights','units':'vector step is one small N-dimensional CPU optimizer update, NOT a 27B training step; p99 over coordinates then max over steps; cosine over full vector then min over steps','toy':'closed-loop exact diagonal convex quadratic gradient, distinct from fixed-gradient replay; same data/init/Adam hyperparameters for each paired configuration','protection':'budget includes compact payload, two FP32 exception sets and their uint32 indices, 128 bytes per selected top-k heap entry, 8 MiB fixed window/codec allowance. Prototype decodes full toy vector, not full model. No measured NPU peak or production streaming kernel. format_only_* files are SUPERSEDED earlier budgets that omit transition scratch.','script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (OUT/('run_'+a.part+'.json')).write_text(json.dumps(config,indent=2),encoding='utf-8');print('completed',config['elapsed_seconds'])

if __name__=='__main__':main()
