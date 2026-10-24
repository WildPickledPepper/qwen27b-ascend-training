"""Independent SOLO-style / ZIP-SR / ZE-EDEN scalar reference experiments.
No third party executable code. Same Adam beta=(.9,.999), lr/epsilon as experiment.py.
"""
import bisect,csv,json,math,struct,time
from pathlib import Path
from experiment import F,pack,unpack,moments,update,gradients,err,P
M=sorted([0.,1.]+[s*x for s in (-1,1) for x in (.0055,.0325,.0775,.2125,.4375,.6625,.8875)])
Q=[0,.00325,.00775,.02125,.04375,.06625,.08875,.15625,.26875,.38125,.49375,.60625,.71875,.83125,.94375,1]
M=list(map(F,M));Q=list(map(F,Q))

def draw(state):
    state ^= state>>12;state ^= (state<<25)&((1<<64)-1);state ^= state>>27
    return state,((state*2685821657736338717)&((1<<64)-1))/2**64

def encode(m,v,kind,rng,t):
    payload=bytearray();n=len(m)
    for state,signed in ((m,True),(v,False)):
        for st in range(0,n,128):
            a=state[st:st+128];scale=F(max(map(abs,a)));x=[z/scale if scale else 0 for z in a]
            if signed:
                codes=[min(range(16),key=lambda i:abs(z-M[i])) for z in x];payload+=pack([scale]);bits=4
            elif kind=='solo42':
                s=sorted(x);pos=.1*(len(s)-1);i=int(pos);quant=s[i]+(s[min(i+1,len(s)-1)]-s[i])*(pos-i)
                # Defined edge policy: all-zero scale=0; constant alpha=1;
                # zero p-quantile in a nonzero block is floored to 1e-30.
                alpha=F(max(quant,1e-30)**(1/3));codes=[]
                for z in x:
                    rng,u=draw(rng)
                    k=round(math.log(max(z,1e-30))/math.log(alpha)+u-.5) if 0<alpha<1 else 0
                    codes.append(max(0,min(3,k)))
                payload+=pack([scale,alpha]);bits=2
            else:
                codes=[]
                for z in x:
                    right=min(15,bisect.bisect_left(Q,z));left=max(0,right-1)
                    if kind=='zip_sr' and left!=right and scale:
                        fun=lambda q:1/(math.sqrt(scale*q/(1-.999**max(1,t)))+1e-8)
                        prob=(fun(z)-fun(Q[right]))/(fun(Q[left])-fun(Q[right]))
                        rng,u=draw(rng);k=left if u<max(0,min(1,prob)) else right
                    else:k=min(range(16),key=lambda i:abs(z-Q[i]))
                    if kind=='ze_eden' and scale:k=max(1,k)
                    codes.append(k)
                if kind=='ze_eden' and scale:
                    c=sum(z*z for z in x)/max(sum(z*Q[k] for z,k in zip(x,codes)),1e-12);scale=F(scale*c)
                payload+=pack([scale]);bits=4
            per=8//bits
            payload+=bytes(sum(codes[i+j]<<(bits*j) for j in range(min(per,len(codes)-i))) for i in range(0,len(codes),per))
    return struct.pack('<QIIQ',t,n,128,rng)+payload

def decode(blob,kind):
    t,n,block,rng=struct.unpack('<QIIQ',blob[:24]);off=24;vals=[]
    for signed in (True,False):
        for st in range(0,n,block):
            count=min(block,n-st);scale=unpack(blob[off:off+4])[0];off+=4;bits=4
            if not signed and kind=='solo42':alpha=unpack(blob[off:off+4])[0];off+=4;bits=2
            per=8//bits;codes=[(blob[off+i//per]>>((i%per)*bits))&((1<<bits)-1) for i in range(count)];off+=(count+per-1)//per
            vals += [F(scale*(M[k] if signed else alpha**k if kind=='solo42' else Q[k])) for k in codes]
    assert off==len(blob)
    return vals[:n],vals[n:],rng

def main():
    out=Path('candidate-results');out.mkdir(exist_ok=True);rows=[];detail=[];started=time.perf_counter()
    for case in ('constant','alternating','scale_jump','near_cancel','sparse_spikes','random'):
        n=128;gs=list(gradients(case,n,256,20261024));rm=rv=[0.0]*n;rw=[F(.1+i/n) for i in range(n)];ref=[]
        for t,g in enumerate(gs,1):rm,rv=moments(rm,rv,g);d,rw=update(rm,rv,rw,t);ref.append((rm,rv,d,rw))
        for kind in ('solo42','zip_sr','ze_eden'):
            for seed in (17,29,43):
                blob=encode([0.0]*n,[0.0]*n,kind,seed,0);w=[F(.1+i/n) for i in range(n)];maxabs=[0.]*4;maxrel=[0.]*4;bits=[0]*4
                for t,(g,rf) in enumerate(zip(gs,ref),1):
                    m,v,rng=decode(blob,kind);m,v=moments(m,v,g);d,w=update(m,v,w,t);blob=encode(m,v,kind,rng,t);m,v,_=decode(blob,kind)
                    ee=[err(x,y) for x,y in zip((m,v,d,w),rf)]
                    for j,(aa,rr,bb) in enumerate(ee):maxabs[j]=max(maxabs[j],aa);maxrel[j]=max(maxrel[j],rr);bits[j]+=bb
                    detail.append([case,kind,seed,t]+[z for e in ee for z in e])
                row={'case':case,'kind':kind,'rounding_seed':seed,'actual_state_bytes':len(blob),'metadata_bytes':24,'codebook_bytes':64 if kind=='solo42' else 128,'codes_bytes':96 if kind=='solo42' else 128,'scale_alpha_bytes':12 if kind=='solo42' else 8,'indices_bytes':0,'fallback_bytes':0,'projected_GiB':((len(blob)-24)/n*P+24+(64 if kind=='solo42' else 128))/2**30}
                for j,name in enumerate(('m','v','update','weight')):row[name+'_max_abs']=maxabs[j];row[name+'_max_relative_L2']=maxrel[j];row[name+'_bit_mismatches']=bits[j]
                rows.append(row)
    with (out/'summary.csv').open('w',newline='',encoding='utf-8') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    with (out/'per_step.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.writer(f);w.writerow(['case','kind','seed','step']+[n+'_'+k for n in ('m','v','update','weight') for k in ('abs','rel_L2','bits')]);w.writerows(detail)
    edge={}
    for kind in ('solo42','zip_sr','ze_eden'):
        for label,v in [('zero',[0.]*128),('constant',[1.]*128),('mixed_zero',[0.]*127+[1.])]:
            b=encode([0.]*128,v,kind,17,1);_,vv,_=decode(b,kind);edge[kind+'_'+label]={'finite':all(math.isfinite(x) for x in vv),'min':min(vv),'max':max(vv),'exact':pack(v)==pack(vv),'bytes':len(b)}
    (out/'edge_cases.json').write_text(json.dumps(edge,indent=2),encoding='utf-8');print('seconds',time.perf_counter()-started);print(json.dumps(edge,indent=2))

if __name__=='__main__':main()
