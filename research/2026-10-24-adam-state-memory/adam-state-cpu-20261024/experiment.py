"""CPU-only, standard-library AdamW representation experiment. No external code.
Run: python experiment.py --n 256 --steps 256 --out results
All arithmetic nodes explicitly round to IEEE binary32; no fused multiply-add.
"""
import argparse, array, csv, hashlib, json, math, platform, random, struct, sys, time, tracemalloc, zlib
from pathlib import Path

F = lambda x: struct.unpack('<f', struct.pack('<f', x))[0]
U = lambda x: struct.unpack('<I', struct.pack('<f', x))[0]
V = lambda x: struct.unpack('<f', struct.pack('<I', x))[0]
pack = lambda xs: struct.pack('<'+'f'*len(xs), *xs)
unpack = lambda b: list(struct.unpack('<'+'f'*(len(b)//4), b))
P = 26895998464
B1, B2, LR, EPS, WD = map(F, (.9,.999,.0001,1e-8,.01))
OM1, OM2 = F(1-.9), F(1-.999)

def moments(m,v,g):
    return ([F(F(B1*x)+F(OM1*y)) for x,y in zip(m,g)],
            [F(F(B2*x)+F(OM2*F(y*y))) for x,y in zip(v,g)])

def update(m,v,w,t):
    # Bias correction: binary64 powers of exact decimal hyperparameters,
    # then round denominators once to FP32. Epsilon outside sqrt.
    c1,c2=F(1-.9**t),F(1-.999**t)
    d=[F(F(x/c1)/F(F(math.sqrt(F(y/c2)))+EPS)) for x,y in zip(m,v)]
    nw=[F(F(x*F(1-F(LR*WD)))-F(LR*y)) for x,y in zip(w,d)]
    return d,nw

def bf(x):
    u=U(x); return V(((u+0x7fff+((u>>16)&1))>>16)<<16)

def shuffle(b): return b''.join(b[i::4] for i in range(4))
def unshuffle(b):
    n=len(b)//4; return bytes(b[k*n+i] for i in range(n) for k in range(4))

def encode(m,v,kind,block=128):
    n=len(m); vals=m+v; raw=pack(vals)
    # All formats have 16 byte header, including step + tensor size.
    h=struct.pack('<QII',0,n,block)
    if kind=='fp32': return h+raw
    if kind=='bit_halves':
        # Two uint16 words per FP32: exactly 4 bytes per state.
        return h+b''.join(struct.pack('<HH',U(x)>>16,U(x)&65535) for x in vals)
    if kind in ('bf16','bf16_residual','bf16_fp32res'):
        out=bytearray(h)
        for x in vals:
            a=bf(x); out+=struct.pack('<H',U(a)>>16)
            if kind=='bf16_residual': out+=struct.pack('<H',U(bf(F(x-a)))>>16)
            if kind=='bf16_fp32res': out+=pack([F(x-a)])
        return bytes(out)
    if kind in ('zlib_raw','zlib_joint','zlib_spatial'):
        if kind=='zlib_joint':
            words=[U(x) for x in m]+[U(x)^U(y) for x,y in zip(m,v)]
            raw=struct.pack('<'+'I'*len(words),*words)
        if kind=='zlib_spatial':
            words=[U(x) for x in vals]; prev=0; delta=[]
            for x in words: delta.append(x^prev); prev=x
            raw=struct.pack('<'+'I'*len(delta),*delta)
        raw=shuffle(raw); comp=zlib.compress(raw,9)
        return h+(b'Z'+comp if len(comp)<len(raw) else b'R'+raw)
    if kind.startswith('q4'):
        out=bytearray(h)
        for state,signed in ((m,True),(v,False)):
            for start in range(0,n,block):
                a=state[start:start+block]; scale=F(max(map(abs,a),default=0))
                if kind=='q4_log' and not signed:
                    pos=[x for x in a if x>0]; lo=F(math.log(min(pos))) if pos else 0.0; hi=F(math.log(max(pos))) if pos else 0.0
                    q=[0 if x==0 else 1+max(0,min(14,round((math.log(x)-lo)/(hi-lo)*14))) if hi>lo else 1 for x in a]
                    out+=pack([lo,hi]); out+=bytes(q[i]|((q[i+1] if i+1<len(q) else 0)<<4) for i in range(0,len(q),2)); continue
                q=[]; rec=[]
                for x in a:
                    if signed:
                        j=max(0,min(14,round(x/scale*7)+7)) if scale else 7
                        y=F((j-7)*scale/7)
                    else:
                        j=max(0,min(15,round(x/scale*15))) if scale else 0
                        y=F(j*scale/15)
                        if kind=='q4_floor' and x>0 and j==0: j=1; y=F(scale/15)
                    q.append(j); rec.append(y)
                # Conservative fallback catches per-element relative errors >1%.
                fallback=kind=='q4_fallback' and any(abs(x-y)>0.01*max(abs(x),1e-30) for x,y in zip(a,rec))
                if kind=='q4_fallback': out+=bytes([int(fallback)])
                if fallback: out+=pack(a); continue
                out+=pack([scale]); out+=bytes(q[i]|((q[i+1] if i+1<len(q) else 0)<<4) for i in range(0,len(q),2))
        return bytes(out)
    if kind=='m_only': return h+pack(m)
    if kind=='u_only': return h+pack([F(x/F(math.sqrt(y)+EPS)) for x,y in zip(m,v)])
    if kind=='u_logv':
        # Two FP32 fields, not one state. Log reconstruction is approximate.
        return h+pack([F(x/F(math.sqrt(y)+EPS)) for x,y in zip(m,v)]+[F(math.log(y)) if y>0 else -math.inf for y in v])
    raise ValueError(kind)

def decode(data,kind):
    _,n,block=struct.unpack('<QII',data[:16]); b=data[16:]
    if kind=='fp32': vals=unpack(b)
    elif kind=='bit_halves': vals=[V((a<<16)|c) for a,c in struct.iter_unpack('<HH',b)]
    elif kind=='bf16': vals=[V(x[0]<<16) for x in struct.iter_unpack('<H',b)]
    elif kind=='bf16_residual': vals=[F(V(a<<16)+V(c<<16)) for a,c in struct.iter_unpack('<HH',b)]
    elif kind=='bf16_fp32res': vals=[F(V(a<<16)+c) for a,c in struct.iter_unpack('<Hf',b)]
    elif kind.startswith('zlib'):
        raw=unshuffle(zlib.decompress(b[1:]) if b[:1]==b'Z' else b[1:])
        words=list(struct.unpack('<'+'I'*(2*n),raw))
        if kind=='zlib_joint': words=words[:n]+[x^y for x,y in zip(words[:n],words[n:])]
        if kind=='zlib_spatial':
            prev=0
            for i,x in enumerate(words): words[i]=x^prev; prev=words[i]
        vals=[V(x) for x in words]
    elif kind.startswith('q4'):
        vals=[]; off=0
        for signed in (True,False):
            for start in range(0,n,block):
                count=min(block,n-start)
                if kind=='q4_log' and not signed:
                    lo,hi=unpack(b[off:off+8]);off+=8
                    q=[(b[off+i//2]>>(4*(i%2)))&15 for i in range(count)];off+=(count+1)//2
                    vals += [0.0 if x==0 else F(math.exp(lo+(x-1)*(hi-lo)/14)) for x in q];continue
                if kind=='q4_fallback':
                    flag=b[off]; off+=1
                    if flag: vals+=unpack(b[off:off+4*count]); off+=4*count; continue
                scale=unpack(b[off:off+4])[0]; off+=4
                q=[(b[off+i//2]>>(4*(i%2)))&15 for i in range(count)]; off+=(count+1)//2
                vals += [F((x-7)*scale/7) if signed else F(x*scale/15) for x in q]
    elif kind=='m_only':
        m=unpack(b); return m,[F(x*x) for x in m]
    elif kind=='u_only':
        # Deliberately false closure v=1; its failure tests information loss.
        return unpack(b),[1.0]*n
    elif kind=='u_logv':
        a=unpack(b); v=[F(math.exp(x)) if math.isfinite(x) else 0.0 for x in a[n:]]
        return [F(x*F(math.sqrt(y)+EPS)) for x,y in zip(a[:n],v)],v
    else: raise ValueError(kind)
    return vals[:n],vals[n:]

def gradients(case,n,steps,seed):
    r=random.Random(seed); base=[F(r.uniform(-1,1)) for _ in range(n)]
    for t in range(steps):
        if case=='constant': a=base
        elif case=='alternating': a=[F(x*(-1 if t%2 else 1)) for x in base]
        elif case=='scale_jump': a=[F(x*(1e-5 if t<steps//3 else 1e3 if t<2*steps//3 else 1e-3)) for x in base]
        elif case=='near_cancel': a=[F(x*((-.9+1e-7) if t%2 else 1)) for x in base]
        elif case=='sparse_spikes': a=[F((100 if i==t%n and t%7==0 else 1e-7)*base[i]) for i in range(n)]
        elif case=='random': a=[F(r.gauss(0,1)*10**r.uniform(-3,1)) for _ in range(n)]
        yield a

def err(a,b):
    dif=[abs(x-y) for x,y in zip(a,b)]
    return max(dif,default=0),math.sqrt(sum(x*x for x in dif)/max(sum(x*x for x in b),1e-60)),sum(U(x)!=U(y) for x,y in zip(a,b))

def counterexamples():
    r=random.Random(917); fail=None; exact=0
    for _ in range(10000):
        x=V(r.randrange(0x00800000,0x7e000000))
        y=F(bf(x)+bf(F(x-bf(x))))
        if U(x)!=U(y) and fail is None: fail={'x':x,'x_hex':hex(U(x)),'reconstructed':y,'reconstructed_hex':hex(U(y))}
        if U(x)==U(y): exact+=1
        assert decode(encode([x],[x],'bit_halves'),'bit_halves')==([x],[x])
    # Same current normalized update (exact FP32) but different next update.
    a=(F(1),F(1)); b=(F(2),F(4)); d1,_=update([a[0]],[a[1]],[0],1); d2,_=update([b[0]],[b[1]],[0],1)
    ma,va=moments([a[0]],[a[1]],[F(.5)]); mb,vb=moments([b[0]],[b[1]],[F(.5)])
    na,_=update(ma,va,[0],2); nb,_=update(mb,vb,[0],2)
    # Histories with identical first moment after two updates (search exact F32).
    m1,v1=moments(*moments([0],[0],[1]),[0])
    m2,v2=moments(*moments([0],[0],[0]),[F(.9)])
    return {'two_bf16_sum':{'equal_count_of_10000':exact,'first_failure':fail},
            'same_current_update_different_future':{'states':[a,b],'current':[d1,d2],'next_gradient':.5,'next':[na,nb], 'current_bit_equal':pack(d1)==pack(d2)},
            'same_m_histories':{'gradients':[[1,0],[0,F(.9)]],'m':[m1,m2],'v':[v1,v2],'m_bit_equal':pack(m1)==pack(m2)}}

def replay_test(n=16,steps=32):
    hist=[]; m=v=[0.0]*n; ok=True
    for g in gradients('random',n,steps,20261024):
        m,v=moments(m,v,g); hist.append(pack(g)); rm=rv=[0.0]*n
        for b in hist: rm,rv=moments(rm,rv,unpack(b))
        ok &= pack(m+v)==pack(rm+rv)
    return {'bit_equal_all_steps':ok,'stored_gradient_bytes':sum(map(len,hist)), 'reference_state_bytes':8*n,'steps':steps,'n':n,'note':'History FP32, no seed/regeneration oracle. Grows 4*N*T; arithmetic O(T^2).'}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--n',type=int,default=256); ap.add_argument('--steps',type=int,default=256); ap.add_argument('--out',default='results'); a=ap.parse_args()
    out=Path(a.out); out.mkdir(parents=True,exist_ok=True)
    kinds=['fp32','bit_halves','zlib_raw','zlib_joint','zlib_spatial','bf16','bf16_residual','bf16_fp32res','q4','q4_floor','q4_log','q4_fallback','m_only','u_only','u_logv']
    cases=['constant','alternating','scale_jump','near_cancel','sparse_spikes','random']; rows=[]; detailed=[]; temporal=[]
    start=time.perf_counter()
    for case in cases:
        gs=list(gradients(case,a.n,a.steps,20261024))
        refs=[]; m=v=[0.0]*a.n; w=[F(.1+i/a.n) for i in range(a.n)]; prev=pack(m+v)
        for t,g in enumerate(gs,1):
            m,v=moments(m,v,g); d,w=update(m,v,w,t); refs.append((m,v,d,w))
            raw=pack(m+v); delta=bytes(x^y for x,y in zip(raw,prev)); enc=zlib.compress(shuffle(delta),9)
            assert bytes(x^y for x,y in zip(unshuffle(zlib.decompress(enc)),prev))==raw
            temporal.append({'case':case,'step':t,'delta_bytes':len(enc),'anchor_bytes':len(prev),'total_bytes':len(enc)+len(prev)+16}); prev=raw
        for kind in kinds:
            data=encode([0.0]*a.n,[0.0]*a.n,kind); w=[F(.1+i/a.n) for i in range(a.n)]
            maxes=[0.0]*8; mismatch=[0]*4; peak=len(data); first=None; before=time.perf_counter()
            for t,(g,ref) in enumerate(zip(gs,refs),1):
                m,v=decode(data,kind); m,v=moments(m,v,g)
                # Use full precision newly computed states for current update,
                # then persist approximation for subsequent recurrence.
                d,w=update(m,v,w,t); data=encode(m,v,kind); sm,sv=decode(data,kind); peak=max(peak,len(data))
                data=struct.pack('<Q',t)+data[8:]
                e=[err(x,y) for x,y in zip((sm,sv,d,w),ref)]
                for i,(abs_e,rel_e,bits) in enumerate(e):
                    maxes[2*i]=max(maxes[2*i],abs_e); maxes[2*i+1]=max(maxes[2*i+1],rel_e); mismatch[i]+=bits
                if first is None and any(ei[2] for ei in e): first=t
                detailed.append([case,kind,t,len(data)]+[q for ei in e for q in ei])
            row={'case':case,'kind':kind,'first_mismatch_step':first,'max_state_bytes':peak,'bytes_per_parameter':peak/a.n,'projected_GiB_including_per_tensor_header':peak/a.n*P/2**30,'seconds':time.perf_counter()-before}
            for i,name in enumerate(('m','v','update','weight')): row[name+'_max_abs']=maxes[2*i]; row[name+'_max_relative_L2']=maxes[2*i+1]; row[name+'_bit_mismatches']=mismatch[i]
            rows.append(row)
        print(case+' complete',flush=True)
    def csvout(name,rr):
        with (out/name).open('w',newline='',encoding='utf-8') as f:
            wr=csv.DictWriter(f,fieldnames=list(rr[0]));wr.writeheader();wr.writerows(rr)
    csvout('summary.csv',rows); csvout('temporal.csv',temporal)
    with (out/'per_step.csv').open('w',newline='',encoding='utf-8') as f:
        wr=csv.writer(f);wr.writerow(['case','kind','step','state_bytes']+[name+'_'+metric for name in ('m','v','update','weight') for metric in ('max_abs','relative_L2','bit_mismatches')]);wr.writerows(detailed)
    # Actual Python allocation peak for codec+decoder; includes interpreter heap,
    # not an NPU workspace claim. Inputs allocated before tracing.
    memory=[]
    for kind in kinds:
        m,v,_,_=refs[-1]; tracemalloc.start(); blob=encode(m,v,kind); dec=decode(blob,kind); cur,pk=tracemalloc.get_traced_memory();tracemalloc.stop()
        memory.append({'kind':kind,'serialized_bytes':len(blob),'bytes_object_allocation':sys.getsizeof(blob),'python_codec_decode_peak_bytes':pk,'decoded_numeric_bytes':8*a.n,'header_bytes':16,'indices_bytes':0,'zlib_native_workspace':'not traced; not measured' if kind.startswith('zlib') else 'N/A'})
    csvout('memory.csv',memory)
    result={'n':a.n,'steps':a.steps,'seed':20261024,'python':sys.version,'platform':platform.platform(),'zlib':zlib.ZLIB_VERSION,'seconds':time.perf_counter()-start,'counterexamples':counterexamples(),'gradient_replay':replay_test(),'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'gradient_sha256':{c:hashlib.sha256(b''.join(pack(g) for g in gradients(c,a.n,a.steps,20261024))).hexdigest() for c in cases}}
    (out/'run.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result['counterexamples'],indent=2)); print('Elapsed',result['seconds'])

if __name__=='__main__':main()
