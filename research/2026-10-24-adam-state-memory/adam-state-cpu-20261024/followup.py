"""Boundary tests and actual fixed-window state storage; stdlib only."""
import csv, ctypes, json, math, random, struct, time, tracemalloc
from pathlib import Path
from experiment import F,U,V,bf,pack,unpack,moments,update,encode,decode,gradients,P

OUT=Path('followup-results');OUT.mkdir(exist_ok=True)

def ratio_test(beta,gs,quant=False):
    b=F(beta); om=F(1-beta); m=v=0.0; rm=y=0.0; fallback_v=None; rows=[]
    code=[.25,.5,.75,1,1.5,2,3,4,6,8,12,16,24,32,48,64]
    for t,g in enumerate(gs,1):
        g=F(g);m=F(F(b*m)+F(om*g));v=F(F(b*v)+F(om*F(g*g)))
        oldm=rm;rm=F(F(b*rm)+F(om*g))
        # Singular fallback stores v in y's 4-byte slot plus a 1-byte tag.
        # Recover v on the next step before returning to ratio coordinates.
        if fallback_v is not None:
            rv=F(F(b*fallback_v)+F(om*F(g*g)))
            y=F(F(om/b)*F(F(rv/F(rm*rm))-1)) if rm else 0.0
        elif rm==0:
            oldv=F(F(oldm*oldm)*F(1+F(F(b*y)/om)))
            rv=F(F(b*oldv)+F(om*F(g*g)))
        else:
            x=F(oldm/rm); y=F(F(F(b*F(x*x))*y)+F(F(1-x)*F(1-x)))
            rv=F(F(rm*rm)*F(1+F(F(b*y)/om)))
        fallback_v=rv if rm==0 else None
        if quant and fallback_v is None:
            y=min(code,key=lambda z:abs(z-y));rv=F(F(rm*rm)*F(1+F(F(b*y)/om)))
        c=F(1-beta**t)
        du=F(F(m/c)/F(F(math.sqrt(F(v/c)))+F(1e-8)))
        ru=F(F(rm/c)/F(F(math.sqrt(max(0,F(rv/c))))+F(1e-8)))
        rows.append({'step':t,'g':g,'m':m,'v_reference':v,'v_ratio':rv,'y':None if rm==0 else y,'fallback':rm==0,'v_bits_equal':U(v)==U(rv),'update_abs_error':abs(du-ru),'serialized_bytes_scalar':9 if not quant or rm==0 else 6})
    return rows

def main():
    boundaries=[]
    for hi in (0x0000,0x8000,0x3f80,0x3f81,0xbf80,0xbf81):
        for lo in (0,0x7fff,0x8000,0x8001):
            raw=(hi<<16)|lo; shifted=(raw+32768)&0xffffffff
            recovered=(shifted-32768)&0xffffffff
            halfup=shifted>>16; rne=U(bf(V(raw)))>>16
            boundaries.append({'raw':hex(raw),'restored_exact':recovered==raw,'halfup_hi':hex(halfup),'rne_hi':hex(rne),'forward_equal':halfup==rne})
    # v's sign cannot be tested with v<0 to retain a -0 tie marker.
    markers={'positive_zero_bits':hex(U(0.0)),'negative_zero_bits':hex(U(-0.0)),'negative_zero_less_than_zero':-0.0<0,'abs_negative_zero_bits':hex(U(abs(-0.0))),'unsigned_v_drops_sign_marker':True}
    rng=random.Random(20261024)
    rr={}
    for b in (.5,.9,.999):
        gs=[rng.gauss(0,1) for _ in range(1024)]
        for q in (False,True):rr[f'beta{b}_quant{q}']=ratio_test(b,gs,q)
    rr['singularity_exact']=ratio_test(.5,[1,-.5,1])
    # Snapshot lossless compression on larger arrays; full tensor and tiled sizes.
    snaps=[]
    for case in ('constant','sparse_spikes','random'):
        n=8192;m=v=[0.0]*n
        for g in gradients(case,n,32,20261024):m,v=moments(m,v,g)
        for kind in ('zlib_raw','zlib_joint','zlib_spatial'):
            blob=encode(m,v,kind);assert pack(sum(decode(blob,kind),[]))==pack(m+v)
            blocks=[encode(m[i:i+128],v[i:i+128],kind) for i in range(0,n,128)]
            # Variable blocks require offsets, counted as uint64, plus global header.
            tiled=sum(map(len,blocks))+8*(len(blocks)+1)+16
            snaps.append({'case':case,'kind':kind,'n':n,'full_bytes':len(blob),'full_bytes_per_param':len(blob)/n,'tiled_bytes_with_offsets':tiled,'tiled_bytes_per_param':tiled/n,'max_decode_numeric_window_bytes':1024})
    streaming=[]
    for kind in ('q4','q4_floor','q4_log'):
        n=8192;window=128;blocks=n//window;stride=len(encode([0.0]*window,[0.0]*window,kind))-16
        state=bytearray(struct.pack('<QII',0,n,window)+b''.join(encode([0.0]*window,[0.0]*window,kind)[16:] for _ in range(blocks)))
        reference=[encode([0.0]*window,[0.0]*window,kind) for _ in range(blocks)]
        initial=len(state);tracemalloc.start();start=time.perf_counter();match=True
        for t in range(1,17):
            for bi in range(blocks):
                off=16+bi*stride;header=struct.pack('<QII',t,window,window)
                m,v=decode(header+state[off:off+stride],kind)
                g=[F(math.sin((bi*window+i+1)*.17+t)) for i in range(window)]
                m,v=moments(m,v,g);blob=encode(m,v,kind)
                state[off:off+stride]=blob[16:]
                am,av=decode(reference[bi],kind);am,av=moments(am,av,g);reference[bi]=encode(am,av,kind)
                match &= reference[bi][16:]==state[off:off+stride]
            state[:8]=struct.pack('<Q',t)
        _,pk=tracemalloc.get_traced_memory();tracemalloc.stop()
        streaming.append({'kind':kind,'n':n,'window':window,'persistent_bytes':len(state),'metadata_bytes':16,'indices_bytes':0,'scale_bytes':blocks*(12 if kind=='q4_log' else 8),'codes_bytes':n,'extra_residual_bytes':0,'decode_pair_numeric_bytes':8*window,'two_old_new_pair_buffers_bytes':16*window,'python_peak_including_audit_reference_changes':pk,'inplace_size_unchanged':len(state)==initial,'window_matches_block_reference':match,'projected_state_GiB':((len(state)-16)/n*P+16)/2**30,'seconds':time.perf_counter()-start})
    def writecsv(name,rows):
        with (OUT/name).open('w',newline='',encoding='utf-8') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    writecsv('master_boundaries.csv',boundaries);writecsv('compression_snapshots.csv',snaps);writecsv('streaming.csv',streaming)
    for k,rows in rr.items():writecsv('ratio_'+k+'.csv',rows)
    summary={'master_recovery_all':all(x['restored_exact'] for x in boundaries),'master_forward_mismatches':sum(not x['forward_equal'] for x in boundaries),'v_zero_marker':markers,'ratio':{k:{'v_mismatch_steps':sum(not x['v_bits_equal'] for x in r),'max_update_abs_error':max(x['update_abs_error'] for x in r),'fallback_steps':sum(x['fallback'] for x in r)} for k,r in rr.items()},'streaming':streaming}
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8');print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
