from pathlib import Path
import csv,hashlib,json,math,os,platform,subprocess,time
import numpy as np
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'results';OUT.mkdir(exist_ok=True)
GRID=[(n,mb) for n in [32,64,128] for mb in [1,4,16]]+[(256,1)]
CASES=[(n,mb,42) for n,mb in GRID]+[(128,4,7)];HORIZONS=[1,4,8];BLOCK=128
FLAGS=['-std=c11','-O2','-fno-fast-math','-ffp-contract=off','-fno-math-errno','-mno-red-zone','-fstack-usage','-Wall','-Wextra','-frecord-gcc-switches']
protocol=dict(grid=GRID,cases=CASES,horizons=HORIZONS,storage_modes={'0':'native BF16 inputs lifted losslessly to FP32 storage; all other factors FP32','1':'native BF16 inputs stored as uint16; all originally FP32 factors remain FP32'},effective_tokens='2*microbatch',tile_size=BLOCK,repeats=3,source_sha256=hashlib.sha256((ROOT/'width_replay.c').read_bytes()).hexdigest(),compiler_flags=FLAGS,scope='Finite capacity/horizon, explicit FP32 scalar K order. Reversed inner-MB K order is a deliberate negative control. Native input data is identical under the two storage choices; no FP32 factor quantization.',raw_output_policy='All raw outputs retained; identical byte sequences are hard-linked after SHA256+byte equality to avoid redundant archive payload.')
(OUT/'protocol.json').write_text(json.dumps(protocol,indent=2));builds={};rows=[];parities=[];controls=[];seen={};start=time.perf_counter()
def dedup(path):
 raw=path.read_bytes();h=hashlib.sha256(raw).hexdigest();key=(len(raw),h)
 if key in seen and seen[key]!=path and seen[key].read_bytes()==raw:path.unlink();os.link(seen[key],path)
 else:seen[key]=path
 return h

def build(n,mb,mixed):
 key=(n,mb,mixed)
 if key in builds:return builds[key]
 d=ROOT/'build'/f'n{n}_mb{mb}_x{mixed}';d.mkdir(parents=True,exist_ok=True);binary=d/'width_replay'
 args=['gcc',*FLAGS,f'-DD={n}u',f'-DH={n}u',f'-DKOUT={n}u',f'-DMB={mb}u',f'-DMIXED_X={mixed}',str(ROOT/'width_replay.c'),'-lm','-o',str(binary)]
 p=subprocess.run(args,text=True,capture_output=True);(d/'build.log').write_text(p.stdout+p.stderr);assert p.returncode==0 and not p.stderr,p.stderr
 (d/'compile_command.json').write_text(json.dumps(args,indent=2));su={}
 for line in (d/'width_replay.su').read_text().splitlines():
  loc,size,qual=line.split('\t');su[loc.rsplit(':',1)[-1]]=int(size)
 stack=su['main']+su['algorithm']+max(su[x] for x in ['capture','reconstruct','update_weight'])
 with (d/'disassembly.txt').open('w') as f:subprocess.run(['objdump','-drwC',str(binary)],stdout=f,check=True)
 info={'binary':str(binary),'stack':stack,'frames':su,'sha256':hashlib.sha256(binary.read_bytes()).hexdigest()};(d/'provenance.json').write_text(json.dumps(info,indent=2));builds[key]=info;return info

def invoke(n,mb,mixed,seed,t,mode,cap=None):
 cap=t if cap is None else cap;bi=build(n,mb,mixed);tag=f'n{n}_mb{mb}_x{mixed}_s{seed}_t{t}_m{mode}_c{cap}';prefix=OUT/tag
 args=[bi['binary'],str(mode),str(seed),str(t),str(BLOCK),str(cap),str(bi['stack']),'0',str(prefix),'3']
 p=subprocess.run(args,text=True,capture_output=True,check=True);q=json.loads(p.stdout)
 q.update(width=n,microbatch=mb,tokens=2*mb,mixed_x=mixed,tag=tag,command=args)
 assert q['status']=='complete' and q['completed_steps']==t
 P=2*n*n+2*n;record=16+(2*mb*n*(2 if mixed else 4))+2*mb*3*n*4+2*n*4
 assert q['record_bytes']==record and q['parameters']==P and q['history_reserved_bytes']==record*(cap if mode else 1)
 assert q['original_fp32_factor_bytes_per_step']==4*(2*mb*3*n+2*n)
 assert q['native_input_storage_bits']==(16 if mixed else 32)
 assert q['optimizer_replacement_bytes']==q['malloc_usable_bytes']+16+296+bi['stack']-4*P
 assert q['moment_transitions']==P*(t*(t+1)//2 if mode else t)
 raw={}
 for ext,size in [('states',12*P),('gradients',4*P),('factors',record)]:
  path=prefix.with_suffix('.'+ext);assert path.stat().st_size==size*t;raw[ext]=path.read_bytes();q[ext+'_sha256']=dedup(path)
 assert np.isfinite(np.frombuffer(raw['states'],dtype='<f4')).all()
 q['optimizer_fraction_raw_m_v']=q['optimizer_replacement_bytes']/(8*P);q['all_state_fraction_raw_w_m_v']=q['charged_algorithm_bytes']/(12*P)
 prefix.with_suffix('.json').write_text(json.dumps(q,indent=2));return q,raw

def decoded_factors(raw,n,mb,mixed,t):
 per=16+2*mb*n*(2 if mixed else 4)+2*mb*3*n*4+8*n;ans=[]
 for step in range(t):
  b=raw[step*per:(step+1)*per];head=np.frombuffer(b[:16],dtype='<u4').copy();xlen=2*mb*n*(2 if mixed else 4)
  if mixed:x=(np.frombuffer(b[16:16+xlen],dtype='<u2').astype('<u4')<<16).tobytes()
  else:x=b[16:16+xlen]
  ans.append(head.tobytes()+x+b[16+xlen:])
 return b''.join(ans)

def save():
 for name,data in [('frontier',rows),('dtype_parity',parities),('reduction_controls',controls)]:
  (OUT/(name+'.json')).write_text(json.dumps(data,indent=2))
  if data:
   with (OUT/(name+'.csv')).open('w',newline='') as fp:
    w=csv.DictWriter(fp,fieldnames=sorted(set().union(*(q.keys() for q in data))));w.writeheader();w.writerows(data)

for n,mb,seed in CASES:
 refstorage={}
 for mixed in [0,1]:
  ref,raw=invoke(n,mb,mixed,seed,8,0);rows.append(ref);refstorage[mixed]=(ref,raw)
  for t in HORIZONS:
   q,actual=invoke(n,mb,mixed,seed,t,2);P=q['parameters'];expected={ext:raw[ext][:t*size] for ext,size in [('states',12*P),('gradients',4*P),('factors',q['record_bytes'])]}
   assert actual==expected,(n,mb,mixed,seed,t)
   q['same_order_raw_exact']=True;q['words_compared']=sum(map(len,actual.values()))//4;q['reference8_mean_ns']=ref['nanoseconds_mean'];rows.append(q)
  save()
 # Strong cross-storage check: same numeric factors, gradients and trajectory, not two different numeric training references.
 a,b=refstorage[0][1],refstorage[1][1];assert a['states']==b['states'] and a['gradients']==b['gradients']
 assert decoded_factors(a['factors'],n,mb,0,8)==decoded_factors(b['factors'],n,mb,1,8)
 parities.append(dict(width=n,microbatch=mb,seed=seed,steps=8,all_weight_m_v_gradient_bits_equal=True,all_decoded_factor_bits_equal=True,original_fp32_factors_unchanged=True,record_bytes_fp32=refstorage[0][0]['record_bytes'],record_bytes_mixed=refstorage[1][0]['record_bytes']))
 # One clear positive no-order-change case, and two deliberately changed-order controls, on a moderate shape.
 if n==64 and mb in [1,4,16]:
  for mixed in [0,1]:
   q,actual=invoke(n,mb,mixed,seed,4,3);ref,raw=refstorage[mixed];P=q['parameters'];expected={ext:raw[ext][:4*size] for ext,size in [('states',12*P),('gradients',4*P),('factors',q['record_bytes'])]}
   c=dict(width=n,microbatch=mb,mixed_x=mixed,steps=4,seed=seed,reverse_K=True)
   for ext in ['states','gradients']:
    av=np.frombuffer(actual[ext],dtype='<u4');bv=np.frombuffer(expected[ext],dtype='<u4');c[ext+'_different_words']=int(np.count_nonzero(av!=bv));c[ext+'_total_words']=len(av)
   c['first_step_gradient_different_words']=int(np.count_nonzero(np.frombuffer(actual['gradients'][:P*4],dtype='<u4')!=np.frombuffer(expected['gradients'][:P*4],dtype='<u4')))
   assert (c['gradients_different_words']==0 and c['states_different_words']==0) if mb==1 else (c['first_step_gradient_different_words']>0 and c['states_different_words']>0)
   controls.append(c);rows.append(q)
 save();print('done',n,mb,seed,flush=True)
summary=dict(builds=len(builds),same_order_candidates=sum(q.get('same_order_raw_exact',False) for q in rows),same_order_words=sum(q.get('words_compared',0) for q in rows),dtype_parity_cases=len(parities),reduction_controls=controls,finite_horizon_under10pct=[{k:q[k] for k in ['width','tokens','mixed_x','requested_steps','capacity_steps','optimizer_replacement_bytes','raw_FP32_m_v_bytes','optimizer_fraction_raw_m_v','charged_algorithm_bytes']} for q in rows if q.get('same_order_raw_exact') and q['optimizer_fraction_raw_m_v']<.1],seconds=time.perf_counter()-start)
(OUT/'summary.json').write_text(json.dumps(summary,indent=2));save();print('DONE',json.dumps(summary),flush=True)
