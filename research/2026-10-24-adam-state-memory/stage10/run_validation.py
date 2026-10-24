"""Declared finite-horizon test protocol, raw outputs and machine-readable evidence."""
from pathlib import Path
import csv,hashlib,json,math,platform,subprocess,sys,time
import numpy as np
from independent_oracle import oracle
ROOT=Path(__file__).resolve().parent;BIN=ROOT/'build/formal/factor_replay';OUT=ROOT/'formal_results';OUT.mkdir(exist_ok=True)
SEEDS=[0,1,42,2026,2147483647];STEPS=[1,2,4,8,16];BLOCKS={1:[128,511],2:[1,32,128,511,6288]};CAPS=[0,1,2,4,8,15]
P=6288;RECORD=10320;su={}
for line in (ROOT/'build/formal/factor_replay.su').read_text().splitlines():
 loc,size,qual=line.split('\t');su[loc.rsplit(':',1)[-1]]=int(size)
STACK=su['main']+su['algorithm']+max(su[x] for x in ['capture','reconstruct','update_weight'])
protocol=dict(seeds=SEEDS,steps=STEPS,blocks_by_mode=BLOCKS,capacity_rejection=CAPS,capacity_seeds=[0,42],oracle_seeds=SEEDS,repeats_reference=15,repeats_candidates=3,static_stack_frames=su,charged_live_call_chain=STACK,core_source_unchanged=True,source_sha256=hashlib.sha256((ROOT/'factor_replay.c').read_bytes()).hexdigest(),binary_sha256=hashlib.sha256(BIN.read_bytes()).hexdigest(),compile_command=(ROOT/'build/formal/compile_command.txt').read_text().strip(),python=sys.version,numpy=np.__version__,platform=platform.platform(),scope='CPU finite-horizon scalar-FP32 mechanism; no hardware GEMM/NPU/Qwen equivalence claim. All memory includes capacity reserved, not just used. Current weights separated but not omitted. Validation buffers excluded from measured algorithm.',budget_fractions_of_raw_m_v=[.1,.15,.25,.5,.75,.99],budget_edge='Every complete candidate preflight-1 rejects; preflight+65536 accepts. Actual malloc capacity checked after allocation.')
# Protocol exists before any formal executable invocation.
(OUT/'protocol.json').write_text(json.dumps(protocol,indent=2));rows=[];budgets=[];parity=[];refs={};started=time.perf_counter()

def invoke(mode,seed,t,b,cap,budget=0,dump=True,repeats=3,tag=None):
 tag=tag or f's{seed}_t{t}_m{mode}_b{b}_c{cap}'
 prefix=OUT/tag
 args=[str(BIN),str(mode),str(seed),str(t),str(b),str(cap),str(STACK),str(budget),str(prefix) if dump else '-',str(repeats)]
 proc=subprocess.run(args,capture_output=True,text=True,check=True)
 with (OUT/'commands.jsonl').open('a') as fp:fp.write(json.dumps(dict(args=args,returncode=proc.returncode,stdout=proc.stdout,stderr=proc.stderr))+'\n')
 d=json.loads(proc.stdout);d['tag']=tag;d['budget_bytes']=budget
 if dump:prefix.with_suffix('.json').write_text(json.dumps(d,indent=2))
 return d,prefix

def outputs(prefix):return {s:prefix.with_suffix('.'+s).read_bytes() for s in ['states','gradients','factors']}
def save():
 for name,data in [('curve',rows),('budget_screen',budgets),('oracle_parity',parity)]:
  (OUT/(name+'.json')).write_text(json.dumps(data,indent=2))
  if data:
   keys=sorted(set().union(*(x.keys() for x in data)))
   with (OUT/(name+'.csv')).open('w',newline='') as fp:
    writer=csv.DictWriter(fp,fieldnames=keys);writer.writeheader();writer.writerows(data)

def check_accounting(d):
 mode=d['mode'];b=d['block'];cap=d['capacity_steps'];done=d['completed_steps']
 assert d['record_bytes']==RECORD and d['parameters']==P and d['current_weight_bytes']==4*P
 assert d['raw_FP32_m_v_bytes']==8*P and d['bias_vector_bytes_per_step']==4*144
 assert d['history_reserved_bytes']==RECORD*(cap if mode else 1)
 assert d['history_used_bytes']==RECORD*(done if mode else 1)
 assert d['unused_history_bytes_charged']==(cap-done)*RECORD if mode else d['unused_history_bytes_charged']==0
 assert d['workspace_bytes']==(4*(2*b+(P if mode==1 else b)) if mode else 12*P)
 assert d['request_bytes']==d['header_bytes']+4*P+d['history_reserved_bytes']+d['workspace_bytes']
 assert d['charged_algorithm_bytes']==d['malloc_usable_bytes']+16+d['constant_reserve_bytes']+STACK
 assert d['optimizer_replacement_bytes']==d['charged_algorithm_bytes']-4*P
 assert d['capture_calls']==done
 transitions=P*(done*(done+1)//2 if mode else done)
 assert d['moment_transitions']==transitions
 rebuild=(math.ceil(P/b)*done*(done+1)//2) if mode else done
 assert d['rebuild_calls']==rebuild
 coords=(rebuild*P if mode==1 else transitions)
 assert d['reconstructed_coordinates']==coords
 assert d['gradient_products']==(P-144)*8*(rebuild if mode==1 else done*(done+1)//2 if mode==2 else done)

for seed in SEEDS:
 for t in STEPS:
  d,p=invoke(0,seed,t,128,t,repeats=15);assert d['status']=='complete';check_accounting(d)
  raw=outputs(p);assert len(raw['states'])==t*P*12 and len(raw['gradients'])==t*P*4 and len(raw['factors'])==t*RECORD
  assert np.isfinite(np.frombuffer(raw['states'],dtype='<f4')).all()
  refs[(seed,t)]=(d,raw);rows.append(d)
  for mode,bs in BLOCKS.items():
   for b in bs:
    q,path=invoke(mode,seed,t,b,t);assert q['status']=='complete';check_accounting(q)
    actual=outputs(path);assert actual==raw,(seed,t,mode,b)
    q['all_raw_bytes_equal']=True;q['uint32_compared']=(len(raw['states'])+len(raw['gradients'])+len(raw['factors']))//4
    q['time_ratio']=q['nanoseconds_mean']/d['nanoseconds_mean'];q['optimizer_ratio_vs_raw_m_v']=q['optimizer_replacement_bytes']/(8*P);q['total_ratio_vs_reference']=q['charged_algorithm_bytes']/d['charged_algorithm_bytes']
    rows.append(q)
  save();print('case done',seed,t,flush=True)
 # Independent full-trajectory comparison, not only replay-to-shared-C-reference comparison.
 op=OUT/f'oracle_s{seed}_t16';info=oracle(seed,16,op);assert outputs(op)==refs[(seed,16)][1],seed
 info['all_raw_bytes_equal']=True;info['uint32_compared']=sum(len(x) for x in outputs(op).values())//4;parity.append(info);save()

for seed in [0,42]:
 for mode in [1,2]:
  for cap in CAPS:
   q,path=invoke(mode,seed,16,128,cap);assert q['status']=='capacity_rejected' and q['completed_steps']==cap and q['rejected_step']==cap+1;check_accounting(q)
   raw=outputs(path);ref=refs[(seed,16)][1]
   for ext,size in [('states',P*12),('gradients',P*4),('factors',RECORD)]:assert raw[ext]==ref[ext][:cap*size]
   q['all_raw_bytes_equal']=True;q['prefix_reference_equal']=True;rows.append(q)
# Reserved capacity must be counted before usage: step1 but capacity16.
for mode in [1,2]:
 q,path=invoke(mode,42,1,128,16);assert q['status']=='complete';check_accounting(q);assert outputs(path)==refs[(42,1)][1];q['all_raw_bytes_equal']=True;rows.append(q)
save()
# Six strict budget levels per representative horizon; no need to rerun all seeds under a seed-independent allocation screen.
for mode in [1,2]:
 for t in STEPS:
  for frac in [.1,.15,.25,.5,.75,.99]:
   q,_=invoke(mode,42,t,128,t,budget=int(8*P*frac),dump=False,repeats=1);q.update(mode_requested=mode,steps_requested=t,budget_fraction=frac);budgets.append(q)
for row in [x for x in rows if x.get('status')=='complete' and x.get('mode') and x.get('seed')==42]:
 preflight=row['request_bytes']+32+row['constant_reserve_bytes']+STACK-4*P
 for label,budget in [('preflight_minus_one',preflight-1),('generous_above_preflight',preflight+65536)]:
  q,_=invoke(row['mode'],42,row['requested_steps'],row['block'],row['capacity_steps'],budget=budget,dump=False,repeats=1)
  assert (q['status']=='budget_rejected') if label=='preflight_minus_one' else (q['status']=='complete')
  q.update(edge=label,mode_requested=row['mode'],steps_requested=row['requested_steps']);budgets.append(q)
save()
summary=dict(reference_cases=25,candidate_complete_cases=175,capacity_rejected_cases=24,reserved_unused_capacity_cases=2,all_raw_bytes_equal=True,total_complete_candidate_words=sum(r.get('uint32_compared',0) for r in rows),independent_oracle_seeds=SEEDS,independent_oracle_words=sum(r['uint32_compared'] for r in parity),budget_checks=len(budgets),budget_rejected=sum(r['status'].startswith('budget_rejected') for r in budgets),budget_accepted=sum(r['status']=='complete' for r in budgets),ten_percent_budget_accepted=sum(r['status']=='complete' and r.get('budget_fraction')==.1 for r in budgets),charged_stack_bytes=STACK,seconds=time.perf_counter()-started)
(OUT/'summary.json').write_text(json.dumps(summary,indent=2));print('DONE',json.dumps(summary),flush=True)
