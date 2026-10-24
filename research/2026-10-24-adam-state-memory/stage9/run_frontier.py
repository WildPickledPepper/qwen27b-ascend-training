"""Predeclared native exact frontier; no third-party code or installation."""
from pathlib import Path
import sys,subprocess,json,csv,hashlib,time,platform
import numpy as np
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'results';OUT.mkdir(exist_ok=True)
CASES=[(4,6),(4,8),(8,4),(8,6),(8,8),(32,2),(32,3),(64,2),(64,3),(124,2)]
CAP=2_000_000;BUDGET_FRACS=[.25,.5,.75,.9,.99]
su={}
for line in (ROOT/'frontier.su').read_text().splitlines():
 loc,size,qual=line.split('\t');su[loc.rsplit(':',1)[-1]]=int(size)
assert set(su)=={'transition','query','algorithm','dump_validation','main'}
# Exact audited static max call chains, including the live caller main frame.
STACK={0:su['main']+su['algorithm']+su['transition'],1:su['main']+su['algorithm']+su['query']+su['transition']}
configs=[]
for p,t in CASES:
 configs.append(dict(P=p,T=t,mode=1,policy='uncached',Q=0,tau=0))
 qs=sorted(set([1,max(1,p//4),p//2,p-1,p]))
 for q in qs:configs.append(dict(P=p,T=t,mode=2,policy='LRU',Q=q,tau=0))
 for q in qs:
  for tau in sorted(set([1,t//2])):configs.append(dict(P=p,T=t,mode=3,policy='pinned_slice',Q=q,tau=tau))
protocol=dict(cases=CASES,configs=configs,call_cap=CAP,budget_fractions_of_raw_dense_w_m_v=BUDGET_FRACS,repeats_reference=31,repeats_candidate=5,static_stack_frames=su,charged_call_chain_bytes=STACK,constant_pool_extra_reserve=64,all_final_weights_retained=True,full_checkpoint_is_over_raw_state_budget=True,context='Fresh zeroed arena per experiment and repetition; P and immutable fixed H/init/optimizer versions define context; no cache reuse across context; no RNG',source_sha256=hashlib.sha256((ROOT/'frontier.c').read_bytes()).hexdigest(),binary_sha256=hashlib.sha256((ROOT/'frontier').read_bytes()).hexdigest(),scope='CPU native mechanism experiment, not Qwen/NPU feasibility. Hardware runtime/code/stdio/timer/validation infrastructure not optimizer arena. Core and caller static stack included.',compile_command='cc -std=c11 -O2 -fno-fast-math -ffp-contract=off -fno-math-errno -mno-red-zone -fstack-usage -Wall -Wextra frontier.c -lm -o frontier',numpy=np.__version__,platform=platform.platform())
# Written BEFORE invoking any experiment subprocess.
(OUT/'protocol.json').write_text(json.dumps(protocol,indent=2));rows=[];budgets=[];refs={};start=time.perf_counter()

def invoke(c,budget=0,dump=False,repeats=5):
 mode=c['mode'];stack=STACK[0 if mode==0 else 1];tag=f"p{c['P']}_t{c['T']}_m{mode}_q{c['Q']}_s{c['tau']}";file=OUT/(tag+'.bin')
 args=[str(ROOT/'frontier'),str(mode),str(c['P']),str(c['T']),str(c['Q']),str(c['tau']),str(CAP),str(stack),str(budget),str(file) if dump else '-',str(repeats)]
 result=subprocess.run(args,check=True,capture_output=True,text=True);d=json.loads(result.stdout);d.update(c);d['command']=' '.join(args);d['budget_bytes']=budget
 return d,file

def save():
 # Some capped/complete variants have same schema; budget outcomes differ.
 for name,data in [('curve',rows),('budget_screen',budgets)]:
  if not data:continue
  keys=sorted(set().union(*(d.keys() for d in data)))
  with (OUT/(name+'.csv')).open('w',newline='') as f:
   w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(data)
  (OUT/(name+'.json')).write_text(json.dumps(data,indent=2))

for p,t in CASES:
 c=dict(P=p,T=t,mode=0,policy='dense_reference',Q=0,tau=0);row,path=invoke(c,dump=True,repeats=31);assert row['status']=='complete';row['raw_output_sha256']=hashlib.sha256(path.read_bytes()).hexdigest();refs[(p,t)]=(row,path.read_bytes());rows.append(row)
for c in configs:
 row,path=invoke(c,dump=True,repeats=5);ref,raw=refs[(c['P'],c['T'])]
 row['reference_charged_bytes']=ref['charged_algorithm_bytes'];row['peak_ratio_vs_dense_reference']=row['charged_algorithm_bytes']/ref['charged_algorithm_bytes'];row['mean_time_ratio_vs_dense_reference']=row['nanoseconds_mean']/ref['nanoseconds_mean'];row['min_time_ratio_vs_dense_reference']=row['nanoseconds_min']/ref['nanoseconds_min'];row['cache_alone_fraction_raw_current_state']=row['cache_bytes']/(12*c['P']);row['net_memory_below_dense_reference']=row['charged_algorithm_bytes']<ref['charged_algorithm_bytes'];row['analysis_time_below_100x']=row['mean_time_ratio_vs_dense_reference']<=100;row['strict_total_below_raw_dense_state']=row['charged_algorithm_bytes']<12*c['P'];row['complete_exact_bits']=False
 if row['status']=='complete':
  actual=path.read_bytes();assert actual==raw and row['checksum']==ref['checksum'];row['complete_exact_bits']=True;row['uint32_compared']=3*c['P'];row['raw_output_sha256']=hashlib.sha256(actual).hexdigest()
 else:assert row['status']=='operation_capped' and row['calls']==CAP
 rows.append(row)
 for fraction in BUDGET_FRACS:
  budget=int(12*c['P']*fraction);br,_=invoke(c,budget=budget,repeats=1);br['budget_fraction']=fraction;budgets.append(br)
 save();print(c['P'],c['T'],c['policy'],c['Q'],c['tau'],row['status'],'bytes',row['charged_algorithm_bytes'],'ref',ref['charged_algorithm_bytes'],'time_ratio',round(row['mean_time_ratio_vs_dense_reference'],2),flush=True)
# Independent NumPy reference parity for the original P4 coupled problem.
sys.path.insert(0,str(ROOT.parent/'stage8'));import exact_replay as e
parity=[]
for p,t in CASES:
 if p!=4:continue
 w,m,v,_=e.reference(t);expected=np.stack([w,m,v],axis=1).astype('<f4').tobytes();assert expected==refs[(p,t)][1];parity.append(dict(P=p,T=t,all_raw_words_equal=True,uint32_compared=3*p))
complete=[r for r in rows if r['mode']!=0 and r['status']=='complete'];capped=[r for r in rows if r['status']=='operation_capped'];joint=[r for r in complete if r['net_memory_below_dense_reference'] and r['analysis_time_below_100x']]
summary=dict(native_reference_cases=len(refs),candidate_cases=len(configs),completed_candidates=len(complete),operation_capped_cases=len(capped),all_completed_raw_bits_equal=True,total_candidate_uint32_compared=sum(r.get('uint32_compared',0) for r in complete),numpy_P4_parity=parity,strict_budget_checks=len(budgets),strict_budget_accepts=sum(r['status']=='complete' for r in budgets),joint_measured_reference_saving_under_100x_cases=[{k:r[k] for k in ['P','T','policy','Q','tau','charged_algorithm_bytes','reference_charged_bytes','mean_time_ratio_vs_dense_reference']} for r in joint],seconds=time.perf_counter()-start,limitations='Fixed-budget rejects include caller stack and numeric constants. Charged block-level arena+static-stack envelope is not whole-process RSS/allocator arena. 100x is an analysis threshold, not a user limit. No external state/history inputs.')
(OUT/'summary.json').write_text(json.dumps(summary,indent=2));save();print('DONE',json.dumps(summary),flush=True)
