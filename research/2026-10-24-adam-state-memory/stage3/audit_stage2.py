"""Read-only audit of imported stage2; original scripts/results are not changed."""
import csv, hashlib, heapq, json, math, struct, sys, tracemalloc
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]/'adam-state-cpu-20261024'
sys.path.insert(0,str(ROOT));import experiment as e, candidates as c
OUT=Path(__file__).resolve().parent
m=[-1.]*128;v=[1.]*128
blob=c.encode(m,v,'zip_sr',17,1);mm,vv,_=c.decode(blob,'zip_sr')
neg=dict(original=-1.,reconstructed=mm[0],relative_error=abs(mm[0]+1),positive_max_reconstructed=c.decode(c.encode([1.]*128,v,'zip_sr',17,1),'zip_sr')[0][0])
# Quantize only m to isolate the codebook's deterministic negative endpoint bias.
raw_m=raw_v=[0.]*128;refm=refv=[0.]*128
for t in range(1,1025):
 raw_m,raw_v=e.moments(raw_m,raw_v,[-1.]*128);refm,refv=e.moments(refm,refv,[-1.]*128)
 before=raw_m[0]
 b=c.encode(raw_m,raw_v,'zip_sr',17,t);raw_m=c.decode(b,'zip_sr')[0]
neg.update(constant_negative_prequant_m_after_1024=before,reference_m_after_1024=refm[0],ratio=before/refm[0])
# Measuring the returned two-entry list is not measuring the selection peak.
heap=[]
for k in [1024,8192,65536,131072]:
 tracemalloc.start()
 ans=heapq.nlargest(k,((i*.000001,i) for i in range(2*k)))
 _,peak=tracemalloc.get_traced_memory();tracemalloc.stop()
 heap.append(dict(k=k,candidate_count=2*k,peak_python_allocation_bytes=peak,peak_bytes_per_selected=peak/k,returned_deep_size_bytes=sys.getsizeof(ans)+sum(sys.getsizeof(t)+sum(sys.getsizeof(a) for a in t) for t in ans)))
P=e.P;k=math.ceil((1/128)*math.ceil(P/128));base=P*1.0625;fixed=24+128+16+8*2**20
old=(base+k*(2056+128)+fixed)/2**30
# 224 bytes/entry conservative rounded peak for the measured Python nlargest,
# not a claim about C/NPU packed selectors, which can be much smaller.
revised=(base+k*(2056+224)+fixed)/2**30
budget=dict(P=P,selected_block_count=k,original_design_peak_GiB=old,original_headroom_MiB=(30-old)*1024,python_heap_peak_allowance_bytes_per_entry=224,python_heap_revised_projection_GiB=revised,python_heap_revised_exceeds_30GiB=revised>30,important='This rejects scaling the current Python heap as written. A packed native bounded selector <=128 bytes/entry could retain the original arithmetic budget, but no such production implementation is present. Full-vector decode and b.join exception construction in the CPU prototype are also not production-streaming implementations.')
rows=list(csv.DictReader((ROOT/'stage2/toy_summary.csv').open()))
refs=[r for r in rows if r['kind']=='fp32']
refstats=[dict(seed=r['data_init_seed'],initial_loss=float(r['initial_loss']),final_loss=float(r['final_loss']),fraction_initial=float(r['final_loss'])/float(r['initial_loss']),ratio_denominator_is_optimum=False) for r in refs]
orig={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in ROOT.rglob('*') if p.is_file()}
result=dict(m_codebook_endpoint_audit=neg,heap_peak_measurements=heap,protection_budget_audit=budget,toy_reference_stats=refstats,conditional_unbiasedness='ZIP-SR probabilities interpolate f(v)=1/(sqrt(v/c2_t)+eps) between adjacent v-grid points. This is one-step conditional real-arithmetic unbiasedness for fixed current v, not unbiased v, not unchanged Adam recurrence, not bitwise FP32 equality. Rounded scale/output and stochastic histories further break exact equivalence.',zero_and_log_handling='Imported ZIP includes zero v codes and may map positive v to zero. Its nonzero eps keeps updates finite but can permit rare large updates; epsilon placement is outside sqrt and matches the scalar reference. Imported SOLO-style floors the zero quantile at 1e-30 and cannot preserve mixed zero v blocks: that is a locally defined edge policy, not a formal complete paper reproduction.',fairness='Hyperparameters and input seeds are paired in stage2. CPU FP32 operation order is explicit. m-only and v-only really retain the other FP32 state and do not satisfy the 30 GiB design. All full-method labels require STYLE/LOCAL VARIANT; m is shared custom DE with deterministic rounding. These experiments cannot refute official SOLO, ZIP-SR, or ZE-EDEN.',original_file_sha256=orig)
(OUT/'stage2_audit.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k!='original_file_sha256'},indent=2))
