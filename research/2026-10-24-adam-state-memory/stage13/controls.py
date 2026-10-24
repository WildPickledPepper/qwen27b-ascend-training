"""Constructed structural positives and lossless exceptional-bit tests, not training."""
from pathlib import Path
import json
import numpy as np
import residual_codec as c
R=Path(__file__).resolve().parent;rows=[];sources=[];rng=np.random.default_rng(130013)
# Stored factors are full FP32 values. The decoder's ordered product is the definition.
known=[];matrices=[]
for state in range(2):
 u=(rng.normal(size=(128,2))*.1).astype(np.float32);v=(rng.normal(size=(128,2))*.1).astype(np.float32)
 if state==1:u=np.abs(u);v=np.abs(v)
 known.append([(u,v)]);matrices.append(c.outer32(u,v))
base=np.stack(matrices).astype('<f4').view('<u4').ravel()
# Every predictor factor is serialized in each frame, never read from these provenance files.
np.concatenate([x.ravel() for item in known for uv in item for x in uv]).astype('<f4').tofile(R/'inputs/control_known_factors.f32')
for kind in ['exact','sparse_ulp','dense_low16']:
 words=base.copy()
 if kind=='sparse_ulp':
  chosen=rng.choice(len(words),size=len(words)//20,replace=False);o=c.order(words).astype(np.int64);o[chosen]+=rng.choice([-1,1],size=len(chosen));words=c.unorder(o)
 elif kind=='dense_low16':words=(words&np.uint32(0xffff0000))|rng.integers(0,65536,size=len(words),dtype=np.uint32)
 raw=words.astype('<u4').tobytes();case='constructed_known_rank2_'+kind;filename=case+'.bin';(R/'inputs'/filename).write_bytes(raw)
 src={'case':case,'family':'constructed_control','step':0,'shapes':[[128,128]],'filename':filename,'sha256':c.sha(raw),'origin':'Synthetic current moments defined by stored known rank2 FP32 factors; not trained Adam trajectory'};sources.append(src)
 c.run_case(raw,src,rows,known=known,only=[('baseline','baseline',0),('bf16_trunc','xor32',0),('bf16_trunc','ordered_delta',0),('lowrank_f32','xor32',2),('lowrank_f32','ordered_delta',2)])
# Distinct signed zeros, subnormals, finite extremes, infinities, and NaN payloads.
edge=np.array([0,0x80000000,1,0x80000001,0x007fffff,0x807fffff,0x00800000,0x80800000,0x3f800000,0xbf800000,0x3f800001,0xbf800001,0x7f7fffff,0xff7fffff,0x7f800000,0xff800000,0x7fc00001,0xffc00001,0x7fa12345,0xffa12345,0x7f800001,0xff800001,0x00008000,0x80008000,0x7fffffff,0xffffffff,0x3f7fffff,0xbf7fffff,0x0000ffff,0x8000ffff,0x3a83126f,0xba83126f],dtype='<u4')
raw=edge.tobytes();case='edge_float_bits';(R/'inputs'/f'{case}.bin').write_bytes(raw);src={'case':case,'family':'edge_control','step':0,'shapes':[[4,4]],'filename':case+'.bin','sha256':c.sha(raw),'origin':'Raw binary32 encoding test, including nonphysical negative/nonfinite v; not a training state'};sources.append(src)
c.run_case(raw,src,rows,only=[('baseline','baseline',0),('bf16_trunc','xor32',0),('bf16_trunc','ordered_delta',0),('zero','xor32',0),('zero','ordered_delta',0)])
try:c.make_frame(raw,src,'lowrank_f32','xor32',1)
except ValueError as err:nonfinite_rejection=str(err)
else:raise AssertionError('Nonfinite SVD should reject')
# Verify the 33-bit ZigZag extremes and signed-zero order continuity.
a=np.array([0xffffffff,0,0x80000000,0x7fffffff],np.uint32);b=a[::-1].copy();d=c.order(a).astype(np.int64)-c.order(b).astype(np.int64);zz=np.where(d<0,-2*d-1,2*d).astype(np.uint64);assert int(zz.max())==8589934590
back=c.unvarints(c.varints(zz),len(zz));dd=(back>>1).astype(np.int64);dd=np.where(back&1,-dd-1,dd);rec=c.unorder(c.order(b).astype(np.int64)+dd);assert np.array_equal(rec,a)
assert int(c.order(np.array([0],np.uint32))[0])-int(c.order(np.array([0x80000000],np.uint32))[0])==1
# Corruption must fail rather than silently deliver changed state.
frame=c.make_frame(raw,src,'bf16_trunc','xor32')[0];failures={}
for label,bad in [('truncated',frame[:-1]),('corrupt_payload',frame[:-1]+bytes([frame[-1]^1])),('bad_magic',b'BADMAGIC'+frame[8:])]:
 try:c.decode(bad)
 except Exception as err:failures[label]=type(err).__name__+': '+str(err)
 else:raise AssertionError(label)
(R/'inputs/control_index.json').write_text(json.dumps(sources,indent=2)+'\n');(R/'results/control_compression.json').write_text(json.dumps(rows,indent=2));c.savecsv(R/'results/control_compression.csv',rows)
result={'all20_control_frame_roundtrips_exact':len(rows)==20 and all(x['roundtrip_exact'] for x in rows),'constructed_cases':3,'edge_raw_words':len(edge),'nonfinite_lowrank_rejection':nonfinite_rejection,'max_abs_ordered_delta':int(np.abs(d).max()),'max_zigzag':int(zz.max()),'signed_zero_distinct_adjacent':True,'corruptions_rejected':failures,'known_factor_provenance_sha256':c.sha((R/'inputs/control_known_factors.f32').read_bytes()),'notes':['Known factors are entirely embedded and charged in each control frame. No external predictor oracle.','The sparse control changes exactly floor(5%*word_count) entries by +/-1 in ordered-float rank.','The dense low16 control replaces all lower16 bits randomly, demonstrating a harder correction stream.','Ordered ranks are reversible for NaN/infinite words, but are not numerical ULP distances for nonfinite values.']}
(R/'results/controls.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
for case in dict.fromkeys(q['case'] for q in rows):
 print(case,[(q['predictor'],q['residual'],round(q['ratio_to_raw_state'],6)) for q in rows if q['case']==case])
