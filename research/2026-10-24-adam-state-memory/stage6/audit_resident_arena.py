"""Functional tests only, not memory benchmarks. Reference phases return hashes,
then release full FP32 states before optimized phases begin."""
from pathlib import Path
import gc,hashlib,json
import numpy as np
import resident_arena as a
s=a.s
OUT=Path(__file__).resolve().parent/'results'

def gradients(case,n,steps):
 yield from (np.array(g,dtype=np.float32) for g in s.old.gradients(case,n,steps,20261024))
def ref_phase(case,n=257,b=64,steps=128):
 m=v=np.zeros(n,dtype=np.float32);w=np.arange(n,dtype=np.float32)/np.float32(n)+np.float32(.1);rows=[]
 for t,g in enumerate(gradients(case,n,steps),1):
  m,v=s.moments(m,v,g);d,w=s.update(m,v,w,t)
  rows.append((a.hash_state_full(m,v,b),hashlib.sha256(memoryview(w).cast('B')).hexdigest()))
 return rows

checks={}
for case in ['random','constant','alternating','scale_jump','near_cancel','sparse_spikes']:
 reference=ref_phase(case);gc.collect();n=257;b=64;arena=a.Arena(n,b,1.0);w=np.arange(n,dtype=np.float32)/np.float32(n)+np.float32(.1)
 for t,g in enumerate(gradients(case,n,128),1):
  response=arena.step(g,w,t);assert response['ok'];assert arena.state_hash()==reference[t-1][0];assert hashlib.sha256(memoryview(w).cast('B')).hexdigest()==reference[t-1][1]
 checks[case]=dict(steps=128,all_steps_state_and_weight_bit_equal=True,capacity_unchanged=len(arena.buffer)==8*n,max_used=arena.max_used,max_prefix_need=arena.max_prefix_need,raw_capacity=8*n,tail_coordinates=n%b)
# Arbitrary bit patterns force a genuine uncompressed fallback, retaining NaN payloads.
rng=np.random.default_rng(41);words=rng.integers(0,2**32,1024,dtype=np.uint32);raw=words.astype('<u4').tobytes();f=words.view(np.float32)
blob,flag=a.block_encode(f[:512],f[512:]);mm,vv=a.block_decode(blob,flag,512);assert mm.tobytes()+vv.tobytes()==raw;assert len(blob)<=len(raw)
checks['arbitrary_word_fallback']=dict(flag=flag,raw_bytes=len(raw),stored_bytes=len(blob),bit_exact=True)
# Budget failure is atomic for the optimizer's decoded state and weights.
arena=a.Arena(257,64,.3);w=np.zeros(257,dtype=np.float32);g=rng.normal(size=257).astype(np.float32);before=arena.state_hash();wh=hashlib.sha256(w.tobytes()).hexdigest();old_t=arena.t;result=arena.step(g,w,1)
assert not result['ok'];assert before==arena.state_hash() and wh==hashlib.sha256(w.tobytes()).hexdigest() and arena.t==old_t
checks['atomic_capacity_rejection']=result
# The streamed FP32 baseline matches the earlier independent full-vector reference.
p=OUT/'superseded_full_vector_reference.json'
if p.exists():
 x=json.load(open(p));y=json.load(open(OUT/'mlp_ref_b4096.json'));assert all(i['state_sha256']==j['state_sha256'] and i['weights_sha256']==j['weights_sha256'] for i,j in zip(x['checkpoints'],y['checkpoints']))
 checks['streamed_FP32_matches_full_vector_FP32']=True
checks['numeric_storage_worst_case']='A + 9*ceil(N/B) + 32 bytes; A=8*N suffices for every raw/compressed mixed prefix because each stored block has length <=8*its_coordinate_count. This is an exact owned-array storage bound, excluding fixed-block codec/runtime workspace.'
checks['workspace_scope']='Only O(B) decoded numeric arrays/bytes and O(1) zlib codec workspace; no N-sized decoded m/v or second state arena in Arena. Python/native allocator total peak is measured separately, not asserted as a formal RSS upper bound.'
(OUT/'functional_audit.json').write_text(json.dumps(checks,indent=2));print(json.dumps(checks,indent=2))
