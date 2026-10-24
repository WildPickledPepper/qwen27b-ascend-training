"""Verify saved numerical invariants and record actual candidate heap windows."""
import csv,hashlib,json,math,struct,tracemalloc
from pathlib import Path
import experiment as e
import candidates as c

root=Path(__file__).parent
r=list(csv.DictReader((root/'results-v2/summary.csv').open()))
for row in r:
    if row['kind'] in ('fp32','bit_halves','zlib_raw','zlib_joint','zlib_spatial'):
        assert all(int(row[k+'_bit_mismatches'])==0 for k in ('m','v','update','weight'))
assert any(int(row['update_bit_mismatches'])>0 for row in r if row['kind']=='bf16_residual')
j=json.loads((root/'results-v2/run.json').read_text())
assert j['script_sha256']==hashlib.sha256((root/'experiment.py').read_bytes()).hexdigest()
assert j['counterexamples']['same_current_update_different_future']['current_bit_equal']
assert j['counterexamples']['same_current_update_different_future']['next'][0]!=j['counterexamples']['same_current_update_different_future']['next'][1]
assert j['counterexamples']['same_m_histories']['m_bit_equal']
assert all(x['window_matches_block_reference'] for x in json.loads((root/'followup-results/summary.json').read_text())['streaming'])
mem=[];g=next(e.gradients('random',128,1,20261024));m,v=e.moments([0.]*128,[0.]*128,g)
for kind in ('solo42','zip_sr','ze_eden'):
    tracemalloc.start();blob=c.encode(m,v,kind,17,1);decoded=c.decode(blob,kind);_,peak=tracemalloc.get_traced_memory();tracemalloc.stop()
    mem.append({'kind':kind,'serialized_bytes':len(blob),'codebook_bytes':len(e.pack(c.M if kind=='solo42' else c.M+c.Q)),'python_encode_decode_peak_bytes':peak,'decoded_pair_payload_bytes':len(e.pack(decoded[0]+decoded[1])),'native_allocator_and_process_baseline':'not included'})
# Degenerate homogeneous data must compress, preventing a compressor test that
# only accidentally exercises raw fallback; arbitrary uint32 halves are exact.
for kind in ('zlib_raw','zlib_joint','zlib_spatial'):
    b=e.encode([1.]*8192,[2.]*8192,kind);assert len(b)<512
    assert e.pack(sum(e.decode(b,kind),[]))==e.pack([1.]*8192+[2.]*8192)
for u in (0,0x80000000,1,0x7f7fffff,0xff7fffff,0x3f808000):
    x=e.V(u);a,b=e.decode(e.encode([x],[x],'bit_halves'),'bit_halves');assert e.U(a[0])==u and e.U(b[0])==u
target=30*2**30/e.P
payload={'all_assertions_passed':True,'checks':'90-row baseline bitwise invariants, subsequent recurrence counterexample, streaming equality, script hash, compressible positive controls, raw bit boundary roundtrips','candidate_memory':mem,'target_bytes_per_parameter':target,'required_ratio':8/target,'q4_ideal_fp32_block_fallback_fraction_budget':(target-1.0625)/(8-1.0625),'solo_ideal_fp32_block_fallback_fraction_budget':(target-.84375)/(8-.84375),'fallback_note':'Ideal optimistic budgets exclude tags, offsets and scratch; production allowance is smaller.'}
(root/'audit.json').write_text(json.dumps(payload,indent=2),encoding='utf-8')
files={str(p.relative_to(root)).replace('\\','/'):hashlib.sha256(p.read_bytes()).hexdigest() for p in root.rglob('*') if p.is_file() and '__pycache__' not in str(p) and p.name!='SHA256SUMS.json'}
(root/'SHA256SUMS.json').write_text(json.dumps(files,indent=2),encoding='utf-8');print(json.dumps(payload,indent=2))
