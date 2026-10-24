"""Dynamic checks of the production transform and rejection paths.
Imports the main module but never runs its experiment or writes its artifacts.
"""
from pathlib import Path
import json
import numpy as np
import residual_codec as main
from independent_verify import ordered,unordered,digest
from independent_frame_verify import decode_frame
ROOT=Path(__file__).resolve().parent
rng=np.random.default_rng(131313)
high=np.arange(65536,dtype=np.uint32)
words=np.concatenate([(high<<np.uint32(16))|np.uint32(low) for low in [0,1,0x7fff,0x8000,0xfffe,0xffff]]+[rng.integers(0,1<<32,262144,dtype=np.uint32)])
assert np.array_equal(main.order(words),ordered(words))
assert np.array_equal(main.unorder(ordered(words)),words)
assert np.array_equal(unordered(main.order(words).astype(np.uint32)),words)
values=np.concatenate([np.array([0,1,127,128,255,256,16383,16384,2**28-1,2**28,8589934589,8589934590],dtype=np.uint64),rng.integers(0,8589934591,10000,dtype=np.uint64)])
encoded=main.varints(values)
assert np.array_equal(main.unvarints(encoded,len(values)),values)
rejections={}
for name,raw,count in [('incomplete_varint',b'\x80',1),('too_wide_varint',b'\x80'*5+b'\x00',1),('too_large_varint',main.varints([8589934591]),1),('wrong_count',b'\x00',2)]:
    try:main.unvarints(raw,count)
    except ValueError as exc:rejections[name]=str(exc)
    else:raise AssertionError(name)
edge=(ROOT/'inputs/edge_float_bits.bin').read_bytes()
src={'case':'audit_specials','family':'audit_only','step':0,'shapes':[[4,4]]}
try:main.make_frame(edge,src,'lowrank_f32','xor32',1)
except ValueError as exc:rejections['nonfinite_matrix_svd']=str(exc)
else:raise AssertionError('SVD accepted nonfinite matrix')
special_frames=0
for pred in ['zero','bf16_trunc']:
    for res in ['xor32','ordered_delta']:
        frame,_=main.make_frame(edge,src,pred,res)
        assert decode_frame(frame)[0]==edge
        special_frames+=1
        for label,bad in [('truncate',frame[:-1]),('append',frame+b'\x00'),('magic',b'BADMAGIC'+frame[8:]),('payload',frame[:-1]+bytes([frame[-1]^1]))]:
            try:main.decode(bad)
            except Exception as exc:rejections[pred+'_'+res+'_'+label]=type(exc).__name__
            else:raise AssertionError((pred,res,label))
# Nonfinite vectors are allowed and stored exactly; SVD sees only finite matrices.
# This includes signaling NaNs with low-only payload and both zero signs.
w=np.array([0,0x80000000,1,0x3f800000,0x7f800001,0xffa12345,0x7fc00001,0x80000000,0x80000001,0xbf800000,0,0x00800000,0xff800001,0x7fa12345,0x7fffffff,0xffffffff],dtype='<u4')
raw=w.tobytes();source={'case':'audit_vector_payloads','family':'audit_only','step':0,'shapes':[[2,2],[4]]}
for res in ['xor32','ordered_delta']:
    frame,_=main.make_frame(raw,source,'lowrank_f32',res,1)
    assert main.decode(frame)[0]==raw
    assert decode_frame(frame)[0]==raw
    special_frames+=1
result={'main_order_matches_independent_word_cases':int(words.size),'varint_roundtrip_values':int(values.size),'special_frame_independent_roundtrips':special_frames,'nonfinite_vector_payloads_preserved':True,'main_code_sha256':digest((ROOT/'residual_codec.py').read_bytes()),'rejections':rejections,'scope':'Dynamic checks only; main experiment artifacts unchanged. Not a security audit or a cross-platform FP32 guarantee.'}
(ROOT/'audit_behavior_results.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
