"""Decode Stage13 artifacts without importing residual_codec or using its decoder.
All output files belong to the audit; no main experiment artifacts are modified.
"""
from __future__ import annotations
import hashlib, json, math, struct, zlib
from pathlib import Path
import numpy as np
from independent_verify import ordered, unordered, digest, TARGET_RATIO
ROOT=Path(__file__).resolve().parent
HEADER_BYTES=28


def strict_zlib(stream):
    obj=zlib.decompressobj()
    data=obj.decompress(stream)+obj.flush()
    assert obj.eof and not obj.unused_data and not obj.unconsumed_tail
    return data


def independent_unshuffle(stream, width):
    data=strict_zlib(stream)
    assert len(data)%width==0
    n=len(data)//width
    result=bytearray(len(data))
    for lane in range(width):
        result[lane::width]=data[lane*n:(lane+1)*n]
    return bytes(result)


def decode_varints(buf, count):
    # The independent implementation walks each integer from its starting offset.
    out=[];pos=0
    while pos<len(buf):
        start=pos;v=0;multiplier=1
        while True:
            assert pos<len(buf)
            byte=buf[pos];pos+=1
            v+=(byte%128)*multiplier
            if byte<128:break
            multiplier*=128
            assert pos-start<5
        assert v<=8589934590
        assert pos-start==max(1,(v.bit_length()+6)//7), 'Noncanonical varint'
        out.append(-(v//2)-1 if v%2 else v//2)
    assert len(out)==count
    return np.array(out,dtype=np.int64)


def reconstruct_factors(raw, spec):
    vals=np.frombuffer(raw,dtype='<f4')
    pos=0;outputs=[];scalar_checks=0;descriptors=[]
    for block in spec['tensors']:
        shape=block['shape'];size=math.prod(shape)
        if block['kind']=='exact_vector':
            assert len(shape)==1
            assert pos+size<=len(vals)
            outputs.append(vals[pos:pos+size].view('<u4').copy())
            descriptors.append({'kind':'exact_vector','shape':shape,'raw_bytes':size*4})
            pos+=size
        else:
            assert block['kind']=='uv' and len(shape)==2
            n,m=shape;r=block['rank']
            assert 1<=r<=min(n,m)
            left=vals[pos:pos+n*r].reshape(n,r);pos+=n*r
            right=vals[pos:pos+m*r].reshape(m,r);pos+=m*r
            assert np.isfinite(left).all() and np.isfinite(right).all()
            matrix=np.empty((n,m),dtype='<f4')
            # Row-first reconstruction, unlike the encoder's matrix-first loop.
            for row in range(n):
                accum=np.zeros(m,dtype=np.float32)
                for k in range(r):
                    product=np.asarray(left[row,k]*right[:,k],dtype=np.float32)
                    accum=np.asarray(accum+product,dtype=np.float32)
                matrix[row]=accum
            # Separate Python binary64 exact product/add then explicit FP32 round.
            # Binary64 exactly represents every binary32 product used here.
            for i,j in {(0,0),(n-1,m-1),(n//2,m//2),(n//3,m//3),(n//4,m//4)}:
                a=np.float32(0)
                for k in range(r):
                    prod=np.float32(float(left[i,k])*float(right[j,k]))
                    a=np.float32(float(a)+float(prod))
                assert a.view(np.uint32)==matrix[i,j].view(np.uint32)
                scalar_checks+=1
            outputs.append(matrix.ravel().view('<u4'))
            descriptors.append({'kind':'uv','shape':shape,'rank':r,'raw_bytes':4*r*(n+m)})
    assert pos==len(vals)
    return np.concatenate(outputs),scalar_checks,descriptors


def decode_frame(frame):
    assert len(frame)>=HEADER_BYTES
    magic=frame[:8]
    ml=int.from_bytes(frame[8:12],'little')
    pl=int.from_bytes(frame[12:20],'little')
    rl=int.from_bytes(frame[20:28],'little')
    assert magic==b'ADMR13\0\0'
    assert HEADER_BYTES+ml+pl+rl==len(frame)
    meta=json.loads(frame[28:28+ml])
    assert meta['version']==1 and meta['layout']=='m_then_v_little_endian_FP32'
    n=meta['word_count'];spec=meta['predictor']
    assert n==2*sum(math.prod(shape) for shape in meta['shapes'])
    predstream=frame[28+ml:28+ml+pl];resstream=frame[28+ml+pl:]
    predraw=b'';scalar_checks=0;descriptors=[];pred=None
    if spec['kind']=='none':
        assert pl==0 and meta['residual']=='baseline'
        raw=independent_unshuffle(resstream,4)
    else:
        if spec['kind']=='zero':
            assert pl==0
            pred=np.zeros(n,dtype=np.uint32)
        elif spec['kind']=='bf16_trunc':
            predraw=independent_unshuffle(predstream,2)
            assert len(predraw)==2*n
            pred=np.frombuffer(predraw,dtype='<u2').astype(np.uint32)*np.uint32(65536)
        else:
            assert spec['kind']=='lowrank_f32'
            assert spec['decoder']=='separate_fp32_mul_add_K_ascending'
            assert [x['shape'] for x in spec['tensors']]==meta['shapes']*2
            predraw=independent_unshuffle(predstream,4)
            pred,scalar_checks,descriptors=reconstruct_factors(predraw,spec)
        assert pred.size==n
        if meta['residual']=='xor32':
            residual=np.frombuffer(independent_unshuffle(resstream,4),dtype='<u4')
            assert residual.size==n
            actual=np.bitwise_xor(residual,pred)
        else:
            assert meta['residual']=='ordered_delta'
            delta=decode_varints(strict_zlib(resstream),n)
            ranks=ordered(pred).astype(np.int64)+delta
            assert np.all(ranks>=0) and np.all(ranks<=0xffffffff)
            actual=unordered(ranks.astype(np.uint32))
        raw=actual.astype('<u4').tobytes()
    assert len(raw)==n*4
    assert digest(raw)==meta['source_sha256']
    return raw,meta,{'header_bytes':HEADER_BYTES,'metadata_bytes':ml,'predictor_stored_bytes':pl,'residual_stored_bytes':rl,'frame_bytes':len(frame),'predictor_raw_bytes':len(predraw),'scalar_outer_checks':scalar_checks,'predictor_descriptors':descriptors},pred


def verify_input(src):
    raw=(ROOT/'inputs'/src['filename']).read_bytes()
    assert digest(raw)==src['sha256']
    assert len(raw)==src.get('raw_state_bytes',len(raw))
    evidence={'case':src['case'],'source_sha256':digest(raw),'source_bytes':len(raw),'training_rerun':False}
    if src['family']=='stage5_trained':
        name=src['filename'].removeprefix('stage5_')
        original_path=ROOT.parent/'stage5/snapshots'/name
        if original_path.is_file():
            assert original_path.read_bytes()==raw
            evidence.update({'route':'original_stage5_snapshot','comparison':'Byte-for-byte against sibling Stage5 snapshot.'})
        else:
            index_path=ROOT/'provenance/stage5_snapshots_index.json'
            index_bytes=index_path.read_bytes()
            matches=[x for x in json.loads(index_bytes) if x['filename']==name]
            assert len(matches)==1
            original=matches[0]
            assert original['sha256']==digest(raw)
            assert original['raw_state_bytes']==len(raw)
            assert original['seed']==src['seed'] and original['step']==src['step']
            assert original['shapes']==src['shapes']
            assert original['active_parameters']*8==len(raw)
            evidence.update({'route':'archived_stage5_index','archived_index_sha256':digest(index_bytes),'original_snapshot_sha256':original['sha256'],'comparison':'Archived input SHA256, size, seed, step and shapes match the archived original Stage5 index; original training and sibling snapshot are not rerun or re-created.'})
    elif src['family']=='stage10_trained':
        original_path=ROOT.parent/'stage10/formal_results/s42_t16_m0_b128_c16.states'
        if original_path.is_file():
            trajectory_path=original_path
            route='original_stage10_trajectory'
        else:
            trajectory_path=ROOT/'provenance/s42_t16_m0_b128_c16.states'
            route='archived_stage10_trajectory'
        original=trajectory_path.read_bytes()
        assert len(original)==16*6288*3*4
        assert src['source_states_sha256']==digest(original)
        a=np.frombuffer(original,dtype='<u4').reshape(16,6288,3)
        independently_extracted=a[src['step']-1,:,1].tobytes()+a[src['step']-1,:,2].tobytes()
        assert raw==independently_extracted
        assert src['source_step_offset_bytes']==(src['step']-1)*6288*12
        evidence.update({'route':route,'original_trajectory_sha256':digest(original),'original_trajectory_bytes':len(original),'comparison':'Independent m/v extraction from the preserved trajectory; trajectory SHA256 matches the input record. This does not rerun original training.'})
    else:
        evidence.update({'route':'archived_control_input','comparison':'Constructed/edge input SHA256 matches its index; independent construction checks follow.'})
    return raw,evidence


def verify_controls(inputs,frames):
    knownraw=(ROOT/'inputs/control_known_factors.f32').read_bytes()
    specs={'tensors':[{'kind':'uv','shape':[128,128],'rank':2} for _ in range(2)]}
    base,checks,_=reconstruct_factors(knownraw,specs)
    exact=np.frombuffer(inputs['constructed_known_rank2_exact'],dtype='<u4')
    sparse=np.frombuffer(inputs['constructed_known_rank2_sparse_ulp'],dtype='<u4')
    dense=np.frombuffer(inputs['constructed_known_rank2_dense_low16'],dtype='<u4')
    assert np.array_equal(exact,base)
    delta=ordered(sparse).astype(np.int64)-ordered(base).astype(np.int64)
    changes=np.count_nonzero(delta)
    assert changes==len(base)//20==1638
    assert set(delta.tolist())=={-1,0,1}
    assert np.array_equal(dense >> np.uint32(16),base >> np.uint32(16))
    factor_frames=[]
    for item in frames:
        if item['frame'].startswith('constructed_') and '.lowrank_f32.' in item['frame']:
            frame=(ROOT/'results/frames'/item['frame']).read_bytes()
            ml=int.from_bytes(frame[8:12],'little');pl=int.from_bytes(frame[12:20],'little')
            predictor=independent_unshuffle(frame[28+ml:28+ml+pl],4)
            assert predictor==knownraw
            assert len(predictor)==4096
            factor_frames.append(item['frame'])
    assert len(factor_frames)==6
    edge=np.frombuffer(inputs['edge_float_bits'],dtype='<u4')
    expected=np.array([0,0x80000000,1,0x80000001,0x007fffff,0x807fffff,0x00800000,0x80800000,0x3f800000,0xbf800000,0x3f800001,0xbf800001,0x7f7fffff,0xff7fffff,0x7f800000,0xff800000,0x7fc00001,0xffc00001,0x7fa12345,0xffa12345,0x7f800001,0xff800001,0x00008000,0x80008000,0x7fffffff,0xffffffff,0x3f7fffff,0xbf7fffff,0x0000ffff,0x8000ffff,0x3a83126f,0xba83126f],dtype=np.uint32)
    assert np.array_equal(edge,expected)
    return {'known_factors_bytes':len(knownraw),'known_factors_sha256':digest(knownraw),'known_factors_embedded_in_all_six_lowrank_control_frames':True,'exact_control_matches_own_factor_reconstruction':True,'scalar_factor_checks':checks,'sparse_corrections_exactly_plus_or_minus_one':True,'sparse_correction_count':int(changes),'sparse_correction_fraction':changes/len(base),'dense_low16_keeps_base_high16':True,'dense_low16_different_word_count':int(np.count_nonzero(dense!=base)),'edge_words':len(edge)}


def main():
    trained_rows=json.loads((ROOT/'results/compression.json').read_text())
    control_rows=json.loads((ROOT/'results/control_compression.json').read_text())
    rows=trained_rows+control_rows
    run=json.loads((ROOT/'results/run.json').read_text())
    assert run['source_sha256']==digest((ROOT/'residual_codec.py').read_bytes())
    assert run['source_index_sha256']==digest((ROOT/'inputs/index.json').read_bytes())
    assert run['cases']==12 and run['frames']==132
    trained_sources=json.loads((ROOT/'inputs/index.json').read_text())
    control_sources=json.loads((ROOT/'inputs/control_index.json').read_text())
    sources={x['case']:x for x in trained_sources+control_sources}
    assert len(sources)==16 and len(rows)==152
    inputs={};source_evidence=[]
    for case,src in sources.items():
        raw,evidence=verify_input(src)
        inputs[case]=raw;source_evidence.append(evidence)
    verified=[]
    for row in rows:
        fpath=ROOT/'results/frames'/row['frame']
        frame=fpath.read_bytes()
        raw,meta,account,pred=decode_frame(frame)
        assert raw==inputs[row['case']]
        assert meta['case']==row['case'] and meta['family']==row['family'] and meta['step']==row['step']
        assert meta['shapes']==sources[row['case']]['shapes']
        assert account['frame_bytes']==row['encoded_frame_bytes']
        assert account['header_bytes']+account['metadata_bytes']==row['header_and_metadata_bytes']
        assert account['predictor_stored_bytes']==row['stored_predictor_bytes']
        assert account['residual_stored_bytes']==row['stored_residual_bytes']
        assert len(raw)==row['raw_state_bytes']
        assert row['ratio_to_raw_state']==len(frame)/len(raw)
        assert row['meets_14_97074684_percent']==(len(frame)/len(raw)<=TARGET_RATIO)
        if pred is not None:
            w=np.frombuffer(raw,dtype='<u4');diff=ordered(w).astype(np.int64)-ordered(pred).astype(np.int64)
            assert float(np.mean(w==pred))==row['exact_predictor_fraction']
            assert int(np.max(np.abs(diff)))==row['ordered_delta_abs_max']
            if row['predictor']=='bf16_trunc':
                assert np.array_equal(pred,w & np.uint32(0xffff0000))
            if row['predictor']=='lowrank_f32':
                assert account['predictor_raw_bytes']==row['factor_or_exact_vector_raw_bytes']
                expected=0
                for shape in meta['shapes']:
                    if len(shape)==1:expected+=math.prod(shape)*4
                    else:expected+=4*min(row['rank_requested'],*shape)*sum(shape)
                assert account['predictor_raw_bytes']==2*expected
                # All vectors, including biases, must be exact before residuals.
                pos=0
                for shape in meta['shapes']*2:
                    size=math.prod(shape)
                    if len(shape)==1:assert np.array_equal(w[pos:pos+size],pred[pos:pos+size])
                    pos+=size
        verified.append({'frame':row['frame'],'frame_sha256':digest(frame),'source_sha256':digest(raw),'raw_state_bytes':len(raw),'full_frame_ratio':len(frame)/len(raw),**account})
    methods={(r['predictor'],r['rank_requested'],r['residual']) for r in trained_rows}
    assert len(methods)==11
    assert {(r['case'],r['predictor'],r['rank_requested'],r['residual']) for r in trained_rows}=={(src['case'],*method) for src in trained_sources for method in methods}
    bycase=[]
    for case in sources:
        sub=[x for x in rows if x['case']==case]
        best=min(sub,key=lambda x:x['encoded_frame_bytes'])
        bycase.append({k:best[k] for k in ['case','family','step','predictor','rank_requested','residual','raw_state_bytes','encoded_frame_bytes','ratio_to_raw_state','header_and_metadata_bytes','stored_predictor_bytes','stored_residual_bytes']})
    result={'all_independent_raw_roundtrips_and_accounting_pass':True,'frames_checked':len(rows),'total_decoded_raw_bytes':sum(x['raw_state_bytes'] for x in rows),'total_frame_bytes':sum(x['encoded_frame_bytes'] for x in rows),'scalar_outer_checks':sum(x['scalar_outer_checks'] for x in verified),'trained_frames_meeting_target':sum(x['meets_14_97074684_percent'] for x in trained_rows),'control_frames_meeting_target':sum(x['meets_14_97074684_percent'] for x in control_rows),'target_ratio':TARGET_RATIO,'main_code_sha256':digest((ROOT/'residual_codec.py').read_bytes()),'input_index_sha256':digest((ROOT/'inputs/index.json').read_bytes()),'compression_json_sha256':digest((ROOT/'results/compression.json').read_bytes()),'case_best':bycase,'verified_frames':verified,'control_source_checks':verify_controls(inputs,verified),'source_evidence':source_evidence,'source_evidence_limit':'Archived fallback verifies preserved evidence and exact frame decoding. It does not rerun or independently recreate the original training trajectories.'}
    (ROOT/'audit_frame_results.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='verified_frames'},indent=2))
if __name__=='__main__':main()
