"""Independent Stage13 auditor; never trains or changes main experiment outputs.
Only NumPy and Python standard library; exact verification is at uint32 bytes.
"""
from __future__ import annotations
import argparse, csv, hashlib, importlib.util, json, math, struct, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parent
TARGET_RATIO = 0.1497074684
MASK = 0xffffffff
SIGN = 0x80000000

def digest(data):
    return hashlib.sha256(data).hexdigest()

def ordered_scalar(word):
    word = int(word)
    return MASK - word if word & SIGN else word + SIGN

def unordered_scalar(rank):
    rank = int(rank)
    return rank - SIGN if rank & SIGN else MASK - rank

def ordered(words):
    words = np.asarray(words, dtype=np.uint32)
    return np.where((words & SIGN) != 0, np.uint32(MASK) - words, words ^ np.uint32(SIGN)).astype(np.uint32)

def unordered(ranks):
    ranks = np.asarray(ranks, dtype=np.uint32)
    return np.where((ranks & SIGN) != 0, ranks ^ np.uint32(SIGN), np.uint32(MASK) - ranks).astype(np.uint32)

def encode_signed_delta(actual, predicted):
    return ordered(actual).astype(np.int64) - ordered(predicted).astype(np.int64)

def decode_signed_delta(delta, predicted):
    rank = ordered(predicted).astype(np.int64) + delta
    assert (rank >= 0).all() and (rank <= MASK).all()
    return unordered(rank.astype(np.uint32))

def zigzag(delta):
    # No signed overflow is possible: |delta| <= 2^32-1.
    return np.where(delta >= 0, 2 * delta, -2 * delta - 1).astype(np.uint64)

def unzigzag(code):
    code = np.asarray(code, dtype=np.uint64)
    return (code >> np.uint64(1)).astype(np.int64) * np.where(code & np.uint64(1), -1, 1) - (code & np.uint64(1)).astype(np.int64)

def test_provenance():
    base = ROOT.parent
    path = base / 'stage5' / 'snapshots'
    index = json.loads((path / 'index.json').read_text())
    run = json.loads((path / 'run.json').read_text())
    assert len(index) == 21
    assert {(x['seed'], x['step']) for x in index} == {(seed, step) for seed in [101,202,303] for step in [1,2,128,1024,2048,4095,4096]}
    source_paths = {'collector':base/'stage5/collect_snapshots.py', 'mlp':base/'stage4/mlp_teacher.py', 'adam':base/'stage3/run_stage3.py'}
    hashes = {k:digest(p.read_bytes()) for k,p in source_paths.items()}
    assert hashes == run['source_hashes']
    rows = []
    for src in index:
        raw = (path / src['filename']).read_bytes()
        assert digest(raw) == src['sha256']
        assert sum(math.prod(shape) for shape in src['shapes']) == src['active_parameters'] == 21508
        assert len(raw) == src['raw_state_bytes'] == 8 * src['active_parameters'] == 172064
        states = np.frombuffer(raw, dtype='<f4')
        assert np.isfinite(states).all()
        assert (states[src['active_parameters']:] >= 0).all()
        rows.append({'filename':src['filename'], 'sha256':digest(raw), 'bytes':len(raw), 'finite':True, 'v_nonnegative':True})
    return {'snapshot_count':len(rows), 'snapshot_total_bytes':sum(r['bytes'] for r in rows), 'index_sha256':digest((path/'index.json').read_bytes()), 'run_sha256':digest((path/'run.json').read_bytes()), 'source_hashes_match':hashes, 'files':rows, 'provenance_limit':'Matches stored Stage5 trajectory metadata and source hashes; training was not rerun and cannot be independently reconstructed from snapshots alone.'}

def test_stage10_provenance():
    base = ROOT.parent/'stage10'
    manifest = json.loads((base/'MANIFEST.json').read_text())
    hashes = {item['path']:item for item in manifest['files']}
    paths = ['formal_results/s42_t16_m0_b128_c16.states','formal_results/oracle_s42_t16.states','formal_results/oracle_parity.json','factor_replay.c']
    validated = []
    for rel in paths:
        data=(base/rel).read_bytes()
        assert len(data)==hashes[rel]['bytes'] and digest(data)==hashes[rel]['sha256']
        validated.append({'path':rel,'bytes':len(data),'sha256':digest(data)})
    ref=(base/paths[0]).read_bytes()
    oracle=(base/paths[1]).read_bytes()
    assert ref==oracle and len(ref)==16*6288*3*4
    states=np.frombuffer(ref,dtype='<u4').reshape(16,6288,3)
    rows=[]
    for step in [1,8,16]:
        m=states[step-1,:,1];v=states[step-1,:,2]
        raw=m.tobytes()+v.tobytes()
        assert len(raw)==50304
        assert np.isfinite(m.view('<f4')).all() and np.isfinite(v.view('<f4')).all()
        assert (v.view('<f4')>=0).all()
        biases=np.r_[m[4096:4224],m[6272:6288],v[4096:4224],v[6272:6288]]
        rows.append({'step':step,'raw_mv_bytes':len(raw),'mv_sha256':digest(raw),'bias_state_bytes':biases.nbytes,'bias_state_sha256':digest(biases.tobytes()),'target_max_integer_full_frame_bytes':math.floor(len(raw)*TARGET_RATIO)})
    return {'reference_and_oracle_raw_bytes_equal':True,'existing_trajectory_verified_bytes':len(ref),'provenance_files':validated,'selected_snapshots':rows,'training_rerun':False}

def test_rank_and_residuals():
    # All high 16-bit patterns, each with six critical low-word cases.
    hi = np.arange(65536, dtype=np.uint32)
    words = np.concatenate([(hi << np.uint32(16)) | np.uint32(lo) for lo in [0,1,0x7fff,0x8000,0xfffe,0xffff]])
    rng = np.random.default_rng(130013)
    words = np.concatenate([words, rng.integers(0, 1 << 32, 262144, dtype=np.uint32)])
    edges = np.array([0,0x80000000,1,0x80000001,0x007fffff,0x807fffff,0x00800000,0x80800000,0x3f800000,0xbf800000,0x7f7fffff,0xff7fffff,0x7f800000,0xff800000,0x7f800001,0xff800001,0x7fc00000,0xffc00000,0x7fffffff,0xffffffff,0x7fa12345,0xffa12345],dtype=np.uint32)
    words = np.concatenate([edges,words])
    ranks = ordered(words)
    assert np.array_equal(unordered(ranks),words)
    # Independent Python-integer definition on every high-word combination.
    assert np.array_equal(ranks[:65558],np.fromiter((ordered_scalar(w) for w in words[:65558]),dtype=np.uint32))
    assert all(unordered_scalar(ordered_scalar(int(w))) == int(w) for w in edges)
    # Finite ranks must be monotone; signed zero occupies distinct adjacent ranks.
    finite_order = np.array([0xff7fffff,0xbf800000,0x80800000,0x807fffff,0x80000001,0x80000000,0,1,0x007fffff,0x00800000,0x3f800000,0x7f7fffff],dtype=np.uint32)
    assert np.all(np.diff(ordered(finite_order).astype(np.int64)) > 0)
    assert ordered_scalar(0) == ordered_scalar(0x80000000) + 1
    assert ordered_scalar(0xffffffff) == 0
    assert ordered_scalar(0x7fffffff) == MASK
    pair_count=0
    mins=[]; maxs=[]; max_zigzag=[]
    # Include opposing rank extremes, arbitrary pairs, and BF16 truncation predictors.
    predictors=[words[::-1].copy(),np.zeros(words.size,dtype=np.uint32),words & np.uint32(0xffff0000),rng.integers(0,1 << 32,words.size,dtype=np.uint32)]
    for pred in predictors:
        delta=encode_signed_delta(words,pred)
        assert np.array_equal(decode_signed_delta(delta,pred),words)
        zz=zigzag(delta)
        assert np.array_equal(unzigzag(zz),delta)
        assert np.array_equal(decode_signed_delta(unzigzag(zz),pred),words)
        assert np.array_equal((words ^ pred) ^ pred,words)
        pair_count+=words.size
        mins.append(int(delta.min()));maxs.append(int(delta.max()));max_zigzag.append(int(zz.max()))
    a=np.array([0x7fffffff,0xffffffff],dtype=np.uint32)
    p=a[::-1].copy();delta=encode_signed_delta(a,p);zz=zigzag(delta)
    assert delta.tolist()==[MASK,-MASK]
    assert zz.tolist()==[2*MASK,2*MASK-1]
    assert np.array_equal(decode_signed_delta(unzigzag(zz),p),a)
    exp=(words >> np.uint32(23)) & np.uint32(255)
    mant=words & np.uint32(0x7fffff)
    return {'word_cases':int(words.size), 'predictor_pair_cases':int(pair_count+2), 'finite_monotone_order':True, 'signed_zero_distinct':True, 'all_inverse_xor_signed_delta_and_zigzag_tests_pass':True, 'special_case_counts':{'zero':int(np.sum((words & np.uint32(0x7fffffff))==0)), 'subnormal':int(np.sum((exp==0)&(mant!=0))), 'infinity':int(np.sum((exp==255)&(mant==0))), 'nan':int(np.sum((exp==255)&(mant!=0)))}, 'negative_delta_extreme':-MASK, 'positive_delta_extreme':MASK, 'zigzag_max':2*MASK, 'required_zigzag_bits':33, 'interpretation':'Ranks are a reversible ordering of all 32-bit words. For NaNs and infinities, rank difference is an encoding quantity, not a meaningful numerical ULP error.'}

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,default=ROOT/'audit_independent_results.json')
    args=parser.parse_args()
    result={'audit_version':1,'python':sys.version,'numpy':np.__version__, 'target_full_frame_ratio_to_raw_mv':TARGET_RATIO, 'target_raw_snapshot_bytes':172064, 'target_max_integer_full_frame_bytes':math.floor(172064*TARGET_RATIO), 'provenance':test_provenance(), 'stage10_provenance':test_stage10_provenance(), 'independent_integer_bijection':test_rank_and_residuals()}
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'output':str(args.output),'provenance_pass':True,'integer_bijection':result['independent_integer_bijection'],'max_bytes':result['target_max_integer_full_frame_bytes']},indent=2))
if __name__=='__main__':
    main()
