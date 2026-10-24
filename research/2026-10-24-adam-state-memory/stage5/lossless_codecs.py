"""Original exact FP32-state transforms plus stdlib zlib; no third-party codecs.
Every encoded frame contains its transform ID, shape/time and complete metadata.
Temporal frames embed a complete reference frame, never an uncharged oracle.
"""
from pathlib import Path
import csv,hashlib,json,math,struct,sys,time,tracemalloc,zlib
import numpy as np
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'results';OUT.mkdir(exist_ok=True)
HEADER=struct.Struct('<8sQQII');MAGIC=b'ADAMCMP1'
KINDS=['raw','zlib1','zlib6','zlib9','byte_shuffle','float_fields','bitplanes','adjacent_xor','cross_m2','cross_m2_fit','temporal_raw_anchor','temporal_best_anchor']

def shuffle(raw):return np.frombuffer(raw,dtype=np.uint8).reshape(-1,4).T.copy().tobytes()
def unshuffle(raw):return np.frombuffer(raw,dtype=np.uint8).reshape(4,-1).T.copy().tobytes()
def planes(words,nbits=32):return np.packbits(((words[None,:]>>np.arange(nbits,dtype=np.uint32)[:,None])&1).astype(np.uint8),axis=1,bitorder='little').tobytes()
def unplanes(raw,count,nbits=32):
 bits=np.unpackbits(np.frombuffer(raw,dtype=np.uint8).reshape(nbits,-1),axis=1,bitorder='little')[:,:count].astype(np.uint32)
 return np.bitwise_or.reduce(bits<<np.arange(nbits,dtype=np.uint32)[:,None],axis=0)

def cross_predict(mwords,scales=None,block=256):
 with np.errstate(all='ignore'):
  m=mwords.view('<f4');pred=m*m
  if scales is not None:pred=pred*np.repeat(scales,block)[:len(m)]
 return pred.astype('<f4').view('<u4')

def metadata(source):
 return json.dumps({'seed':int(source['seed']),'shapes':source['shapes'],'layout':'m_then_v_little_endian_FP32','source_sha256':source['sha256']},sort_keys=True,separators=(',',':')).encode()

def encode(raw,source,kind,anchor=None):
 n=int(source['active_parameters']);words=np.frombuffer(raw,dtype='<u4');extra=b''
 if kind=='raw':payload=raw
 elif kind.startswith('zlib'):payload=zlib.compress(raw,int(kind[-1]))
 elif kind=='byte_shuffle':payload=zlib.compress(shuffle(raw),9)
 elif kind=='bitplanes':payload=zlib.compress(planes(words),9)
 elif kind=='float_fields':
  signs=np.packbits((words>>31).astype(np.uint8),bitorder='little').tobytes();exps=((words>>23)&255).astype(np.uint8).tobytes()
  payload=zlib.compress(signs+exps+planes(words&0x7fffff,23),9)
 elif kind=='adjacent_xor':
  ww=words.reshape(2,n);d=ww.copy();d[:,1:]=ww[:,1:]^ww[:,:-1];payload=zlib.compress(shuffle(d.astype('<u4').tobytes()),9)
 elif kind in ['cross_m2','cross_m2_fit']:
  scales=None
  if kind=='cross_m2_fit':
   block=256;mm=words[:n].view('<f4').astype(np.float64);vv=words[n:].view('<f4').astype(np.float64);scales=[]
   for start in range(0,n,block):
    a=mm[start:start+block]**2;b=vv[start:start+block];den=float(np.dot(a,a));scale=float(np.dot(a,b))/den if den>0 else 0
    scales.append(scale if np.isfinite(scale) and abs(scale)<np.finfo(np.float32).max else 0.)
   scales=np.array(scales,dtype='<f4');extra=struct.pack('<II',block,len(scales))+scales.tobytes()
  pred=cross_predict(words[:n],scales);d=np.r_[words[:n],words[n:]^pred].astype('<u4').tobytes();payload=extra+zlib.compress(shuffle(d),9)
 elif kind.startswith('temporal_'):
  if anchor is None:raise ValueError('Complete anchor frame is required')
  ar,am=decode(anchor);assert len(ar)==len(raw)
  delta=np.bitwise_xor(words,np.frombuffer(ar,dtype='<u4')).astype('<u4').tobytes()
  payload=struct.pack('<Q',len(anchor))+anchor+hashlib.sha256(ar).digest()+zlib.compress(shuffle(delta),9)
 else:raise ValueError(kind)
 meta=metadata(source);return HEADER.pack(MAGIC,n,int(source['step']),KINDS.index(kind),len(meta))+meta+payload

def decode(frame):
 magic,n,t,k,mlen=HEADER.unpack_from(frame);assert magic==MAGIC
 meta=json.loads(frame[HEADER.size:HEADER.size+mlen]);payload=frame[HEADER.size+mlen:];kind=KINDS[k];count=2*n
 if kind=='raw':raw=payload
 elif kind.startswith('zlib'):raw=zlib.decompress(payload)
 elif kind=='byte_shuffle':raw=unshuffle(zlib.decompress(payload))
 elif kind=='bitplanes':raw=unplanes(zlib.decompress(payload),count).astype('<u4').tobytes()
 elif kind=='float_fields':
  b=zlib.decompress(payload);sb=(count+7)//8;sg=np.unpackbits(np.frombuffer(b[:sb],dtype=np.uint8),bitorder='little')[:count].astype(np.uint32)
  ex=np.frombuffer(b[sb:sb+count],dtype=np.uint8).astype(np.uint32);ma=unplanes(b[sb+count:],count,23)
  raw=((sg<<31)|(ex<<23)|ma).astype('<u4').tobytes()
 elif kind=='adjacent_xor':
  d=np.frombuffer(unshuffle(zlib.decompress(payload)),dtype='<u4').reshape(2,n)
  raw=np.bitwise_xor.accumulate(d,axis=1).astype('<u4').tobytes()
 elif kind in ['cross_m2','cross_m2_fit']:
  scales=None;block=256
  if kind=='cross_m2_fit':
   block,nn=struct.unpack_from('<II',payload);scales=np.frombuffer(payload,dtype='<f4',count=nn,offset=8);payload=payload[8+4*nn:]
  d=np.frombuffer(unshuffle(zlib.decompress(payload)),dtype='<u4');pred=cross_predict(d[:n],scales,block)
  raw=np.r_[d[:n],d[n:]^pred].astype('<u4').tobytes()
 elif kind.startswith('temporal_'):
  alen=struct.unpack_from('<Q',payload)[0];ar,am=decode(payload[8:8+alen]);ah=payload[8+alen:8+alen+32];assert hashlib.sha256(ar).digest()==ah
  delta=np.frombuffer(unshuffle(zlib.decompress(payload[8+alen+32:])),dtype='<u4');raw=np.bitwise_xor(delta,np.frombuffer(ar,dtype='<u4')).astype('<u4').tobytes()
 else:raise ValueError(kind)
 assert len(raw)==8*n;assert hashlib.sha256(raw).hexdigest()==meta['source_sha256']
 return raw,meta

def entropy(values):
 counts=np.bincount(values.astype(np.int64).ravel());p=counts[counts>0]/values.size
 return float(-np.sum(p*np.log2(p)))

def distributions(raw,source):
 n=int(source['active_parameters']);words=np.frombuffer(raw,dtype='<u4');rows=[]
 for name,w in zip(['m','v'],[words[:n],words[n:]]):
  f=w.view('<f4');bit_h=sum(entropy((w>>i)&1) for i in range(32))
  rows.append(dict(seed=source['seed'],step=source['step'],state=name,parameters=n,zero_fraction=float(np.mean((w&0x7fffffff)==0)),negative_fraction=float(np.mean(w>>31)),finite_fraction=float(np.mean(np.isfinite(f))),mean=float(np.mean(f.astype(np.float64))),RMS=float(np.sqrt(np.mean(f.astype(np.float64)**2))),exponent_entropy_bits=entropy((w>>23)&255),mantissa_low8_entropy_bits=entropy(w&255),mantissa_middle8_entropy_bits=entropy((w>>8)&255),mantissa_high7_entropy_bits=entropy((w>>16)&127),independent_bit_entropy_sum_bits=bit_h,unique_low8=int(len(np.unique(w&255))),exponent_histogram_json=json.dumps(np.bincount(((w>>23)&255).astype(np.int64),minlength=256).tolist(),separators=(',',':'))))
 return rows

def savecsv(path,rows):
 with path.open('w',newline='') as f:wr=csv.DictWriter(f,fieldnames=list(rows[0]));wr.writeheader();wr.writerows(rows)

def audit():
 words=np.array([0,0x80000000,1,0x80000001,0x7f800000,0xff800000,0x7fc00001,0x7fa12345,0x3f800000,0xbf800000,0x00800000,0x7f7fffff],dtype='<u4');raw=words.tobytes();source=dict(seed=0,step=1,active_parameters=len(words)//2,sha256=hashlib.sha256(raw).hexdigest(),shapes=[[len(words)//2]])
 for kind in KINDS[:10]:
  frame=encode(raw,source,kind);back,_=decode(frame);assert back==raw,kind
 anchor=encode(raw,source,'byte_shuffle')
 for kind in KINDS[10:]:assert decode(encode(raw,source,kind,anchor))[0]==raw
 return dict(all_12_codecs_bit_exact_on_zero_signed_zero_subnormal_infinity_NaN_payloads=True,warning='Predictive arithmetic transforms reproduced in the same NumPy/IEEE environment; cross-platform arithmetic canonicalization is not separately certified.')

def main():
 started=time.perf_counter();checks=audit();index=json.load(open(ROOT/'snapshots/index.json'));rows=[];dist=[];previous={};best_saved=OUT/'best_frames';best_saved.mkdir(exist_ok=True)
 for src in index:
  raw=(ROOT/'snapshots'/src['filename']).read_bytes();assert hashlib.sha256(raw).hexdigest()==src['sha256'];rawframe=encode(raw,src,'raw');frames={}
  for kind in KINDS[:10]:
   before=time.perf_counter();frame=encode(raw,src,kind);encode_s=time.perf_counter()-before;before=time.perf_counter();back,_=decode(frame);decode_s=time.perf_counter()-before;assert back==raw
   frames[kind]=frame
   tracemalloc.start();f=encode(raw,src,kind);b,_=decode(f);_,peak=tracemalloc.get_traced_memory();tracemalloc.stop()
   rows.append(dict(seed=src['seed'],step=src['step'],codec=kind,active_parameters=src['active_parameters'],raw_state_bytes=len(raw),raw_frame_bytes=len(rawframe),encoded_frame_bytes=len(frame),ratio_to_raw_state=len(frame)/len(raw),ratio_to_raw_frame=len(frame)/len(rawframe),frame_header_and_source_metadata_bytes=HEADER.size+len(metadata(src)),predictor_scale_bytes=8+4*math.ceil(src['active_parameters']/256) if kind=='cross_m2_fit' else 0,embedded_anchor_bytes=0,anchor_step='',roundtrip_bit_exact=True,encode_seconds=encode_s,decode_seconds=decode_s,tracemalloc_encode_decode_peak_bytes=peak))
  best=min(frames,key=lambda k:len(frames[k]));(best_saved/f"seed{src['seed']}_step{src['step']:04d}_{best}.bin").write_bytes(frames[best])
  if src['seed'] in previous:
   prev=previous[src['seed']]
   for kind,anchor in [('temporal_raw_anchor',prev['raw']),('temporal_best_anchor',prev['best'])]:
    before=time.perf_counter();frame=encode(raw,src,kind,anchor);es=time.perf_counter()-before;before=time.perf_counter();back,_=decode(frame);ds=time.perf_counter()-before;assert back==raw
    tracemalloc.start();f=encode(raw,src,kind,anchor);b,_=decode(f);_,pk=tracemalloc.get_traced_memory();tracemalloc.stop()
    rows.append(dict(seed=src['seed'],step=src['step'],codec=kind,active_parameters=src['active_parameters'],raw_state_bytes=len(raw),raw_frame_bytes=len(rawframe),encoded_frame_bytes=len(frame),ratio_to_raw_state=len(frame)/len(raw),ratio_to_raw_frame=len(frame)/len(rawframe),frame_header_and_source_metadata_bytes=HEADER.size+len(metadata(src)),predictor_scale_bytes=0,embedded_anchor_bytes=len(anchor),anchor_step=prev['step'],roundtrip_bit_exact=True,encode_seconds=es,decode_seconds=ds,tracemalloc_encode_decode_peak_bytes=pk))
  previous[src['seed']]=dict(raw=rawframe,best=frames[best],step=src['step'])
  dist+=distributions(raw,src);savecsv(OUT/'compression.csv',rows);savecsv(OUT/'distributions.csv',dist)
  print(src['seed'],src['step'],'best',best,'ratio',round(len(frames[best])/len(rawframe),4),flush=True)
 run=dict(seconds=time.perf_counter()-started,checks=checks,zlib_version=zlib.ZLIB_VERSION,python=sys.version,numpy=np.__version__,script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),snapshot_index_sha256=hashlib.sha256((ROOT/'snapshots/index.json').read_bytes()).hexdigest(),scope='Exact current-state storage on real small-model CPU trajectories. No arbitrary input or 27B compressibility claim. All temporal anchors and all predictor scales are embedded and charged. tracemalloc does not measure native zlib workspace or process RSS.',target_ratio_reference=30/(26895998464*8/2**30),codecs=KINDS)
 (OUT/'run.json').write_text(json.dumps(run,indent=2));print('completed',run['seconds'])
if __name__=='__main__':main()
