"""Own self-contained offline FP32 predictor+lossless residual frames.
No current gradient, factor history, or outside predictor is available to decode.
SVD is encoder-only; decoder uses stored FP32 factors and explicit K-order.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
from pathlib import Path
import json,struct,hashlib,zlib,time,csv,sys,platform
import numpy as np
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'results';FRAMES=OUT/'frames'
HEADER=struct.Struct('<8sIQQ');MAGIC=b'ADMR13\0\0';TARGET=0.14970746839495358

def sha(raw):return hashlib.sha256(raw).hexdigest()
def shuffle(raw,width):return np.frombuffer(raw,np.uint8).reshape(-1,width).T.copy().tobytes()
def unshuffle(raw,width):return np.frombuffer(raw,np.uint8).reshape(width,-1).T.copy().tobytes()
def compress(raw,width):return zlib.compress(shuffle(raw,width),9)
def expand(raw,width):return unshuffle(zlib.decompress(raw),width)
def order(w):
 w=np.asarray(w,dtype=np.uint32)
 return np.where((w>>31)!=0,np.bitwise_not(w),w^np.uint32(0x80000000)).astype(np.uint64)
def unorder(u):
 u=np.asarray(u,dtype=np.uint32)
 return np.where((u>>31)!=0,u^np.uint32(0x80000000),np.bitwise_not(u)).astype(np.uint32)
def varints(values):
 out=bytearray()
 for x in values:
  x=int(x)
  while x>=128:out.append((x&127)|128);x>>=7
  out.append(x)
 return bytes(out)
def unvarints(raw,count):
 vals=[];value=shift=0
 for b in raw:
  value|=(b&127)<<shift
  if b<128:
   if value>8589934590:raise ValueError('Ordered delta exceeds 33-bit signed range')
   vals.append(value);value=shift=0
  else:
   shift+=7
   if shift>=35:raise ValueError('Oversized residual varint')
 if shift or len(vals)!=count:raise ValueError('Residual count/truncation')
 return np.array(vals,dtype=np.uint64)
def outer32(u,v):
 out=np.zeros((u.shape[0],v.shape[0]),dtype=np.float32)
 for k in range(u.shape[1]):
  out=np.add(out,np.multiply(u[:,k,None],v[None,:,k],dtype=np.float32),dtype=np.float32)
 return out

def make_predictor(raw,shapes,kind,rank=0,known=None):
 w=np.frombuffer(raw,dtype='<u4');p=len(w)//2
 if kind=='zero':return {'kind':'zero'},b'',np.zeros_like(w),{}
 if kind=='bf16_trunc':
  high=(w>>16).astype('<u2');return {'kind':kind},compress(high.tobytes(),2),(high.astype(np.uint32)<<16),{}
 if kind!='lowrank_f32':raise ValueError(kind)
 f=w.view('<f4');specs=[];parts=[];preds=[];offset=0;errors=[]
 for state in range(2):
  for ti,shape in enumerate(shapes):
   size=int(np.prod(shape));a=f[offset:offset+size].reshape(shape);offset+=size
   if len(shape)!=2:
    parts.append(a.ravel());preds.append(a.ravel());specs.append({'kind':'exact_vector','shape':shape});continue
   n,m=shape;k=min(rank,n,m)
   if not np.all(np.isfinite(a)):raise ValueError('SVD predictor rejects nonfinite matrix')
   if known is None:
    u,s,vh=np.linalg.svd(a.astype(np.float64),full_matrices=False)
    uu=(u[:,:k]*s[:k]).astype(np.float32);vv=vh[:k,:].T.astype(np.float32)
   else:uu,vv=known[state][ti];k=uu.shape[1]
   pred=outer32(uu,vv);parts.extend([uu.ravel(),vv.ravel()]);preds.append(pred.ravel());specs.append({'kind':'uv','shape':shape,'rank':k})
   denom=float(np.sum(a.astype(np.float64)**2));err=float(np.sum((a.astype(np.float64)-pred.astype(np.float64))**2))
   errors.append({'state':state,'tensor':ti,'rank':k,'relative_l2_error':(err/denom)**.5 if denom else err**.5})
 assert offset==len(f)
 plain=np.concatenate(parts).astype('<f4').tobytes();pred=np.concatenate(preds).astype('<f4').view('<u4')
 return {'kind':kind,'tensors':specs,'decoder':'separate_fp32_mul_add_K_ascending'},compress(plain,4),pred,{'factor_or_exact_vector_raw_bytes':len(plain),'matrix_errors':errors}

def predictor_decode(spec,payload,count):
 kind=spec['kind']
 if kind=='zero':
  if payload:raise ValueError('Nonempty zero predictor')
  return np.zeros(count,dtype=np.uint32)
 if kind=='bf16_trunc':
  h=np.frombuffer(expand(payload,2),dtype='<u2')
  if len(h)!=count:raise ValueError('BF16 predictor count')
  return h.astype(np.uint32)<<16
 if kind!='lowrank_f32':raise ValueError('Unknown predictor')
 f=np.frombuffer(expand(payload,4),dtype='<f4');offset=0;parts=[]
 for spec in spec['tensors']:
  shape=spec['shape'];size=int(np.prod(shape))
  if spec['kind']=='exact_vector':parts.append(f[offset:offset+size]);offset+=size
  elif spec['kind']=='uv':
   n,m=shape;k=spec['rank'];u=f[offset:offset+n*k].reshape(n,k);offset+=n*k;v=f[offset:offset+m*k].reshape(m,k);offset+=m*k
   parts.append(outer32(u,v).ravel())
  else:raise ValueError('Unknown tensor predictor')
 if offset!=len(f):raise ValueError('Unconsumed predictor bytes')
 out=np.concatenate(parts).astype('<f4').view('<u4')
 if len(out)!=count:raise ValueError('Factor predictor count')
 return out

def make_frame(raw,source,predkind,residual,rank=0,known=None):
 words=np.frombuffer(raw,dtype='<u4');diag={}
 if predkind=='baseline':
  spec={'kind':'none'};predpayload=b'';pred=None;payload=compress(raw,4)
 else:
  spec,predpayload,pred,diag=make_predictor(raw,source['shapes'],predkind,rank,known)
  if residual=='xor32':payload=compress((words^pred).astype('<u4').tobytes(),4)
  elif residual=='ordered_delta':
   delta=order(words).astype(np.int64)-order(pred).astype(np.int64)
   zigzag=np.where(delta<0,-2*delta-1,2*delta).astype(np.uint64)
   payload=zlib.compress(varints(zigzag),9)
  else:raise ValueError(residual)
  d=order(words).astype(np.int64)-order(pred).astype(np.int64);ad=np.abs(d)
  diag.update({'exact_predictor_fraction':float(np.mean(words==pred)),'nonzero_correction_fraction':float(np.mean(words!=pred)),'ordered_delta_abs_median':float(np.median(ad)),'ordered_delta_abs_p95':float(np.quantile(ad,.95)),'ordered_delta_abs_max':int(ad.max())})
 meta={'version':1,'case':source['case'],'family':source['family'],'step':source['step'],'shapes':source['shapes'],'word_count':len(words),'layout':'m_then_v_little_endian_FP32','source_sha256':sha(raw),'predictor':spec,'residual':residual,'predictor_compression':'zlib9_byte_shuffle','residual_compression':'zlib9'}
 metadata=json.dumps(meta,sort_keys=True,separators=(',',':')).encode();frame=HEADER.pack(MAGIC,len(metadata),len(predpayload),len(payload))+metadata+predpayload+payload
 return frame,{'header_and_metadata_bytes':HEADER.size+len(metadata),'stored_predictor_bytes':len(predpayload),'stored_residual_bytes':len(payload),**diag}

def decode(frame):
 if len(frame)<HEADER.size:raise ValueError('Short header')
 magic,ml,pl,rl=HEADER.unpack_from(frame)
 if magic!=MAGIC or HEADER.size+ml+pl+rl!=len(frame):raise ValueError('Bad frame magic/size')
 meta=json.loads(frame[HEADER.size:HEADER.size+ml]);n=meta['word_count'];off=HEADER.size+ml;pb=frame[off:off+pl];rb=frame[off+pl:];spec=meta['predictor']
 if spec['kind']=='none':
  if pl or meta['residual']!='baseline':raise ValueError('Baseline header')
  raw=expand(rb,4)
 else:
  p=predictor_decode(spec,pb,n)
  if meta['residual']=='xor32':
   residual=np.frombuffer(expand(rb,4),dtype='<u4')
   if len(residual)!=n:raise ValueError('XOR count')
   w=p^residual
  elif meta['residual']=='ordered_delta':
   zz=unvarints(zlib.decompress(rb),n);d=(zz>>1).astype(np.int64);d=np.where(zz&1,-d-1,d)
   o=order(p).astype(np.int64)+d
   if np.any(o<0) or np.any(o>0xffffffff):raise ValueError('Ordered reconstruction overflow')
   w=unorder(o)
  else:raise ValueError('Unknown residual')
  raw=w.astype('<u4').tobytes()
 if len(raw)!=4*n or sha(raw)!=meta['source_sha256']:raise ValueError('Decoded raw SHA mismatch')
 return raw,meta

def entropy(vals):
 counts=np.bincount(vals.astype(np.int64));pr=counts[counts>0]/len(vals)
 return float(-np.sum(pr*np.log2(pr)))
def diagnostic(raw,source):
 w=np.frombuffer(raw,dtype='<u4');p=len(w)//2;rows=[]
 for state,x in zip(['m','v'],[w[:p],w[p:]]):
  rows.append({'case':source['case'],'family':source['family'],'step':source['step'],'state':state,'words':p,'zero_fraction':float(np.mean((x&0x7fffffff)==0)),'negative_fraction':float(np.mean(x>>31)),'subnormal_fraction':float(np.mean(((x&0x7f800000)==0)&((x&0x7fffff)!=0))),'low8_marginal_entropy_bits':entropy(x&255),'next8_marginal_entropy_bits':entropy((x>>8)&255),'sum_low16_individual_bit_entropies':sum(entropy((x>>i)&1) for i in range(16)),'low16_exact_zero_fraction':float(np.mean((x&65535)==0))})
 return rows

def savecsv(path,rows):
 keys=list(dict.fromkeys(k for row in rows for k in row))
 with path.open('w',newline='') as stream:
  writer=csv.DictWriter(stream,fieldnames=keys);writer.writeheader();writer.writerows(rows)
def run_case(raw,src,rows,known=None,only=None):
 methods=[('baseline','baseline',0),('bf16_trunc','xor32',0),('bf16_trunc','ordered_delta',0)]+[('lowrank_f32',res,r) for r in [1,2,4,8] for res in ['xor32','ordered_delta']]
 if only is not None:methods=only
 for pred,res,r in methods:
  started=time.perf_counter();frame,account=make_frame(raw,src,pred,res,r,known if pred=='lowrank_f32' else None);encode_s=time.perf_counter()-started
  started=time.perf_counter();back,meta=decode(frame);decode_s=time.perf_counter()-started;assert back==raw
  name=f"{src['case']}.{pred}.r{r}.{res}.frame";(FRAMES/name).write_bytes(frame)
  row={'case':src['case'],'family':src['family'],'step':src['step'],'predictor':pred,'rank_requested':r,'residual':res,'raw_state_bytes':len(raw),'encoded_frame_bytes':len(frame),'ratio_to_raw_state':len(frame)/len(raw),'meets_14_97074684_percent':len(frame)/len(raw)<=TARGET,'roundtrip_exact':True,'encode_seconds':encode_s,'decode_seconds':decode_s,'frame':name,**account}
  assert row['header_and_metadata_bytes']+row['stored_predictor_bytes']+row['stored_residual_bytes']==len(frame)
  if 'matrix_errors' in row:row['matrix_errors']=json.dumps(row['matrix_errors'],separators=(',',':'))
  rows.append(row)

if __name__=='__main__':
 OUT.mkdir(exist_ok=True);FRAMES.mkdir(exist_ok=True);start=time.perf_counter();rows=[];dist=[]
 sources=json.loads((ROOT/'inputs/index.json').read_text())
 for src in sources:
  raw=(ROOT/'inputs'/src['filename']).read_bytes();assert sha(raw)==src['sha256'];run_case(raw,src,rows);dist+=diagnostic(raw,src)
  savecsv(OUT/'compression.csv',rows);print(src['case'],'best ratio',min(q['ratio_to_raw_state'] for q in rows if q['case']==src['case']),flush=True)
 (OUT/'compression.json').write_text(json.dumps(rows,indent=2));savecsv(OUT/'distributions.csv',dist);(OUT/'distributions.json').write_text(json.dumps(dist,indent=2))
 run={'source_sha256':sha(Path(__file__).read_bytes()),'source_index_sha256':sha((ROOT/'inputs/index.json').read_bytes()),'python':sys.version,'numpy':np.__version__,'zlib':zlib.ZLIB_VERSION,'platform':platform.platform(),'cases':len(sources),'frames':len(rows),'all_roundtrips_exact':True,'seconds':time.perf_counter()-start,'target_ratio':TARGET,'target_interpretation':'30GiB divided by earlier26895998464 text parameters*8B raw FP32 m/v, not a measured feasible whole-model budget.','scope':'Offline snapshot representation; every predictor is in frame. No online memory peak certification, no full training replay, no NPU.'}
 (OUT/'run.json').write_text(json.dumps(run,indent=2));print(json.dumps(run,indent=2))
