"""Exact rational algebra versus the originally prescribed stepwise FP32 recurrence.
No symbolic/numerical linear algebra package is used for exact rank/factorization.
"""
from pathlib import Path
from fractions import Fraction as F
import csv,hashlib,json,math,platform,random,struct,time
import numpy as np
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'results';OUT.mkdir(exist_ok=True)
N=12;STEPS=[1,2,4,8];SEEDS=[1,7,42];RANKS=[1,2,4];f=np.float32
B1=f(.9);B2=f(.999);A1=f(.1);A2=f(.001)
def frac(x):return F(float(x))
FB1,FB2,FA1,FA2=map(frac,[B1,B2,A1,A2])
def word(x):return struct.unpack('<I',struct.pack('<f',x))[0]
def asf(w):return struct.unpack('<f',struct.pack('<I',w))[0]
def rne_integer(num,den):
 q,r=divmod(num,den)
 return q+(2*r>den or (2*r==den and q%2==1))
def round32(x):
 """Direct integer nearest-even binary32 rounding; avoids double-rounding through float64."""
 if not x:return f(0)
 sign=0x80000000 if x<0 else 0;x=abs(x);num,den=x.numerator,x.denominator
 e=num.bit_length()-den.bit_length()
 if (num < (den<<e)) if e>=0 else ((num<<(-e))<den):e-=1
 if e < -126:
  sig=rne_integer(num<<149,den);return f(asf(sign|sig))
 shift=23-e;sig=rne_integer(num<<shift,den) if shift>=0 else rne_integer(num,den<<(-shift))
 if sig==1<<24:sig>>=1;e+=1
 if e>127:return f(asf(sign|0x7f800000))
 return f(asf(sign|((e+127)<<23)|(sig-(1<<23))))
def zmat(n,m):return [[F(0) for _ in range(m)] for _ in range(n)]
def outer_sum(A,B):
 n=len(A);m=len(B);r=len(A[0]);return [[sum((A[i][k]*B[j][k] for k in range(r)),F(0)) for j in range(m)] for i in range(n)]
def to_fraction(a):return [[frac(x) for x in row] for row in a]
def rounded(a):return np.array([[round32(x) for x in row] for row in a],dtype=np.float32)
def rref(a):
 a=[row[:] for row in a];nr=len(a);nc=len(a[0]);row=0;piv=[]
 for c in range(nc):
  p=next((i for i in range(row,nr) if a[i][c]),None)
  if p is None:continue
  a[row],a[p]=a[p],a[row];v=a[row][c];a[row]=[x/v for x in a[row]]
  for i in range(nr):
   if i!=row and a[i][c]:
    v=a[i][c];a[i]=[x-v*y for x,y in zip(a[i],a[row])]
  piv.append(c);row+=1
  if row==nr:break
 return a,piv
def factorize(a):
 rr,piv=rref(a);U=[[row[j] for j in piv] for row in a];V=[[rr[k][j] for k in range(len(piv))] for j in range(len(a[0]))]
 assert outer_sum(U,V)==a
 return len(piv),U,V
def varint(n):
 out=bytearray()
 while n>=128:out.append((n&127)|128);n>>=7
 out.append(n);return bytes(out)
def encode_rationals(values):
 out=bytearray()
 for row in values:
  for x in row:
   n=x.numerator;out+=varint(2*n if n>=0 else -2*n-1);out+=varint(x.denominator)
 return bytes(out)
def rank(a):return len(rref(a)[1])
def fp32_gradient(A,B):
 g=np.zeros((len(A),len(B)),dtype=np.float32)
 for k in range(A.shape[1]):g=np.add(g,np.multiply(A[:,k,None],B[None,:,k],dtype=np.float32),dtype=np.float32)
 return g
def moment_step(m,v,g):
 m=np.add(np.multiply(B1,m,dtype=np.float32),np.multiply(A1,g,dtype=np.float32),dtype=np.float32)
 gg=np.multiply(g,g,dtype=np.float32)
 v=np.add(np.multiply(B2,v,dtype=np.float32),np.multiply(A2,gg,dtype=np.float32),dtype=np.float32)
 return m,v

def mismatch(a,b):
 ua=a.view(np.uint32);ub=b.view(np.uint32)
 def order(u):return np.where((u&0x80000000)!=0,0xffffffff-u,u+np.uint32(0x80000000)).astype(np.int64)
 dist=np.abs(order(ua)-order(ub));return {'different_words':int(np.count_nonzero(ua!=ub)),'total_words':a.size,'max_ulp_distance':int(dist.max()),'max_abs_error':float(np.max(np.abs(a.astype(np.float64)-b.astype(np.float64))))}
def update(m,v,t):
 c1=f(1-.9**t);c2=f(1-.999**t);den=np.add(np.sqrt(np.divide(v,c2,dtype=np.float32)),f(1e-8),dtype=np.float32)
 direction=np.divide(np.divide(m,c1,dtype=np.float32),den,dtype=np.float32)
 # Same common current weights (zero), so this isolates the moment-induced update difference.
 return np.multiply(f(-.003),direction,dtype=np.float32)

def save_matrix(prefix,name,a):
 a.astype('<f4').tofile(OUT/f'{prefix}.{name}.f32')
def lifted(A,B):
 r=len(A[0]);pairs=[(k,l) for k in range(r) for l in range(k,r)]
 U=[[row[k]*row[l]*(1 if k==l else 2) for k,l in pairs] for row in A]
 V=[[row[k]*row[l] for k,l in pairs] for row in B]
 return U,V

def concatenate(history,weights):
 U=[[] for _ in range(N)];V=[[] for _ in range(N)]
 for (A,B),weight in zip(history,weights):
  for i in range(N):U[i].extend(weight*x for x in A[i]);V[i].extend(B[i])
 return U,V

protocol={'shape':[N,N],'seeds':SEEDS,'rank_per_step':RANKS,'reported_steps':STEPS,'factor_source':'Independent deterministic small integer factors /8; all gradient and squared-gradient values exactly representable in binary32 in the main grid.','coefficient_words':{k:word(v) for k,v in [('beta1',B1),('alpha1',A1),('beta2',B2),('alpha2',A2)]},'coefficients_are_actual_stored_fp32':True,'alpha_is_not_recomputed_as_one_minus_beta':True,'exact_arithmetic':'fractions.Fraction, exact Gaussian elimination, directly rounded integer-ratio binary32 nearest-even','scope':'Algebra and state-representation experiment, not a coupled full training or memory peak benchmark. FP32 operations use separate elementwise multiply/add. Rational codec bytes are serialization sizes, not Python RSS.','numpy':np.__version__,'python_platform':platform.platform(),'raw_dense_m_v_bytes':8*N*N,'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
(OUT/'protocol.json').write_text(json.dumps(protocol,indent=2));rows=[];start=time.perf_counter()
for seed in SEEDS:
 for r in RANKS:
  rng=random.Random(seed*100+r);m=np.zeros((N,N),np.float32);v=m.copy();M=zmat(N,N);V=zmat(N,N);hist=[];vhist=[];f64m=np.zeros((N,N),np.float64);f64v=f64m.copy()
  for t in range(1,9):
   ai=np.array([[rng.randrange(-7,8) for _ in range(r)] for _ in range(N)],dtype=np.int32);bi=np.array([[rng.randrange(-7,8) for _ in range(r)] for _ in range(N)],dtype=np.int32)
   A=ai.astype(np.float32)/f(8);B=bi.astype(np.float32)/f(8);AF=to_fraction(A);BF=to_fraction(B);G=outer_sum(AF,BF);g=fp32_gradient(A,B)
   assert to_fraction(g)==G
   gg=np.multiply(g,g,dtype=np.float32);assert to_fraction(gg)==[[x*x for x in row] for row in G]
   AU,BV=lifted(AF,BF);assert outer_sum(AU,BV)==[[x*x for x in row] for row in G]
   hist.append((AF,BF));vhist.append((AU,BV));m,v=moment_step(m,v,g)
   M=[[FB1*M[i][j]+FA1*G[i][j] for j in range(N)] for i in range(N)];V=[[FB2*V[i][j]+FA2*G[i][j]*G[i][j] for j in range(N)] for i in range(N)]
   f64m=float(B1)*f64m+float(A1)*g.astype(np.float64);f64v=float(B2)*f64v+float(A2)*g.astype(np.float64)**2
   prefix=f's{seed}_r{r}_t{t}';save_matrix(prefix,'A',A);save_matrix(prefix,'B',B);save_matrix(prefix,'gradient',g);save_matrix(prefix,'m_seq',m);save_matrix(prefix,'v_seq',v)
   if t not in STEPS:continue
   cm,dm=concatenate(hist,[FA1*FB1**(t-1-s) for s in range(t)]);cv,dv=concatenate(vhist,[FA2*FB2**(t-1-s) for s in range(t)])
   assert outer_sum(cm,dm)==M and outer_sum(cv,dv)==V
   mr,mu,mv=factorize(M);vr,vu,vv=factorize(V);mf=rounded(M);vf=rounded(V)
   codec=struct.pack('<4I',N,N,mr,vr)+encode_rationals(mu)+encode_rationals(mv)+encode_rationals(vu)+encode_rationals(vv)
   (OUT/f'{prefix}.real_m_v_rank_factor.rational').write_bytes(codec)
   for name,a in [('m_real_once',mf),('v_real_once',vf),('update_seq',update(m,v,t)),('update_real_once',update(mf,vf,t))]:save_matrix(prefix,name,a)
   row={'seed':seed,'r':r,'steps':t,'gradient_rank_real':rank(G),'gradient_rank_fp32_exact':rank(to_fraction(g)),'m_rank_real':mr,'v_rank_real':vr,'m_rank_sequential_fp32_exact':rank(to_fraction(m)),'v_rank_sequential_fp32_exact':rank(to_fraction(v)),'m_concat_columns':t*r,'v_lift_columns':t*r*(r+1)//2,'real_factor_identity_exact':True,'gradient_and_square_no_rounding_confound':True,'m_bits':mismatch(m,mf),'v_bits':mismatch(v,vf),'update_bits':mismatch(update(m,v,t),update(mf,vf,t)),'fp64_vs_rational_m_bits':mismatch(f64m.astype(np.float32),mf),'fp64_vs_rational_v_bits':mismatch(f64v.astype(np.float32),vf),'optimistic_fp32_rank_factor_payload_bytes':4*(N+N)*(mr+vr),'uncompressed_concat_factor_fp32_payload_bytes':4*(N+N)*(t*r+t*r*(r+1)//2),'original_gradient_history_factor_payload_bytes':4*(N+N)*t*r,'actual_rational_codec_bytes':len(codec),'raw_dense_fp32_m_v_bytes':8*N*N}
   rows.append(row)
  print('done',seed,r,flush=True)
(OUT/'grid.json').write_text(json.dumps(rows,indent=2))
flat=[]
for q in rows:
 a={}
 for k,v in q.items():
  if isinstance(v,dict):a.update({k+'_'+x:y for x,y in v.items()})
  else:a[k]=v
 flat.append(a)
with (OUT/'grid.csv').open('w',newline='') as fp:w=csv.DictWriter(fp,fieldnames=list(flat[0]));w.writeheader();w.writerows(flat)
summary={'cases':len(rows),'real_factor_identity_all_exact':True,'m_mismatch_cases':sum(q['m_bits']['different_words']>0 for q in rows),'v_mismatch_cases':sum(q['v_bits']['different_words']>0 for q in rows),'update_mismatch_cases':sum(q['update_bits']['different_words']>0 for q in rows),'m_different_words':sum(q['m_bits']['different_words'] for q in rows),'v_different_words':sum(q['v_bits']['different_words'] for q in rows),'total_words_per_moment_compared':len(rows)*N*N,'fp64_consolidation_matches_direct_rational_rounding':all(q['fp64_vs_rational_m_bits']['different_words']==0 and q['fp64_vs_rational_v_bits']['different_words']==0 for q in rows),'seconds':time.perf_counter()-start}
(OUT/'summary.json').write_text(json.dumps(summary,indent=2));print('DONE',json.dumps(summary),flush=True)
