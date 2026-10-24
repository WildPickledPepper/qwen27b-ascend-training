"""Single resident compressed arena and fixed-block FP32 optimizer updates.
CPU emulation only. No state offload, external backing, or full decoded m/v
inside Arena. Reference mode is a separate process and writes only hashes.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1');os.environ.setdefault('OMP_NUM_THREADS','1')
from pathlib import Path
import argparse,ctypes,hashlib,json,math,resource,struct,sys,time,tracemalloc,zlib
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'stage4'))
import mlp_teacher as model
s=model.s
HEADER=struct.Struct('<QQQQ') # N, B, t, used

def shuffle(raw):return np.frombuffer(raw,dtype=np.uint8).reshape(-1,4).T.copy().tobytes()
def unshuffle(raw):return np.frombuffer(raw,dtype=np.uint8).reshape(4,-1).T.copy().tobytes()
def block_encode(m,v):
 raw=m.astype('<f4',copy=False).tobytes()+v.astype('<f4',copy=False).tobytes()
 compressed=zlib.compress(shuffle(raw),6)
 return (compressed,1) if len(compressed)<len(raw) else (raw,0)
def block_decode(data,flag,count):
 raw=unshuffle(zlib.decompress(data)) if flag else data
 assert len(raw)==8*count
 a=np.frombuffer(raw,dtype='<f4');return a[:count],a[count:]
def rss():
 with open('/proc/self/statm') as f:return int(f.read().split()[1])*os.sysconf('SC_PAGE_SIZE')
def peakrss():return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024

def hash_state_full(m,v,b):
 h=hashlib.sha256()
 for start in range(0,len(m),b):
  h.update(memoryview(m[start:start+b]).cast('B'));h.update(memoryview(v[start:start+b]).cast('B'))
 return h.hexdigest()

class Arena:
 def __init__(self,n,b,fraction):
  self.n=n;self.b=b;self.nb=(n+b-1)//b;self.capacity=math.floor(8*n*fraction)
  self.buffer=bytearray(self.capacity) # never resized
  self.lengths=np.zeros(self.nb,dtype=np.uint32);self.flags=np.zeros(self.nb,dtype=np.uint8)
  self.proposed_lengths=np.zeros(self.nb,dtype=np.uint32) # charged preflight workspace
  self.header=bytearray(HEADER.size);self.t=0;self.used=0;self.max_used=0;self.max_prefix_need=0;self.moves=0;self.moved_bytes=0;self.fallback_blocks=0
  # Create initial zero state one block at a time; no full-sized m/v exists.
  for i,start in enumerate(range(0,n,b)):
   count=min(b,n-start);z=np.zeros(count,dtype=np.float32);data,flag=block_encode(z,z)
   if self.used+len(data)>self.capacity:raise MemoryError('Initial zero encoding cannot fit fixed arena')
   self.buffer[self.used:self.used+len(data)]=data;self.lengths[i]=len(data);self.flags[i]=flag;self.used+=len(data)
  self.max_used=self.used;self._write_header()
 def _write_header(self):HEADER.pack_into(self.header,0,self.n,self.b,self.t,self.used)
 def metadata_bytes(self):return len(self.header)+self.lengths.nbytes+self.flags.nbytes+self.proposed_lengths.nbytes
 def accounted_python_storage(self):return sys.getsizeof(self.buffer)+sys.getsizeof(self.header)+sys.getsizeof(self.lengths)+sys.getsizeof(self.flags)+sys.getsizeof(self.proposed_lengths)
 def numeric_reserved(self):return self.capacity+self.metadata_bytes()
 def block(self,i,offset):
  count=min(self.b,self.n-i*self.b);data=bytes(memoryview(self.buffer)[offset:offset+int(self.lengths[i])])
  return block_decode(data,int(self.flags[i]),count)
 def state_hash(self):
  h=hashlib.sha256();off=0
  for i in range(self.nb):
   m,v=self.block(i,off);h.update(memoryview(m).cast('B'));h.update(memoryview(v).cast('B'));off+=int(self.lengths[i])
  assert off==self.used;return h.hexdigest()
 def step(self,g,w,t):
  # Preflight all growth without mutating state or weights. Charges an O(N/B)
  # array of lengths, but no new state arena and no decoded full m/v.
  offset=0;delta=0;need=self.used;raw_fallback=0
  for i,start in enumerate(range(0,self.n,self.b)):
   count=min(self.b,self.n-start);m,v=self.block(i,offset);mm,vv=s.moments(m,v,g[start:start+count]);data,flag=block_encode(mm,vv)
   self.proposed_lengths[i]=len(data);delta+=len(data)-int(self.lengths[i]);need=max(need,self.used+delta);offset+=int(self.lengths[i]);raw_fallback+=int(flag==0)
  self.max_prefix_need=max(self.max_prefix_need,need)
  if need>self.capacity:return dict(ok=False,required_prefix_bytes=need,capacity=self.capacity,old_state_unchanged=True)
  # Drop last preflight block before commit. Every temporary is block sized.
  del m,v,mm,vv,data
  offset=0
  for i,start in enumerate(range(0,self.n,self.b)):
   count=min(self.b,self.n-start);oldlen=int(self.lengths[i]);m,v=self.block(i,offset);mm,vv=s.moments(m,v,g[start:start+count]);d,neww=s.update(mm,vv,w[start:start+count],t);data,flag=block_encode(mm,vv)
   newlen=len(data);assert newlen==int(self.proposed_lengths[i]);change=newlen-oldlen;tail=self.used-offset-oldlen
   if change and tail:
    assert 0<=offset+newlen and self.used+change<=self.capacity
    # C memmove is overlap-safe and creates no suffix-sized Python copy.
    ptr=ctypes.addressof(ctypes.c_char.from_buffer(self.buffer))
    ctypes.memmove(ptr+offset+newlen,ptr+offset+oldlen,tail);self.moves+=1;self.moved_bytes+=tail
   self.buffer[offset:offset+newlen]=data # same-length assignment never resizes
   w[start:start+count]=neww;self.used+=change;self.max_used=max(self.max_used,self.used);self.lengths[i]=newlen;self.flags[i]=flag;offset+=newlen
  assert offset==self.used and len(self.buffer)==self.capacity
  self.t=t;self.fallback_blocks+=raw_fallback;self._write_header()
  return dict(ok=True,used=self.used,required_prefix_bytes=need)

def get_inputs(mode,n,seed):
 if mode=='mlp':
  model.H=1024;model.SHAPES=[(model.DIN,model.H),(model.H,),(model.H,model.K),(model.K,)];model.SIZES=[int(np.prod(x)) for x in model.SHAPES];model.P=sum(model.SIZES);model.PAD=model.P
  parts,w,batches,hashes=model.make(seed)
  return len(w),w,dict(parts=parts,batches=batches,hashes=hashes)
 r=np.random.default_rng(seed+9000);w=r.normal(0,.1,n).astype(np.float32);g=np.empty(n,dtype=np.float32)
 return n,w,dict(rng=np.random.default_rng(seed+5000),g=g)

def next_gradient(mode,t,w,inputs):
 if mode=='mlp':
  train=inputs['parts'][0];idx=inputs['batches'][t-1];_,_,g=model.lossgrad(w,train[0][idx],train[1][idx]);s.LR=np.float32(model.lr_at(t,.001));return g
 inputs['rng'].standard_normal(size=len(w),dtype=np.float32,out=inputs['g']);s.LR=np.float32(1e-4)
 return inputs['g']

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--mode',choices=['mlp','layout'],default='mlp');ap.add_argument('--impl',choices=['reference','arena'],required=True);ap.add_argument('--n',type=int,default=4194304);ap.add_argument('--steps',type=int,default=4096);ap.add_argument('--block',type=int,default=4096);ap.add_argument('--capacity',type=float,default=.9);ap.add_argument('--seed',type=int,default=101);ap.add_argument('--out',required=True);ap.add_argument('--reference');args=ap.parse_args();out=Path(args.out);out.parent.mkdir(parents=True,exist_ok=True)
 n,w,inputs=get_inputs(args.mode,args.n,args.seed);raw_bytes=8*n
 # Inputs/model allocations are outside the optimizer measurement. Generate and
 # keep the first gradient before measurement to charge both paths consistently.
 g=next_gradient(args.mode,1,w,inputs);baseline_rss=rss();baseline_peak=peakrss();before=time.perf_counter();tracemalloc.start();samples=[];status='completed';failure=None;checkpoint_set={1,2,args.steps}|({128,1024,2048,4095} if args.mode=='mlp' else set())
 reference=json.load(open(args.reference)) if args.reference else None
 if args.impl=='reference':m=np.zeros(n,dtype=np.float32);v=np.zeros(n,dtype=np.float32);arena=None
 else:arena=Arena(n,args.block,args.capacity);m=v=None
 max_step_peak=0;initial_step_peak=0;max_later_step_peak=0;all_exact=True;opt_seconds=0;input_seconds=0
 for t in range(1,args.steps+1):
  if t>1:
   # Temporarily stop tracing gradient work; reset before optimizer scope.
   tracemalloc.stop();tick=time.perf_counter();g=next_gradient(args.mode,t,w,inputs);input_seconds+=time.perf_counter()-tick;tracemalloc.start()
  tick=time.perf_counter()
  if args.impl=='reference':
   # Fair FP32 baseline also uses the same fixed block workspace and in-place weights.
   for start in range(0,n,args.block):
    sl=slice(start,min(n,start+args.block));mm,vv=s.moments(m[sl],v[sl],g[sl]);dd,ww=s.update(mm,vv,w[sl],t);m[sl]=mm;v[sl]=vv;w[sl]=ww
  else:
   prevhash=arena.state_hash() if args.capacity<.85 else None;prevw=hashlib.sha256(memoryview(w).cast('B')).hexdigest() if args.capacity<.85 else None
   response=arena.step(g,w,t)
   if not response['ok']:
    assert prevhash is None or arena.state_hash()==prevhash
    assert prevw is None or hashlib.sha256(memoryview(w).cast('B')).hexdigest()==prevw
    status='capacity_rejected_atomically';failure=dict(step=t,**response);break
  opt_seconds+=time.perf_counter()-tick
  _,pk=tracemalloc.get_traced_memory();max_step_peak=max(max_step_peak,pk)
  if t==1:initial_step_peak=pk
  else:max_later_step_peak=max(max_later_step_peak,pk)
  if t in checkpoint_set:
   state_hash=hash_state_full(m,v,args.block) if args.impl=='reference' else arena.state_hash();wh=hashlib.sha256(memoryview(w).cast('B')).hexdigest();row=dict(step=t,state_sha256=state_hash,weights_sha256=wh,used_bytes=raw_bytes if arena is None else arena.used)
   if reference:
    ref=next(x for x in reference['checkpoints'] if x['step']==t);match=state_hash==ref['state_sha256'] and wh==ref['weights_sha256'];all_exact &=match;assert match,('reference mismatch',t);row['bit_equal_reference']=match
   samples.append(row);print(args.mode,args.impl,args.block,args.capacity,t,'bytes',row['used_bytes'],flush=True)
  # No checkpoint digest or reference file contains recoverable m/v; hashes only.
 tracemalloc.stop();current_rss=rss();max_rss=peakrss()
 if arena:
  reserved=arena.numeric_reserved();storage=arena.accounted_python_storage();meta=arena.metadata_bytes();maxused=arena.max_used;prefix=arena.max_prefix_need;moved=arena.moved_bytes;fallback=arena.fallback_blocks
 else:reserved=raw_bytes;storage=sys.getsizeof(m)+sys.getsizeof(v);meta=0;maxused=raw_bytes;prefix=raw_bytes;moved=0;fallback=0
 result=dict(mode=args.mode,impl=args.impl,n=n,steps_requested=args.steps,steps_completed=t if status=='completed' else t-1,block=args.block,capacity_fraction=args.capacity,status=status,failure=failure,raw_state_bytes=raw_bytes,numeric_reserved_storage_bytes=reserved,python_accounted_storage_bytes=storage,metadata_and_preflight_lengths_bytes=meta,max_observed_payload_bytes=maxused,max_prefix_requirement_bytes=prefix,unused_reserved_capacity_bytes=(arena.capacity-arena.used) if arena else 0,fragmentation_bytes=0,second_arena_bytes=0,raw_fallback_coordinate_blocks=fallback,total_inplace_memmove_bytes=moved,optimizer_seconds=opt_seconds,input_seconds=input_seconds,total_seconds=time.perf_counter()-before,max_optimizer_scope_tracemalloc_bytes=max_step_peak,initial_optimizer_step_tracemalloc_bytes=initial_step_peak,max_later_optimizer_scratch_tracemalloc_bytes=max_later_step_peak,storage_plus_later_traced_scratch_bytes=storage+max_later_step_peak,rss_before_optimizer=baseline_rss,rss_after_optimizer=current_rss,process_peak_rss_before=baseline_peak,process_peak_rss_after=max_rss,process_peak_increase_bytes=max(0,max_rss-baseline_peak),all_recorded_checkpoints_bit_equal=all_exact,checkpoints=samples,script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),limits='CPU emulation, not GPU/NPU execution. tracemalloc peak resets per optimizer scope: initial step includes storage creation; later steps include newly allocated scratch but not pre-existing resident storage. Compare storage plus max later scratch, not just a single allocator counter. Process RSS includes model/gradient generation; references run separately. No state reads from disk or external backing during updates. Layout mode is synthetic scaling, not training-quality evidence.')
 out.write_text(json.dumps(result,indent=2));print('RESULT',out,status,'seconds',result['total_seconds'],flush=True)
if __name__=='__main__':main()
