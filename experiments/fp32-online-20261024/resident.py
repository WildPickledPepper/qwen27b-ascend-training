"""Resident ENEC FP32 weights + packed 4-bit AdamW moments; no host state fallback.

Custom backward consumes each weight gradient before discarding it. CPU objects
contain names, shapes, headers and input-token IDs only. No BF16 compute weights.
"""
import ctypes, json, math, struct, time
from pathlib import Path
import torch
import torch.nn.functional as F

NF4=[-1.,-.6961928009986877,-.5250730514526367,-.39491748809814453,-.28444138169288635,
     -.18477343022823334,-.09105003625154495,0.,.07958029955625534,.16093020141124725,
     .24611230194568634,.33791524171829224,.44070982933044434,.5626170039176941,.7229568362236023,1.]
GUARD=65536

class Codec:
    def __init__(self, path):
        self.lib=ctypes.CDLL(str(path)); self.lib.online_error.restype=ctypes.c_char_p
        self.lib.online_capacity.argtypes=[ctypes.c_uint32];self.lib.online_capacity.restype=ctypes.c_uint64
        self.lib.online_decompress.argtypes=[ctypes.c_void_p,ctypes.c_int,ctypes.c_void_p,ctypes.c_void_p,ctypes.c_void_p]
        self.lib.online_compress.argtypes=[ctypes.c_void_p,ctypes.c_int,ctypes.c_void_p,ctypes.c_uint32,
                                        ctypes.c_void_p,ctypes.c_void_p,ctypes.c_void_p,ctypes.c_void_p,ctypes.c_void_p]
        self.work={};self.fallbacks=0;self.encodes=0;self.decodes=0
    def check(self,status):
        if status:raise RuntimeError(self.lib.online_error().decode())
    def workspace(self,device,nbytes):
        cap=self.lib.online_capacity(nbytes)
        old=self.work.get(device)
        if old is None or old['cap']<cap:
            self.work.pop(device,None)
            self.work[device]={'cap':cap,'scratch':torch.zeros(cap,dtype=torch.uint8,device=device),
                'packed':torch.zeros(cap,dtype=torch.uint8,device=device),
                'hist':torch.zeros(48*256*4,dtype=torch.uint8,device=device),
                'sizes':torch.zeros(48*32,dtype=torch.uint8,device=device),
                'verify':torch.empty((64<<20)//4,dtype=torch.float32,device=device)}
        return self.work[device]
    def decode(self,frame,output=None):
        device=frame['data'].device
        with torch.npu.device(device):
            n=frame['n'];out=output if output is not None else torch.empty(n,dtype=torch.float32,device=device)
            if frame['kind']=='raw':out.view(torch.uint8).copy_(frame['data'][:n*4])
            else:
                h=ctypes.create_string_buffer(frame['header'],32)
                self.check(self.lib.online_decompress(torch.npu.current_stream(device).npu_stream,device.index,
                           h,frame['data'].data_ptr(),out.data_ptr()))
                self.decodes+=1
            return out
    @torch.no_grad()
    def encode(self,w):
        w=w.contiguous().view(-1);n=w.numel();nb=n*4;device=w.device
        if nb<65536 or nb%65536:
            return {'kind':'raw','n':n,'bytes':nb,'data':w.view(torch.uint8).clone(),'header':None}
        with torch.npu.device(device):
            ws=self.workspace(device,nb);h=ctypes.create_string_buffer(32)
            self.check(self.lib.online_compress(torch.npu.current_stream(device).npu_stream,device.index,w.data_ptr(),nb,
                       ws['scratch'].data_ptr(),ws['packed'].data_ptr(),ws['hist'].data_ptr(),ws['sizes'].data_ptr(),h))
            header=bytes(h.raw);size=struct.unpack_from('<I',header,20)[0]
            assert 32<=size<=ws['cap']
            candidate={'kind':'enec','n':n,'bytes':size,'data':ws['packed'],'header':header}
            restored=self.decode(candidate,ws['verify'][:n])
            exact=torch.equal(restored.view(torch.int32),w.view(torch.int32))
            del restored
            self.encodes+=1
            if not exact or size>=nb:
                self.fallbacks+=1
                return {'kind':'raw','n':n,'bytes':nb,'data':w.view(torch.uint8).clone(),'header':None}
            saved=torch.zeros(size+GUARD,dtype=torch.uint8,device=device)
            saved[:size].copy_(ws['packed'][:size])
            torch.npu.current_stream(device).synchronize()
            return {'kind':'enec','n':n,'bytes':size,'data':saved,'header':header}

class QuantMoments:
    def __init__(self,n,device,block=128,lr=1e-5,wd=.01,window=1048576):
        self.n=n;self.block=block;self.lr=lr;self.wd=wd;self.window=window//block*block
        self.pad=math.ceil(n/block)*block
        self.codes=torch.full((self.pad,),7,dtype=torch.uint8,device=device)
        self.scales=torch.zeros((3,self.pad//block),dtype=torch.float32,device=device)
        self.book=torch.tensor(NF4,dtype=torch.float32,device=device)
        self.updated=0
    def bytes(self):return self.codes.numel()+self.scales.numel()*4+64
    @torch.no_grad()
    def update(self,start,w,g,step):
        assert start%self.block==0 and w.numel()==g.numel()
        b=self.block
        for off in range(0,w.numel(),self.window):
            length=min(self.window,w.numel()-off);end=start+off+length
            # Matrices align to full blocks; tiny native parameters use separate FP32 moments.
            assert length%b==0
            chunk=w[off:off+length];grad=g[off:off+length].reshape(-1,b)
            if not torch.isfinite(grad).all().item():raise FloatingPointError('Nonfinite gradient; update aborted')
            cp=self.codes[start+off:end].float().reshape(-1,b)
            vi=torch.floor(cp/16).long();mi=(cp-vi.float()*16).long()
            meta=self.scales[:,(start+off)//b:end//b];scale,lo,hi=meta.unbind(0)
            m=self.book[mi]*scale[:,None]
            safe_lo=lo.clamp_min(torch.finfo(torch.float32).tiny)
            safe_hi=hi.clamp_min(torch.finfo(torch.float32).tiny)
            logs=safe_lo.log();span=safe_hi.log()-logs
            v=torch.exp(logs[:,None]+span[:,None]*(vi.float()-1).clamp_min(0)/14)
            v=torch.where(vi==0,0.,v)
            v=torch.where(vi==1,lo[:,None],v);v=torch.where(vi==15,hi[:,None],v)
            m.mul_(.9).add_(grad,alpha=.1);v.mul_(.999).addcmul_(grad,grad,value=.001)
            chunk.mul_(1-self.lr*self.wd).addcdiv_(m.reshape(-1)/(1-.9**step),
                (v.reshape(-1)/(1-.999**step)).sqrt().add_(1e-8),value=-self.lr)
            newscale=m.abs().amax(1)
            norm=torch.where(newscale[:,None]>0,m/newscale[:,None].clamp_min(torch.finfo(torch.float32).tiny),0.).clamp(-1,1)
            right=torch.bucketize(norm.contiguous(),self.book).clamp(0,15)
            left=(right-1).clamp_min(0);lower=self.book[left];upper=self.book[right]
            prob=torch.where(upper>lower,(norm-lower)/(upper-lower).clamp_min(1e-30),0.).clamp(0,1)
            mcode=torch.where(torch.rand_like(prob)<prob,right,left)
            pos=v>0
            newlo=torch.where(pos,v,torch.inf).amin(1);newlo=torch.where(torch.isfinite(newlo),newlo,0.)
            newhi=v.amax(1)
            logs=newlo.clamp_min(torch.finfo(torch.float32).tiny).log()
            span=newhi.clamp_min(torch.finfo(torch.float32).tiny).log()-logs
            location=torch.where(span[:,None]>0,(v.clamp_min(torch.finfo(torch.float32).tiny).log()-logs[:,None])/(span[:,None].clamp_min(1e-30))*14,0.)
            vr=location.ceil().clamp(0,14);vl=(vr-1).clamp_min(0)
            lower=torch.exp(logs[:,None]+span[:,None]*vl/14)
            upper=torch.exp(logs[:,None]+span[:,None]*vr/14)
            lower=torch.where(vl==0,newlo[:,None],lower);upper=torch.where(vr==14,newhi[:,None],upper)
            c2=1-.999**step
            fl=1/((lower/c2).sqrt()+1e-8);fu=1/((upper/c2).sqrt()+1e-8);fx=1/((v/c2).sqrt()+1e-8)
            pl=torch.where(fl>fu,(fx-fu)/(fl-fu).clamp_min(1e-30),1.).clamp(0,1)
            vcode=torch.where(pos,torch.where(torch.rand_like(pl)<pl,vl,vr)+1,0.)
            self.codes[start+off:end].copy_((mcode.float()+16*vcode).reshape(-1).to(torch.uint8))
            meta[0].copy_(newscale);meta[1].copy_(newlo);meta[2].copy_(newhi)
            self.updated+=length
            if getattr(self,'progress',None) and (self.updated%(8<<20)==0 or off+length==w.numel()):self.progress()

class WeightStore:
    def __init__(self,name,shape,frames,codec,device,block,lr,wd,window):
        self.name=name;self.shape=tuple(shape);self.frames=frames;self.codec=codec;self.device=torch.device(device)
        self.n=math.prod(shape);self.step=0;self.enabled=False;self.changed_steps=0
        self.moments=QuantMoments(self.n,self.device,block,lr,wd,window)
        self.moments.progress=None
        assert sum(x['n'] for x in frames)==self.n
    def stored_bytes(self):return sum(f['data'].numel() for f in self.frames)
    def read_all(self):
        with torch.npu.device(self.device):
            out=torch.empty(self.n,dtype=torch.float32,device=self.device);offset=0
            for f in self.frames:self.codec.decode(f,out[offset:offset+f['n']]);offset+=f['n']
            return out.reshape(self.shape)
    def read_range(self,start,n):
        with torch.npu.device(self.device):
            out=torch.empty(n,dtype=torch.float32,device=self.device);offset=0
            for f in self.frames:
                end=offset+f['n'];lo=max(start,offset);hi=min(start+n,end)
                if hi>lo:
                    data=self.codec.decode(f)
                    out[lo-start:hi-start].copy_(data[lo-offset:hi-offset]);del data
                offset=end
                if offset>=start+n:break
            return out
    def write_range(self,start,w):
        offset=0
        for i,f in enumerate(self.frames):
            end=offset+f['n'];lo=max(start,offset);hi=min(start+w.numel(),end)
            if hi>lo:
                if lo==offset and hi==end:part=w[lo-start:hi-start]
                else:
                    part=self.codec.decode(f);part[lo-offset:hi-offset].copy_(w[lo-start:hi-start])
                new=self.codec.encode(part);self.frames[i]=new
                del part,f
            offset=end
            if offset>=start+w.numel():break
    @torch.no_grad()
    def update_all(self,w,g):
        offset=0
        for i,frame in enumerate(self.frames):
            n=frame['n'];part=w.view(-1)[offset:offset+n];grad=g.view(-1)[offset:offset+n]
            self.moments.update(offset,part,grad,self.step)
            self.frames[i]=self.codec.encode(part)
            if hasattr(self,'progress'):self.progress()
            offset+=n
        self.changed_steps+=1

class LinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x,proxy,s):
        ctx.s=s;ctx.input_device=x.device;ctx.shape=x.shape
        with torch.npu.device(s.device):
            z=x.detach().to(s.device).reshape(-1,s.shape[1]);ctx.save_for_backward(z)
            if s.n<=128_000_000 and not getattr(s,'force_chunked',False):
                w=s.read_all();y=F.linear(z,w);del w
            else:
                outputs=[]
                for row in range(0,s.shape[0],4096):
                    count=min(4096,s.shape[0]-row);w=s.read_range(row*s.shape[1],count*s.shape[1]).view(count,-1)
                    outputs.append(F.linear(z,w));del w
                y=torch.cat(outputs,dim=-1);del outputs
            return y.reshape(*x.shape[:-1],s.shape[0]).to(x.device)
    @staticmethod
    def backward(ctx,dy):
        s=ctx.s;(x,)=ctx.saved_tensors
        with torch.no_grad(),torch.npu.device(s.device):
            gy=dy.to(s.device).reshape(-1,s.shape[0])
            if s.n<=128_000_000 and not getattr(s,'force_chunked',False):
                w=s.read_all();gx=gy@w;g=gy.T@x
                if s.enabled:s.update_all(w,g)
                del w,g
            else:
                gx=torch.zeros_like(x)
                for row in range(0,s.shape[0],4096):
                    count=min(4096,s.shape[0]-row);start=row*s.shape[1]
                    w=s.read_range(start,count*s.shape[1]).view(count,-1);slice_grad=gy[:,row:row+count]
                    gx.addmm_(slice_grad,w);g=slice_grad.T@x
                    if s.enabled:
                        s.moments.update(start,w.view(-1),g.view(-1),s.step);s.write_range(start,w.view(-1))
                    del w,g
                if s.enabled:s.changed_steps+=1
            return gx.reshape(ctx.shape).to(ctx.input_device),None,None

class CompressedLinear(torch.nn.Module):
    def __init__(self,s):
        super().__init__();self.store=s;self.in_features=s.shape[1];self.out_features=s.shape[0]
        self.proxy=torch.nn.Parameter(torch.zeros((),dtype=torch.float32,device=s.device))
    def forward(self,x):return LinearFunction.apply(x,self.proxy,self.store)

class EmbeddingFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx,ids,proxy,s):
        ctx.s=s;ctx.input_device=ids.device;ctx.ids=ids.detach().cpu().reshape(-1).tolist();ctx.shape=ids.shape
        with torch.npu.device(s.device):
            result=torch.empty((len(ctx.ids),s.shape[1]),dtype=torch.float32,device=s.device)
            for i,row in enumerate(ctx.ids):result[i].copy_(s.read_range(row*s.shape[1],s.shape[1]))
            return result.reshape(*ids.shape,s.shape[1]).to(ids.device)
    @staticmethod
    def backward(ctx,dy):
        s=ctx.s
        if s.enabled:
            with torch.no_grad(),torch.npu.device(s.device):
                gradient=dy.to(s.device).reshape(-1,s.shape[1]);offset=0
                for index,f in enumerate(s.frames):
                    n=f['n'];w=s.codec.decode(f);g=torch.zeros_like(w)
                    for i,row in enumerate(ctx.ids):
                        start=row*s.shape[1];lo=max(start,offset);hi=min(start+s.shape[1],offset+n)
                        if hi>lo:g[lo-offset:hi-offset].add_(gradient[i,lo-start:hi-start])
                    s.moments.update(offset,w,g,s.step);s.frames[index]=s.codec.encode(w)
                    offset+=n;del w,g,f
                s.changed_steps+=1
        return None,None,None

class CompressedEmbedding(torch.nn.Module):
    def __init__(self,s):
        super().__init__();self.store=s;self.num_embeddings=s.shape[0];self.embedding_dim=s.shape[1]
        self.proxy=torch.nn.Parameter(torch.zeros((),dtype=torch.float32,device=s.device))
    def forward(self,ids):return EmbeddingFunction.apply(ids,self.proxy,self.store)

def load_frames(folder,report,name,device):
    frames=[]
    with (Path(folder)/'weights.enec.parts').open('rb') as archive:
        for row in sorted((r for r in report['rows'] if r['name']==name),key=lambda r:r['element_offset']):
            archive.seek(row['archive_offset']);raw=archive.read(row['stored_bytes'])
            assert len(raw)==row['stored_bytes']
            # Checkpoint loading only; CPU bytes are released before training.
            blob=torch.frombuffer(bytearray(raw),dtype=torch.uint8)
            if row['storage_kind']=='enec':
                data=torch.zeros(len(raw)+GUARD,dtype=torch.uint8,device=device);data[:len(raw)].copy_(blob)
                header=raw[:32];kind='enec'
            else:data=blob.to(device);header=None;kind='raw'
            frames.append({'kind':kind,'n':row['elements'],'bytes':len(raw),'data':data,'header':header})
            del raw,blob
    return frames
