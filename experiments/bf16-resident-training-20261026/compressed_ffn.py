"""Lossless resident FFN payloads, native BF16 Linear/autograd, FP32 CPU Adam.

No model weights are paged from CPU during forward/backward. Only original CPU
optimizer updates supply new BF16 chunks at the original global boundary.
"""
import ctypes,hashlib,json,struct,time
from pathlib import Path
import torch
from torch.profiler import record_function as scope

GUARD=65536
FRAME_ELEMENTS=32*1024*1024

class Codec:
    def __init__(self,path,device):
        self.device=torch.device(device);self.lib=ctypes.CDLL(str(path));self.lib.online_error.restype=ctypes.c_char_p
        self.lib.online_capacity.argtypes=[ctypes.c_uint32];self.lib.online_capacity.restype=ctypes.c_uint64
        self.lib.online_decompress.argtypes=[ctypes.c_void_p,ctypes.c_int,ctypes.c_void_p,ctypes.c_void_p,ctypes.c_void_p]
        self.lib.online_compress.argtypes=[ctypes.c_void_p,ctypes.c_int,ctypes.c_void_p,ctypes.c_uint32,
            ctypes.c_void_p,ctypes.c_void_p,ctypes.c_void_p,ctypes.c_void_p,ctypes.c_void_p]
        self.ws=None;self.reset_stats()
    def reset_stats(self):self.decodes=0;self.encodes=0;self.fallbacks=0;self.decode_host=0.;self.encode_wall=0.
    def check(self,status):
        if status:raise RuntimeError(self.lib.online_error().decode())
    def workspace(self,nb):
        cap=int(self.lib.online_capacity(nb))
        if self.ws is None or self.ws['cap']<cap:
            self.ws=None
            self.ws=dict(cap=cap,scratch=torch.zeros(cap,dtype=torch.uint8,device=self.device),
                packed=torch.zeros(cap,dtype=torch.uint8,device=self.device),
                hist=torch.zeros(48*256*4,dtype=torch.uint8,device=self.device),
                sizes=torch.zeros(48*32,dtype=torch.uint8,device=self.device),
                verify=torch.empty(nb//2,dtype=torch.bfloat16,device=self.device))
        return self.ws
    @torch.no_grad()
    def decode_into(self,frame,out):
        assert out.numel()==frame['n'] and out.dtype==torch.bfloat16
        begin=time.monotonic()
        with torch.npu.device(self.device),scope('weight/decode'):
            if frame['kind']=='raw':out.view(torch.uint8).copy_(frame['data'][:out.numel()*2])
            else:
                h=ctypes.create_string_buffer(frame['header'],32)
                self.check(self.lib.online_decompress(torch.npu.current_stream(self.device).npu_stream,self.device.index,
                    h,frame['data'].data_ptr(),out.data_ptr()))
                self.decodes+=1
        self.decode_host+=time.monotonic()-begin
    @torch.no_grad()
    def encode(self,w):
        begin=time.monotonic();w=w.contiguous().view(-1);nb=w.numel()*2
        with torch.npu.device(self.device),scope('weight/encode_verify'):
            if nb<65536 or nb%32768:
                return dict(kind='raw',n=w.numel(),bytes=nb,data=w.view(torch.uint8).clone(),header=None)
            ws=self.workspace(nb);h=ctypes.create_string_buffer(32)
            self.check(self.lib.online_compress(torch.npu.current_stream(self.device).npu_stream,self.device.index,
                w.data_ptr(),nb,ws['scratch'].data_ptr(),ws['packed'].data_ptr(),ws['hist'].data_ptr(),ws['sizes'].data_ptr(),h))
            header=bytes(h.raw);size=struct.unpack_from('<I',header,20)[0];assert 32<=size<=ws['cap']
            candidate=dict(kind='enec',n=w.numel(),bytes=size,data=ws['packed'],header=header)
            self.decode_into(candidate,ws['verify'][:w.numel()])
            exact=torch.equal(ws['verify'][:w.numel()].view(torch.int16),w.view(torch.int16));self.encodes+=1
            if not exact or size>=nb:
                self.fallbacks+=1
                frame=dict(kind='raw',n=w.numel(),bytes=nb,data=w.view(torch.uint8).clone(),header=None)
            else:
                data=torch.empty(size+GUARD,dtype=torch.uint8,device=self.device)
                data[:size].copy_(ws['packed'][:size]);data[size:].zero_()
                frame=dict(kind='enec',n=w.numel(),bytes=size,data=data,header=header)
            self.encode_wall+=time.monotonic()-begin
            return frame

class CompressedFFN:
    def __init__(self,model,params,manifest,device,library,archive=None):
        self.params=params;self.device=torch.device(device);self.codec=Codec(library,device)
        self.records={};self.live={};self.module_indices={};self.hooks=[];self.packs=0;self.unpacks=0
        self.stub=torch.empty(1,device=device,dtype=torch.bfloat16)
        by_id={id(p):i for i,p in enumerate(params)}
        folder=Path(archive) if archive else None
        metadata=json.loads((folder/'report.json').read_text()) if folder else None
        if metadata:assert metadata['complete']
        archive_rows={}
        if metadata:
            for row in metadata['rows']:archive_rows.setdefault(row['name'],[]).append(row)
        f=(folder/'weights.bf16.enec.parts').open('rb') if folder else None
        before=torch.npu.memory_allocated()
        try:
            for name,module in model.named_modules():
                if not name.endswith(('.mlp.gate_proj','.mlp.up_proj','.mlp.down_proj')):continue
                idx=by_id[id(module.weight)];p=params[idx];shape=tuple(p.shape);n=p.numel();frames=[]
                if f:
                    rows=sorted(archive_rows[manifest[idx]['name']],key=lambda r:r['element_offset']);offset=0
                    for row in rows:
                        assert row['element_offset']==offset;f.seek(row['archive_offset']);raw=f.read(row['stored_bytes'])
                        assert hashlib.sha256(raw).hexdigest()==row['stored_sha256']
                        kind='enec' if row['storage_kind']=='enec' else 'raw'
                        header=raw[:32] if kind=='enec' else None
                        if header:assert struct.unpack_from('<I',header,16)[0]==row['elements']*2
                        data=torch.empty(len(raw)+(GUARD if kind=='enec' else 0),dtype=torch.uint8,device=device)
                        data[:len(raw)].copy_(torch.frombuffer(bytearray(raw),dtype=torch.uint8))
                        if kind=='enec':data[len(raw):].zero_()
                        frames.append(dict(kind=kind,n=row['elements'],bytes=len(raw),header=header,data=data));offset+=row['elements']
                    assert offset==n
                else:
                    for start in range(0,n,FRAME_ELEMENTS):frames.append(self.codec.encode(p.detach().view(-1)[start:start+FRAME_ELEMENTS]))
                self.records[idx]=dict(shape=shape,n=n,frames=frames,name=manifest[idx]['name'])
                decoded=self.read(idx)
                assert torch.equal(decoded.view(torch.int16),p.detach().view(torch.int16)),manifest[idx]['name']
                del decoded
                p.data=self.stub.expand(shape)
                self.module_indices[id(module)]=idx
                self.hooks.append(module.register_forward_pre_hook(self.pre))
                self.hooks.append(module.register_forward_hook(self.post,always_call=True))
        finally:
            if f:f.close()
        assert self.records and any(f['kind']=='enec' for r in self.records.values() for f in r['frames'])
        torch.npu.synchronize();torch.npu.empty_cache()
        self.initial_audit=dict(selected_parameters=len(self.records),
            original_ffn_weight_bytes=sum(r['n']*2 for r in self.records.values()),
            resident_payload_bytes=self.bytes(),all_initial_weights_bit_exact=True,
            allocated_before=before,allocated_after=torch.npu.memory_allocated(),
            original_weight_storage_released=True,gradient_leaf_shapes_preserved=True)
    def bytes(self):return sum(f['data'].numel() for r in self.records.values() for f in r['frames'])
    @torch.no_grad()
    def read(self,idx):
        r=self.records[idx];flat=torch.empty(r['n'],dtype=torch.bfloat16,device=self.device);offset=0
        for frame in r['frames']:
            self.codec.decode_into(frame,flat[offset:offset+frame['n']]);offset+=frame['n']
        return flat.view(r['shape'])
    @torch.no_grad()
    def pre(self,module,args):
        idx=self.module_indices[id(module)];p=self.params[idx]
        p.data=self.read(idx);self.live[p.untyped_storage().data_ptr()]=idx
    @torch.no_grad()
    def post(self,module,args,output):
        idx=self.module_indices[id(module)];p=self.params[idx]
        self.live.pop(p.untyped_storage().data_ptr(),None)
        p.data=self.stub.expand(self.records[idx]['shape'])
    def pack(self,tensor):
        idx=self.live.get(tensor.untyped_storage().data_ptr()) if tensor.device.type=='npu' else None
        if idx is None:return tensor.detach() # Avoid retaining autograd graph cycles.
        self.packs+=1
        return ('compressed_ffn',idx,tuple(tensor.shape),tuple(tensor.stride()),tensor.storage_offset())
    def unpack(self,item):
        if isinstance(item,torch.Tensor):return item
        tag,idx,shape,stride,offset=item;assert tag=='compressed_ffn';self.unpacks+=1
        return self.read(idx).as_strided(shape,stride,offset)
    @torch.no_grad()
    def update_chunk(self,idx,start,cpu):
        r=self.records[idx];end=start+cpu.numel();offset=0;covered=0
        for i,frame in enumerate(r['frames']):
            nxt=offset+frame['n']
            if nxt>start and offset<end:
                assert start<=offset and nxt<=end,'CPU update chunks must cover complete codec frames'
                part=cpu[offset-start:nxt-start].to(self.device,non_blocking=True)
                r['frames'][i]=self.codec.encode(part);covered+=frame['n'];del part
            offset=nxt
        assert covered==cpu.numel()
    def reset_stats(self):self.codec.reset_stats();self.packs=0;self.unpacks=0
    def stats(self):
        return dict(payload_bytes=self.bytes(),decode_calls=self.codec.decodes,encode_calls=self.codec.encodes,
            encode_raw_fallbacks=self.codec.fallbacks,decode_host_enqueue_seconds=self.codec.decode_host,
            encode_verify_wall_seconds=self.codec.encode_wall,saved_weight_packs=self.packs,saved_weight_unpacks=self.unpacks)
