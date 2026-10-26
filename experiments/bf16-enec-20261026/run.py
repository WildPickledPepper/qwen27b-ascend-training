"""Actual BF16 compute-weight archive, original author's codec plus raw fallback.

CPU checkpoint input and verification are offline staging, not training offload.
Author kernels stay unchanged. Only physical cards 2/3; never reset a device.
"""
import argparse,contextlib,ctypes,hashlib,json,math,os,subprocess,time,traceback
from pathlib import Path
os.environ['TORCH_DEVICE_BACKEND_AUTOLOAD']='0'
os.environ.pop('ASCEND_RT_VISIBLE_DEVICES',None) # ACL wrapper uses physical IDs.
import numpy as np
import torch
from safetensors import safe_open
p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--reference',required=True)
p.add_argument('--device',required=True,type=int,choices=[2,3]);p.add_argument('--mode',choices=['probe','all'],required=True)
p.add_argument('--output',required=True);p.add_argument('--chunk-mib',type=int,default=64);a=p.parse_args()
assert 1<=a.chunk_mib<=128
torch.set_num_threads(4);root=Path(__file__).resolve().parent;out=Path(a.output)
out.mkdir(parents=True,exist_ok=False)
smi=subprocess.check_output(['npu-smi','info'],text=True)
assert f'No running processes found in NPU {a.device}' in smi,'Selected card is occupied'
logs=[json.loads(x) for x in (Path(a.reference)/f'rank{a.device-2}.jsonl').read_text().splitlines()]
manifest=next(x['manifest'] for x in logs if x['event']=='model_ready')
index_path=Path(a.model)/'model.safetensors.index.json';index_bytes=index_path.read_bytes()
mapping=json.loads(index_bytes)['weight_map']
def key(name):return name.replace('model.','model.language_model.',1) if name.startswith('model.') else name
shards={mapping[key(x['name'])] for x in manifest}
stats={n:(Path(a.model,n).stat().st_size,Path(a.model,n).stat().st_mtime_ns) for n in shards}
report=dict(mode=a.mode,device=a.device,dtype='BF16',scope='Offline BF16 computational-weight archive; no training integration',
    source_index_sha256=hashlib.sha256(index_bytes).hexdigest(),source_checkpoint_read_only=True,
    build=json.loads((root/'build_manifest.json').read_text()),rows=[],complete=False,
    parameters=sum(x['numel'] for x in manifest),tensor_index={x['name']:x['shape'] for x in manifest},
    source_dtype_elements={},chunk_bytes=a.chunk_mib*2**20)
lib=ctypes.CDLL(str(root/'build/libenec_bf16_bridge.so'));lib.enec_error.restype=ctypes.c_char_p
lib.enec_init.argtypes=[ctypes.c_int]
lib.enec_roundtrip.argtypes=[ctypes.c_void_p,ctypes.c_uint32,ctypes.c_void_p,ctypes.c_void_p,ctypes.c_uint32,
    ctypes.POINTER(ctypes.c_uint32),ctypes.POINTER(ctypes.c_double)]
def check(code):
    if code:raise RuntimeError(lib.enec_error().decode())
def save():
    tmp=out/'report.tmp';tmp.write_text(json.dumps(report,indent=2));tmp.replace(out/'report.json')
def encode(data,name,offset,f,diagnostic=False):
    assert data.dtype==np.uint8 and data.flags.c_contiguous and data.nbytes%2==0
    size=data.nbytes;digest=hashlib.sha256(data).hexdigest();metrics=[0.]*6
    begin=time.monotonic();exact=True;encoded_size=size
    if size<65536:
        stored=data;kind='raw_small_tensor'
    else:
        restored=np.empty(size,dtype=np.uint8);encoded=np.empty(size*2+16*2**20,dtype=np.uint8)
        encoded_n=ctypes.c_uint32();timings=(ctypes.c_double*6)()
        check(lib.enec_roundtrip(data.ctypes.data,size,restored.ctypes.data,encoded.ctypes.data,encoded.nbytes,
            ctypes.byref(encoded_n),timings))
        metrics=list(timings);encoded_size=encoded_n.value;exact=bool(np.array_equal(data,restored))
        if exact and encoded_size<size:stored=encoded[:encoded_size];kind='enec'
        else:stored=data;kind='raw_author_mismatch' if not exact else 'raw_expansion'
    row=dict(name=name,element_offset=offset,elements=size//2,raw_bf16_bytes=size,
        stored_bytes=stored.nbytes,author_encoded_bytes=encoded_size,storage_kind=kind,
        author_roundtrip_bit_exact=exact,stored_lossless=True,input_sha256=digest,
        stored_sha256=hashlib.sha256(stored).hexdigest(),diagnostic=diagnostic,
        compress_event_ms=metrics[0],merge_event_ms=metrics[1],decompress_event_ms=metrics[2],
        explicit_codec_device_allocation_bytes=int(metrics[3]),roundtrip_wall_seconds=time.monotonic()-begin)
    if f is not None:row['archive_offset']=f.tell();f.write(memoryview(stored))
    report['rows'].append(row)
    if len(report['rows'])%16==0:save();print(json.dumps(dict(event='progress',frames=len(report['rows']),name=name)),flush=True)
started=time.monotonic();check(lib.enec_init(a.device))
try:
    if a.mode=='probe':
        for label,words in [('all_uint16_patterns',np.arange(65536,dtype=np.uint16)),
                            ('zeros',np.zeros(131072,dtype=np.uint16)),
                            ('unaligned_tail',np.arange(32768+17,dtype=np.uint16))]:
            encode(words.view(np.uint8),label,0,None,True)
    selected=manifest
    if a.mode=='probe':
        categories={}
        for info in manifest:
            name=info['name'];group='embedding_head' if name in ['model.embed_tokens.weight','lm_head.weight'] else (
                'gate_up' if '.mlp.up_proj.' in name else 'down' if '.mlp.down_proj.' in name else
                'gdn' if '.linear_attn.' in name else 'attention' if '.self_attn.' in name else 'norm')
            if group not in categories and info['numel']>=32768:categories[group]=info
        selected=list(categories.values())
    with contextlib.ExitStack() as stack:
        files={n:stack.enter_context(safe_open(str(Path(a.model)/n),framework='pt',device='cpu')) for n in shards}
        f=stack.enter_context((out/'weights.bf16.enec.parts').open('xb')) if a.mode=='all' else None
        for info in selected:
            name=info['name'];tensor=files[mapping[key(name)]].get_tensor(key(name))
            assert list(tensor.shape)==info['shape'] and tensor.numel()==info['numel']
            dtype=str(tensor.dtype);report['source_dtype_elements'][dtype]=report['source_dtype_elements'].get(dtype,0)+tensor.numel()
            # Same BF16 cast as the verified training initializer, no FP32 expansion.
            tensor=tensor.to(dtype=torch.bfloat16).contiguous().view(-1)
            count=tensor.numel() if a.mode=='all' else min(tensor.numel(),2*2**20)
            for start in range(0,count,a.chunk_mib*2**19):
                data=tensor[start:min(start+a.chunk_mib*2**19,count)].view(torch.uint8).numpy()
                encode(data,name,start,f)
            del tensor
    ordinary=[r for r in report['rows'] if not r['diagnostic']]
    raw=sum(r['raw_bf16_bytes'] for r in ordinary);stored=sum(r['stored_bytes'] for r in ordinary)
    if a.mode=='all':assert raw==report['parameters']*2 and (out/'weights.bf16.enec.parts').stat().st_size==stored
    assert index_path.read_bytes()==index_bytes
    assert all((Path(a.model,n).stat().st_size,Path(a.model,n).stat().st_mtime_ns)==value for n,value in stats.items())
    report.update(complete=True,source_file_sizes_and_mtimes_unchanged=True,
        aggregate=dict(raw_bf16_bytes=raw,stored_frame_bytes=stored,frames=len(ordinary),
            author_mismatch_fallbacks=sum(r['storage_kind']=='raw_author_mismatch' for r in ordinary),
            expansion_fallbacks=sum(r['storage_kind']=='raw_expansion' for r in ordinary),
            raw_small_tensors=sum(r['storage_kind']=='raw_small_tensor' for r in ordinary),
            all_stored_lossless=all(r['stored_lossless'] for r in ordinary),
            compress_event_seconds=sum(r['compress_event_ms'] for r in ordinary)/1000,
            merge_event_seconds=sum(r['merge_event_ms'] for r in ordinary)/1000,
            decompress_event_seconds=sum(r['decompress_event_ms'] for r in ordinary)/1000,
            roundtrip_wall_seconds=sum(r['roundtrip_wall_seconds'] for r in ordinary)))
except BaseException:
    report['error']=traceback.format_exc();raise
finally:
    report['close_status']=lib.enec_close();report['elapsed_seconds']=time.monotonic()-started;save()
    if report['complete']:
        for _ in range(4):
            report['mapping_bytes']=(out/'report.json').stat().st_size
            report['archive_and_mapping_bytes']=report['aggregate']['stored_frame_bytes']+report['mapping_bytes'];save()
        assert report['mapping_bytes']==(out/'report.json').stat().st_size
        print(json.dumps({k:v for k,v in report.items() if k not in ['rows','tensor_index']}),flush=True)
