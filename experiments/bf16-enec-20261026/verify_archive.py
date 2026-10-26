"""Independent saved-frame reload on NPU; original checkpoint never opened."""
import argparse,ctypes,hashlib,json,math,os,subprocess,time
from pathlib import Path
os.environ['TORCH_DEVICE_BACKEND_AUTOLOAD']='0';os.environ.pop('ASCEND_RT_VISIBLE_DEVICES',None)
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--archive',required=True);p.add_argument('--device',type=int,choices=[2,3],required=True);a=p.parse_args()
root=Path(__file__).resolve().parent;folder=Path(a.archive);d=json.loads((folder/'report.json').read_text())
assert d['complete'] and d['mode']=='all'
assert f'No running processes found in NPU {a.device}' in subprocess.check_output(['npu-smi','info'],text=True)
assert d['archive_and_mapping_bytes']==(folder/'weights.bf16.enec.parts').stat().st_size+(folder/'report.json').stat().st_size
expected_offset=0
for r in d['rows']:
    assert r['archive_offset']==expected_offset;expected_offset+=r['stored_bytes']
assert expected_offset==(folder/'weights.bf16.enec.parts').stat().st_size
for name,shape in d['tensor_index'].items():
    rows=sorted((r for r in d['rows'] if r['name']==name),key=lambda r:r['element_offset']);end=0
    for r in rows:assert r['element_offset']==end and r['raw_bf16_bytes']==r['elements']*2;end+=r['elements']
    assert end==math.prod(shape)
lib=ctypes.CDLL(str(root/'build/libenec_bf16_bridge.so'));lib.enec_error.restype=ctypes.c_char_p
lib.enec_init.argtypes=[ctypes.c_int];lib.enec_restore.argtypes=[ctypes.c_void_p,ctypes.c_uint32,ctypes.c_void_p,ctypes.c_uint32]
def check(code):
    if code:raise RuntimeError(lib.enec_error().decode())
check(lib.enec_init(a.device));started=time.monotonic();passed=0
try:
    with (folder/'weights.bf16.enec.parts').open('rb') as f:
        for r in d['rows']:
            f.seek(r['archive_offset']);raw=f.read(r['stored_bytes']);assert len(raw)==r['stored_bytes']
            assert hashlib.sha256(raw).hexdigest()==r['stored_sha256']
            if r['storage_kind']=='enec':
                inp=np.frombuffer(raw,dtype=np.uint8);decoded=np.empty(r['raw_bf16_bytes'],dtype=np.uint8)
                check(lib.enec_restore(inp.ctypes.data,len(raw),decoded.ctypes.data,decoded.nbytes))
                digest=hashlib.sha256(decoded).hexdigest()
            else:digest=hashlib.sha256(raw).hexdigest()
            assert digest==r['input_sha256'],(r['name'],r['element_offset']);passed+=1
            if passed%64==0:print(json.dumps(dict(event='verified',frames=passed)),flush=True)
    result=dict(passed=True,device=a.device,frames_verified=passed,total_frames=len(d['rows']),
        original_weights_loaded=False,all_tensor_ranges_complete=True,archive_ranges_complete=True,seconds=time.monotonic()-started)
    (folder/'archive_verification.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
finally:check(lib.enec_close())
