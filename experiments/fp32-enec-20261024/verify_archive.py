"""Reload saved compressed frames on NPU and compare source SHA, without original weights."""
import argparse, ctypes, hashlib, json, os, time
from pathlib import Path
os.environ['TORCH_DEVICE_BACKEND_AUTOLOAD'] = '0'
os.environ.pop('ASCEND_RT_VISIBLE_DEVICES', None)
import numpy as np

p = argparse.ArgumentParser()
p.add_argument('--archive', required=True)
p.add_argument('--device', type=int, choices=[2,3], required=True)
p.add_argument('--limit', type=int, default=0)
a = p.parse_args()
root = Path(__file__).resolve().parent; folder = Path(a.archive)
d = json.loads((folder / 'report.json').read_text())
assert d['complete']
if d['mode'] == 'all':
    for name, shape in d['tensor_index'].items():
        tensor_rows = sorted((r for r in d['rows'] if r['name'] == name), key=lambda r: r['element_offset'])
        end = 0
        for row in tensor_rows:
            assert row['element_offset'] == end, (name, 'missing or overlapping chunk')
            assert row['raw_fp32_bytes'] == 4 * row['elements']
            end += row['elements']
        assert end == int(np.prod(shape)), (name, 'incomplete tensor')
lib = ctypes.CDLL(str(root / 'build/libenec_fp32_bridge.so'))
lib.enec_error.restype = ctypes.c_char_p
lib.enec_init.argtypes = [ctypes.c_int]
lib.enec_restore.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_void_p, ctypes.c_uint32]
def check(code):
    if code: raise RuntimeError(lib.enec_error().decode())
check(lib.enec_init(a.device))
results = []; started = time.time()
try:
    rows = [r for r in d['rows'] if 'archive_offset' in r]
    if a.limit and a.limit < len(rows):
        rows = [rows[i] for i in sorted(set(np.linspace(0,len(rows)-1,a.limit).astype(int)))]
    with (folder / 'weights.enec.parts').open('rb') as f:
        for row in rows:
            f.seek(row['archive_offset']); raw = f.read(row['stored_bytes'])
            assert len(raw) == row['stored_bytes']
            if 'stored_sha256' in row: assert hashlib.sha256(raw).hexdigest() == row['stored_sha256']
            if row['storage_kind'] == 'enec':
                inp = np.frombuffer(raw,dtype=np.uint8); output = np.empty(row['raw_fp32_bytes'],dtype=np.uint8)
                check(lib.enec_restore(inp.ctypes.data,len(raw),output.ctypes.data,output.nbytes))
                digest = hashlib.sha256(output).hexdigest()
            else: digest = hashlib.sha256(raw).hexdigest()
            assert digest == row['input_sha256'], row['name']
            results.append({'name':row['name'],'element_offset':row['element_offset'],'passed':True})
    report = {'passed': True, 'frames_verified': len(results), 'total_frames': len([r for r in d['rows'] if 'archive_offset' in r]),
              'original_weights_loaded': False, 'device': a.device, 'seconds': time.time()-started,
              'results': results}
    (folder / 'archive_verification.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({k:v for k,v in report.items() if k!='results'}),flush=True)
finally: check(lib.enec_close())
