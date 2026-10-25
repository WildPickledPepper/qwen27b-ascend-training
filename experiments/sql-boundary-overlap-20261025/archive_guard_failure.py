"""Keep failed pre-update source-guard attempt; never overwrite failed evidence."""
import json,subprocess
from pathlib import Path
r=Path(__file__).resolve().parent
s=json.loads((r/'job.json').read_text());assert s['stage']=='failed'
assert len(s['runs'])==1 and s['runs'][0]['name']=='tiny-cache'
assert 'source.count(boundary_call)' in (r/'tiny-cache.log').read_text()
assert not list((r/'tiny-cache').glob('probe-state*'))
info=subprocess.check_output(['npu-smi','info'],text=True)
assert all(f'No running processes found in NPU {i}' in info for i in [2,3])
archive=r/'guard-failure-1';archive.mkdir()
for name in ['job.json','controller.log','tiny-cache.log','tiny-cache','cpu_accumulation.py']:
 if (r/name).exists():(r/name).rename(archive/name)
print('Archived guard failure before any training update; devices idle')
