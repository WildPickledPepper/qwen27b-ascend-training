"""Preserve the CPU benchmark setup failure; numeric probe had already passed."""
import json
from pathlib import Path
r=Path(__file__).resolve().parent
assert json.loads((r/'cpu-job.json').read_text())['stage']=='failed'
assert 'no pinned memory allocator' in (r/'probe_cpu.py.log').read_text()
assert json.loads((r/'cpu-probe.json').read_text())['passed']
a=r/'cpu-setup-failure.private';a.mkdir()
for name in ['cpu-job.json','cpu-controller.log','build_fused.py.log','probe_cpu.py.log','cpu-probe.json']:
 (r/name).rename(a/name)
print('Preserved missing pinned-allocator setup failure; no training was attempted')
