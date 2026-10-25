"""Preserve our failed setup only after checking devices have been released."""
import json,subprocess
from pathlib import Path
r=Path(__file__).resolve().parent.resolve()
assert json.loads((r/'capture-job.json').read_text())['stage']=='failed'
info=subprocess.check_output(['npu-smi','info'],text=True)
assert all(f'No running processes found in NPU {i}' in info for i in [2,3])
dest=r/'capture-setup-failure.private';dest.mkdir(exist_ok=False)
for name in ['capture-job.json','tiny-smoke.private.log','tiny-smoke','capture-controller.private.log']:
 p=r/name;target=dest/name
 assert p.resolve().is_relative_to(r) and target.resolve().is_relative_to(r)
 if p.exists():p.rename(target)
print('Preserved setup failure before model loading; devices released')
