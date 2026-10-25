from pathlib import Path
r=Path(__file__).resolve().parent
dest=r/'guard-failure';dest.mkdir(exist_ok=False)
for name in ['job.json','tiny-native.log','tiny-native','supervisor.log']:
 p=r/name
 if p.exists():
  assert p.resolve().is_relative_to(r.resolve())
  p.rename(dest/name)
