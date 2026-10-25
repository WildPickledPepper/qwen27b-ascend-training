"""Cancel only this controller, before it has launched any GPU child."""
import json,os,signal,time
from pathlib import Path
r=Path(__file__).resolve().parent;job=json.loads((r/'job.json').read_text())
assert job['stage']=='waiting-for-prior-prefetch-run' and not job['runs']
matches=[]
for proc in Path('/proc').iterdir():
 if not proc.name.isdigit():continue
 try:
  if (proc/'cwd').resolve()!=r:continue
  args=(proc/'cmdline').read_bytes().split(b'\0')
  if b'run_experiment.py' in args:matches.append(int(proc.name))
 except (FileNotFoundError,PermissionError):pass
assert len(matches)==1,matches
os.kill(matches[0],signal.SIGTERM)
job.update(stage='cancelled_before_gpu_use',reason='Prioritize measured exposed ReduceScatter; FFN leaf grouping may delay gradient readiness.',finished_unix=time.time())
(r/'job.json').write_text(json.dumps(job,indent=2));print(json.dumps(job))
