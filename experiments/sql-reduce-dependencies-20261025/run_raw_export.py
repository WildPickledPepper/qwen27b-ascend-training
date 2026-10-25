"""Export a private copy of existing raw capture; never run an NPU application."""
import json,os,shutil,signal,subprocess,time
from pathlib import Path
r=Path(__file__).resolve().parent;base=r.parent/'qwen-sql-boundary-overlap-20261025/cpu-full/profile'
sources=[p for p in base.rglob('PROF_*') if p.is_dir()]
assert len(sources)==1
source=sources[0];dest=r/'raw-export.private'/source.name
assert not dest.exists() and not (r/'raw-export-job.json').exists()
size=sum(p.stat().st_size for p in source.rglob('*') if p.is_file());assert shutil.disk_usage(r).free>5*size
state={'stage':'copying','source_bytes':size,'npu_application_started':False}
def save():(r/'raw-export-job.json').write_text(json.dumps(state,indent=2))
save();shutil.copytree(source,dest)
with (r/'raw-export.private.log').open('x') as log:
 p=subprocess.Popen(['msprof','--export=on','--type=text','--output='+str(dest)],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 state.update(stage='exporting',pid=p.pid);save()
 try:code=p.wait(timeout=300)
 except subprocess.TimeoutExpired:
  os.killpg(p.pid,signal.SIGTERM)
  try:p.wait(timeout=20)
  except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
  code=-15
state.update(stage='complete' if code==0 else 'failed',returncode=code)
state['exports']=[{'name':p.name,'bytes':p.stat().st_size} for p in dest.rglob('*') if p.is_file() and p.suffix in ['.json','.csv','.db']]
save();print(json.dumps(state));assert code==0
