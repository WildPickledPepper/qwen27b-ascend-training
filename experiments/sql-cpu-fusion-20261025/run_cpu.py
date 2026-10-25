import json,os,subprocess,sys,time
from pathlib import Path
r=Path(__file__).resolve().parent;os.chdir(r);assert not (r/'cpu-job.json').exists()
s={'stage':'starting','runs':[],'started_unix':time.time()}
def save():(r/'cpu-job.json').write_text(json.dumps(s,indent=2))
try:
 for script,limit in [('build_fused.py',600),('probe_cpu.py',600)]:
  s['stage']=script;save()
  with (r/(script+'.log')).open('x') as output:
   p=subprocess.Popen([sys.executable,str(r/script)],cwd=r,stdout=output,stderr=subprocess.STDOUT,preexec_fn=(lambda:os.sched_setaffinity(0,{96,97})) if script=='build_fused.py' else None)
   entry={'script':script,'pid':p.pid};s['runs'].append(entry);save();entry['returncode']=p.wait(timeout=limit);save()
   assert entry['returncode']==0,(script,entry['returncode'])
 s['stage']='complete'
except Exception as e:s['stage']='failed';s['failure_type']=type(e).__name__;raise
finally:s['finished_unix']=time.time();save()
