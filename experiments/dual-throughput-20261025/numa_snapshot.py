"""Read-only anonymous resident-page placement of this experiment's workers."""
import json,os,re
from pathlib import Path
r=Path(__file__).resolve().parent
state=json.loads((r/'sweep-job.json').read_text());parent=state['runs'][-1]['pid']
children=Path(f'/proc/{parent}/task/{parent}/children').read_text().split()
rows=[]
for pid in children:
 path=Path(f'/proc/{pid}')
 if 'train_throughput.py' not in (path/'cmdline').read_text():continue
 counts={}
 for line in (path/'numa_maps').read_text().splitlines():
  if 'anon=' not in line and 'heap ' not in line:continue
  for node,n in re.findall(r' N(\d+)=(\d+)',line):counts[node]=counts.get(node,0)+int(n)
 status=(path/'status').read_text()
 rows.append({'pid':int(pid),'affinity':sorted(os.sched_getaffinity(int(pid))),
  'anonymous_resident_gib_by_node':{n:pages*os.sysconf('SC_PAGE_SIZE')/2**30 for n,pages in counts.items()},
  'rss_kib':int(re.search(r'VmRSS:\s+(\d+)',status).group(1))})
print(json.dumps({'run':state['runs'][-1]['name'],'workers':rows}),flush=True)
