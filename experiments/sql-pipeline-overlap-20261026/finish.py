"""Wait for this case's own job exit, then run read-only evidence collection."""
import argparse,json,subprocess,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--reference',required=True);p.add_argument('--zero-reference',required=True)
p.add_argument('--cache4',action='store_true');a=p.parse_args();root=Path(__file__).resolve().parent
job=root/('cache4-job.private.json' if a.cache4 else 'job.private.json')
deadline=time.monotonic()+1200
while True:
    state=json.loads(job.read_text())
    assert state['state']!='failed',state
    if state['state']=='complete' and state['phase']==('cache4-full' if a.cache4 else 'async2-full'):break
    assert time.monotonic()<deadline,'This job did not exit within 20 minutes'
    time.sleep(15)
name='cache4' if a.cache4 else 'async'
folder='cache4-full.private' if a.cache4 else 'async2-full.private'
command=['python',str(root/'collect_result.py'),'--root',str(root/folder),'--reference',a.reference,
    '--zero-reference',a.zero_reference,'--output',str(root/(name+'-results.json'))]
if a.cache4:command.append('--cache-change')
subprocess.run(command,check=True,stdout=subprocess.DEVNULL)
subprocess.run(['python',str(root/'diagnose.py'),'--root',str(root/folder),'--output',str(root/('after-'+name+'.json'))],
    check=True,stdout=subprocess.DEVNULL)
result=json.loads((root/(name+'-results.json')).read_text())
print(json.dumps({k:v for k,v in result.items() if k!='rank_results'}))
