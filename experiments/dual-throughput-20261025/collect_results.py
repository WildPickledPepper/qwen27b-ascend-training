"""Export aggregate training evidence without messages, weights or private paths."""
import argparse,json,statistics
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--root',default='.');p.add_argument('--output',required=True);a=p.parse_args()
r=Path(a.root);cases=[]
for folder in sorted(r.iterdir()):
 if not folder.is_dir():continue
 files=sorted(folder.glob('rank*-result.json'))
 if not files:continue
 ranks=[json.loads(f.read_text()) for f in files]
 assert len(ranks)==2 and all(x['completed'] and x['fp32_master'] and x['fp32_m_v'] for x in ranks)
 steps=[]
 for i in range(len(ranks[0]['steps'])):
  rows=[x['steps'][i] for x in ranks]
  steps.append({'step':i+1,'seconds':max(x['seconds'] for x in rows),
    'useful_input_tokens':sum(x['input_tokens'] for x in rows),
    'padded_input_tokens':sum(x['padded_input_tokens'] for x in rows),
    'peak_allocated_gib_by_rank':[x['peak_allocated']/2**30 for x in rows],
    'peak_reserved_gib_by_rank':[x['peak_reserved']/2**30 for x in rows],
    'fp32_m_v_bytes_total':sum(x['fp32_m_v_bytes'] for x in rows),
    'profiled':any(x['profiled'] for x in rows),
    'phase_wall_seconds_by_rank':[x['phase_wall_seconds'] for x in rows]})
 warm=[s for s in steps[1:] if not s['profiled']]
 events=[]
 for logfile in folder.glob('*jsonl'):
  for line in logfile.read_text().splitlines():
   try:event=json.loads(line)
   except json.JSONDecodeError:continue
   if event.get('event') in ['begin','data','checkpoint_policy','numa_placement','validation_after2']:
    events.append(event)
 row={'name':folder.name,'parameters':ranks[0]['parameters'],'global_batch':ranks[0]['global_batch'],
      'micro_batch':ranks[0]['micro_batch'],'checkpoint':ranks[0]['checkpoint'],
      'numa_spread':ranks[0].get('numa_spread',False),'config':ranks[0]['config'],
      'validation_before':ranks[0]['validation_before'],'validation_after':ranks[0]['validation_after'],
      'steps':steps,'events':events,'fp32_master':True,'fp32_m_v':True,'state_cpu_offload':True}
 row['ranking_eligible']=not any(s['profiled'] for s in steps)
 if warm and row['ranking_eligible']:
  row['warm_mean_seconds']=statistics.mean(s['seconds'] for s in warm)
  row['warm_useful_input_tokens_per_second']=sum(s['useful_input_tokens'] for s in warm)/sum(s['seconds'] for s in warm)
 cases.append(row)
states={}
for name in ['sweep-job.json','sweep2-job.json','advanced-job.json','matched-job.json']:
 if (r/name).exists():states[name]=json.loads((r/name).read_text())
result={'protocol':'Original BF16 text model, CPU FP32 master and Adam m/v; independent short train/validation split',
 'timing':'Maximum rank wall time, exclude cold first update and profiled updates',
 'cases':cases,'supervisors':states,
 'limits':['Batch-32 and historical batch-2 have different update frequency; compare separately',
 'Short repeated training and eight validation examples do not establish long-run training quality',
 'Stage wall times overlap asynchronous device work; do not interpret as pure kernel time']}
Path(a.output).write_text(json.dumps(result,indent=2));print(json.dumps({'cases':len(cases),'output':a.output}))
