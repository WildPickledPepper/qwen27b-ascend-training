"""Export aggregate experiment evidence; no messages, tensors or private paths."""
import argparse,json,statistics
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--root',default='.');p.add_argument('--output',required=True);a=p.parse_args()
r=Path(a.root);result={'supervisor':json.loads((r/'research2-job.json').read_text()),'initial_failure':json.loads((r/'research-job.json').read_text()),'cases':[],'probes':{}}
for name in ['identity-probe','cast-probe','pipeline-probe','gdn-probe','gdn-actual-probe','gdn-actual-fp64-probe']:
 if (r/(name+'.json')).exists():result['probes'][name]=json.loads((r/(name+'.json')).read_text())
if 'gdn-actual-probe' in result['probes']:
 result['probes']['gdn-actual-probe']['metric_warning']='Original CPU FP32 cosine reductions were inaccurate for large vectors; use gdn-actual-fp64-probe for final numerical metrics'
for folder in sorted(r.iterdir()):
 if not folder.is_dir() or not (folder/'rank0-result.json').exists():continue
 ranks=[json.loads((folder/f'rank{i}-result.json').read_text()) for i in [0,1]]
 assert all(x['completed'] and x['fp32_master'] and x['fp32_m_v'] for x in ranks)
 steps=[]
 for i in range(len(ranks[0]['steps'])):
  rows=[x['steps'][i] for x in ranks]
  steps.append({'step':i+1,'seconds':max(x['seconds'] for x in rows),'input_tokens':sum(x['input_tokens'] for x in rows),
   'padded_input_tokens':sum(x['padded_input_tokens'] for x in rows),'peak_allocated_gib':[x['peak_allocated']/2**30 for x in rows],
   'peak_reserved_gib':[x['peak_reserved']/2**30 for x in rows],
   'phase_wall_seconds_by_rank':[x['phase_wall_seconds'] for x in rows],
   'cpu_update_stats_by_rank':[x['cpu_update_stats'] for x in rows],
   'fp32_m_v_bytes_total':sum(x['fp32_m_v_bytes'] for x in rows),'profiled':any(x['profiled'] for x in rows)})
 events=[]
 for logfile in folder.glob('rank*.jsonl'):
  for line in logfile.read_text().splitlines():
   try:e=json.loads(line)
   except json.JSONDecodeError:continue
   if e['event'] in ['begin','data','fusion_enabled','checkpoint_policy','cpu_update_policy','numa_placement']:events.append(e)
 row={'name':folder.name,'parameters':ranks[0]['parameters'],'cpu_mode':ranks[0]['cpu_mode'],'fuse_gdn':ranks[0]['fuse_gdn'],
  'config':ranks[0]['config'],'global_batch':ranks[0]['global_batch'],'micro_batch':ranks[0]['micro_batch'],
  'checkpoint':ranks[0]['checkpoint'],'validation_before':ranks[0]['validation_before'],'validation_after':ranks[0]['validation_after'],
  'fp32_master':True,'fp32_m_v':True,'steps':steps,'events':events,'ranking_eligible':not any(s['profiled'] for s in steps)}
 if row['ranking_eligible']:
  row['warm_mean_seconds']=statistics.mean(s['seconds'] for s in steps[1:])
  row['warm_input_tokens_per_second']=sum(s['input_tokens'] for s in steps[1:])/sum(s['seconds'] for s in steps[1:])
 result['cases'].append(row)
result['limits']=['Short training and eight validation examples do not establish long-run convergence',
 'Community fused BF16 intermediates differ numerically from the original FP32 GDN decomposition',
 'CPU statistics are wall-time stages; asynchronous copy submission is not DMA duration']
Path(a.output).write_text(json.dumps(result,indent=2));print(json.dumps({'successful_cases':len(result['cases'])}))
