import json,sys
from pathlib import Path
r=Path(__file__).resolve().parent
state=json.loads((r/(sys.argv[1] if len(sys.argv)>1 else 'sweep-job.json')).read_text())
for item in state['runs']:
 f=r/item['name']/'rank0.jsonl'
 if not f.exists():continue
 rows=[]
 for line in f.read_text().splitlines():
  try:rows.append(json.loads(line))
  except json.JSONDecodeError:pass
 item['last_event']=rows[-1]['event']
 micro=[x for x in rows if x['event']=='micro_complete']
 if micro:item['micro_progress']={k:micro[-1][k] for k in ['step','micro','total_micros']}
 data=[x for x in rows if x['event']=='data']
 if data:
  d=data[-1]
  item['data']={'corpus_size':d['training_corpus_size'],'exposures':len(d['training_hashes']),
    'min_length':min(d['training_lengths']),'max_length':max(d['training_lengths'])}
 item['steps']=[{k:x[k] for k in ['step','seconds','input_tokens','peak_allocated','phase_wall_seconds']} for x in rows if x['event']=='step_complete']
 for e in ['validation_before','validation_after']:
  entries=[x['metrics'] for x in rows if x['event']==e]
  if entries:item[e]=entries[-1]
print(json.dumps(state),flush=True)
