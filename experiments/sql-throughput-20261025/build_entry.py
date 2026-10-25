"""Derive the SQL benchmark from the already validated process-local training entry."""
import json
from pathlib import Path
r=Path(__file__).resolve().parent;base=r.parent/'training-fusion-20261025'
def replace(text,old,new):
 assert text.count(old)==1,old
 return text.replace(old,new,1)
s=(base/'train_fusion.py').read_text(encoding='utf-8')
s=replace(s,"p.add_argument('--steps',type=int,default=3)","p.add_argument('--steps',type=int,default=3)\np.add_argument('--pad-mode',choices=['dynamic64','fixed512'],default='dynamic64')")
s=replace(s,'torch.manual_seed(20261024)','torch.manual_seed(42)')
start=s.index(' def examples(filename):');end=s.index(" train=examples('train.jsonl');val=examples('validation.jsonl')",start)
s=s[:start]+''' def examples(filename):
  rows=[]
  for line in (Path(a.data)/filename).read_text().splitlines():
   row=json.loads(line)
   assert 0<row['length']<=512 and len(row['ids'])==len(row['labels'])==row['length']
   rows.append((row['length'],row['hash'],row['ids'],row['labels']))
  return rows
'''+s[end:]
s=replace(s,' size=max(row[0] for row in rows)'," size=512 if a.pad_mode=='fixed512' else max(128,math.ceil(max(row[0] for row in rows)/64)*64)")
s=replace(s,' started=time.monotonic()', ''' started=time.monotonic()
 # Match the author's 698-update cosine schedule with 5% warmup (35 steps).
 warmup=35;total_updates=698
 factor=step/warmup if step<warmup else .5*(1+math.cos(math.pi*(step-warmup)/(total_updates-warmup)))
 for group in optimizer.optimizer.param_groups:group['lr']=2e-5*factor
 log('learning_rate',step=step+1,learning_rate=2e-5*factor)''')
s=replace(s,"        'cpu_mode':a.cpu_mode,", "        'benchmark':'public-text-to-sql','pad_mode':a.pad_mode,'max_sequence_length':512,'seed':42,\n        'cpu_mode':a.cpu_mode,")
(r/'train_sql.py').write_text(s,encoding='utf-8',newline='\n')
for name in ['cpu_update.py','bounded_grad_copy.py']:
 (r/name).write_text((base/name).read_text(encoding='utf-8'),encoding='utf-8',newline='\n')
g=(base/'community_gdn.py').read_text(encoding='utf-8')
g=g.replace('length=q.shape[1];assert 0<length<=128','length=q.shape[1];assert 0<length<=512')
g=g.replace('pad=128-length','pad=max(128,((length+63)//64)*64)-length')
g=g.replace('short, no-cache training experiment','bounded length-512, no-cache SQL training experiment')
(r/'community_gdn.py').write_text(g,encoding='utf-8',newline='\n')
warm=(base/'warm_gdn_fp64.py').read_text(encoding='utf-8')
warm=warm.replace('[(1,31),(1,128),(16,125)]','[(2,512),(4,512)]')
warm=warm.replace("'gdn-actual-fp64-probe.json'","'gdn-sql-fp64-probe.json'")
(r/'probe_gdn_512.py').write_text(warm,encoding='utf-8',newline='\n')
cfg=json.loads((base/'prefetch-large-groups.json').read_text())
cfg['gradient_clipping']=1.0;cfg['optimizer']['params']['lr']=2e-5
(r/'zero3-sql.json').write_text(json.dumps(cfg,indent=2)+'\n',encoding='utf-8')
print('SQL entry generated; original training files unchanged')
