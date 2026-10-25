"""Build a deterministic public SQL short benchmark; never use inference evaluation data."""
import argparse,hashlib,json,random,re,statistics
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
from transformers import AutoTokenizer
p=argparse.ArgumentParser();p.add_argument('--raw',required=True);p.add_argument('--model',required=True);p.add_argument('--output',required=True);p.add_argument('--samples',type=int,default=1024);a=p.parse_args()
root=Path(a.raw);out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
tok=AutoTokenizer.from_pretrained(a.model,local_files_only=True)
system='You are a SQL expert. Given a database schema and a natural language question, write the correct SQL query.'
def normalize(row,source):
 if source=='bmc':schema,q,sql=row['context'],row['question'],row['answer']
 else:schema,q,sql=row['sql_context'],row['sql_prompt'],row['sql']
 # Remove complete INSERT statements; no SQL execution. Both public sources are untrusted data.
 schema=re.sub(r"(?is)\bINSERT\s+INTO\b(?:[^;'\"]|'(?:''|[^'])*'|\"(?:\"\"|[^\"])*\")*;",'',schema).strip()
 q=q.strip();sql=sql.strip()
 key=hashlib.sha256((re.sub(r'\s+',' ',schema)+'\n'+re.sub(r'\s+',' ',q)).encode()).hexdigest()
 prompt=f'<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\nSchema: {schema}\nQuestion: {q}<|im_end|>\n<|im_start|>assistant\n'
 return key,prompt,sql+'<|im_end|>\n',source
test=[normalize(r,'gretel') for r in pq.read_table(root/'synthetic_text_to_sql_test.snappy.parquet').to_pylist()]
heldout={r[0] for r in test};seen=set();records=[];duplicates=0;overlaps=0
for source,raw in [('bmc',json.loads((root/'sql_create_context_v4.json').read_text())),('gretel',pq.read_table(root/'synthetic_text_to_sql_train.snappy.parquet').to_pylist())]:
 for r in raw:
  item=normalize(r,source);key=item[0]
  if key in heldout:overlaps+=1;continue
  if key in seen:duplicates+=1;continue
  seen.add(key);records.append(item)
random.Random(42).shuffle(records)
def tokenize(items):
 rows=[];lengths=[];skipped=0;truncated=0
 for start in range(0,len(items),512):
  group=items[start:start+512]
  prefixes=tok([r[1] for r in group],add_special_tokens=False)['input_ids']
  full=tok([r[1]+r[2] for r in group],add_special_tokens=False)['input_ids']
  for item,prefix,ids in zip(group,prefixes,full):
   assert ids[:len(prefix)]==prefix
   lengths.append(len(ids))
   if len(prefix)>=511 or len(ids)<=len(prefix):skipped+=1;continue
   truncated+=len(ids)>512;ids=ids[:512]
   rows.append({'length':len(ids),'hash':item[0],'ids':ids,'labels':[-100]*len(prefix)+ids[len(prefix):],'source':item[3]})
 return rows,lengths,skipped,truncated
train,lengths,skip,trunc=tokenize(records);val,_,_,_=tokenize(test[:64])
assert len(train)>=a.samples and len(val)>=8
train=train[:a.samples];val=val[:8]
assert not {r['hash'] for r in train}&{r['hash'] for r in val}
for name,rows in [('train',train),('validation',val)]:
 (out/(name+'.jsonl')).write_text(''.join(json.dumps(r)+'\n' for r in rows),encoding='utf-8')
stats={'seed':42,'max_length':512,'training_samples':len(train),'validation_samples':len(val),
 'unique_training_before_length_filter':len(records),'raw_length_percentiles':dict(zip(['p50','p90','p99','max'],[float(x) for x in np.percentile(lengths,[50,90,99,100])])),
 'duplicates_removed':duplicates,'validation_overlap_removed':overlaps,'prompt_too_long_skipped':skip,'answer_truncated':trunc,
 'selected_source_counts':{s:sum(r['source']==s for r in train) for s in ['bmc','gretel']},
 'selected_mean_length':statistics.mean(r['length'] for r in train),'manual_chat_template':True,'loss_on_sql_only':True,
 'source_manifest':json.loads((root/'source-manifest.json').read_text()),
 'training_hashes':[r['hash'] for r in train],'validation_hashes':[r['hash'] for r in val]}
(out/'data-summary.json').write_text(json.dumps(stats,indent=2),encoding='utf-8')
print(json.dumps({k:v for k,v in stats.items() if not k.endswith('hashes') and k!='source_manifest'}),flush=True)
