import os,json,random,hashlib
from pathlib import Path
from transformers import AutoTokenizer
r=Path(os.environ['SFT_RUN']); model=os.environ['MODEL_DIR']
tok=AutoTokenizer.from_pretrained(model,local_files_only=True)
raw=(r/'alpaca_gpt4_data_zh.json').read_bytes(); rows=json.loads(raw);random.Random(20261023).shuffle(rows)
chosen=[];seen=set()
for row in rows:
 prompt=row['instruction']+ ('\n'+row['input'] if row.get('input') else '')
 messages=[{'role':'user','content':prompt},{'role':'assistant','content':row['output']}]
 key=hashlib.sha256(prompt.encode()).hexdigest()
 ids=tok.apply_chat_template(messages,tokenize=True,enable_thinking=False,return_dict=False)
 if len(ids)<=128 and len(ids)>=24 and key not in seen:
  chosen.append({'messages':messages});seen.add(key)
 if len(chosen)==40:break
assert len(chosen)==40
for name,data in [('train',chosen[:32]),('validation',chosen[32:])]:
 (r/(name+'.jsonl')).write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in data))
(r/'data-provenance.json').write_text(json.dumps({'source':'Instruction-Tuning-with-GPT-4/GPT-4-LLM/data/alpaca_gpt4_data_zh.json','sha256':hashlib.sha256(raw).hexdigest(),'train':32,'validation':8,'seed':20261023,'max_tokens':128,'question_hashes':list(seen),'gsm8k_used':False},indent=2))
print('DATA_READY',len(chosen))
