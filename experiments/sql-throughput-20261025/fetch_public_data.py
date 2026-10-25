"""Download public SQL sources and extract primitive author settings without unpickling."""
import hashlib,io,json,pickletools,urllib.request,zipfile
from pathlib import Path
import argparse
p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args()
out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
def get(url):
 with urllib.request.urlopen(url,timeout=45) as f:return f.read()
sources=[]
for repo,files in [('b-mc2/sql-create-context',['sql_create_context_v4.json']),('gretelai/synthetic_text_to_sql',['synthetic_text_to_sql_train.snappy.parquet','synthetic_text_to_sql_test.snappy.parquet'])]:
 meta=json.loads(get('https://huggingface.co/api/datasets/'+repo));revision=meta['sha']
 for filename in files:
  url=f'https://huggingface.co/datasets/{repo}/resolve/{revision}/{filename}'
  data=get(url);(out/filename).write_bytes(data)
  sources.append({'repository':repo,'revision':revision,'file':filename,'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()})
  print(json.dumps(sources[-1]),flush=True)
repo='mahernaija/Qwen3.5-27B-Text2SQL'
revision=json.loads(get('https://huggingface.co/api/models/'+repo))['sha']
base=f'https://huggingface.co/{repo}/resolve/{revision}/checkpoint-698/'
state=json.loads(get(base+'trainer_state.json'))
data=get(base+'training_args.bin');z=zipfile.ZipFile(io.BytesIO(data))
ops=list(pickletools.genops(z.read(next(n for n in z.namelist() if n.endswith('data.pkl')))))
keys={'learning_rate','weight_decay','adam_beta1','adam_beta2','adam_epsilon','max_grad_norm','bf16','fp16','per_device_train_batch_size','gradient_accumulation_steps','seed','data_seed','gradient_checkpointing','warmup_ratio','dataloader_drop_last','num_train_epochs','average_tokens_across_devices'}
settings={}
for i,(op,arg,pos) in enumerate(ops):
 if isinstance(arg,str) and arg in keys:
  values=[(o.name,v) for o,v,_ in ops[i+1:i+5] if o.name not in {'BINPUT','LONG_BINPUT','MEMOIZE'}]
  name,value=values[0]
  if name=='NEWTRUE':value=True
  elif name=='NEWFALSE':value=False
  elif name=='NONE':value=None
  elif not name.startswith(('BININT','BINFLOAT','BINUNICODE')):continue
  settings[arg]=value
reference={'source_model':repo,'revision':revision,'training_args_sha256':hashlib.sha256(data).hexdigest(),
 'training_args_read_method':'pickletools opcode inspection only; no torch.load/pickle execution',
 'settings':settings,'completed_steps':state['global_step'],'author_reported_total_seconds':11160,
 'author_reported_samples':178000,'author_reported_cards':16,'author_reported_sequence_limit':512,
 'author_has_per_step_wall_times':False,'author_eval_final':[x for x in state['log_history'] if 'eval_loss' in x][-1],
 'limitations':['No public preprocessing/training script found in model file listing.', 'Public model card gives total wall time, not isolated steady-state step times.']}
(out/'source-manifest.json').write_text(json.dumps({'datasets':sources,'reference':reference},indent=2),encoding='utf-8')
print(json.dumps(reference),flush=True)
