"""Read-only audit of padding and paired-rank shape imbalance within fixed updates."""
import argparse,json,math,hashlib
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--data',required=True);p.add_argument('--output',required=True);a=p.parse_args()
rows=[json.loads(line) for line in (Path(a.data)/'train.jsonl').read_text().splitlines()]
def pad(group):return max(128,math.ceil(max(x['length'] for x in group)/64)*64)
records=[]
for step in range(3):
 batch=rows[step*256:(step+1)*256];assert len(batch)==256
 sorted_batch=sorted(enumerate(batch),key=lambda x:(-x[1]['length'],x[0]));sorted_batch=[x for _,x in sorted_batch]
 native=[];scheduled=[]
 for micro in range(8):
  block=batch[micro*32:(micro+1)*32];new=sorted_batch[micro*32:(micro+1)*32]
  old_shapes=[pad(block[rank::2]) for rank in [0,1]]
  common=pad(new)
  native.append(old_shapes);scheduled.append([common,common])
 before=sum(sum(shape)*16 for shape in native);after=sum(sum(shape)*16 for shape in scheduled)
 old_critical=sum(max(shape)*16 for shape in native);new_critical=sum(max(shape)*16 for shape in scheduled)
 assert sorted(x['hash'] for x in batch)==sorted(x['hash'] for x in sorted_batch)
 records.append({'step':step+1,'samples':256,'effective_input_tokens':sum(x['length'] for x in batch),
 'assistant_tokens':sum(sum(y!=-100 for y in x['labels'][1:]) for x in batch),
 'native_pair_sequence_lengths':native,'scheduled_pair_sequence_lengths':scheduled,
 'native_padded_tokens':before,'scheduled_padded_tokens':after,
 'native_sum_of_microbatch_max_padded_tokens_per_rank':old_critical,'scheduled_sum_of_microbatch_max_padded_tokens_per_rank':new_critical,
 'native_shape_mismatch_microbatches':sum(x[0]!=x[1] for x in native),
 'unchanged_sample_multiset_sha256':hashlib.sha256(''.join(sorted(x['hash'] for x in batch)).encode()).hexdigest()})
v={'records':records,'padding_reduction_percent':100*(1-sum(x['scheduled_padded_tokens'] for x in records)/sum(x['native_padded_tokens'] for x in records)),
 'limits':'CPU metadata audit, not measured speedup. Reordering microbatches changes BF16 accumulation grouping; dtype/master/m/v and update frequency need not change.'}
Path(a.output).write_text(json.dumps(v,indent=2));print(json.dumps(v))
