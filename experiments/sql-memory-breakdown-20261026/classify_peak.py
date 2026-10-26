"""Group live peak blocks by inspected allocation stacks; export no addresses/private paths."""
import argparse,collections,json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--output',required=True);a=p.parse_args();root=Path(a.root);ranks=[]
for rank in [0,1]:
 d=json.loads((root/f'peak-rank{rank}.private.json').read_text());groups=collections.defaultdict(lambda:dict(bytes=0,blocks=0));origins=collections.defaultdict(lambda:dict(bytes=0,blocks=0))
 for b in d['blocks']:
  group=b['category'];fs=b['frames'];names=[f['name'] for f in fs];py=[(Path(f['filename']).name,f.get('line',0),f['name']) for f in fs if f.get('filename','').endswith('.py')]
  qlines={line for file,line,name in py if file=='modeling_qwen3_5.py'}
  if group=='dynamic_unclassified':
   if not py and any('nll_loss_backward' in n or '_log_softmax_backward_data' in n for n in names):group='loss_temporary_gradients'
   elif not py and any('AccumulateGrad::apply' in n for n in names):group='parameter_gradient_accumulation_temporary'
   elif not py and any('MatmulBackward0::apply' in n for n in names):group='matmul_backward_temporary_gradients'
   elif not py and any('InputBuffer::add' in n for n in names):group='activation_gradient_merge'
   elif any(file=='checkpoint.py' and name=='recompute_fn' for file,line,name in py):group='checkpoint_recompute_intermediates'
   elif 736 in qlines:group='ffn_forward_intermediates'
   elif qlines & {747,750,753,754}:group='rmsnorm_forward_intermediates'
   elif any(file=='loss_utils.py' for file,line,name in py):group='loss_forward_intermediates'
   elif any(600<=line<=721 for line in qlines):group='attention_forward_intermediates'
   elif any(file=='triton.py' for file,line,name in py) or any(file=='community_gdn.py' for file,line,name in py):group='gdn_forward_intermediates'
   elif 811 in qlines or 1174 in qlines or (py and py[0][0]=='train_memory.py' and py[0][1] in [280,289]):group='hidden_states_checkpoint_boundaries_and_pipeline'
   elif any(file=='train_memory.py' and name=='batch' for file,line,name in py) or qlines & {169,1182} or any(file=='distributed_c10d.py' for file,line,name in py) or any(file=='__init__.py' and name=='_make_grads' for file,line,name in py) or any(file=='train_memory.py' and line==199 for file,line,name in py):group='inputs_positions_masks_small_control'
   else:group='unclassified'
  groups[group]['bytes']+=b['bytes'];groups[group]['blocks']+=1
  signature=' > '.join(f'{file}:{line}:{name}' for file,line,name in py[:5])
  if not signature:signature=' > '.join(n for n in names if '::_ops::' in n or 'Backward' in n or 'AccumulateGrad' in n or 'InputBuffer' in n)[:900]
  origins[(group,signature)]['bytes']+=b['bytes'];origins[(group,signature)]['blocks']+=1
 groups['allocator_block_reconciliation']={'bytes':d['allocator_block_reconciliation_bytes'],'blocks':None}
 assert sum(x['bytes'] for x in groups.values())==d['reported_peak']
 rows=[dict(category=k,**v,gib=v['bytes']/2**30) for k,v in groups.items()]
 ranks.append(dict(rank=rank,physical_card=rank+2,allocated_peak_bytes=d['reported_peak'],allocated_peak_gib=d['reported_peak']/2**30,reconstructed_final_exact=d['replayed_final']==d['reported_final'],categories=rows,origins=[dict(category=k[0],stack=k[1],**v) for k,v in sorted(origins.items(),key=lambda kv:-kv[1]['bytes'])],trace_events=d['action_counts']))
 print('RANK',rank,json.dumps(rows))
Path(a.output).write_text(json.dumps(dict(ranks=ranks,unit='GiB = 2**30 bytes',scope='Simultaneous allocator peak in first original 256-sample update; standalone instrumentation, no speed claim',limits=['Allocator trace sizes are requested bytes; rounding to 512 bytes reconstructs final allocated exactly. A 0.13-0.27 MiB difference at peak is retained explicitly as allocator block reconciliation, not assigned to activations or workspace.','Forward intermediates mean allocations produced in forward and still live at peak, not independently measured saved-for-backward tensors.','Operator workspace outside PyTorch allocator and driver/runtime HBM are not measured by this snapshot. No independently identified workspace block exists in the two reconstructed allocated peaks; this is not a claim that all operators use zero workspace over the whole iteration.','The two ranks reach their peaks at different times; do not sum them as one simultaneous server peak.']),indent=2))
