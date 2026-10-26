"""Replay allocator lifetimes; classify the simultaneous peak, never sum independent peaks."""
import argparse,collections,json,pickle
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--output',required=True);a=p.parse_args()
root=Path(a.root);result=[]
for rank in [0,1]:
 initial=pickle.loads((root/f'memory-initial-rank{rank}.private.pickle').read_bytes())
 final=pickle.loads((root/f'memory-step1-rank{rank}.private.pickle').read_bytes())
 live={}
 for segment in initial['snapshot']['segments']:
  for b in segment['blocks']:
   if b['state']=='active_allocated':live[b['address']]=dict(size=b['size'],requested=b['requested_size'],frames=b['frames'],initial=True)
 initial_bytes=sum(b['size'] for b in live.values());assert initial_bytes==initial['allocated'],(initial_bytes,initial['allocated'])
 # First snapshot marker fixes the replay origin; history was enabled immediately before it.
 trace=final['snapshot']['device_traces'][rank]
 starts=[i for i,e in enumerate(trace) if e['action']=='snapshot'];assert starts
 live_bytes=initial_bytes;peak=live_bytes;peak_live=live.copy();peak_index=starts[0];action_counts=collections.Counter()
 for i,e in enumerate(trace[starts[0]+1:],starts[0]+1):
  action_counts[e['action']]+=1;addr=e.get('addr')
  if e['action']=='alloc':
   assert addr not in live,('duplicate live allocation',i,addr)
   sz=(e['size']+511)//512*512
   live[addr]=dict(size=sz,requested=e['size'],frames=e.get('frames',[]),initial=False);live_bytes+=sz
  elif e['action']=='free_requested':
   if addr not in live:raise AssertionError(('free missing',i,addr))
   live_bytes-=live.pop(addr)['size']
  if live_bytes>peak:peak=live_bytes;peak_live=live.copy();peak_index=i
 tags=initial['tags'];groups=collections.defaultdict(lambda:dict(bytes=0,blocks=0));blocks=[]
 for addr,b in peak_live.items():
  if addr in tags and b['initial']:group=tags[addr]['category'];origin=tags[addr].get('name','')
  else:
   py=[f for f in b['frames'] if f.get('filename','').endswith('.py')]
   names=[f['name'] for f in b['frames']]
   # Allocation origin is evidence; semantic roles will be refined after inspecting full peak stacks.
   user=[f for f in py if Path(f['filename']).name not in ['_tensor.py','_utils.py','module.py','graph.py']]
   origin=' > '.join(f"{Path(f['filename']).name}:{f.get('line',0)}:{f['name']}" for f in user[:8])
   group='dynamic_unclassified'
  groups[group]['bytes']+=b['size'];groups[group]['blocks']+=1
  blocks.append(dict(bytes=b['size'],requested_bytes=b['requested'],category=group,origin=origin,frames=b['frames'],initial=b['initial']))
 blocks.sort(key=lambda b:-b['bytes'])
 raw=dict(rank=rank,peak=peak,allocator_block_reconciliation_bytes=final['peak']-peak,reported_peak=final['peak'],replayed_final=live_bytes,reported_final=final['allocated'],initial_bytes=initial_bytes,peak_event=peak_index,groups=dict(groups),action_counts=dict(action_counts),blocks=blocks)
 (root/f'peak-rank{rank}.private.json').write_text(json.dumps(raw,indent=2))
 print(json.dumps({k:v for k,v in raw.items() if k!='blocks'}))
 assert 0<=final['peak']-peak<2**20,('peak mismatch exceeds 1MiB',peak,final['peak'])
 assert live_bytes==final['allocated'],('final mismatch',live_bytes,final['allocated'])
 result.append({k:v for k,v in raw.items() if k!='blocks'})
Path(a.output).write_text(json.dumps(result,indent=2))
