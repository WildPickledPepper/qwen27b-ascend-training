"""Read installed reducer source and checkpoint tensor headers only; no model/NPU."""
import argparse,collections,hashlib,inspect,json,struct
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--model',required=True);a=p.parse_args()
r=Path(__file__).resolve().parent
import deepspeed.runtime.zero.stage3 as stage3
import deepspeed.runtime.comm.coalesced_collectives as collectives
names=['_DeepSpeedZeroOptimizer_Stage3__reduce_and_partition_ipg_grads','_DeepSpeedZeroOptimizer_Stage3__avg_scatter_grads','partition_grads']
def unwrap(function):
 for _ in range(8):
  if function.__module__!='deepspeed.utils.nvtx':break
  function=inspect.getclosurevars(function).nonlocals['func']
 return inspect.getsource(function)
source={name:unwrap(getattr(stage3.DeepSpeedZeroOptimizer_Stage3,name)) for name in names}
source['reduce_scatter_coalesced']=unwrap(collectives.reduce_scatter_coalesced)
(r/'native-source.private.json').write_text(json.dumps(source,indent=2))
model=Path(a.model);parameters=collections.defaultdict(list)
for path in sorted(model.glob('*.safetensors')):
 with path.open('rb') as f:
  size=struct.unpack('<Q',f.read(8))[0];assert size<10000000
  header=json.loads(f.read(size))
 for name,item in header.items():
  if name=='__metadata__' or not (name.startswith('model.language_model.') or name=='lm_head.weight'):continue
  shape=item['shape'];numel=1
  for x in shape:numel*=x
  parameters[numel].append({'name':name,'shape':shape,'dtype':item['dtype']})
total=sum(n*len(v) for n,v in parameters.items());assert total==26895998464,total
targets=[89128960,1271398400,52428800,31457280,62914560,10486272]
v={'no_model_loaded':True,'checkpoint_headers_only':True,'trained_text_parameters':total,'source_sha256':{k:hashlib.sha256(v.encode()).hexdigest() for k,v in source.items()},
 'matching_parameter_groups':{str(n):{'parameter_count':len(parameters[n]),'examples':parameters[n][:6]} for n in targets}}
(r/'native-inspection.json').write_text(json.dumps(v,indent=2));print(json.dumps(v,indent=2))
