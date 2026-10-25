"""Add isolated checkpoint/NUMA and profiler diagnostics to the frozen batch sweep."""
from pathlib import Path
r=Path(__file__).resolve().parent
s=(r/'train_throughput_v2.py').read_text()
s=s.replace("choices=['on','off']", "choices=['on','off','selective']")
s=s.replace("p.add_argument('--profile',action='store_true')", "p.add_argument('--profile',action='store_true')\np.add_argument('--numa-spread',action='store_true')")
s=s.replace("a=p.parse_args()", "a=p.parse_args()\nif a.profile:os.environ['HCCL_EXEC_TIMEOUT']='600'")
s=s.replace('assert chosen;os.sched_setaffinity(0,chosen)', '''if a.numa_spread:
 import ctypes
 chosen=[i for i in range(96+local*24,96+local*24+a.cpu_threads) if i in allowed]
 libnuma=ctypes.CDLL('libnuma.so.1',use_errno=True)
 mask=ctypes.c_ulong(1<<(4+local))
 libnuma.set_mempolicy.argtypes=[ctypes.c_int,ctypes.POINTER(ctypes.c_ulong),ctypes.c_ulong]
 assert libnuma.set_mempolicy(1,ctypes.byref(mask),8)==0,('NUMA preferred policy unavailable',ctypes.get_errno())
assert chosen;os.sched_setaffinity(0,chosen)''')
s=s.replace(" CPUAdamBuilder.load=lambda self,verbose=True:cpu_adam_extension", ''' if a.profile:
  real_adam_update=cpu_adam_extension.adam_update
  def profiled_adam_update(*args,**kwargs):
   with torch.profiler.record_function('optimizer/CPUAdam'):
    return real_adam_update(*args,**kwargs)
  cpu_adam_extension.adam_update=profiled_adam_update
 CPUAdamBuilder.load=lambda self,verbose=True:cpu_adam_extension''')
s=s.replace("checkpoint=a.checkpoint)", "checkpoint=a.checkpoint,numa_spread=a.numa_spread)",1)
s=s.replace(" if a.checkpoint=='on':model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})", ''' if a.checkpoint!='off':
  model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
  if a.checkpoint=='selective':
   for i,layer in enumerate(model.model.layers):
    if i%4==3:layer.gradient_checkpointing=False
 log('checkpoint_policy',layers=len(model.model.layers),enabled_layers=sum(bool(getattr(layer,'gradient_checkpointing',False)) for layer in model.model.layers))''')
s=s.replace('steps.append(row);log(\'step_complete\',**row)', '''steps.append(row);log('step_complete',**row)
 if a.profile and step==1:
  log('validation_after2',metrics=evaluate())
 if step==a.steps-1:
  import re
  numa_pages={}
  for line in Path('/proc/self/numa_maps').read_text().splitlines():
   if 'anon=' not in line and 'heap ' not in line:continue
   for node,n in re.findall(r' N(\\d+)=(\\d+)',line):numa_pages[node]=numa_pages.get(node,0)+int(n)
  log('numa_placement',anonymous_resident_gib={node:pages*os.sysconf('SC_PAGE_SIZE')/2**30 for node,pages in numa_pages.items()})''')
s=s.replace("'config':cfg,", "'numa_spread':a.numa_spread,'config':cfg,")
(r/'train_advanced.py').write_text(s,encoding='utf-8',newline='\n')
import ast
ast.parse(s)
print('Built isolated advanced/profiler entry')
