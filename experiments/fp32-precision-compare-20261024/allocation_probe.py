import ctypes,os,time,json,resource
from pathlib import Path
lib=ctypes.CDLL(None,use_errno=True)
if os.environ.get('DISABLE_THP')=='1':assert lib.prctl(41,1,0,0,0)==0
import torch
torch.set_num_threads(4)
def counters():return {k:int(v) for k,v in (l.split() for l in Path('/proc/vmstat').read_text().splitlines()) if k in ['compact_stall','allocstall_movable']}
before=counters();times=[]
for i in range(4):
 t=time.monotonic();x=torch.zeros(134217728,dtype=torch.float32);times.append(time.monotonic()-t);del x
print(json.dumps({'disabled':bool(lib.prctl(42,0,0,0,0)),'seconds':times,'counter_delta':{k:v-before[k] for k,v in counters().items()}}),flush=True)
