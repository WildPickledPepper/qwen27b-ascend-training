import ctypes,time,json
assert ctypes.CDLL(None).prctl(41,1,0,0,0)==0
import torch,torch_npu
from pathlib import Path
torch.set_num_threads(4);torch.manual_seed(20261024);torch.npu.set_device(0)
n=89128960;window=1048576;source=torch.randn(window,device='npu:0')
rows=[];reference=None
for route in ['direct_to_cpu_slice','new_cpu_tensor_then_cpu_copy']:
 destination=torch.zeros(n,dtype=torch.float32)
 start=time.monotonic()
 for i in range(1,33):
  chunk=destination[i*window:(i+1)*window]
  if route=='direct_to_cpu_slice':chunk.copy_(source)
  else:chunk.copy_(source.cpu())
 elapsed=time.monotonic()-start
 result={'route':route,'copy_bytes':32*window*4,'seconds':elapsed}
 if reference is None:reference=destination
 else:result['same_bits']=torch.equal(destination.view(torch.int32),reference.view(torch.int32));assert result['same_bits']
 rows.append(result)
 del destination
Path(__file__).with_suffix('.json').write_text(json.dumps({'passed':True,'results':rows},indent=2));print(json.dumps(rows),flush=True)
