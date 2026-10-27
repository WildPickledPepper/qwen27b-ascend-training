import datetime,json,os
from pathlib import Path
import torch,torch_npu
import torch.distributed as dist
rank=int(os.environ['RANK']);torch.set_num_threads(2);torch.npu.set_device(rank)
dist.init_process_group('hccl',timeout=datetime.timedelta(seconds=120))
x=torch.ones((64,64),device=f'npu:{rank}',dtype=torch.bfloat16)
y=x@x;assert float(y[0,0])==64
v=torch.tensor(float(rank+1),device=f'npu:{rank}');dist.all_reduce(v);assert float(v)==3
if rank==0:dist.send(y,1)
else:
 z=torch.empty_like(y);dist.recv(z,0);assert torch.equal(y,z)
torch.npu.synchronize()
Path(f'init-rank{rank}.private.json').write_text(json.dumps(dict(rank=rank,physical_card=[3,4][rank],matmul=True,hccl=True,p2p=True,torch=torch.__version__,torch_npu=torch_npu.__version__)))
dist.destroy_process_group()
