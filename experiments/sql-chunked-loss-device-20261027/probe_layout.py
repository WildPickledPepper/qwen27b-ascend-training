import json,subprocess
from pathlib import Path
import torch,torch_npu
from chunked_loss_diagnostic import chunked_linear_ce,records
smi=subprocess.check_output(['npu-smi','info'],text=True)
assert all(f'No running processes found in NPU {i}' in smi for i in [3,4])
torch.set_num_threads(4);torch.npu.set_device(1);torch.manual_seed(42)
x=(torch.randn(16,256,5120,device='npu:1',dtype=torch.bfloat16)*.1).requires_grad_()
w=(torch.randn(248320,5120,device='npu:1',dtype=torch.bfloat16)*.02).requires_grad_()
target=torch.randint(0,248320,(16,256),device='npu:1');target[:,:5]=-100
torch.npu.synchronize();torch.npu.reset_peak_memory_stats()
loss=chunked_linear_ce(x,w,target,512);loss.backward();torch.npu.synchronize()
result=dict(physical_card=4,shape=[16,256,5120,248320],loss=float(loss),allocated_peak_bytes=torch.npu.max_memory_allocated(),records=records)
Path('layout-diagnostic.private.json').write_text(json.dumps(result,indent=2))
print(json.dumps(dict(peak_gib=result['allocated_peak_bytes']/2**30,records=[dict(stage=r['stage'],start=r['start'],allocated_gib=r['allocated_bytes']/2**30,peak_gib=r['peak_bytes']/2**30,dw=r['dw'],gw=r['gw']) for r in records[:5]])))
