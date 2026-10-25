"""Verify observation preserves cross-device tensor values and counts each copy once."""
import collections,json,os
os.environ['ASCEND_RT_VISIBLE_DEVICES']='2,3'
import torch,torch_npu
from gradient_scopes import TransferObserver
torch.npu.set_device(0);torch.manual_seed(42)
source=torch.randn(16384,dtype=torch.float32);stats=collections.defaultdict(float)
expected=source.npu().cpu();torch.npu.synchronize()
with TransferObserver(stats):
 device=source.to('npu');copied=device.to('cpu')
 target=torch.empty_like(source);target.copy_(device,non_blocking=True)
torch.npu.synchronize()
assert torch.equal(copied,expected) and torch.equal(target,source)
assert stats['h2d_source_bytes']==source.numel()*4
assert stats['d2h_source_bytes']==source.numel()*8
result={'passed':True,'values_equal':True,'counts_verified':True,'stats':dict(stats)}
open('observer-probe.json','w').write(json.dumps(result,indent=2))
print(json.dumps(result),flush=True)
