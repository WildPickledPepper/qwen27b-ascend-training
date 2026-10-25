"""Compare CPU BF16 staging with existing direct FP32-host to BF16-NPU copy."""
import json,os
from pathlib import Path
os.environ['ASCEND_RT_VISIBLE_DEVICES']='3'
import torch,torch_npu
torch.npu.set_device(0);torch.set_num_threads(12)
bits=torch.arange(65536,dtype=torch.int64)
bits=bits[(bits&0x7f80)!=0x7f80]
samples=torch.cat([(bits<<16)+x for x in [0,32767,32768,32769,65535]])
samples=samples.to(torch.int32).view(torch.float32)
samples=samples[torch.isfinite(samples)]
direct=torch.empty(samples.shape,device='npu',dtype=torch.bfloat16);direct.copy_(samples)
staged=samples.to(torch.bfloat16).to('npu');torch.npu.synchronize()
d=int((direct.cpu().view(torch.int16)!=staged.cpu().view(torch.int16)).sum())
result={'completed':True,'coordinates':samples.numel(),'bf16_bit_differences':d,'device':torch.npu.get_device_name(0)}
Path('cast-probe.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
