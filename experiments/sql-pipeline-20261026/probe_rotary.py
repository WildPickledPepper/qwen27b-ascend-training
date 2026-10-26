"""Compare native zero.Init rotary buffers with direct CPU/NPU construction."""
import hashlib,json,os
from pathlib import Path
import torch,torch_npu,deepspeed
from transformers import AutoConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextRotaryEmbedding
rank=int(os.environ['LOCAL_RANK']);torch.npu.set_device(rank)
deepspeed.init_distributed(dist_backend='hccl')
cfg=AutoConfig.from_pretrained(os.environ['MODEL_DIR'],local_files_only=True).text_config
native_cfg={'train_batch_size':2,'train_micro_batch_size_per_gpu':1,'bf16':{'enabled':True},'zero_optimization':{'stage':3}}
with deepspeed.zero.Init(config_dict_or_path=native_cfg,dtype=torch.bfloat16):
    native=Qwen3_5TextRotaryEmbedding(config=cfg)
cpu=Qwen3_5TextRotaryEmbedding(config=cfg)
npu=Qwen3_5TextRotaryEmbedding(config=cfg,device=f'npu:{rank}')
def data(m):return m.inv_freq.detach().cpu()
def sha(t):return hashlib.sha256(t.numpy().tobytes()).hexdigest()
x,y,z=map(data,[native,cpu,npu])
result=dict(rank=rank,native_dtype=str(x.dtype),native_sha256=sha(x),cpu_sha256=sha(y),npu_sha256=sha(z),
            native_cpu_exact=torch.equal(x,y),native_npu_exact=torch.equal(x,z),
            cpu_max_abs=float((x-y).abs().max()),npu_max_abs=float((x-z).abs().max()))
Path(f'rotary-rank{rank}.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
torch.distributed.destroy_process_group()
