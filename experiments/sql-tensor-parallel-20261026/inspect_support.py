"""Read-only library/model support inventory, no model tensors or NPU jobs."""
import hashlib,importlib.metadata,json,os
from pathlib import Path
from transformers import AutoConfig
cfg=AutoConfig.from_pretrained(os.environ['MODEL_DIR'],local_files_only=True).text_config
fields=['hidden_size','intermediate_size','num_hidden_layers','num_attention_heads','num_key_value_heads','head_dim',
        'linear_num_key_heads','linear_num_value_heads','linear_key_head_dim','linear_value_head_dim','vocab_size']
result=dict(versions={name:importlib.metadata.version(name) for name in ['torch','torch-npu','transformers','deepspeed']},
            architecture={key:getattr(cfg,key) for key in fields},hf_tensor_parallel_plan=cfg.base_model_tp_plan,
            limits=['TP rules present does not establish correct Ascend training; hardware forward/backward probe required.',
                    'GDN gathered-column rules shard projection weights but retain full GDN activations on each rank.'],
            sources=['https://raw.githubusercontent.com/deepspeedai/DeepSpeed/master/docs/_tutorials/autotp-training.md',
                     'https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_5/configuration_qwen3_5.py'])
Path('support.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
