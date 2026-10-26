"""Seal the measured residency candidate's source/config hashes."""
import hashlib,json
from pathlib import Path
root=Path(__file__).resolve().parent
names=['train_cache.py','resident_gradients.py','community_gdn.py']
result=dict(name='sql-two-stage-bf16-resident-gradient-2gib',global_batch=256,micro_batch=16,pipeline_microbatches=16,
            resident_bytes_per_rank=2*2**30,compute='BF16',master_m_v='FP32 CPU',checkpointed_layers=48,
            seconds_per_256=85.0901912350673,source_lf_sha256={name:hashlib.sha256((root/name).read_bytes().replace(b'\r\n',b'\n')).hexdigest() for name in names},
            scope='Measured SQL short corpus; no general convergence or hardware-limit claim')
(root/'selected-profile.json').write_text(json.dumps(result,indent=2))
