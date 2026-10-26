"""Seal only the faster, checked 4GiB residency result and portable replay sources."""
import argparse,hashlib,json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--result',required=True);a=p.parse_args();root=Path(__file__).resolve().parent
r=json.loads(Path(a.result).read_text());assert r['time_reduction_percent']>0
assert all(x['all_four_gradient_norms_exact'] and x['validation_before_and_after_768_exact'] for x in r['checks'])
cfg=dict(name='sql-two-stage-bf16-resident-gradient-4gib',global_batch=256,micro_batch=16,
    pipeline_microbatches=16,resident_bytes_per_rank=4*2**30,compute='BF16',master_m_v='FP32 CPU',
    checkpointed_layers=48,seconds_per_256=r['seconds_per_256'],
    source_lf_sha256={n:hashlib.sha256((root/n).read_bytes().replace(b'\r\n',b'\n')).hexdigest() for n in ['train_resident.py','resident_gradients.py','community_gdn.py']},
    scope='Measured same-workload SQL short run, not a convergence or hardware-limit claim; original copy path, no async variant')
(root/'selected-profile.json').write_text(json.dumps(cfg,indent=2));print(json.dumps(cfg))
