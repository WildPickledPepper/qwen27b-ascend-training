"""Create a guarded replay configuration from the successful measured run."""
import hashlib,json
from pathlib import Path
r=Path(__file__).resolve().parent;repo=r.parents[1]
result=json.loads((r/'rank0-result.private.json').read_text())
evidence=json.loads((repo/'evidence/2026-10-25/sql-reduce-schedule-results.json').read_text())
reference=json.loads((repo/'experiments/sql-bottleneck-20261025/rank0-result.private.json').read_text())
assert result['completed'] and result['config']==reference['config']
assert evidence['time_reduction_percent']>0 and evidence['validation_loss_difference']==0
assert result['reduce_schedule']['event_limit']==4
(r/'zero3-selected.json').write_text(json.dumps(result['config'],indent=2),encoding='utf-8',newline='\n')
names=['train_reduce.py','cpu_accumulation.py','cpu_update.py','bounded_grad_copy.py','community_gdn.py','gradient_cache.py','gradient_scopes.py','zero3-selected.json']
v={'name':'sql-fixed256-reduce-lookahead4','global_batch':256,'micro_batch_per_rank':16,'accumulation':8,
 'physical_devices':[2,3],'cpu_threads_per_rank':24,'numa_spread':True,'cpu_update_mode':'pipeline',
 'cpu_gradient_receive_slots':32,'cpu_gradient_pinned_bytes_per_rank':6569328640,
 'native_reduce_event_limit':4,'max_queued_reduce_events':5,'max_sequence_length_cap':512,
 'padding':'dynamic64, minimum128; unchanged','training_sample_order':'unchanged',
 'compute_dtype':'BF16','master_m_v_dtype':'FP32 on CPU; unchanged',
 'seconds_per_256':evidence['scheduled_seconds_per_256'],'reference_seconds_per_256':evidence['baseline_seconds_per_256'],
 'time_reduction_percent':evidence['time_reduction_percent'],'config':'zero3-selected.json',
 'source_hash_normalization':'LF newlines',
 'files_sha256':{name:hashlib.sha256((r/name).read_bytes().replace(b'\r\n',b'\n')).hexdigest() for name in names},
 'scope':'Measured on the current SQL short corpus; not a capacity or convergence claim for arbitrary data.'}
(r/'selected-profile.json').write_text(json.dumps(v,indent=2),encoding='utf-8',newline='\n');print(json.dumps({'selected':v['name'],'seconds_per_256':v['seconds_per_256']}))
