"""Export filtered compatibility evidence and failed native norm-hook diagnosis."""
import argparse,json
from pathlib import Path
root=Path(__file__).resolve().parent
p=argparse.ArgumentParser();p.add_argument('--probe-root',default='probe.private');p.add_argument('--output',default='probe-evidence.json');a=p.parse_args()
probe=root/a.probe_root
rows=[]
for rank in [0,1]:
    result=json.loads((probe/f'rank{rank}-result.json').read_text())
    logs=[json.loads(x) for x in (probe/f'rank{rank}.jsonl').read_text().splitlines()]
    init=next(x for x in logs if x['event']=='initialized')
    forward=next(x for x in logs if x['event']=='forward_check')
    calls=next(x['calls'] for x in logs if x['event']=='reattached_hook_calls')
    assert result['passed'] and all(c==16 for c in calls.values())
    assert all(x['relative_l2']<.08 or x['max_abs']<1e-6 for x in result['gradient_checks'])
    rows.append(dict(rank=rank,passed=True,forward=forward,gradient_checks=result['gradient_checks'],
                     hook_calls=calls,fp32_cpu_master_m_v=result['fp32_cpu_master_m_v'],
                     hybrid_layers=result['hybrid_layers'],data_parallel_size=init['data_parallel_size'],
                     tensor_parallel_size=init['tensor_parallel_size'],gradient_dtype=init['gradient_dtype'],config=result['config'],
                     nonboundary_gradient_storage=next((x for x in logs if x['event']=='nonboundary_grad_residence'),None),
                     local_shapes=init['local_shapes'],original_shapes=init['original_shapes']))
native=json.loads((root/'norm-diagnostic.json').read_text())
support=json.loads((root/'support.json').read_text())
export=dict(passed=True,production_path_without_observing_hooks=True,ranks=rows,native_failure=native,
            support=support,repair='Re-register only the two attention-norm SUM hooks on final Parameters after optimizer initialization; process-local',
            limits=['Tiny hybrid-model compatibility test, not a 27B speed or convergence result.',
                    'BF16 TP arithmetic differs; relative-gradient checks are not bitwise equivalence.'])
(root/a.output).write_text(json.dumps(export,indent=2));print('Both ranks passed production-path hybrid gradient gate')
