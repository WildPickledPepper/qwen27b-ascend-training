"""Compare completed single/dual BF16 runs; export metrics, never raw probes."""
import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument('--single', default='bf16-single-run2')
p.add_argument('--dual', default='bf16-dual-run3')
p.add_argument('--output', required=True)
a = p.parse_args()
root = Path(__file__).resolve().parent

def collect(name):
    status = json.loads((root/name/'status.json').read_text())
    job = json.loads((root/(name+'-job.json')).read_text())
    assert status['completed'] and status['full_parameter_steps'] == 20
    assert job['stage'] == 'complete' and job['returncode'] == 0
    steps = [r for r in status['logs'] if r['event'] == 'step_complete']
    assert len(steps) == 20
    assert all(r['updated_parameters'] == 26895998464 and r['remaining_parameter_gradients'] == 0 for r in steps)
    seconds = [r['seconds'] for r in steps]
    metrics = {}
    for key in ['validation_before', 'validation_after_first4_fp32', 'validation_after', 'validation_after_bf16_execution']:
        v = status[key]
        assert math.isfinite(v['loss'])
        metrics[key] = {k:v[k] for k in ['loss','assistant_token_accuracy','assistant_tokens','canonical_fp32']}
    result = {'physical_devices':status['physical_devices'], 'task_queue_enable':status.get('task_queue_enable','not explicitly recorded'), 'steps':20,
              'step_seconds':seconds, 'total_training_seconds':sum(seconds),
              'mean_seconds':statistics.mean(seconds), 'median_seconds':statistics.median(seconds),
              'steps2_4_mean_seconds':statistics.mean(seconds[1:4]),
              'steps2_20_mean_seconds':statistics.mean(seconds[1:]),
              'peak_allocated_gib':[max(r['memory'][i]['peak_allocated'] for r in status['logs'] if 'memory' in r)/2**30 for i in [0,1]],
              'host_peak_rss_gib':max(r['host_max_rss_kib'] for r in steps)/2**20,
              'validation':metrics, 'source_sha256':job['source_sha256']}
    result['overlapping_pipeline_work_seconds_per_step_2_20'] = {
        key:(steps[-1]['pipeline_timing'][key]-steps[0]['pipeline_timing'][key])/19
        for key in ['d2h_event_seconds','h2d_event_seconds','cpu_adam_seconds','weight_stage_seconds','slot_wait_seconds']}
    result['synchronous_embedding_work_seconds_per_step_2_20'] = {
        key:(steps[-1]['cpu_master_timing'][key]-steps[0]['cpu_master_timing'][key])/19
        for key in ['gradient_d2h_seconds','cpu_adam_seconds','weight_stage_seconds','weight_h2d_seconds']}
    return status, result

s, single = collect(a.single)
d, dual = collect(a.dual)
assert s['data'] == d['data']
assert all(s[k] == d[k] for k in ['lr','seed','gradient_accumulation','global_gradient_clipping'])
assert not set(s['data']['train_hashes']) & set(s['data']['validation_hashes'])
for key in ['bench_bf16_native.py','bench_bf16_master.py','bench_cpu_pipeline.py','bench_cpu_master.py']:
    assert single['source_sha256'][key] == dual['source_sha256'][key], key

def probes(name):
    records = json.loads((root/name/'parameter-probes.json').read_text())
    result = {}
    for row in records:
        if row['step'] != 1: continue
        for i,w,g in zip(row['indices'], row['weight_before'], row['gradient']):
            k = row['name'], i
            assert k not in result
            result[k] = w,g
    return result

sp, dp = probes(a.single), probes(a.dual)
assert sp.keys() == dp.keys()
ref2 = sum(v[1]**2 for v in sp.values())
diff2 = sum((sp[k][1]-dp[k][1])**2 for k in sp)
result = {'completed':True, 'same_20_examples_and_validation':True,
          'same_compute_and_optimizer_implementation':True,
          'configuration_difference':'dual run sets TASK_QUEUE_ENABLE=0 to avoid reproduced invalid stream/event error; single run used original environment',
          'scope':'one process with model weights split across two devices; CPU FP32 master/Adam; not two-way data parallel',
          'single':single, 'dual':dual,
          'single_over_dual_mean_speed_ratio':single['mean_seconds']/dual['mean_seconds'],
          'single_over_dual_steps2_20_speed_ratio':single['steps2_20_mean_seconds']/dual['steps2_20_mean_seconds'],
          'first_step_gradient_comparison':{'coordinates':len(sp),
              'relative_l2':math.sqrt(diff2/max(ref2,1e-300)),
              'initial_weight_max_abs_difference':max(abs(sp[k][0]-dp[k][0]) for k in sp)},
          'checkpoint_saved':False}
(root/a.output).write_text(json.dumps(result,indent=2))
print(json.dumps(result),flush=True)
