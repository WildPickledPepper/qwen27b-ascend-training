"""Portable results for original NPU offload, CPU master and pipelined CPU master."""
import argparse, hashlib, json, statistics
from pathlib import Path
p = argparse.ArgumentParser(); p.add_argument('--root', required=True); p.add_argument('--output', required=True)
a = p.parse_args(); root = Path(a.root); groups = {}; data = None
for label, folder in [('original_npu', 'fp32-npuadam-run1'), ('largewindow_npu', 'fp32-largewindow-numa-run1'),
                      ('cpu_master_sync', 'cpu-master-run1'), ('cpu_master_pipeline', 'cpu-pipeline-run1')]:
    status = json.loads((root/folder/'status.json').read_text())
    assert status['completed'] and status['full_parameter_steps'] == 4 and not status.get('error')
    if data is None: data = status['data']
    else: assert status['data'] == data
    rows = []; previous = {}
    for event in status['logs']:
        if event['event'] != 'step_complete': continue
        assert event['updated_parameters'] == 26895998464 and event['remaining_parameter_gradients'] == 0
        row = {'step': event['step'], 'seconds': event['seconds'], 'loss': event['loss']}
        for section in ['host_timing', 'cpu_master_timing', 'pipeline_timing']:
            if section not in event: continue
            old = previous.get(section, {})
            row[section] = {k: v-old.get(k, 0) for k, v in event[section].items()}
            previous[section] = event[section]
        rows.append(row)
    memory = [m for e in status['logs'] for m in e.get('memory', [])]
    groups[label] = {'steps': rows, 'warm_mean_seconds': statistics.mean(row['seconds'] for row in rows[1:]),
                     'validation_before': status['validation_before'], 'validation_after': status['validation_after'],
                     'peak_allocated_gib': [max(m['peak_allocated'] for m in memory if m['logical_device'] == d)/2**30 for d in [0, 1]],
                     'host_peak_rss_gib': max(e['host_max_rss_kib'] for e in status['logs'] if e['event'] == 'step_complete')/2**20,
                     'optimizer_compute_device': status.get('optimizer_compute_device', 'npu'),
                     'affinity': status.get('host_affinity'),
                     'pipeline_runtime': next((e['pipeline_runtime'] for e in reversed(status['logs']) if 'pipeline_runtime' in e), None)}
analyses = {name: json.loads((root/name).read_text()) for name in ['cpu-master-analysis.json', 'cpu-pipeline-analysis.json', 'pipeline-vs-sync-analysis.json']}
for step in analyses['pipeline-vs-sync-analysis.json']['per_step'].values():
    assert step['sampled_coordinates'] == 52928
    assert all(step[k]['max_abs_difference'] == 0 for k in ['weight', 'update', 'gradient'])
assert abs(groups['cpu_master_sync']['validation_after']['loss']-groups['cpu_master_pipeline']['validation_after']['loss']) < 1e-5
assert len({g['validation_after']['assistant_token_accuracy'] for g in groups.values()}) == 1
result = {'scope': 'Same independent four complete short training examples and eight validation examples; full FP32 text parameters, gradients and m/v. Original completed baselines reused.',
          'groups': groups,
          'speedup_pipeline_vs_previous_best': groups['largewindow_npu']['warm_mean_seconds']/groups['cpu_master_pipeline']['warm_mean_seconds'],
          'speedup_pipeline_vs_original': groups['original_npu']['warm_mean_seconds']/groups['cpu_master_pipeline']['warm_mean_seconds'],
          'analyses': analyses,
          'microprobes': {name: json.loads((root/name).read_text()) for name in ['cpu-master-probe.json', 'cpu-pipeline-probe.json']},
          'cpuadam_provenance': json.loads((root/'cpu-adam-async-build/provenance.json').read_text()),
          'notes': ['CPUAdam uses its own FP32 operation order and beta rounding; not claimed bitwise equal to NPU reference.',
                    'Pipeline and synchronous CPU paths must be compared by sampled real gradients/weights and independent validation; sampled equality is not exhaustive full-model state equality.',
                    'Pipeline microtest compares entire tested weights, master, m and v for three updates with tails, intersecting frame/window boundaries and slot reuse.',
                    'Parallel phase work sums/events overlap; do not sum them as disjoint portions of end-to-end time.',
                    'Pipeline changes include persistent device-frame writes and 24-thread worker initialization; speedup is not attributed exclusively to overlap.',
                    'CPU master adds roughly100.20GiB RAM; full FP32 m/v remain roughly200.39GiB. No whole-model BF16 copy or low-bit optimizer states.',
                    'Initial cpu-master-probe included a CPU finite-gradient scan. Full CPU-master and pipeline runs check the same gradient on NPU before transfer; the first probe adapter is not the final adapter snapshot.',
                    'Shared server and four short steps: retain cold and each warm step; not long-run throughput or long-run quality certification.']}
files = ['bench_cpu_master.py', 'bench_cpu_pipeline.py', 'probe_cpu_master.py', 'probe_cpu_pipeline.py',
         'build_async_cpuadam.py', 'build_cpu_master_entry.py', 'build_cpu_pipeline_entry.py',
         'train_cpu_master.py', 'train_cpu_pipeline.py', 'run_cpu_pipeline_sequence.py', 'summarize_cpu_pipeline.py']
result['source_sha256'] = {name: hashlib.sha256((root/name).read_bytes()).hexdigest() for name in files}
result['source_lf_sha256'] = {name: hashlib.sha256((root/name).read_bytes().replace(b'\r\n', b'\n')).hexdigest() for name in files}
Path(a.output).write_text(json.dumps(result, indent=2))
print(json.dumps({'warm_seconds': {k: g['warm_mean_seconds'] for k, g in groups.items()},
                  'loss': {k: g['validation_after']['loss'] for k, g in groups.items()},
                  'speedup': result['speedup_pipeline_vs_previous_best']}, indent=2))
