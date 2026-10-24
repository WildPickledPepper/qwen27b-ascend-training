"""Validate the completed controlled runs and export portable, weight-free metrics."""
import argparse
import hashlib
import json
import statistics
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument('--root', required=True)
p.add_argument('--output', required=True)
p.add_argument('--bf16-folder', default='bf16-cachecheck-run1')
a = p.parse_args()
root = Path(a.root)
names = {'fp32_cpu_adam_auxiliary': 'fp32-run3', 'fp32_full': 'fp32-npuadam-run1',
         'bf16_full': a.bf16_folder, 'compressed_b128': 'compressed-run1'}
runs, summary = {}, {}
for label, folder in names.items():
    d = json.loads((root/folder/'status.json').read_text())
    assert d['completed'] and d['full_parameter_steps'] == 4, (label, 'incomplete')
    assert d['full_text_parameters'] == 26_895_998_464
    assert not d.get('error'), label
    steps = [x for x in d['logs'] if x['event'] == 'step_complete']
    assert len(steps) == 4
    assert all(x['updated_parameters'] == d['full_text_parameters'] and x['remaining_parameter_gradients'] == 0 for x in steps)
    for key in ['validation_before', 'validation_after']:
        evaluation = d[key]
        assert evaluation['canonical_fp32'] and len(evaluation['examples']) == 8
        total = sum(x['assistant_tokens'] for x in evaluation['examples'])
        weighted = sum(x['assistant_tokens']*x['loss'] for x in evaluation['examples'])/total
        accuracy = sum(x['correct'] for x in evaluation['examples'])/total
        assert total == evaluation['assistant_tokens']
        assert abs(weighted-evaluation['loss']) < 1e-10
        assert abs(accuracy-evaluation['assistant_token_accuracy']) < 1e-10
    seconds = [x['seconds'] for x in steps]
    forward = []
    for step in range(1, 5):
        begin = next(x for x in d['logs'] if x['event'] == 'step_begin' and x['step'] == step)
        end = next(x for x in d['logs'] if x['event'] == 'forward_done' and x['step'] == step)
        forward.append(end['elapsed']-begin['elapsed'])
    memory = [m for row in d['logs'] for m in row.get('memory', [])]
    summary[label] = {
        'step_seconds': seconds, 'mean_step_seconds': statistics.mean(seconds),
        'warm_step_seconds_mean_2_to_4': statistics.mean(seconds[1:]),
        'forward_seconds': forward,
        'peak_allocated_gib': [max(m['peak_allocated'] for m in memory if m['logical_device'] == device)/2**30 for device in [0, 1]],
        'peak_reserved_gib': [max(m['peak_reserved'] for m in memory if m['logical_device'] == device)/2**30 for device in [0, 1]],
        'host_peak_rss_gib': max(x['host_max_rss_kib'] for x in steps)/2**20,
        'weight_payload_gib': (steps[-1]['weights_and_guards_bytes']+d['native_parameters']*4)/2**30,
        'moment_payload_gib': steps[-1]['moment_bytes']/2**30,
        'cpu_state_offload': d['cpu_state_offload'],
        'validation_before': d['validation_before'], 'validation_after': d['validation_after'],
        'host_timing_cumulative': steps[-1]['host_timing'],
        'validation_after_bf16_execution': d.get('validation_after_bf16_execution')}
    runs[label] = d

reference = runs['fp32_full']
for label, d in runs.items():
    assert d['data'] == reference['data'], (label, 'data mismatch')
    assert abs(d['validation_before']['loss']-reference['validation_before']['loss']) < 1e-5
    row = summary[label]
    row['speedup_vs_full_fp32_warm'] = summary['fp32_full']['warm_step_seconds_mean_2_to_4']/row['warm_step_seconds_mean_2_to_4']
    row['validation_loss_delta_vs_full_fp32'] = d['validation_after']['loss']-reference['validation_after']['loss']
    row['validation_token_accuracy_delta_pp_vs_full_fp32'] = 100*(d['validation_after']['assistant_token_accuracy']-reference['validation_after']['assistant_token_accuracy'])

source_names = ['train_compare.py', 'bench_resident.py', 'train_compare_npu.py',
                'bench_resident_npu.py', 'bf16_precision_guards.py', 'train_compare_bf16_guarded.py', 'train_compare_bf16_cachecheck.py', 'analyze_probes.py', 'summarize_comparison.py']
result = {'scope': 'Four updates, batch1, complete short examples; eight independent validation examples. Custom harness, not a production-framework speed ceiling or long-term quality certification.',
          'timing_scope': 'Full-state controls instrument transfer/Adam regions. Compressed quantization and codec work is included in total step time but not decomposed by those counters; zero compressed counters do not imply zero NPU optimizer work.',
          'physical_devices': [2, 3], 'summary': summary, 'runs': runs,
          'source_sha256': {n: hashlib.sha256((root/n).read_bytes()).hexdigest() for n in source_names},
          'source_lf_sha256': {n: hashlib.sha256((root/n).read_bytes().replace(b'\r\n', b'\n')).hexdigest() for n in source_names},
          'source_hash_note': 'Raw executed bytes and LF-normalized content hashes both provided because Git normalizes Python text line endings.'}
Path(a.output).write_text(json.dumps(result, ensure_ascii=False, indent=2))
print(json.dumps({k: {'warm_seconds': v['warm_step_seconds_mean_2_to_4'], 'loss': v['validation_after']['loss'],
                       'token_accuracy': v['validation_after']['assistant_token_accuracy']} for k, v in summary.items()}, indent=2))
