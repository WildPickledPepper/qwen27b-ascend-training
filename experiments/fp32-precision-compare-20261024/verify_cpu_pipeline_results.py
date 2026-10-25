"""Check recorded source identity, numerical invariants and released resources."""
import argparse, hashlib, json, subprocess
from pathlib import Path
p = argparse.ArgumentParser(); p.add_argument('--root', required=True); p.add_argument('--output', required=True)
a = p.parse_args(); root = Path(a.root)
result = json.loads((root/'cpu-pipeline-results.json').read_text())
job = json.loads((root/'cpu-pipeline-run1-job.json').read_text())
matched = {name: hashlib.sha256((root/name).read_bytes()).hexdigest() == digest
           for name, digest in job['source_sha256'].items()}
assert all(matched.values()), matched
sequence = json.loads((root/'cpu-pipeline-sequence.json').read_text())
assert sequence['stage'] == 'complete'
assert not Path('/proc/'+str(job['pid'])).exists()
smi = subprocess.check_output(['npu-smi', 'info'], text=True, timeout=30)
idle = {str(d): 'No running processes found in NPU '+str(d) in smi for d in [2, 3]}
assert all(idle.values()), idle
sources = result['source_sha256']
assert all(hashlib.sha256((root/name).read_bytes()).hexdigest() == digest for name, digest in sources.items())
cleanup = {'own_pipeline_process_exited': True, 'physical_cards_no_active_processes': idle,
           'source_matches_launch_snapshot': matched, 'sequence_complete': True}
result['verification_and_cleanup'] = cleanup
Path(a.output).write_text(json.dumps(result, indent=2))
print(json.dumps({'verification_and_cleanup': cleanup,
                  'groups': {name: {'step_seconds': [r['seconds'] for r in g['steps']],
                                    'warm_seconds': g['warm_mean_seconds'], 'peak_allocated_gib': g['peak_allocated_gib'],
                                    'host_peak_rss_gib': g['host_peak_rss_gib']} for name, g in result['groups'].items()}}, indent=2))
