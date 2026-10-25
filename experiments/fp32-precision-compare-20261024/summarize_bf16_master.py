"""Export aggregate evidence only: no weights, data text or private paths."""
import argparse, hashlib, json, statistics
from pathlib import Path
p = argparse.ArgumentParser(); p.add_argument('--root',default='.'); p.add_argument('--output',required=True)
p.add_argument('--name',default='bf16-single-run2')
a = p.parse_args(); root = Path(a.root)
status = json.loads((root/a.name/'status.json').read_text())
rows = [r for r in status['logs'] if r['event']=='step_complete']
before = status.get('validation_before',{})
after = status.get('validation_after',{})
bf16_after = status.get('validation_after_bf16_execution',{})
reference = json.loads((root/'cpu-pipeline-run1/status.json').read_text())
old = [r for r in reference['logs'] if r['event']=='step_complete']
assert status['completed'] and status['full_parameter_steps']==20 and len(rows)==20
assert status['data']['train_hashes'][:4]==reference['data']['train_hashes']
assert status['data']['validation_hashes']==reference['data']['validation_hashes']
assert all(r['updated_parameters']==26_895_998_464 and r['remaining_parameter_gradients']==0 for r in rows)
assert not set(status['data']['train_hashes']) & set(status['data']['validation_hashes'])
def metrics(row):
    result={k:row[k] for k in ['loss','assistant_token_accuracy','assistant_tokens','canonical_fp32'] if k in row}
    if 'examples' in row:
        result['examples']=[{k:r[k] for k in ['hash','tokens','assistant_tokens','loss','correct']} for r in row['examples']]
    return result
job = json.loads((root/(a.name+'-job.json')).read_text())
for name, digest in job['source_sha256'].items():
    assert hashlib.sha256((root/name).read_bytes()).hexdigest()==digest,name
result = {'completed':True,'full_parameter_steps':20,'physical_devices':status['physical_devices'],
          'precision':{'linear_compute_weights':'bf16','linear_forward_backward_matmuls':'bf16',
                       'linear_outputs':'fp32','linear_backward':'native BF16 autograd; input gradients cast to FP32 activation dtype',
                       'other_operations_embedding_tiny_parameters':'fp32',
                       'master_weights':'fp32','adam_m_v':'fp32','master_optimizer_device':'cpu',
                       'global_autocast':False,'state_quantization':False},
          'dataset':status['data'],'unique_training_examples':status['training_unique_examples'],
          'repeated_training_examples':status['training_repeats_if_needed'],
          'checkpoint_saved':False,'source_sha256':job['source_sha256'],
          'source_sha256_lf':{name:hashlib.sha256((root/name).read_bytes().replace(b'\r\n',b'\n')).hexdigest() for name in job['source_sha256']},
          'optimizer_probe':json.loads((root/'bf16-master-probe.json').read_text()),
          'validation_before_fp32':metrics(before),'validation_after_fp32':metrics(after),
          'validation_before_bf16_preflight':metrics(status['validation_before_bf16_preflight']),
          'validation_after_bf16_execution':metrics(bf16_after),
          'validation_after_first4_fp32':metrics(status.get('validation_after_first4_fp32',{})),
          'first_step_sampled_gradient_l2':status['first_step_sampled_gradient_l2'],
          'reference_fp32_validation_after':metrics(reference['validation_after']),
          'steps':[{k:r[k] for k in ['step','loss','seconds','updated_parameters','remaining_parameter_gradients',
                                     'weights_and_guards_bytes','moment_bytes','cpu_master_bytes','memory','host_max_rss_kib',
                                     'pipeline_timing','cpu_master_timing','bf16_matmul_counts']} for r in rows],
          'progress_validation':[{'step':r['step'],'metrics':metrics(r['metrics'])} for r in status['logs'] if r['event']=='validation_progress'],
          'comparison':{'reference_fp32_first4_seconds':[r['seconds'] for r in old],
                        'bf16_first4_seconds':[r['seconds'] for r in rows[:4]],
                        'reference_fp32_steps2_4_mean_seconds':statistics.mean(r['seconds'] for r in old[1:]),
                        'bf16_steps2_4_mean_seconds':statistics.mean(r['seconds'] for r in rows[1:4]),
                        'bf16_last5_mean_seconds':statistics.mean(r['seconds'] for r in rows[-5:]),
                        'same_first4_training_examples':True,'same_validation_examples':True,
                        'reference_not_rerun':True,'short_sample_only':True,
                        'reference_physical_devices':[2,3],'candidate_physical_devices':status['physical_devices'],
                        'placement_and_backward_implementation_changed':True}}
result['comparison']['first4_speedup_excluding_cold_step'] = result['comparison']['reference_fp32_steps2_4_mean_seconds']/result['comparison']['bf16_steps2_4_mean_seconds']
if (root/'bf16-master-run1/status.json').exists():
    initial=json.loads((root/'bf16-master-run1/status.json').read_text())
    result['interrupted_custom_backward_attempt']={'completed':False,
        'completed_steps':initial['full_parameter_steps'],
        'reason':'Manually stopped after step-10 validation loss worsened and sampled gradients were abnormal; not a valid mixed-precision baseline',
        'step_seconds':[r['seconds'] for r in initial['logs'] if r['event']=='step_complete'],
        'progress_validation':[{'step':r['step'],'loss':r['metrics']['loss']} for r in initial['logs'] if r['event']=='validation_progress']}
for filename in ['bf16-ffn-native-probe.json','bf16-ffn-native-cross-probe.json','bf16-backward-diagnostic1/backward-diagnostic.json']:
    result[filename.replace('/','_')]=json.loads((root/filename).read_text())
result['optimizer_probe_fixed']=json.loads((root/'bf16-master-probe-fixed.json').read_text())
result['native_cross_probe_fixed']=json.loads((root/'bf16-ffn-native-cross-fixed-probe.json').read_text())
result['native_persistent_probe']=json.loads((root/'bf16-ffn-native-persistent-probe.json').read_text())
failed_native=json.loads((root/'bf16-native-run1/status.json').read_text())
result['failed_native_multidevice_attempt']={'completed':False,'completed_steps':failed_native['full_parameter_steps'],
    'reason':'Runtime reported invalid stream/event handles during first backward; candidate was discarded, not treated as a valid precision comparison'}
failed_single=json.loads((root/'bf16-single-run1/status.json').read_text())
result['failed_single_ephemeral_attempt']={'completed':False,'completed_steps':failed_single['full_parameter_steps'],
    'reason':'OOM allocating 2.37 GiB in native BF16 backward with a temporary BF16 weight copy; replaced by direct autograd on persistent compute-weight storage'}
for label,filename in [('custom_backward_initial','bf16-master-job.json'),('native_multidevice','bf16-native-run1-job.json'),('native_single_ephemeral','bf16-single-run1-job.json')]:
    result.setdefault('failed_attempt_source_sha256',{})[label]=json.loads((root/filename).read_text())['source_sha256']
Path(a.output).write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps({'completed':True,'comparison':result['comparison'],'validation_after_fp32':metrics(after),
                  'validation_after_bf16_execution':metrics(bf16_after)}),flush=True)
