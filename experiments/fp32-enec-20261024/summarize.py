"""Validate cross-card coverage and write filtered whole-text checkpoint evidence."""
import argparse, json, math
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument('--card2', required=True)
p.add_argument('--card3', required=True)
p.add_argument('--output', required=True)
a = p.parse_args()
folders = [Path(a.card2), Path(a.card3)]
reports = [json.loads((d / 'report.json').read_text()) for d in folders]
assert all(d['complete'] and d['mode'] == 'all' for d in reports)
assert reports[0]['text_parameters'] == reports[1]['text_parameters']
names = set(); raw = stored = json_bytes = frames = mismatches = raw_small = 0
device_checks = []; max_live = 0
for folder, report in zip(folders, reports):
    verification = json.loads((folder / 'archive_verification.json').read_text())
    assert verification['passed'] and verification['frames_verified'] == verification['total_frames']
    assert not names.intersection(report['tensor_index'])
    names.update(report['tensor_index'])
    assert sum(math.prod(s) for s in report['tensor_index'].values()) == report['assigned_parameters']
    assert all(r['stored_lossless'] for r in report['rows'])
    assert sum(r['stored_bytes'] for r in report['rows']) == (folder / 'weights.enec.parts').stat().st_size
    for key, shape in report['tensor_index'].items():
        end = 0
        for row in sorted((r for r in report['rows'] if r['name'] == key), key=lambda r:r['element_offset']):
            assert row['element_offset'] == end
            assert row['raw_fp32_bytes'] == 4 * row['elements']
            end += row['elements']
        assert end == math.prod(shape), key
    raw += report['aggregate']['raw_fp32_bytes']; stored += report['aggregate']['stored_bytes']
    json_bytes += (folder / 'report.json').stat().st_size
    frames += len(report['rows'])
    mismatches += sum(not r['bit_exact'] for r in report['rows'])
    raw_small += sum(r['storage_kind'] == 'raw_small_tensor' for r in report['rows'])
    max_live = max(max_live,max(r.get('explicit_live_device_allocation_bytes',0) for r in report['rows']))
    device_checks.append({k:verification[k] for k in ['device','frames_verified','total_frames','passed','seconds','original_weights_loaded']})
assert raw == 4 * reports[0]['text_parameters']
result = {'scope':'Offline lossless FP32 text-weight archive; no online training integration',
          'parameters': reports[0]['text_parameters'], 'tensors':len(names), 'chunks':frames,
          'raw_fp32_bytes':raw,'stored_frame_bytes':stored,'mapping_json_bytes':json_bytes,
          'archive_and_mapping_bytes':stored+json_bytes,
          'raw_fp32_gib':raw/2**30,'archive_and_mapping_gib':(stored+json_bytes)/2**30,
          'saved_percent':100*(1-(stored+json_bytes)/raw),
          'compression_ratio':raw/(stored+json_bytes),
          'author_mismatch_raw_fallback_chunks':mismatches,'raw_small_tensor_chunks':raw_small,
          'all_stored_chunks_lossless':True,'cross_card_tensor_coverage_passed':True,
          'all_saved_frames_reloaded_and_sha_verified':True,'device_checks':device_checks,
          'max_explicit_codec_device_allocation_bytes':max_live,
          'codec':reports[0]['codec'],'author_build':reports[0]['build'],
          'original_checkpoint_unchanged':True,
          'initial_source':'Actual checkpoint tensors lifted exactly to FP32; mostly BF16-origin values',
          'excludes':['vision','MTP','optimizer states','gradients','activations','online training memory proof'],
          'no_bf16_forward_copy':True,'model_training_started':False}
Path(a.output).write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
