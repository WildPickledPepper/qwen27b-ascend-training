"""CPU-only formula checks and a held-out historical sequence-length check."""
import csv
import hashlib
import json
from pathlib import Path
from memory_model import ROOT, CONFIG, GIB, predict

REPO = ROOT.parents[1]
evidence = REPO/'evidence/2026-10-26'
observed = json.loads((evidence/'sql-memory-breakdown.json').read_text(encoding='utf-8-sig'))
old = json.loads((evidence/'sql-pipeline-overlap-cache4-results.json').read_text(encoding='utf-8-sig'))

# Independent parameter manifest, not the architecture equation under test.
counts = [0,0]
manifest = REPO/'research/2026-10-24-adam-state-memory/stage11/qwen_audit.parameters.csv'
for row in csv.DictReader(manifest.open(encoding='utf-8-sig')):
    name = row['name']
    if name.startswith('layers.'):
        rank = int(name.split('.')[1])//32
    else:
        rank = 0 if name == 'embed_tokens.weight' else 1
    counts[rank] += int(row['numel'])

baseline = predict()
checks = []
for rank in [0,1]:
    formula = {x['category']:x for x in baseline['ranks'][rank]['categories']}
    actual = {x['category']:x for x in observed['ranks'][rank]['categories']}
    assert formula['weight']['bytes'] == counts[rank]*2
    check = dict(rank=rank,architecture_parameter_count_exact=True,formula_categories={})
    for name in ['ffn_forward_intermediates','rmsnorm_forward_intermediates',
                 'attention_forward_intermediates','hidden_states_checkpoint_boundaries_and_pipeline']:
        exact = formula[name]['bytes'] == actual[name]['bytes']
        assert exact, name
        check['formula_categories'][name] = dict(bytes=formula[name]['bytes'],reference_exact=exact)
    if rank == 1:
        assert formula['loss_forward_intermediates']['bytes'] == 16*256*248320*4
        assert formula['temporary_gradients']['bytes'] == 2*16*256*248320*4
    checks.append(check)

cases = []
for sequence, step, label in [(256,1,'calibrated_reference_not_independent'),
                              (192,2,'held_out_historical_shape_not_used_for_calibration')]:
    result = predict(s=sequence)
    for rank in [0,1]:
        expected = old['rank_results'][rank]['steps'][step-1]['allocated_peak']/GIB
        estimate = result['ranks'][rank]['estimated_allocated_peak_template_gib']
        error = estimate-expected
        assert abs(error) < 0.002
        cases.append(dict(case=label,micro_batch=16,padded_sequence=sequence,rank=rank,
                          predicted_gib=estimate,observed_gib=expected,
                          error_mib=error*1024,error_percent=100*error/expected))

assert baseline['ranks'][0]['estimated_physical_with_supplied_overheads_gib'] is None
for args in [dict(b=0),dict(s=257),dict(s=64),dict(resident_gib=-1)]:
    try:
        predict(**args)
    except ValueError:
        continue
    raise AssertionError(args)

what_if = [dict(micro_batch=b,sequence=s,status='not_run_on_hardware',
                predicted_allocated_template_gib=[r['estimated_allocated_peak_template_gib']
                                                 for r in predict(b=b,s=s)['ranks']])
           for b,s in [(8,256),(16,128),(16,512)]]
report = dict(parameter_manifest_checks=checks,cases=cases,what_if=what_if,
              sequence_evidence=dict(source='original cache4-full logs, groups(step) and dynamic64/min128 padding',
                                     step2_padded_micro_sequences=[192]*16),
              config_sha256=hashlib.sha256(CONFIG.read_bytes()).hexdigest(),
              calibration_sha256=hashlib.sha256((ROOT/'calibration.json').read_bytes()).hexdigest(),
              limits=['256 reference is calibrated; it is not an independent accuracy test.',
                      '192 held-out case checks only one additional length at unchanged B, implementation and schedule.',
                      '8-batch,128/512 length predictions are unvalidated; 512 may trigger other peak/workspace behavior.',
                      'No NPU job was started; physical HBM prediction requires runtime and allocator slack inputs.'])
(evidence/'sql-memory-model-validation.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
(evidence/'sql-memory-model-reference.json').write_text(json.dumps(baseline,indent=2)+'\n',encoding='utf-8')
print(json.dumps(dict(cases=cases,what_if=what_if),indent=2))
