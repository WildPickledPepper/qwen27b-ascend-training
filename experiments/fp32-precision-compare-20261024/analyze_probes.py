"""Compare sampled training trajectories without exporting sampled model weights."""
import argparse
import json
import math
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument('--reference', required=True)
p.add_argument('--candidate', required=True)
p.add_argument('--output', required=True)
p.add_argument('--replay', action='store_true', help='Replay sampled real gradients with full FP32 CPU Adam states')
a = p.parse_args()


def load(path):
    data = {}
    for row in json.loads(Path(path).read_text()):
        for i, index in enumerate(row['indices']):
            key = (row['step'], row['name'], index)
            assert key not in data, key
            data[key] = (row['weight_before'][i], row['gradient'][i], row['weight_after'][i])
    return data


def compare(rows):
    count = 0
    weight_sum = 0.
    squared_error = {'weight': 0., 'update': 0., 'gradient': 0.}
    squared_reference = dict(squared_error)
    squared_candidate = dict(squared_error)
    dots = dict(squared_error)
    maxima = dict(squared_error)
    for r, c, weight in rows:
        count += 1
        weight_sum += weight
        values = {'weight': (r[2], c[2]), 'update': (r[2]-r[0], c[2]-c[0]),
                  'gradient': (r[1], c[1])}
        for name, (x, y) in values.items():
            squared_error[name] += (x-y)**2 * weight
            squared_reference[name] += x*x * weight
            squared_candidate[name] += y*y * weight
            dots[name] += x*y * weight
            maxima[name] = max(maxima[name], abs(x-y))
    result = {'sampled_coordinates': count, 'represented_coordinates': weight_sum}
    for name in squared_error:
        denominator = math.sqrt(squared_reference[name]*squared_candidate[name])
        result[name] = {'max_abs_difference': maxima[name],
                        'rmse': math.sqrt(squared_error[name]/max(weight_sum, 1)),
                        'relative_l2_error': math.sqrt(squared_error[name]/squared_reference[name]) if squared_reference[name] else None,
                        'cosine': dots[name]/denominator if denominator else None}
    return result


reference, candidate = load(a.reference), load(a.candidate)
assert reference.keys() == candidate.keys()
steps = sorted({key[0] for key in reference})
strata = {}
for step, name, index in reference:
    entry = strata.setdefault((step, name), [0, 0])
    entry[0] = max(entry[0], index+1)
    entry[1] += 1
weights = {key: total/count for key, (total, count) in strata.items()}
report = {'scope': '64 deterministic coordinates per real tensor (or all coordinates for smaller tensors); not full-weight verification',
          'weighting': 'Each sampled coordinate weighted by tensor size / sample count; an estimate, not full-weight L2 verification',
          'per_step': {str(step): compare((reference[k], candidate[k], weights[k[:2]]) for k in reference if k[0] == step) for step in steps}}


def full_state_replay(data):
    import torch
    torch.set_num_threads(4)
    coordinates = sorted((name, index) for step, name, index in data if step == steps[0])
    m = torch.zeros(len(coordinates), dtype=torch.float32)
    v = torch.zeros_like(m)
    output = {}
    for step in steps:
        keys = [(step, name, index) for name, index in coordinates]
        before = torch.tensor([data[k][0] for k in keys], dtype=torch.float32)
        grad = torch.tensor([data[k][1] for k in keys], dtype=torch.float32)
        m.mul_(.9).add_(grad, alpha=.1)
        v.mul_(.999).addcmul_(grad, grad, value=.001)
        before.mul_(1-1e-5*.01).addcdiv_(m/(1-.9**step), (v/(1-.999**step)).sqrt().add_(1e-8), value=-1e-5)
        expected = before.tolist()
        rows = (((data[k][0], data[k][1], expected[i]), data[k], weights[k[:2]]) for i, k in enumerate(keys))
        metrics = compare(rows)
        metrics.pop('gradient')
        output[str(step)] = metrics
    return output


if a.replay:
    report['replay_scope'] = 'Same sampled real gradient trajectory, replayed with complete FP32 Adam moments from zero; compares update mechanics, not a new full-model training run. CPU/NPU arithmetic differences are included.'
    report['reference_full_state_replay_control'] = full_state_replay(reference)
    report['candidate_same_gradient_full_state_replay'] = full_state_replay(candidate)
Path(a.output).write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))
