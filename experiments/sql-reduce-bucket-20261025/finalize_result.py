"""Keep the faster reference and archive a numerically valid rejected candidate."""
import hashlib,json,statistics
from pathlib import Path
r=Path(__file__).resolve().parent;out=r.parents[1]/'evidence/2026-10-25'
load=lambda name:json.loads((out/name).read_text())
reference=load('sql-boundary-overlap-results.json');candidate=load('sql-reduce-bucket-results.json')
assert candidate['validation_loss_difference']==0 and load('sql-reduce-bucket-verification.json')['passed']
hist=lambda name:sorted((x['hccl_count'],x['dtype_id'],x['calls']) for x in load(name)['results']['reduce_count_distribution'])
a=hist('sql-reduce-size-analysis.json');b=hist('sql-reduce-bucket-size-analysis.json')
assert a==b and sum(x[2] for x in a)==4504
base=r.parent/'sql-boundary-overlap-20261025';selected=json.loads((base/'selected-profile.json').read_text())
for name,digest in selected['files_sha256'].items():
 assert hashlib.sha256((base/name).read_bytes().replace(b'\r\n',b'\n')).hexdigest()==digest
assert selected['seconds_per_256']==candidate['reference_seconds_per_256']
assert candidate['time_reduction_percent']<1
old_profile=load('sql-boundary-overlap-profile.json');new_profile=load('sql-reduce-bucket-profile.json')
assert old_profile['results']['compute_total'][0]['calls']==new_profile['results']['compute_total'][0]['calls']==219114
for key in ['communication_top20']:
 assert sorted((x['op_type'],x['calls']) for x in old_profile['results'][key])==sorted((x['op_type'],x['calls']) for x in new_profile['results'][key])
def phases(rows):return {k:statistics.mean(rows[i]['phase_wall_seconds'][k] for i in [1,2]) for k in ['forward_wall','backward_wall','step_wall']}
v={'adopted':False,'kept_reference':'experiments/sql-boundary-overlap-20261025/selected-profile.json',
 'kept_seconds_per_256':candidate['reference_seconds_per_256'],'candidate_seconds_per_256':candidate['candidate_seconds_per_256'],
 'candidate_time_increase_percent':-candidate['time_reduction_percent'],
 'same_hccl_count_dtype_frequency_distribution_verified':True,'same_compute_and_collective_counts_verified':True,
 'unchanged_reference_source_and_configuration_hashes_verified':True,
 'reference_phase_wall_rank0':phases(reference['ranks'][0]['steps']),'candidate_phase_wall_rank0':phases(candidate['ranks'][0]['steps']),
 'reference_overlap':load('sql-boundary-overlap-communication-overlap.json')['summary'],
 'candidate_overlap':load('sql-reduce-bucket-communication-overlap.json')['summary'],
 'reason':'Earlier host submission did not improve measured whole-model time; reduction without compute increased in the matched profile.',
 'limits':'The unprofiled third-update forward regression is observed but not hardware-attributed by the fourth-update profile. No claim that all regression is causally explained.'}
(out/'sql-reduce-bucket-decision.json').write_text(json.dumps(v,indent=2));print(json.dumps(v,indent=2))
