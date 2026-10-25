"""Report CPU-local savings separately from fixed-work whole-model speed."""
import hashlib,json,statistics
from pathlib import Path
r=Path(__file__).resolve().parent;out=r.parents[1]/'evidence/2026-10-25'
load=lambda name:json.loads((out/name).read_text())
reference=load('sql-boundary-overlap-results.json');candidate=load('sql-cpu-fusion-results.json')
assert candidate['validation_loss_difference']==0 and load('sql-cpu-fusion-verification.json')['passed']
base=r.parent/'sql-boundary-overlap-20261025';selected=json.loads((base/'selected-profile.json').read_text())
for name,digest in selected['files_sha256'].items():
 assert hashlib.sha256((base/name).read_bytes().replace(b'\r\n',b'\n')).hexdigest()==digest
assert selected['seconds_per_256']==candidate['reference_seconds_per_256']
old=load('sql-boundary-overlap-profile.json');new=load('sql-cpu-fusion-profile.json')
assert old['results']['compute_total'][0]['calls']==new['results']['compute_total'][0]['calls']==219114
assert sorted((x['op_type'],x['calls']) for x in old['results']['communication_top20'])==sorted((x['op_type'],x['calls']) for x in new['results']['communication_top20'])
def phases(rows):return {k:statistics.mean(rows[i]['phase_wall_seconds'][k] for i in [1,2]) for k in ['forward_wall','backward_wall','step_wall']}
a=phases(reference['ranks'][0]['steps']);b=phases(candidate['ranks'][0]['steps'])
v={'adopted':candidate['time_reduction_percent']>=1,
 'reference':'experiments/sql-boundary-overlap-20261025/selected-profile.json',
 'reference_seconds_per_256':candidate['reference_seconds_per_256'],
 'candidate_seconds_per_256':candidate['candidate_seconds_per_256'],
 'candidate_time_reduction_percent':candidate['time_reduction_percent'],
 'promotion_threshold_percent':1,'unchanged_reference_source_and_config_hashes_verified':True,
 'same_compute_and_collective_counts_verified':True,
 'reference_phase_wall_rank0':a,'candidate_phase_wall_rank0':b,
 'cpu_step_seconds_saved_rank0':a['step_wall']-b['step_wall'],
 'cpu_step_time_reduction_percent_rank0':100*(1-b['step_wall']/a['step_wall']),
 'reference_overlap':load('sql-boundary-overlap-communication-overlap.json')['summary'],
 'candidate_overlap':load('sql-cpu-fusion-communication-overlap.json')['summary'],
 'limits':['The fourth profiling update does not identify the cause of the unprofiled third-update forward increase.',
 'The fused CPUAdam range also includes scaling and BF16 output; compare whole step, not native CPUAdam range alone.',
 'Two post-warmup updates do not establish a statistically significant whole-model speed improvement.']}
(out/'sql-cpu-fusion-decision.json').write_text(json.dumps(v,indent=2));print(json.dumps(v,indent=2))
