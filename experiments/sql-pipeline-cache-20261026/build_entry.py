"""Derive candidate from the committed matched pipeline, keeping it unchanged."""
import hashlib,json,shutil
from pathlib import Path
root=Path(__file__).resolve().parent
base=root.parent/'sql-pipeline-20261026'
source=(base/'train_pipeline.py').read_text()
expected='p.add_argument(\'--slots\', type=int, default=16)'
assert source.count(expected)==1
source=source.replace(expected,expected+"\np.add_argument('--resident-gib',type=float,default=0.)")
marker="spec=importlib.util.spec_from_file_location('cpu_adam',os.environ['CPU_ADAM_EXTENSION'])"
assert source.count(marker)==1
source=source.replace(marker,"from resident_gradients import create_offload_class\nOffload=create_offload_class(Offload,params,rank,device,a,scope)\n\n"+marker)
source=source.replace('off=Offload()',"off=Offload()\nlog('resident_plan',**off.resident_plan)")
marker="with scope('pipeline/gradient_drain'): off.drain()"
assert source.count(marker)==1
source=source.replace(marker,"with scope('pipeline/gradient_drain'): off.finish()")
marker='worker_copy_wait_seconds=off.copy_wait,worker_cpu_add_seconds=off.cpu_add,'
assert source.count(marker)==1
source=source.replace(marker,marker+'\n             resident_add_calls=off.resident_add_calls,resident_add_bytes=off.resident_add_bytes,')
(root/'train_cache.py').write_text(source)
shutil.copy2(base/'community_gdn.py' if (base/'community_gdn.py').exists() else root.parent/'sql-boundary-overlap-20261025'/'community_gdn.py',root/'community_gdn.py')
for name in ['verify_smoke.py','analyze_profile.py']:
    shutil.copy2(base/name,root/name)
manifest=dict(base_lf_sha256=hashlib.sha256((base/'train_pipeline.py').read_bytes().replace(b'\r\n',b'\n')).hexdigest(),
              candidate_lf_sha256=hashlib.sha256(source.replace('\r\n','\n').encode()).hexdigest())
(root/'source-manifest.json').write_text(json.dumps(manifest,indent=2));print(json.dumps(manifest))
