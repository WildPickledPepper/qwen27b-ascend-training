"""Isolated derivative of the verified 2GiB pipeline; no original files changed."""
import hashlib,json,shutil
from pathlib import Path
root=Path(__file__).resolve().parent;base=root.parent/'sql-pipeline-cache-20261026'
source=(base/'train_cache.py').read_text()
marker='Offload=create_offload_class(Offload,params,rank,device,a,scope)'
assert source.count(marker)==1
source=source.replace(marker,"from async_offload import make_async_base\nOffload=make_async_base(Offload,params,rank,device,a,scope)\n"+marker)
marker='resident_add_calls=off.resident_add_calls,resident_add_bytes=off.resident_add_bytes,'
assert source.count(marker)==1
source=source.replace(marker,marker+'\n             offload_parameter_stats=off.parameter_stats,')
(root/'train_overlap.py').write_text(source)
shutil.copy2(base/'train_cache.py',root/'train_resident.py')
for name in ['resident_gradients.py','community_gdn.py','compare_states.py','verify_smoke.py']:
    shutil.copy2(base/name,root/name)
manifest=dict(base_lf_sha256=hashlib.sha256((base/'train_cache.py').read_bytes().replace(b'\r\n',b'\n')).hexdigest(),
    candidate_lf_sha256=hashlib.sha256(source.replace('\r\n','\n').encode()).hexdigest())
(root/'source-manifest.json').write_text(json.dumps(manifest,indent=2));print(json.dumps(manifest))
