"""Copy versioned CPU/meta dependencies into a fresh private run directory."""
import argparse,hashlib,json,shutil
from pathlib import Path
p=argparse.ArgumentParser()
p.add_argument('--destination',type=Path,required=True)
p.add_argument('--config',type=Path,required=True)
a=p.parse_args()
root=Path(__file__).resolve().parent
base=root.parent/'sql-memory-liveness-20261027'
a.destination.mkdir(parents=True,exist_ok=False)
files=[(root,name) for name in ['predict_meta.py','chunked_loss.py','test_chunked_loss.py','run_predictions.py']]
files += [(base,name) for name in ['storage_tracker.py','launch_stub.py','npu_linear_meta.py','community_gdn.py','inputs.json','backend-rule-sources.json']]
manifest={}
for directory,name in files:
    data=(directory/name).read_bytes().replace(b'\r\n',b'\n')
    (a.destination/name).write_bytes(data)
    manifest[name]=hashlib.sha256(data).hexdigest()
shutil.copyfile(a.config,a.destination/'config.json')
manifest['config.json']=hashlib.sha256((a.destination/'config.json').read_bytes()).hexdigest()
(a.destination/'source-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
