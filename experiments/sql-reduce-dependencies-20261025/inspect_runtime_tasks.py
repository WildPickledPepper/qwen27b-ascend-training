"""Inspect raw host task timestamps and their parser definition."""
import json,sqlite3,shutil
from pathlib import Path
r=Path(__file__).resolve().parent;p=next((r/'raw-export.private').rglob('runtime.db'))
c=sqlite3.connect(f'file:{p}?mode=ro',uri=True);c.row_factory=sqlite3.Row
v={'samples':[dict(x) for x in c.execute('SELECT * FROM HostTask WHERE stream_id=466 ORDER BY timestamp LIMIT 5')],
 'kinds':[dict(x) for x in c.execute('SELECT task_type,COUNT(*) AS calls,MIN(timestamp) AS min_timestamp,MAX(timestamp) AS max_timestamp FROM HostTask GROUP BY task_type ORDER BY calls DESC LIMIT 20')]}
tool=Path(shutil.which('msprof')).resolve()
root=tool.parent.parent
files=[]
for pattern in ['*runtime*parser*.py','*runtime*model*.py','*task*parser*.py']:
 for q in root.rglob(pattern):
  text=q.read_text(errors='replace')
  if 'HostTask' in text or 'connection_id' in text and 'timestamp' in text:
   files.append(q);(r/(q.name+'.private.txt')).write_text(text)
v['parser_files']=[x.name for x in files]
(r/'runtime-inspection.json').write_text(json.dumps(v,indent=2));print(json.dumps(v,indent=2))
