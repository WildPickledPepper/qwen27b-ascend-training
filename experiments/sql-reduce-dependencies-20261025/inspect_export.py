"""Inspect newly parsed raw schemas and task CSV fields; no NPU execution."""
import csv,json,sqlite3
from pathlib import Path
r=Path(__file__).resolve().parent;root=r/'raw-export.private'
schemas={}
for pattern in ['runtime.db','ascend_task.db','msprof_*.db','rts_track.db','api_event.db']:
 for p in root.rglob(pattern):
  c=sqlite3.connect(f'file:{p}?mode=ro',uri=True)
  schemas[p.name]=[x[0] for x in c.execute("SELECT sql FROM sqlite_master WHERE type='table' ORDER BY name")];c.close()
csvdata={}
for p in root.rglob('task_time_*.csv'):
 with p.open(newline='') as f:
  reader=csv.DictReader(f);csvdata={'columns':reader.fieldnames,'sample':next(reader)}
v={'schemas':schemas,'task_time':csvdata}
(r/'export-schema.private.json').write_text(json.dumps(v,indent=2));print(json.dumps(v,indent=2))
