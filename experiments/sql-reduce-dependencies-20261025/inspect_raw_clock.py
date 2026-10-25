import json,sqlite3
from pathlib import Path
r=Path(__file__).resolve().parent;root=r/'raw-export.private'
p=next(root.rglob('api_event.db'));c=sqlite3.connect(f'file:{p}?mode=ro',uri=True);c.row_factory=sqlite3.Row
v={'api_sample':[dict(x) for x in c.execute('SELECT * FROM ApiData WHERE connection_id=957163 ORDER BY start LIMIT 5')]}
times=[]
for p in root.rglob('time.db'):
 c=sqlite3.connect(f'file:{p}?mode=ro',uri=True);c.row_factory=sqlite3.Row
 for (name,) in c.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"):
  times.append({'table':name,'rows':[dict(x) for x in c.execute('SELECT * FROM "'+name+'" ORDER BY 1 LIMIT 2')]})
v['time_metadata']=times
(r/'raw-clock.private.json').write_text(json.dumps(v,indent=2));print(json.dumps(v,indent=2))
