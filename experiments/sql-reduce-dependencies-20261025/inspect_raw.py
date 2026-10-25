"""List only our captured profiler artifacts and source metadata."""
import json,sqlite3
from pathlib import Path
r=Path(__file__).resolve().parent;base=r.parent/'qwen-sql-boundary-overlap-20261025/cpu-full/profile'
items=[]
for p in base.rglob('*'):
 if p.is_file() and (p.suffix=='.db' or 'task' in p.name.lower() or 'event' in p.name.lower()):
  item={'relative':str(p.relative_to(base)),'bytes':p.stat().st_size}
  if p.suffix=='.db':
   c=sqlite3.connect(f'file:{p}?mode=ro',uri=True)
   item['tables']=[x[0] for x in c.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")];c.close()
  items.append(item)
(r/'raw-inventory.private.json').write_text(json.dumps(items,indent=2))
print(json.dumps([{'name':Path(x['relative']).name,'bytes':x['bytes'],'tables':x.get('tables')} for x in items],indent=2))
