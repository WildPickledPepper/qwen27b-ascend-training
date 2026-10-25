import sqlite3,json
from pathlib import Path
root=Path(__file__).resolve().parent
db=next((root/'profile-g256-m16/profile').rglob('ascend_pytorch_profiler_0.db'))
c=sqlite3.connect(f'file:{db}?mode=ro',uri=True)
for table in ['TASK','COMMUNICATION_OP','PYTORCH_API']:
 print(table,c.execute(f'SELECT MIN(startNs),MAX(endNs),(MAX(endNs)-MIN(startNs))/1e9 FROM {table}').fetchone())
print('host',c.execute("SELECT n.value,MIN(p.startNs),MAX(p.endNs) FROM PYTORCH_API p JOIN STRING_IDS n ON p.name=n.id WHERE n.value IN ('optimizer/CPUAdam','ProfilerStep#0') GROUP BY n.value ORDER BY MIN(p.startNs) LIMIT 20").fetchall())
