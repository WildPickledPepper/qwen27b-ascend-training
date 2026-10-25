"""Verify parent runtime stream of HCCL task groups against device substreams."""
import json,sqlite3
from pathlib import Path
r=Path(__file__).resolve().parent;root=r/'raw-export.private'
db=next(root.rglob('msprof_*.db'));runtime=next(root.rglob('runtime.db'))
c=sqlite3.connect(f'file:{db}?mode=ro',uri=True);c.row_factory=sqlite3.Row
c.execute('ATTACH DATABASE ? AS runtime',(f'file:{runtime}?mode=ro',))
sql="""SELECT n.value AS collective,h.stream_id AS runtime_parent_stream,h.task_type AS parent_task_type,COUNT(*) AS matches,COUNT(DISTINCT o.opId) AS distinct_ops FROM COMMUNICATION_OP o JOIN STRING_IDS n ON n.id=o.opType JOIN runtime.HostTask h ON h.connection_id=o.connectionId GROUP BY n.value,h.stream_id,h.task_type ORDER BY matches DESC LIMIT 20"""
v={'query':sql,'results':[dict(x) for x in c.execute(sql)]}
(r/'parent-streams.json').write_text(json.dumps(v,indent=2));print(json.dumps(v,indent=2))
