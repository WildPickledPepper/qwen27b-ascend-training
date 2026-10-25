"""Compare raw runtime task stamps with their own launch API clock domain."""
import json,sqlite3
from pathlib import Path
r=Path(__file__).resolve().parent;root=r/'raw-export.private'
c=sqlite3.connect(f"file:{next(root.rglob('runtime.db'))}?mode=ro",uri=True);c.row_factory=sqlite3.Row
c.execute('ATTACH DATABASE ? AS api',(f"file:{next(root.rglob('api_event.db'))}?mode=ro",))
sql="""SELECT h.task_type,COUNT(*) AS matches,SUM(CASE WHEN h.timestamp BETWEEN a.start AND a.end THEN 1 ELSE 0 END) AS task_stamp_inside_launch_api,SUM(CASE WHEN h.timestamp>a.end THEN 1 ELSE 0 END) AS stamp_after_api FROM HostTask h JOIN api.ApiData a ON h.connection_id=a.connection_id WHERE h.stream_id=466 AND a.struct_type='launch' GROUP BY h.task_type ORDER BY matches DESC LIMIT 20"""
v={'query':sql,'results':[dict(x) for x in c.execute(sql)],'limits':'Raw timestamps are compared in their own CPU counter domain, not guessed as nanoseconds.'}
(r/'runtime-clock-verification.json').write_text(json.dumps(v,indent=2));print(json.dumps(v,indent=2))
