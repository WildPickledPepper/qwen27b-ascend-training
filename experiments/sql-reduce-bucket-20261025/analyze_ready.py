"""Reuse validated skill SQL, then summarize the new deferred host range."""
import json,runpy
from pathlib import Path
r=Path(__file__).resolve().parent
ns=runpy.run_path(str(r/'analyze_gradient.py'),run_name='__main__')
ns['q']('boundary_finalize_host',"SELECT n.value AS name,COUNT(*) AS calls,SUM(CAST(p.endNs AS INTEGER)-CAST(p.startNs AS INTEGER))/1e9 AS host_seconds FROM PYTORCH_API p LEFT JOIN STRING_IDS n ON n.id=p.name WHERE n.value IN ('zero/deferred_boundary','zero/partition_gradients','zero/boundary_gradient_offload','zero/flush_oversized_ready') GROUP BY n.value ORDER BY host_seconds DESC LIMIT 20")
out=Path(ns['ns']['a'].output)
result=json.loads(out.read_text());result['queries']=ns['ns']['queries'];result['results']=ns['ns']['outputs']
out.write_text(json.dumps(result,indent=2))
print(json.dumps({'boundary_finalize_host':result['results']['boundary_finalize_host']}))
