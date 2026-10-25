"""Read-only scoped host API diagnostics, using documented ns timestamps."""
import json,sqlite3
from pathlib import Path
r=Path(__file__).resolve().parent
db=next((r/'cpu-full/profile').rglob('ascend_pytorch_profiler_0.db'))
c=sqlite3.connect(f'file:{db}?mode=ro',uri=True);c.row_factory=sqlite3.Row
# PYTORCH_API is absent from the bundled schema reference; allowed fallback.
columns=[x['name'] for x in c.execute('PRAGMA table_info(PYTORCH_API)')]
assert {'name','startNs','endNs'}<=set(columns)
scopes="""WITH scopes AS (
 SELECT n.value AS scope,CAST(p.startNs AS INTEGER) AS s,CAST(p.endNs AS INTEGER) AS e
 FROM PYTORCH_API p JOIN STRING_IDS n ON n.id=p.name
 WHERE n.value LIKE 'diagnosis/%/3/%'
)
"""
queries={
 'scopes':scopes+'SELECT scope,COUNT(*) AS calls,SUM(e-s)/1e9 AS host_seconds FROM scopes GROUP BY scope ORDER BY scope LIMIT 20',
 'first_engine_cann':scopes+"SELECT n.value AS name,COUNT(*) AS calls,SUM(MIN(a.endNs,s.e)-MAX(a.startNs,s.s))/1e9 AS clipped_host_seconds,MAX(a.endNs-a.startNs)/1e9 AS longest_full_call_seconds FROM scopes s JOIN CANN_API a ON a.startNs<s.e AND a.endNs>s.s JOIN STRING_IDS n ON n.id=a.name WHERE s.scope='diagnosis/engine_forward/3/1' GROUP BY n.value ORDER BY clipped_host_seconds DESC LIMIT 20",
 'allocator_calls':"SELECT n.value AS name,COUNT(*) AS calls,SUM(a.endNs-a.startNs)/1e9 AS host_seconds,MAX(a.endNs-a.startNs)/1e9 AS longest_seconds FROM CANN_API a JOIN STRING_IDS n ON n.id=a.name WHERE n.value LIKE '%Malloc%' OR n.value LIKE '%Free%' OR n.value LIKE '%MemMap%' OR n.value LIKE '%MemUnmap%' GROUP BY n.value ORDER BY host_seconds DESC LIMIT 20",
 'first_engine_pytorch':scopes+"SELECT n.value AS name,COUNT(*) AS calls,SUM(MIN(CAST(a.endNs AS INTEGER),s.e)-MAX(CAST(a.startNs AS INTEGER),s.s))/1e9 AS clipped_inclusive_host_seconds FROM scopes s JOIN PYTORCH_API a ON CAST(a.startNs AS INTEGER)<s.e AND CAST(a.endNs AS INTEGER)>s.s JOIN STRING_IDS n ON n.id=a.name WHERE s.scope='diagnosis/engine_forward/3/1' AND n.value NOT LIKE 'diagnosis/%' GROUP BY n.value ORDER BY clipped_inclusive_host_seconds DESC LIMIT 20"
}
result={'read_only':True,'schema_reference':'CANN_API official schema; PYTORCH_API absent, PRAGMA fallback','queries':queries,'results':{name:[dict(x) for x in c.execute(sql)] for name,sql in queries.items()},'limits':['Temporal API overlap is not a proven task dependency mapping.','Nested inclusive PyTorch/API ranges may overlap; do not add them as elapsed time.','This targeted run has Level1 plus first-micro cProfile overhead and is not a throughput ranking.']}
(r/'forward-analysis.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
