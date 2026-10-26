"""Compact archive progress, no source paths or data content."""
import json
from pathlib import Path
root=Path(__file__).resolve().parent;job=root/'job.private.json'
print(job.read_text() if job.exists() else 'not started')
for name in ['probe-card2','full-card2','full-card3']:
    file=root/(name+'.private')/'report.json'
    if not file.exists():continue
    r=json.loads(file.read_text());rows=r['rows'];raw=sum(x['raw_bf16_bytes'] for x in rows);stored=sum(x['stored_bytes'] for x in rows)
    print(json.dumps(dict(name=name,complete=r['complete'],frames=len(rows),raw_gib=raw/2**30,stored_frame_gib=stored/2**30,
        mismatches=sum(x['storage_kind']=='raw_author_mismatch' for x in rows),error=r.get('error'))))
    if name=='probe-card2':print(json.dumps([dict(name=x['name'],kind=x['storage_kind'],exact=x['author_roundtrip_bit_exact'],fraction=x['stored_bytes']/x['raw_bf16_bytes']) for x in rows]))
