"""Filtered original-kernel probe, explicitly separating raw fallback from success."""
import argparse,json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--output',required=True);a=p.parse_args()
r=json.loads((Path(a.root)/'probe-card2.private/report.json').read_text());assert r['complete'] and r['close_status']==0
out=dict(build=r['build'],dtype=r['dtype'],stored_format_lossless=True,
    rows=[{k:v for k,v in row.items() if k in ['name','raw_bf16_bytes','stored_bytes','author_encoded_bytes','storage_kind',
        'author_roundtrip_bit_exact','stored_lossless','diagnostic','compress_event_ms','merge_event_ms','decompress_event_ms']} for row in r['rows']],
    limits=['Author codec fails some exact bit-pattern tests; raw fallback is required.',
        'Samples are a gate, not an estimate of full-model compression or online speed.'])
Path(a.output).write_text(json.dumps(out,indent=2));print('Filtered probe exported')
