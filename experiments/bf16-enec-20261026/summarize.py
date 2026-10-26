"""Filter archive size/integrity and NPU codec timings, no weights or host paths."""
import argparse,json,subprocess
from collections import defaultdict
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--output',required=True);a=p.parse_args()
root=Path(a.root);cards=[];names=set();parameters=0;modules=defaultdict(lambda:dict(raw_bytes=0,stored_frame_bytes=0))
for d in [2,3]:
    folder=root/f'full-card{d}.private';r=json.loads((folder/'report.json').read_text());v=json.loads((folder/'archive_verification.json').read_text())
    assert r['complete'] and v['passed'] and v['frames_verified']==v['total_frames']==len(r['rows'])
    assert not names.intersection(r['tensor_index']);names.update(r['tensor_index']);parameters+=r['parameters']
    for row in r['rows']:
        name=row['name'];module='FFN' if '.mlp.' in name else 'GDN' if '.linear_attn.' in name else 'attention' if '.self_attn.' in name else 'embedding_head' if name in ['model.embed_tokens.weight','lm_head.weight'] else 'other'
        modules[module]['raw_bytes']+=row['raw_bf16_bytes'];modules[module]['stored_frame_bytes']+=row['stored_bytes']
    cards.append(dict(device=d,parameters=r['parameters'],tensors=len(r['tensor_index']),aggregate=r['aggregate'],
        mapping_bytes=r['mapping_bytes'],archive_and_mapping_bytes=r['archive_and_mapping_bytes'],
        source_dtype_elements=r['source_dtype_elements'],source_index_sha256=r['source_index_sha256'],
        source_file_sizes_and_mtimes_unchanged=r['source_file_sizes_and_mtimes_unchanged'],
        all_frames_reloaded=v,elapsed_seconds=r['elapsed_seconds'],
        explicit_max_codec_device_bytes=max(x['explicit_codec_device_allocation_bytes'] for x in r['rows'])))
assert len(names)==851 and parameters==26895998464
raw=sum(c['aggregate']['raw_bf16_bytes'] for c in cards);stored=sum(c['archive_and_mapping_bytes'] for c in cards)
assert raw==parameters*2 and cards[0]['source_index_sha256']==cards[1]['source_index_sha256']
probe=json.loads((root/'probe-card2.private/report.json').read_text())
smi=subprocess.check_output(['npu-smi','info'],text=True)
assert all(f'No running processes found in NPU {d}' in smi for d in [2,3])
result=dict(scope='Offline actual BF16 computational-weight archive, complete text backbone only',
    codec=probe['build'],parameters=parameters,tensors=len(names),raw_bf16_bytes=raw,archive_and_mapping_bytes=stored,
    raw_gib=raw/2**30,stored_gib=stored/2**30,saved_gib=(raw-stored)/2**30,saved_percent=(1-stored/raw)*100,
    cards=cards,module_totals=dict(modules),diagnostics=[r for r in probe['rows'] if r['diagnostic']],
    source_checkpoint_read_only=True,model_training_started=False,original_bf16_dtype_retained=True,
    resource_release=dict(cards_2_3_no_processes=True),
    limits=['Source is the original checkpoint cast to the same BF16 computational weights as current training, not an unsaved trained state.',
        'Mapping/header/raw fallbacks counted; online decoder working set and training integration are not included.',
        'Offline harness holds source, encoded, scratch and decoded buffers together; not proof of online HBM savings.',
        'Event totals and offline roundtrip wall include different work and must not be added together.',
        'No claim of training speedup; unchanged BF16 matmul requires decoding before use.'])
Path(a.output).write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k not in ['diagnostics','codec','cards']}))
