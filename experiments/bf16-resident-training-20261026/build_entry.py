"""Derive isolated pipeline with native Linear paging; preserve original math."""
import hashlib,json,shutil
from pathlib import Path
root=Path(__file__).resolve().parent;base=root.parent/'sql-pipeline-overlap-20261026'
s=(base/'train_resident.py').read_text()
marker="p.add_argument('--resident-gib',type=float,default=0.)";assert s.count(marker)==1
s=s.replace(marker,marker+"\np.add_argument('--archive-root')\np.add_argument('--codec-library',required=True)")
marker='off=Offload()';assert s.count(marker)==1
s=s.replace(marker,"from compressed_ffn import CompressedFFN\nstore=CompressedFFN(model,params,manifest,device,a.codec_library,\n    str(Path(a.archive_root)/f'full-card{rank+2}.private') if not a.tiny else None)\nlog('compressed_store',**store.initial_audit)\nsaved_weight_context=torch.autograd.graph.saved_tensors_hooks(store.pack,store.unpack)\nsaved_weight_context.__enter__()\n"+marker)
marker="with scope('pipeline/gradient_zero'): off.reset()";assert s.count(marker)==1
s=s.replace(marker,marker+"\n    store.reset_stats()")
marker='params[idx].detach().view(-1)[start:start+n].copy_(staging[slot][:n],non_blocking=True)';assert s.count(marker)==1
s=s.replace(marker,"if idx in store.records:store.update_chunk(idx,start,staging[slot][:n])\n            else:"+marker)
marker='resident_add_calls=off.resident_add_calls,resident_add_bytes=off.resident_add_bytes,';assert s.count(marker)==1
s=s.replace(marker,marker+'\n             compressed_weight_stats=store.stats(),')
s=s.replace('off.jobs.put(None); off.jobs.join(); off.thread.join(); dist.destroy_process_group()',
    'saved_weight_context.__exit__(None,None,None)\noff.jobs.put(None); off.jobs.join(); off.thread.join(); dist.destroy_process_group()')
(root/'train.py').write_text(s)
for name in ['community_gdn.py','verify_smoke.py','compare_states.py']:
    shutil.copy2(base/name,root/name)
s=(base/'resident_gradients.py').read_text().replace('budget<=4*2**30','budget<=8*2**30')
(root/'resident_gradients.py').write_text(s)
(root/'source-manifest.json').write_text(json.dumps(dict(base_lf_sha256=hashlib.sha256((base/'train_resident.py').read_bytes().replace(b'\r\n',b'\n')).hexdigest(),
    candidate_lf_sha256=hashlib.sha256((root/'train.py').read_bytes().replace(b'\r\n',b'\n')).hexdigest()),indent=2))
