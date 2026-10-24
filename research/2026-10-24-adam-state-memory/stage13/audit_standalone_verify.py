"""Verify archived-frame fallback in an isolated copy with no Stage5/Stage10 siblings.
This reruns only frame validation, never training or the compression grid.
"""
from pathlib import Path
from collections import Counter
import json,os,shutil,subprocess,sys,tempfile
ROOT=Path(__file__).resolve().parent
reference=json.loads((ROOT/'audit_frame_results.json').read_text())
with tempfile.TemporaryDirectory(prefix='stage13_standalone_audit_') as temp:
    copy=Path(temp)/'stage13';copy.mkdir()
    for name in ['independent_verify.py','independent_frame_verify.py','residual_codec.py']:
        shutil.copy2(ROOT/name,copy/name)
    for name in ['inputs','provenance','results']:
        shutil.copytree(ROOT/name,copy/name)
    assert not (copy.parent/'stage5').exists()
    assert not (copy.parent/'stage10').exists()
    env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
    env.pop('PYTHONPATH',None)
    command=[sys.executable,str(copy/'independent_frame_verify.py')]
    p=subprocess.run(command,cwd=copy,env=env,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=60)
    log='Standalone frame validation in a temporary copy. No Stage5 or Stage10 siblings exist.\n'
    log+='Only independent_frame_verify.py ran; no training or grid rerun.\n'
    log+=f'Return code: {p.returncode}\n\n'+p.stdout
    (ROOT/'audit_standalone_verification.log').write_text(log)
    assert p.returncode==0
    result=json.loads((copy/'audit_frame_results.json').read_text())
    routes=dict(Counter(x['route'] for x in result['source_evidence']))
    assert routes=={'archived_stage5_index':9,'archived_stage10_trajectory':3,'archived_control_input':4}
    assert result['all_independent_raw_roundtrips_and_accounting_pass']
    assert result['verified_frames']==reference['verified_frames']
    assert result['case_best']==reference['case_best']
    assert result['control_source_checks']==reference['control_source_checks']
    summary={'standalone_frame_audit_pass':True,'frames_checked':result['frames_checked'],'decoded_bytes':result['total_decoded_raw_bytes'],'no_stage5_or_stage10_sibling_directories':True,'evidence_routes':routes,'all_frame_hashes_decode_and_accounting_match_original_audit':True,'training_or_compression_grid_rerun':False,'return_code':p.returncode,'provenance_limit':result['source_evidence_limit'],'source_evidence':result['source_evidence']}
    (ROOT/'audit_standalone_results.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps({k:v for k,v in summary.items() if k!='source_evidence'},indent=2))
