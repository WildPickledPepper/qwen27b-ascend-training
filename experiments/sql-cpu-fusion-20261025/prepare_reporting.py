"""Derive evidence collection for native-state CPU fusion."""
from pathlib import Path
r=Path(__file__).resolve().parent;b=r.parent/'sql-boundary-overlap-20261025'
s=(b/'collect_result.py').read_text().replace('sql-reduce-schedule-results.json','sql-boundary-overlap-results.json').replace("prior['scheduled_seconds_per_256']","prior['candidate_seconds_per_256']")
s=s.replace("(out/'sql-boundary-overlap-results.json').write_text", "(out/'sql-cpu-fusion-results.json').write_text")
s=s.replace('selected 101.28-second reference', 'selected 95.74-second reference')
s=s.replace("assert result['completed']", "assert result['cpu_fusion']['native_adam_simd_math_reused']\n assert result['completed']")
s=s.replace("assert a['cpu_gradient_accumulation_stats']['deferred_boundary_parameters']==851", "assert a['cpu_gradient_accumulation_stats']['deferred_boundary_parameters']==851\n  assert a['cpu_update_stats']['fused_scale_adam_bf16_calls']==44\n  assert a['cpu_update_stats']['async_copy_bytes']==26895998464")
s=s.replace("'cpu_accumulation_plan':x['cpu_gradient_accumulation']", "'cpu_accumulation_plan':x['cpu_gradient_accumulation'],'cpu_fusion':x['cpu_fusion']")
(r/'collect_result.py').write_text(s,encoding='utf-8',newline='\n')
s=(b/'postprocess.py').read_text().replace('qwen-sql-reduce-schedule-20261025','qwen-sql-boundary-overlap-20261025')
s=s.replace("assert result['completed']", "assert result['cpu_fusion']['native_adam_simd_math_reused']\n assert result['completed']")
s=s.replace("assert row['fp32_m_v_bytes']==107583993856", "assert row['fp32_m_v_bytes']==107583993856\n  assert row['cpu_update_stats']['fused_scale_adam_bf16_calls']==44\n  assert row['cpu_update_stats']['async_copy_bytes']==26895998464")
start=s.index(" 'guard_failure':");end=s.index(" 'limits':",start)
s=s[:start]+" 'original_cpu_probe_passed':json.loads((r/'cpu-probe.json').read_text())['passed'],\n 'fused_optimizer_calls_44_all_full_steps':True,\n"+s[end:]
(r/'postprocess.py').write_text(s,encoding='utf-8',newline='\n')
s=(b/'create_selected.py').read_text().replace('sql-boundary-overlap-results.json','sql-cpu-fusion-results.json').replace("'sql-reduce-schedule-20261025'","'sql-boundary-overlap-20261025'")
s=s.replace("e['time_reduction_percent']>0", "e['time_reduction_percent']>=1")
s=s.replace("name='sql-fixed256-deferred-boundary'", "name='sql-fixed256-cpu-fused-update'")
s=s.replace("names=['train_boundary.py' if n=='train_reduce.py' else n for n in v['files_sha256']]", "names=['train_fused.py' if n=='train_boundary.py' else 'cpu_update_reference.py' if n=='cpu_update.py' else n for n in v['files_sha256']]+['cpu_update_fused.py','build_fused.py','cpu-probe.json','build-info.json']\nv['cpu_fusion']=raw['cpu_fusion']")
old="(r/'run_selected.py').write_text((previous/'run_selected.py').read_text().replace('train_reduce.py','train_boundary.py'),encoding='utf-8',newline='\\n')"
assert old in s
new="""script=(previous/'run_selected.py').read_text().replace('train_boundary.py','train_fused.py').replace("'--cpu-gradient-accum']","'--cpu-gradient-accum','--fused-cpu-update']")
anchor="assert os.environ.get('ASCEND_RT_VISIBLE_DEVICES')"
script=script.replace(anchor,"binary=r/'build.private/cpu_adam_fused.so'\\nassert binary.exists() and hashlib.sha256(binary.read_bytes()).hexdigest()==selected['cpu_fusion']['extension_sha256'],'Build/revalidate the tested CPU extension before running'\\n"+anchor)
(r/'run_selected.py').write_text(script,encoding='utf-8',newline='\\n')"""
s=s.replace(old,new)
(r/'create_selected.py').write_text(s,encoding='utf-8',newline='\n')
print('Prepared source/config/state audits and guarded selected replay generation')
