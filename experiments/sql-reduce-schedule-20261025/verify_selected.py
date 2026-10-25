import hashlib,json,sqlite3,subprocess,time
from pathlib import Path
r=Path(__file__).resolve().parent
ranks=[json.loads((r/'cpu-full'/f'rank{i}-result.json').read_text()) for i in [0,1]]
assert all(x['completed'] for x in ranks)
reference=json.loads((r.parent/'qwen-sql-bottleneck-20261025/profile-g256-m16/rank0-result.json').read_text())
assert all(x['config']==reference['config'] for x in ranks)
selected=json.loads((r/'selected-profile.json').read_text())
for name,digest in selected['files_sha256'].items():
 assert hashlib.sha256((r/name).read_bytes().replace(b'\r\n',b'\n')).hexdigest()==digest,name
assert json.loads((r/'zero3-selected.json').read_text())==ranks[0]['config']
root=r/'cpu-full/profile';assert list(root.rglob('profiler_info_0.json'))
db=next(root.rglob('ascend_pytorch_profiler_0.db'));c=sqlite3.connect(f'file:{db}?mode=ro',uri=True)
sql="SELECT n.value AS name,COUNT(*) AS calls,SUM(c.endNs-c.startNs)/1e9 AS host_seconds FROM CANN_API c LEFT JOIN STRING_IDS n ON n.id=c.name WHERE n.value IN ('aclrtMallocPhysical','aclrtFreePhysical') GROUP BY n.value ORDER BY host_seconds DESC LIMIT 20"
memory=[dict(zip(['name','calls','host_seconds'],x)) for x in c.execute(sql)]
probe=json.loads((r/'probe-comparison.json').read_text());assert probe['passed']
info=subprocess.check_output(['npu-smi','info'],text=True)
released=all(f'No running processes found in NPU {i}' in info for i in [2,3]);assert released
states={x.name:json.loads((x/'job.json').read_text())['stage'] for x in [r,r.parent/'qwen-sql-layer-gather-20261025',r.parent/'qwen-sql-gradient-pipeline-20261025',r.parent/'qwen-sql-gradient-cpu-20261025',r.parent/'qwen-sql-gradient-cache-20261025',r.parent/'qwen-sql-ffn-gather-20261025']}
assert all(x=='complete' for k,x in states.items() if k!='qwen-sql-ffn-gather-20261025')
assert states['qwen-sql-ffn-gather-20261025']=='cancelled_before_gpu_use'
v={'passed':True,'unix':time.time(),'both_ranks_complete':True,'same_deepspeed_config_as_native_reference':True,'selected_source_and_config_hashes_match':True,
 'full_tiny_gradient_master_m_v_bitwise_comparison':True,'normal_stop_metadata_present':True,'parsed_db_present':True,
 'physical_memory_api_query':sql,'physical_memory_api_rows':memory,'physical_devices_2_3_released':released,
 'other_known_service_still_present':'1424210' in info,'experiment_states':states,
 'limits':'Short validation and tiny bitwise comparison do not establish full-model long-run bitwise identity or convergence.'}
(r/'selected-verification.json').write_text(json.dumps(v,indent=2));print(json.dumps(v))
