"""Validate one stopped rank-0 profile and execute read-only skill CTEs."""
import argparse,json,sqlite3
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--output',required=True);a=p.parse_args()
root=Path(a.root)
folders=sorted(root.rglob('*_ascend_pt'))
assert folders,'No PyTorch framework profiler directory'
folder=folders[0]
infos=list(folder.glob('profiler_info*.json'))
assert infos,'Invalid profile: profiler stop metadata missing'
config=json.loads(infos[0].read_text())
capture_flags={}
def flags(value):
 if not isinstance(value,dict):return
 for key,item in value.items():
  if key in ['profiler_level','with_stack','with_modules','record_shapes','profile_memory','export_type']:
   capture_flags[key]=item
  elif isinstance(item,dict):flags(item)
flags(config)
dbs=list(folder.rglob('*pytorch*profiler*.db'))
assert dbs,'Unparsed/incomplete DB export: parse before analysis'
db=dbs[0];connection=sqlite3.connect(f'file:{db}?mode=ro',uri=True)
tables={x[0] for x in connection.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")}
queries={};outputs={}
def query(name,sql,params=()):
 queries[name]={'sql':sql,'parameters':list(params)}
 cursor=connection.execute(sql,params);cols=[x[0] for x in cursor.description]
 outputs[name]=[dict(zip(cols,row)) for row in cursor.fetchall()]
compute='''WITH compute_view AS (
    SELECT c.globalTaskId, ROUND(t.endNs - t.startNs) AS duration_ns, n.value AS op_name, type_str.value AS op_type
    FROM COMPUTE_TASK_INFO c
    LEFT JOIN TASK t ON t.globalTaskId = c.globalTaskId
    LEFT JOIN STRING_IDS n ON n.id = c.name
    LEFT JOIN STRING_IDS type_str ON type_str.id = c.opType
)
'''
comm='''WITH comm_view AS (
    SELECT ROUND(c.endNs - c.startNs) AS duration_ns, n.value AS op_name, t.value AS op_type, g.value AS group_name
    FROM COMMUNICATION_OP c
    LEFT JOIN STRING_IDS n ON n.id = c.opName
    LEFT JOIN STRING_IDS t ON t.id = c.opType
    LEFT JOIN STRING_IDS g ON g.id = c.groupName
)
'''
if {'COMPUTE_TASK_INFO','TASK','STRING_IDS'}<=tables:
 query('compute_top20',compute+'SELECT op_type, COUNT(*) AS calls, SUM(duration_ns)/1e9 AS summed_seconds FROM compute_view GROUP BY op_type ORDER BY summed_seconds DESC LIMIT 20')
 query('compute_total',compute+'SELECT COUNT(*) AS calls, SUM(duration_ns)/1e9 AS summed_seconds FROM compute_view')
 query('compute_interval_union',compute+''', intervals AS (
 SELECT t.startNs AS s,t.endNs AS e FROM compute_view c JOIN TASK t ON t.globalTaskId=c.globalTaskId
), ordered AS (
 SELECT s,e,MAX(e) OVER (ORDER BY s,e ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) AS previous_end FROM intervals
), marked AS (
 SELECT s,e,CASE WHEN previous_end IS NULL OR s>previous_end THEN 1 ELSE 0 END AS new_group FROM ordered
), grouped AS (
 SELECT s,e,SUM(new_group) OVER (ORDER BY s,e ROWS UNBOUNDED PRECEDING) AS group_id FROM marked
), merged AS (
 SELECT MIN(s) AS s,MAX(e) AS e FROM grouped GROUP BY group_id
)
 SELECT COUNT(*) AS compute_islands,SUM(e-s)/1e9 AS nonoverlapping_compute_seconds FROM merged''')
if {'COMMUNICATION_OP','STRING_IDS'}<=tables:
 query('communication_top20',comm+'SELECT op_type,COUNT(*) AS calls,SUM(duration_ns)/1e9 AS summed_seconds FROM comm_view GROUP BY op_type ORDER BY summed_seconds DESC LIMIT 20')
if 'TASK' in tables:
 query('device_window','SELECT COUNT(*) AS tasks,MIN(startNs) AS start_ns,MAX(endNs) AS end_ns,(MAX(endNs)-MIN(startNs))/1e9 AS window_seconds FROM TASK')
# The bundled get_schema.py has no PYTORCH_API documentation. This is the
# explicitly allowed schema fallback, not a guessed field/meaning.
if {'PYTORCH_API','STRING_IDS'}<=tables:
 columns=[x[1] for x in connection.execute('PRAGMA table_info(PYTORCH_API)')]
 if {'startNs','endNs','name'}<=set(columns):
  query('cpu_adam_ranges',"SELECT COUNT(*) AS calls,SUM(p.endNs-p.startNs)/1e9 AS summed_seconds FROM PYTORCH_API p LEFT JOIN STRING_IDS n ON n.id=p.name WHERE n.value=?",('optimizer/CPUAdam',))
  query('host_api_top20',"SELECT n.value AS name,COUNT(*) AS calls,SUM(p.endNs-p.startNs)/1e9 AS summed_seconds FROM PYTORCH_API p LEFT JOIN STRING_IDS n ON n.id=p.name GROUP BY n.value ORDER BY summed_seconds DESC LIMIT 20")
wall_check=None
rank_result=root.parent/'rank0-result.json'
if rank_result.exists() and outputs.get('device_window'):
 profiled=[s for s in json.loads(rank_result.read_text())['steps'] if s.get('profiled')]
 assert len(profiled)==1
 wall=profiled[0]['seconds'];window=outputs['device_window'][0]['window_seconds']
 wall_check={'recorded_step_seconds':wall,'device_window_seconds':window,'difference_seconds':window-wall}
 assert abs(window-wall)<max(1,wall*.1),'Device window inconsistent with recorded update'
result={'valid':True,'type':'PyTorch framework profiler','rank_sample':'rank0 only','db_basename':db.name,
        'normal_stop_metadata':True,'db_export_present':True,'tables':sorted(tables),
        'requested_capture_config':{'profiler_level':'Level1','record_shapes':False,'profile_memory':False,'with_stack':False,'export_type':'Db'},
        'capture_metadata_flags':capture_flags,
        'queries':queries,'results':outputs,
        'schema_reference':'Skill get_schema.py TASK; PYTORCH_API absent in reference, PRAGMA fallback used',
        'execution':'Read-only SQLite fallback; msprof_mcp tools not configured',
        'wall_window_check':wall_check,
        'capture_note':'Single active RECORD_AND_SAVE schedule; verify stop metadata, DB export and update time window',
        'limits':['Summed kernel/API/communication durations may overlap; do not add them as critical-path time',
                  'CPUAdam is a user record_function range around unmodified library math',
                  'Profiling run excluded from unprofiled throughput ranking']}
Path(a.output).write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
