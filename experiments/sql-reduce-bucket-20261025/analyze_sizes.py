"""Read-only communication payload statistics using documented HCCL tables."""
import argparse,json,sqlite3
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--output',required=True);a=p.parse_args()
r=Path(a.root);assert list(r.rglob('profiler_info_0.json'))
db=next(r.rglob('ascend_pytorch_profiler_0.db'));c=sqlite3.connect(f'file:{db}?mode=ro',uri=True)
queries={};results={}
def q(name,sql):
 queries[name]=sql;cur=c.execute(sql);cols=[x[0] for x in cur.description];results[name]=[dict(zip(cols,x)) for x in cur.fetchall()]
comm='''WITH comm_view AS (
    SELECT ROUND(c.endNs - c.startNs) AS duration_ns, n.value AS op_name, t.value AS op_type, g.value AS group_name
    FROM COMMUNICATION_OP c
    LEFT JOIN STRING_IDS n ON n.id = c.opName
    LEFT JOIN STRING_IDS t ON t.id = c.opType
    LEFT JOIN STRING_IDS g ON g.id = c.groupName
)
'''
q('reduce_duration_histogram',comm+"SELECT CASE WHEN duration_ns<100000 THEN '<100us' WHEN duration_ns<1000000 THEN '100us-1ms' WHEN duration_ns<10000000 THEN '1-10ms' ELSE '>=10ms' END AS duration_class,COUNT(*) AS calls,SUM(duration_ns)/1e9 AS summed_seconds FROM comm_view WHERE op_type='hcom_reduceScatter_' GROUP BY duration_class ORDER BY summed_seconds DESC LIMIT 20")
# Track B: count/size/task metadata are outside the communication macro.
# Fields checked against the skill's official COMMUNICATION_OP/TASK_INFO reference.
q('reduce_count_distribution',"SELECT c.count AS hccl_count,c.dataType AS dtype_id,COUNT(*) AS calls,SUM(c.endNs-c.startNs)/1e9 AS summed_seconds,AVG(c.endNs-c.startNs)/1e6 AS mean_ms FROM COMMUNICATION_OP c JOIN STRING_IDS n ON n.id=c.opType WHERE n.value='hcom_reduceScatter_' GROUP BY c.count,c.dataType ORDER BY summed_seconds DESC LIMIT 20")
q('reduce_count_classes',"SELECT CASE WHEN c.count<262144 THEN '<256K elements' WHEN c.count<4194304 THEN '256K-4M' WHEN c.count<16777216 THEN '4M-16M' ELSE '>=16M' END AS count_class,COUNT(*) AS calls,SUM(c.endNs-c.startNs)/1e9 AS summed_seconds FROM COMMUNICATION_OP c JOIN STRING_IDS n ON n.id=c.opType WHERE n.value='hcom_reduceScatter_' GROUP BY count_class ORDER BY summed_seconds DESC LIMIT 20")
q('reduce_transport_tasks',"SELECT t.value AS task_type,COUNT(*) AS calls,SUM(k.size) AS task_size_sum_bytes,SUM(s.endNs-s.startNs)/1e9 AS task_summed_seconds FROM COMMUNICATION_TASK_INFO k JOIN COMMUNICATION_OP o ON o.opId=k.opId JOIN STRING_IDS n ON n.id=o.opType JOIN STRING_IDS t ON t.id=k.taskType LEFT JOIN TASK s ON s.globalTaskId=k.globalTaskId WHERE n.value='hcom_reduceScatter_' GROUP BY t.value ORDER BY task_summed_seconds DESC LIMIT 20")
v={'rank':'rank0','schema_reference':'Skill get_schema.py COMMUNICATION_OP and COMMUNICATION_TASK_INFO; original comm CTE retained','queries':queries,'results':results,'limits':['Communication ranges include waits. HCCL count is not automatically the full input-gradient size. Transport-task size sums include multiple tasks/stages and are not unique physical-link bytes.']}
Path(a.output).write_text(json.dumps(v,indent=2));print(json.dumps(results,indent=2))
