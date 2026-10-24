"""Sequential descriptive timing after correctness, with seeded order and no overlapping tests."""
from pathlib import Path
import json,random,statistics,subprocess
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'formal_results';rows=[]
protocol={'seeds':[0,1,42,2026,2147483647],'steps':[1,2,4,8,16],'modes':[(0,128),(2,128),(1,128)],'repeats':7,'order_seed':3120,'stack_bytes':968,'scope':'Sequential CPU elapsed timing, includes initialization/capture/replay/update/digests, excludes cold raw-output proof pass and process launch; not CPU-affinity pinned, shared virtual host, descriptive only.'}
(OUT/'benchmark_protocol.json').write_text(json.dumps(protocol,indent=2))
configs=[(s,t,m,b) for s in protocol['seeds'] for t in protocol['steps'] for m,b in protocol['modes']];random.Random(3120).shuffle(configs)
for seed,t,mode,b in configs:
 args=[str(ROOT/'build/formal/factor_replay'),str(mode),str(seed),str(t),str(b),str(t),'968','0','-','7']
 run=subprocess.run(args,check=True,text=True,capture_output=True);d=json.loads(run.stdout);assert d['status']=='complete';d['command']=args;rows.append(d)
 with (OUT/'benchmark_raw.jsonl').open('a') as fp:fp.write(json.dumps(d)+'\n')
(OUT/'benchmark_raw.json').write_text(json.dumps(rows,indent=2));summary=[]
for t in protocol['steps']:
 for m,b in protocol['modes']:
  q=[x for x in rows if x['requested_steps']==t and x['mode']==m]
  ratios=[x['nanoseconds_mean']/next(z['nanoseconds_mean'] for z in rows if z['requested_steps']==t and z['mode']==0 and z['seed']==x['seed']) for x in q]
  summary.append(dict(steps=t,mode=m,block=b,mean_ms_median=statistics.median(x['nanoseconds_mean']/1e6 for x in q),time_ratio_median=statistics.median(ratios),time_ratio_min=min(ratios),time_ratio_max=max(ratios),optimizer_bytes=sorted(set(x['optimizer_replacement_bytes'] for x in q)),algorithm_bytes=sorted(set(x['charged_algorithm_bytes'] for x in q))))
(OUT/'benchmark_summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
