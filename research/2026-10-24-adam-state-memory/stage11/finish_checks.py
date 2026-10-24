from pathlib import Path
import hashlib,json,subprocess,os
import numpy as np
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'results';rows=json.loads((OUT/'frontier.json').read_text())
checks=[]
for ext in ['states','gradients','factors']:
 a=(OUT/f'independent_n64_mb4.{ext}').read_bytes();b=(OUT/f'n64_mb4_x0_s42_t8_m0_c8.{ext}').read_bytes();assert a==b;checks.append(dict(artifact=ext,bytes=len(a),uint32_words=len(a)//4,sha256=hashlib.sha256(a).hexdigest(),all_bits_equal=True))
(OUT/'independent_oracle_parity.json').write_text(json.dumps(checks,indent=2))
frontier=[]
for n,mb in sorted(set((q['width'],q['microbatch']) for q in rows)):
 for mixed in [0,1]:
  q=next(q for q in rows if q['width']==n and q['microbatch']==mb and q['mixed_x']==mixed and q.get('same_order_raw_exact') and q['requested_steps']==8 and q['seed']==42)
  overhead=448+q['static_stack_reserve']+12*128
  entry={k:q[k] for k in ['width','microbatch','tokens','mixed_x','parameters','record_bytes','raw_FP32_m_v_bytes','static_stack_reserve']};entry['nonhistory_nonweight_bytes_before_allocator_slack']=overhead
  for frac in [.1,.15,1.]:
   # Strict < budget; delta>=0, so this is only an optimistic necessary capacity limit.
   lim=0
   while (lim+1)*q['record_bytes']+overhead < frac*q['raw_FP32_m_v_bytes']:lim+=1
   entry[f'optimistic_max_capacity_below_{frac:g}_raw_m_v']=lim
  tested=[r for r in rows if r.get('same_order_raw_exact') and r['seed']==42 and r['width']==n and r['microbatch']==mb and r['mixed_x']==mixed]
  entry['tested_horizons']=[{'steps':r['requested_steps'],'optimizer_bytes':r['optimizer_replacement_bytes'],'optimizer_fraction_raw_m_v':r['optimizer_fraction_raw_m_v'],'all_bits_equal':r['same_order_raw_exact'],'malloc_slack':r['malloc_usable_bytes']-r['request_bytes']} for r in tested]
  frontier.append(entry)
(OUT/'capacity_frontier.json').write_text(json.dumps(frontier,indent=2))
sanitizers=[]
for n,mb in [(64,16),(256,1)]:
 for mixed in [0,1]:
  d=ROOT/'build'/f'san_n{n}_mb{mb}_x{mixed}';d.mkdir(exist_ok=True);binary=d/'width_replay'
  args=['gcc','-std=c11','-O1','-g','-fno-fast-math','-ffp-contract=off','-fno-math-errno','-mno-red-zone','-fsanitize=address,undefined','-fno-omit-frame-pointer',f'-DD={n}u',f'-DH={n}u',f'-DKOUT={n}u',f'-DMB={mb}u',f'-DMIXED_X={mixed}',str(ROOT/'width_replay.c'),'-lm','-o',str(binary)]
  p=subprocess.run(args,text=True,capture_output=True);(d/'build.log').write_text(p.stdout+p.stderr);assert p.returncode==0
  env={**os.environ,'ASAN_OPTIONS':'detect_leaks=0:halt_on_error=1','UBSAN_OPTIONS':'halt_on_error=1'}
  command=[str(binary),'2','42','8','128','8','4096','0','-','1'];p=subprocess.run(command,env=env,text=True,capture_output=True)
  (d/'stdout.json').write_text(p.stdout);(d/'stderr.txt').write_text(p.stderr)
  assert p.returncode==0 and not p.stderr
  q=json.loads(p.stdout);ref=next(r for r in rows if r.get('same_order_raw_exact') and r['width']==n and r['microbatch']==mb and r['mixed_x']==mixed and r['seed']==42 and r['requested_steps']==8)
  assert all(q[key]==ref[key] for key in ['state_digest','gradient_digest','factor_digest'])
  result=dict(width=n,microbatch=mb,mixed_x=mixed,steps=8,exit_code=p.returncode,stderr_empty=not bool(p.stderr),digests_equal=True,build_command=args,run_command=command,leak_check='disabled because known unsupported ptrace runtime from Stage10',scope='Sanitizer memory and timing are not production accounting or benchmark numbers.')
  (d/'provenance.json').write_text(json.dumps(result,indent=2));sanitizers.append(result)
(OUT/'sanitizer_checks.json').write_text(json.dumps(sanitizers,indent=2))
summary=json.loads((OUT/'summary.json').read_text());summary['independent_oracle_words']=sum(q['uint32_words'] for q in checks);summary['independent_oracle_all_bits_equal']=True;summary['address_undefined_sanitizer_cases']=len(sanitizers);(OUT/'summary.json').write_text(json.dumps(summary,indent=2))
print('DONE independent oracle, analytical capacities, and four ASan/UBSan checks')
