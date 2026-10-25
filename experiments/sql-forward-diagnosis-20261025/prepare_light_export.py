from pathlib import Path
r=Path(__file__).resolve().parent
s=(r/'export_diagnosis.py').read_text().replace("r/'job.json'","r/'light-job.json'").replace("r/'cpu-full'","r/'light-full'")
s=s.replace("['forward_parts','forward_cpu_profile','step_complete','sample_checkpoint']","['forward_light','step_complete','sample_checkpoint']").replace("r/'diagnosis.json'","r/'light-diagnosis.json'")
s=s.replace("'limits':'Instrumentation changes timing; not a replacement for unprofiled throughput.'", "'limits':'Light counters perturb Python allocation/GC timing; cannot retroactively attribute older untraced stalls.'")
s=s.replace('out=[]','out=[];validation=[]')
s=s.replace(' assert actual==expected', " assert actual['assistant_tokens']==expected['assistant_tokens'] and actual['correct']==expected['correct']\n validation.append({'rank':rank,'actual':actual,'reference':expected,'exactly_equal':actual==expected,'loss_difference':actual['loss']-expected['loss']})")
s=s.replace("'same_validation_at_768':True", "'same_validation_at_768':all(x['exactly_equal'] for x in validation),'validation_comparisons':validation")
(r/'export_light.py').write_text(s,encoding='utf-8',newline='\n')
