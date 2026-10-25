"""Print a compact run snapshot without private paths or sample contents."""
import json,sys
from pathlib import Path
root = Path(__file__).resolve().parent
s = json.loads((root/(sys.argv[1] if len(sys.argv)>1 else 'bf16-master-run1')/'status.json').read_text())
steps = [r for r in s['logs'] if r['event']=='step_complete']
result = {'completed':s['completed'],'full_parameter_steps':s['full_parameter_steps'],
          'last_event':s['logs'][-1]['event'],'elapsed':s['logs'][-1]['elapsed'],
          'steps':[{'step':r['step'],'seconds':r['seconds'],'loss':r['loss']} for r in steps],
          'peak_allocated_gib':[max(r['memory'][d]['peak_allocated'] for r in s['logs'] if 'memory' in r)/2**30 for d in [0,1]],
          'progress_validation':[{'step':r['step'],'loss':r['metrics']['loss']} for r in s['logs'] if r['event']=='validation_progress']}
result['forward_losses']=[{'step':r['step'],'loss':r['loss']} for r in s['logs'] if r['event']=='forward_done']
if 'first_step_sampled_gradient_l2' in s:result['first_step_sampled_gradient_l2']=s['first_step_sampled_gradient_l2']
if 'error' in s: result['failed'] = True
for k in ['validation_after_first4_fp32','validation_after','validation_after_bf16_execution']:
    if k in s: result[k] = {f:s[k][f] for f in ['loss','assistant_token_accuracy','assistant_tokens']}
print(json.dumps(result),flush=True)
