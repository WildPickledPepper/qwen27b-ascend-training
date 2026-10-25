"""Derive a no-update full-model backward audit; inspect real FFN activations."""
from pathlib import Path
root=Path(__file__).resolve().parent
s=(root/'train_bf16_master.py').read_text()
s=s.replace('configure("fp32");bench_bf16_master.install()', 'configure("fp32");bench_bf16_master.install();install_diagnostic_moments()')
injection='''
class DiagnosticMoments(bench_bf16_master.MixedMoments):
    def __init__(self,n,device,block=128,lr=1e-5,wd=.01,window=16777216):
        self.n=n;self.lr=lr;self.wd=wd;self.window=window
        self.master=torch.empty(n,dtype=torch.float32);self.initialized=False
        self.updated=0;self.progress=None
    def bytes(self):return self.n*4
    def update(self,start,w,g,step):
        assert torch.isfinite(g).all()
        token=probe_before(self.name,self.n,start,w.reshape(-1).float(),g.reshape(-1).float(),step)
        if token:
            chosen,ix,before,gradient=token
            PROBES.append({'name':self.name,'step':step,'indices':chosen,'weight_before':before,'gradient':gradient,'weight_after':before})
        self.updated+=g.numel()
def install_diagnostic_moments():
    import bench_resident
    bench_resident.base.QuantMoments=DiagnosticMoments
'''
pos=s.index('p=argparse.ArgumentParser()');s=s[:pos]+injection+'\n'+s[pos:]
begin=s.index('def install_native_hook(');end=s.index('def get_examples(',begin)
s=s[:begin]+'''def install_native_hook(name,param):
    native[name]=(param,{})
    def hook(p):
        if p.grad is None:return
        token=probe_before(name,p.numel(),0,p.detach().view(-1),p.grad.float().view(-1),1)
        if token:
            chosen,ix,before,gradient=token
            PROBES.append({'name':name,'step':1,'indices':chosen,'weight_before':before,'gradient':gradient,'weight_after':before})
        p.grad=None
    param.register_post_accumulate_grad_hook(hook)

'''+s[end:]
begin=s.index('    baseline=run_eval(val);');end=s.index('\nexcept Exception:',begin)
s=s[:begin]+'''    diagnostic={'stages':{}}
    for canonical in [True,False]:
        label='fp32' if canonical else 'bf16';snap={};handles=[];PROBES.clear()
        def capture(name):
            def forward(module,inputs,output):
                if not output.requires_grad:return
                snap[name+'_output']=output.detach().cpu().clone()
                snap[name+'_input']=inputs[0].detach().cpu().clone()
                output.register_hook(lambda g:snap.__setitem__(name+'_output_gradient',g.detach().cpu().clone()))
                if inputs[0].requires_grad:
                    inputs[0].register_hook(lambda g:snap.__setitem__(name+'_input_gradient',g.detach().cpu().clone()))
            return forward
        for part in ['gate_proj','up_proj','down_proj']:
            mod=model.get_submodule('model.layers.63.mlp.'+part)
            handles.append(mod.register_forward_hook(capture(part)))
        for st in stores.values():st.enabled=True;st.step=1
        native_step[0]=1;model.train()
        with evaluation_context(canonical):
            loss=model(**tensors(train[0])).loss
            diagnostic['stages'][label]={'loss':float(loss)}
            loss.backward()
        for d in [0,1]:torch.npu.synchronize(d)
        for h in handles:h.remove()
        torch.save(snap,out/(label+'-last-ffn-private.pt'))
        (out/(label+'-probes-private.json')).write_text(json.dumps(PROBES))
        g=snap['gate_proj_output'];up=snap['up_proj_output'];dy=snap['down_proj_input_gradient']
        sigmoid=torch.sigmoid(g);expected=dy*up*(sigmoid+g*sigmoid*(1-sigmoid))
        actual=snap['gate_proj_output_gradient']
        diagnostic['stages'][label]['silu_gradient']={
            'gate_input_max_abs':float(g.abs().max()),'up_output_max_abs':float(up.abs().max()),
            'actual_l2':float(actual.norm()),'expected_l2':float(expected.norm()),
            'relative_l2':float((actual-expected).norm()/expected.norm().clamp_min(1e-30)),
            'actual_max_abs':float(actual.abs().max()),'expected_max_abs':float(expected.abs().max())}
        del loss,snap;gc.collect()
        log('backward_diagnostic_stage',stage=label,metrics=diagnostic['stages'][label])
        (out/'backward-diagnostic.json').write_text(json.dumps(diagnostic,indent=2))
    status['diagnostic_complete']=True;log('diagnostic_complete',metrics=diagnostic)
'''+s[end:]
(root/'diagnose_bf16_backward.py').write_text(s,encoding='utf-8',newline='\n')
