"""Select native linear autograd and add controlled quality checks."""
from pathlib import Path
root=Path(__file__).resolve().parent
s=(root/'train_bf16_master.py').read_text()
s=s.replace('import bench_bf16_master\nfrom bench_bf16_master import precision_context,evaluation_context',
            'import bench_bf16_native as bench_bf16_master\nfrom bench_bf16_native import precision_context,evaluation_context')
s=s.replace('        with torch.no_grad():\n            g=p.grad.float()',
            '        with bench_bf16_master.SUBMIT_LOCK,torch.npu.device(p.device),torch.no_grad():\n            g=p.grad.float()')
s=s.replace('        assert all(p.grad is None for p in model.parameters())',
            '        assert all(p.grad is None for p in model.parameters())\n        assert all(st.compute_weight.grad is None for st in stores.values() if hasattr(st,"compute_weight"))')
s=s.replace("        (out/'parameter-probes.json').write_text(json.dumps(PROBES))", """        (out/'parameter-probes.json').write_text(json.dumps(PROBES))
        if step==1:
            probe_gradient_l2=sum(g*g for row in PROBES for g in row['gradient'])**.5
            status['first_step_sampled_gradient_l2']=probe_gradient_l2
            log('first_step_gradient_check',sampled_l2=probe_gradient_l2,coordinates=sum(len(r['indices']) for r in PROBES))
            assert probe_gradient_l2<20, 'Abnormal sampled first-step gradient; stop before extending run'""")
s=s.replace("        if step%5==0:\n            log('validation_progress',step=step,metrics=run_eval(val[:2],canonical=False))", """        if step==4:
            status['validation_after_first4_fp32']=run_eval(val,canonical=True)
            log('validation_after_first4_fp32',metrics=status['validation_after_first4_fp32'])
        if step%5==0:
            progress=run_eval(val[:2],canonical=False)
            log('validation_progress',step=step,metrics=progress)
            assert progress['loss']<native_before['loss']+.5, 'Validation degraded; stop and retain diagnostic'""")
s=s.replace("status['optimization']='", "status['linear_backward']='native PyTorch BF16 autograd, FP32 CPU master update'\nstatus['optimization']='",1)
(root/'train_bf16_native.py').write_text(s,encoding='utf-8',newline='\n')
single=s.replace("name=rename(key);assert tuple(shape)","name=rename(key);device='npu:0';assert tuple(shape)")
single=single.replace("status['linear_backward']=", "status['physical_devices']=[2]\nstatus['visible_physical_devices']=[2,3]\nstatus['placement']='all model weights on physical NPU2, CPU FP32 master/Adam offload'\nstatus['linear_backward']=",1)
(root/'train_bf16_single.py').write_text(single,encoding='utf-8',newline='\n')
