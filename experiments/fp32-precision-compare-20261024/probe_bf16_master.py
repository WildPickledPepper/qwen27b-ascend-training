"""Check both devices: master preservation, BF16 refresh, slot reuse, linear grads."""
import argparse, ctypes, json, os
from pathlib import Path
assert ctypes.CDLL(None).prctl(41, 1, 0, 0, 0) == 0
os.sched_setaffinity(0, set(range(96, 120)) & os.sched_getaffinity(0))
import torch, torch_npu
import bench_bf16_master as mixed
import bench_cpu_pipeline as pipeline
from bench_cpu_master import cpu_op
from bench_resident import WeightStore, configure
p = argparse.ArgumentParser(); p.add_argument('--output', required=True); a = p.parse_args()
assert os.environ['ASCEND_RT_VISIBLE_DEVICES'] == '2,3'
torch.set_num_threads(24); torch.manual_seed(20261024)
torch.npu.config.allow_internal_format = False
configure('fp32'); mixed.install(); rows = []
for device in ['npu:0', 'npu:1']:
    with torch.npu.device(device):
        shape = (512, 513); n = 512*513
        initial = torch.randn(n, device=device)*.01
        frames = []
        for start in range(0, n, 32768):
            raw = initial[start:start+32768].clone()
            frames.append({'kind':'raw','n':raw.numel(),'bytes':raw.numel()*4,'data':raw.view(torch.uint8),'header':None})
        s = WeightStore('mixed.probe',shape,frames,None,device,128,1e-5,.01,65536)
        mixed.install_store(s)
        master = initial.cpu(); m = torch.zeros_like(master); v = torch.zeros_like(master)
        assert torch.equal(s.moments.master.view(torch.int32), master.view(torch.int32))
        assert all(f['data'].numel() == f['n']*2 for f in frames)
        x = torch.randn((3,513),device=device,requires_grad=True)
        reference_x = x.detach().clone().requires_grad_(True)
        proxy = torch.zeros((), device=device, requires_grad=True)
        y = mixed.MixedLinear.apply(x,proxy,s)
        ref = torch.nn.functional.linear(reference_x.to(torch.bfloat16),initial.view(shape).to(torch.bfloat16)).float()
        assert torch.equal(y,ref)
        y.sum().backward(); ref.sum().backward()
        assert torch.equal(x.grad,reference_x.grad)
        for step in [1,2,3]:
            gradient = (torch.randn(n,device=device)*.001).to(torch.bfloat16).float()
            host_gradient = gradient.cpu()
            cpu_op().adam_update(7201,step,1e-5,.9,.999,1e-8,.01,True,master,host_gradient,m,v)
            s.moments.update(0,s.read_all().view(-1),gradient,step)
            pipeline.flush()
            actual = s.read_all().view(-1).cpu()
            checks = {'master_bits':torch.equal(s.moments.master.view(torch.int32),master.view(torch.int32)),
                      'm_bits':torch.equal(s.moments.m.view(torch.int32),m.view(torch.int32)),
                      'v_bits':torch.equal(s.moments.v.view(torch.int32),v.view(torch.int32)),
                      'bf16_compute_weight_bits':torch.equal(actual.view(torch.int16),master.to(torch.bfloat16).view(torch.int16)),
                      'coordinate_count':s.moments.updated==n*step}
            assert all(checks.values()),checks
            rows.append({'device':device,'step':step,'checks':checks})
Path(a.output).write_text(json.dumps({'complete':True,'rows':rows,'linear_forward_backward_equal':True},indent=2))
print(json.dumps({'complete':True,'rows':rows}),flush=True)
