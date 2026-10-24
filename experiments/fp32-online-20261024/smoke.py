"""NPU codec, gradient routing and quantized update checks before 27B allocation."""
import argparse,json,time
from pathlib import Path
import torch,torch_npu
from resident import Codec,WeightStore,CompressedLinear,CompressedEmbedding
p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args()
torch.npu.config.allow_internal_format=False
torch.npu.matmul.allow_hf32=False;torch.npu.conv.allow_hf32=False
torch.manual_seed(20261024);torch.npu.manual_seed_all(20261024)
root=Path(__file__).resolve().parent;codec=Codec(root/'libonline_codec.so')
result={'tests':[],'passed':False,'no_bf16':True};start=time.time()
def test(row):result['tests'].append(row);print(json.dumps(row),flush=True)
for device in ['npu:0','npu:1']:
    with torch.npu.device(device):
        w=torch.randn(256,256,device=device,dtype=torch.float32)*.02
        frame=codec.encode(w);restored=codec.decode(frame).view_as(w)
        assert torch.equal(restored.view(torch.int32),w.view(torch.int32))
        test({'case':'codec','device':device,'raw_bytes':w.numel()*4,'stored_bytes':frame['bytes'],'kind':frame['kind']})
        s=WeightStore('smoke.weight',w.shape,[frame],codec,device,128,1e-5,.01,65536)
        module=CompressedLinear(s)
        x=torch.randn(3,256,device='npu:0',requires_grad=True)
        reference_x=x.detach().to(device).requires_grad_();reference_w=w.detach().clone().requires_grad_()
        target=torch.randn(3,256,device='npu:0')
        reference_loss=(torch.nn.functional.linear(reference_x,reference_w)-target.to(device)).square().mean()
        reference_loss.backward()
        y=module(x);loss=(y-target).square().mean();s.enabled=False;loss.backward()
        forward_error=float((y.detach()-torch.nn.functional.linear(reference_x.detach(),reference_w.detach()).to('npu:0')).abs().max())
        gradient_error=float((x.grad-reference_x.grad.to('npu:0')).abs().max())
        assert forward_error<1e-5 and gradient_error<1e-5
        test({'case':'linear_backward','device':device,'forward_max_abs':forward_error,'input_gradient_max_abs':gradient_error})
        x.grad=None;s.enabled=True;s.step=1
        before=s.read_all().clone();loss=(module(x)-target).square().mean();loss.backward()
        after=s.read_all();assert s.changed_steps==1 and not torch.equal(before,after)
        # First update starts with exact zero moments, so quantization only affects later updates.
        g=reference_w.grad;expected=before*(1-1e-5*.01)-1e-5*(g/(g.abs()+1e-8))
        error=float((after-expected).abs().max());assert error<2e-7
        assert s.moments.codes.dtype==torch.uint8
        test({'case':'first_adam_update','device':device,'max_abs_vs_fp32':error,'moment_bytes':s.moments.bytes(),'parameters':s.n})
        del module,s,x,reference_x,reference_w,w,restored,frame,after,before,g,expected,target,y,loss,reference_loss
        torch.npu.synchronize(device);torch.npu.empty_cache()

device='npu:0'
with torch.npu.device(device):
    w=torch.randn(128,128,device=device)*.01;frame=codec.encode(w)
    s=WeightStore('embed',w.shape,[frame],codec,device,128,1e-5,.01,16384)
    emb=CompressedEmbedding(s);ids=torch.tensor([[1,2,1,120]],device=device)
    s.enabled=True;s.step=1;emb(ids).square().mean().backward()
    assert s.changed_steps==1 and s.moments.updated==s.n
    test({'case':'embedding_backward','all_coordinates_updated':True,'repeat_token':True})
result['passed']=True;result['seconds']=time.time()-start
result['codec_fallbacks']=codec.fallbacks
Path(a.output).write_text(json.dumps(result,indent=2));print('ALL_PASSED',flush=True)
