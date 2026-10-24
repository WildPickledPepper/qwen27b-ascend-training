import argparse,json
from pathlib import Path
import torch,torch_npu
import bench_resident_npu as bench
import profile_codec_scopes as profile
p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args()
bench.configure('compressed');profile.install()
torch.npu.config.allow_internal_format=False
for d in [0,1]:
 with torch.npu.device(d):
  codec=bench.Codec(Path(__file__).resolve().parent/'libonline_codec.so')
  x=torch.randn(1<<20,device='npu:'+str(d),dtype=torch.float32)*.01
  profile.begin('smoke')
  frame=codec.encode(x);y=codec.decode(frame)
  assert torch.equal(x.view(torch.int32),y.view(torch.int32))
  profile.finish(Path(a.output+str(d)+'.json'))
  print('codec observer smoke passed',d,flush=True)
