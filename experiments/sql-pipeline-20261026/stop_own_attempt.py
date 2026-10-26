"""Stop one explicitly identified experiment torchrun group, never other jobs."""
import argparse,os,signal
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--pid',type=int,required=True);a=p.parse_args()
root=Path(__file__).resolve().parent
args=Path(f'/proc/{a.pid}/cmdline').read_bytes().decode().split('\0')
assert 'torch.distributed.run' in args and str(root/'train_pipeline.py') in args,args
assert os.getpgid(a.pid)==a.pid,'Refuse to signal a shared process group'
assert Path(f'/proc/{a.pid}').stat().st_uid==os.getuid()
os.killpg(a.pid,signal.SIGTERM)
print('Sent TERM only to verified own experiment torchrun group',a.pid)
