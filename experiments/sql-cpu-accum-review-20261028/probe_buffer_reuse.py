"""CPU-only deterministic lifetime probe for the unmodified FusedOffload.

The injected completion gate holds the consumer before marking a slice staged.
No NPU initialization, pinned allocation, compiled kernel or model is needed.
"""
import argparse,ctypes,importlib.util,json,threading
from types import SimpleNamespace
import torch

p=argparse.ArgumentParser();p.add_argument('--module',required=True);p.add_argument('--output');a=p.parse_args()
spec=importlib.util.spec_from_file_location('reviewed_cpu_fused',a.module);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
class CPUReferenceKernel:
    def accum1(self,acc,src,n):
        dst=torch.frombuffer((ctypes.c_uint16*n.value).from_address(acc.value),dtype=torch.bfloat16)
        source=torch.frombuffer((ctypes.c_uint16*n.value).from_address(src.value),dtype=torch.bfloat16)
        dst.add_(source)
def case(overlap):
    entered=threading.Event();release=threading.Event()
    class Event:
        def record(self):pass
        def synchronize(self):
            entered.set()
            assert release.wait(10),'Gate timed out'
    param=torch.zeros(8,dtype=torch.bfloat16)
    off=module.FusedOffload([param],mode='k1',chunk=16,threads=1,pinned=False,lib=CPUReferenceKernel(),event_factory=Event)
    try:
        off.begin_micro(1);off.hook(0)(SimpleNamespace(grad=torch.ones_like(param)))
        assert entered.wait(5),'Worker did not reach event gate'
        staged_before_second=off._stage_count[0][0]
        if not overlap:release.set();off.drain()
        off.begin_micro(2);off.hook(0)(SimpleNamespace(grad=torch.full_like(param,2)))
        release.set();off.drain()
        actual=off.accum[0].tolist()
        return dict(overlap=overlap,stage_count_before_next_micro=staged_before_second,expected=[3.0]*8,actual=actual,correct=actual==[3.0]*8)
    finally:
        release.set();off.close();off.thread.join(timeout=5)
        assert not off.thread.is_alive()
r={'serial_control':case(False),'next_micro_before_completion':case(True),'scope':'CPU-only simulated legal completion delay; does not assert historical NPU runs encountered this schedule.'}
assert r['serial_control']['correct']
assert not r['next_micro_before_completion']['correct'],'Suspected overwrite did not reproduce'
if a.output:
    from pathlib import Path
    Path(a.output).write_text(json.dumps(r,indent=2)+'\n')
print(json.dumps(r,indent=2))
