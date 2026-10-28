"""Only CPU BF16 addition; owns no gradient buffers, queues or NPU streams."""
import ctypes,subprocess,threading
from pathlib import Path
import torch

class SlotAdd:
    def __init__(self,threads=24):
        self.threads=threads;self.local=threading.local();self.calls=0
        root=Path(__file__).resolve().parent
        source=root/'cpu_fused_accum.c';library=root/'slot_add.so'
        if not library.exists() or library.stat().st_mtime<source.stat().st_mtime:
            subprocess.run(['gcc','-O3','-fopenmp','-march=armv8.2-a','-fPIC','-shared','-o',str(library),str(source)],check=True)
        self.lib=ctypes.CDLL(str(library));self.lib.accum1.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_size_t];self.lib.accum1.restype=None
        self.omp=ctypes.CDLL('libgomp.so.1');self.omp.omp_set_num_threads.argtypes=[ctypes.c_int]
    def add(self,dst,src):
        assert dst.device.type==src.device.type=='cpu'
        assert dst.dtype==src.dtype==torch.bfloat16 and dst.is_contiguous() and src.is_contiguous() and dst.numel()==src.numel()
        if not getattr(self.local,'configured',False):
            self.omp.omp_set_num_threads(self.threads);self.local.configured=True
        self.lib.accum1(dst.data_ptr(),src.data_ptr(),dst.numel());self.calls+=1

def validate():
    torch.set_num_threads(4);torch.manual_seed(42);kernel=SlotAdd(4)
    n=2**20
    # Cover all finite BF16 exponent/sign ranges; sums may legitimately overflow.
    def values():
        bits=torch.randint(0,65536,(n,),dtype=torch.int32)
        bits[(bits&0x7f80)==0x7f80]=0
        return bits.to(torch.uint16).view(torch.bfloat16)
    dst=values();src=values();ref=dst.clone();ref.add_(src);kernel.add(dst,src)
    diff=int((dst.view(torch.uint16)!=ref.view(torch.uint16)).sum());assert diff==0,diff
    dst=torch.randn(4096).bfloat16();ref=dst.clone()
    for _ in range(16):
        src=torch.randn_like(dst);ref.add_(src);kernel.add(dst,src)
    sequential=torch.equal(dst.view(torch.uint16),ref.view(torch.uint16));assert sequential
    return dict(random_finite_bf16_elements=n,differing_bits=diff,sequential_16_additions_exact=sequential,threads_configured_inside_calling_thread=True,scope='Finite BF16 inputs; no NaN payload equivalence claim')
if __name__=='__main__':
    import json
    print(json.dumps(validate()))
