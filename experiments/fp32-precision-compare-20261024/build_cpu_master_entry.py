"""Generate the controlled CPUAdam entry from the frozen NPU reference entry."""
from pathlib import Path

root = Path(__file__).resolve().parent
source = (root/'train_fp32_large_window.py').read_text()

def replace(old, new):
    global source
    assert old in source, old
    source = source.replace(old, new)

replace('set(range(96,100))', 'set(range(96,120))')
replace('import resource', 'import resource\nimport bench_cpu_master\nfrom bench_cpu_master import DETAIL as CPU_MASTER_DETAIL')
replace('configure(a.mode)\n', 'configure(a.mode);bench_cpu_master.install()\n')
replace('torch.set_num_threads(4)', 'torch.set_num_threads(24)')
replace("status['optimizer_compute_device']='npu'", "status['optimizer_compute_device']='cpu (DeepSpeed CPUAdam); tiny native tensors retain reference NPU update'")
replace("status['optimization']='Original FP32 NPU Adam math; larger offload window and NUMA4 CPU affinity; no staged pipeline adopted'",
        "status['optimization']='Persistent CPU FP32 master/m/v; DeepSpeed CPUAdam NEON; 24 local cores; bounded pinned staging'\nstatus['rounding']='FP32 library AdamW, not bitwise identical to reference NPU operations'")
replace("'full FP32 m/v on CPU; updated in NPU windows'", "'full FP32 m/v and persistent master on CPU; DeepSpeed CPUAdam'")
replace("            stores[name]=store;", "            store.moments.initialize_frames(store.frames)\n            stores[name]=store;")
replace("            moment_bytes=sum(s.moments.bytes() for s in stores.values())+sum(p.numel()*8 for p,_ in native.values()),",
        "            moment_bytes=sum(s.n*8 for s in stores.values())+sum(p.numel()*8 for p,_ in native.values()),\n            cpu_master_bytes=sum(s.n*4 for s in stores.values()),cpu_master_timing=dict(CPU_MASTER_DETAIL),")
replace('same custom autograd harness; full FP32 moments offloaded to CPU with NPU compute, not tuned DeepSpeed',
        'same custom autograd harness; persistent CPU master and installed DeepSpeed CPUAdam, not a DeepSpeed training engine')
(root/'train_cpu_master.py').write_text(source)
