"""Derive pipeline entry from the controlled synchronous CPU-master entry."""
from pathlib import Path
root = Path(__file__).resolve().parent
source = (root/'train_cpu_master.py').read_text()

def replace(old, new):
    global source
    assert old in source, old
    source = source.replace(old, new)

replace('from bench_cpu_master import DETAIL as CPU_MASTER_DETAIL',
        'from bench_cpu_master import DETAIL as CPU_MASTER_DETAIL\nimport bench_cpu_pipeline')
replace('configure(a.mode);bench_cpu_master.install()', 'configure(a.mode);bench_cpu_pipeline.install()')
replace('store.moments.initialize_frames(store.frames)',
        'bench_cpu_pipeline.install_store(store,is_embedding=isinstance(old,torch.nn.Embedding))')
replace('Persistent CPU FP32 master/m/v; DeepSpeed CPUAdam NEON; 24 local cores; bounded pinned staging',
        'CPU FP32 master/m/v; GIL-free unmodified DeepSpeed CPUAdam math; 4-slot gradient/update/weight pipeline; persistent device frames')
replace('        loss.backward()\n', '        loss.backward()\n        bench_cpu_pipeline.flush()\n')
replace('cpu_master_timing=dict(CPU_MASTER_DETAIL),',
        'cpu_master_timing=dict(CPU_MASTER_DETAIL),pipeline_timing=dict(bench_cpu_pipeline.DETAIL),pipeline_runtime=dict(bench_cpu_pipeline.RUNTIME),')
(root/'train_cpu_pipeline.py').write_text(source)
