"""Inspect installed optimizer and arithmetic implementation without training."""
import hashlib,inspect,json
from pathlib import Path
import deepspeed
from deepspeed.runtime.zero.stage3 import DeepSpeedZeroOptimizer_Stage3 as Z
from deepspeed.ops.adam import DeepSpeedCPUAdam
root=Path(deepspeed.__file__).resolve().parent
for name in ['unscale_and_clip_grads','_optimizer_step','_prepare_sub_group','_release_sub_group','step']:
 fn=getattr(Z,name)
 for _ in range(8):
  if fn.__module__!='deepspeed.utils.nvtx':break
  fn=inspect.getclosurevars(fn).nonlocals['func']
 print('METHOD',name);print(inspect.getsource(fn))
print('CPUADAM_STEP');print(inspect.getsource(DeepSpeedCPUAdam.step))
for relative in ['ops/csrc/adam/cpu_adam.cpp','ops/csrc/adam/cpu_adam_impl.cpp','ops/csrc/includes/cpu_adam.h','ops/csrc/includes/simd.h']:
 p=root/relative
 if p.exists():
  print('SOURCE',relative,hashlib.sha256(p.read_bytes()).hexdigest());print(p.read_text())
