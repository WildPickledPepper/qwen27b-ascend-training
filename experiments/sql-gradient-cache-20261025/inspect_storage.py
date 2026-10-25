import inspect
from deepspeed.runtime.zero.stage3 import DeepSpeedZeroOptimizer_Stage3 as Z
lines=inspect.getsource(Z).splitlines()
for i,line in enumerate(lines):
 if '__param_id_to_grad_partition' in line:
  print('\n'.join(lines[max(0,i-4):i+5]))
