import inspect,json
from deepspeed.accelerator import get_accelerator
from deepspeed.runtime.zero.stage3 import DeepSpeedZeroOptimizer_Stage3 as Z
a=get_accelerator()
print(json.dumps({'handles_memory_backpressure':a.handles_memory_backpressure(),'resolves_data_dependency':a.resolves_data_dependency(),'is_synchronized_device':a.is_synchronized_device()}))
for line in inspect.getsource(Z).splitlines():
 if 'max_param_reduce_events' in line:print(line.strip())
