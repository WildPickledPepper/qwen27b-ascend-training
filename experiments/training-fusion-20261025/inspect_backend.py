"""Read the actual bound implementation after framework decorators/patches."""
import inspect
import deepspeed.runtime.zero.stage3 as stage3
fn=stage3.DeepSpeedZeroOptimizer_Stage3.unscale_and_clip_grads
print(fn.__module__,inspect.getsourcefile(fn))
print(inspect.getsource(inspect.unwrap(fn)))
from cpu_update import verified_source
print('Verified underlying source:')
print(verified_source(fn))
