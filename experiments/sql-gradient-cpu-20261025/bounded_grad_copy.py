"""Process-local adapter: bound a full-gradient FP32 cast, preserving values."""
import inspect,textwrap
import deepspeed.runtime.zero.stage3 as stage3

def install():
 cls=stage3.DeepSpeedZeroOptimizer_Stage3
 original=cls._offload_grad_partition_at_boundary
 source=textwrap.dedent(inspect.getsource(original))
 old='fp32_grad_tensor.copy_(grad_buffer.to(dtype=self.master_weights_and_grads_dtype), non_blocking=True)'
 assert source.count(old)==1,'Unexpected DeepSpeed version; do not patch silently'
 new='''for start in range(0, grad_buffer.numel(), 16777216):
            count = min(16777216, grad_buffer.numel() - start)
            converted = grad_buffer.narrow(0, start, count).to(dtype=self.master_weights_and_grads_dtype)
            fp32_grad_tensor.narrow(0, start, count).copy_(converted, non_blocking=True)
            del converted'''
 namespace={}
 exec(source.replace(old,new),stage3.__dict__,namespace)
 cls._offload_grad_partition_at_boundary=namespace[original.__name__]
