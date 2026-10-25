"""Process-local earlier submission of already-singleton oversized gradient buckets."""
import collections,hashlib,inspect,time,types
import torch
def install(optimizer):
 original=optimizer.reduce_independent_p_g_buckets_and_remove_grads
 source=inspect.getsource(original)
 for part in ['bucket.elements + param.ds_numel > self.reduce_bucket_size and bucket.elements > 0','self.__reduce_and_partition_ipg_grads(comm_dtype)','self.__add_grad_to_ipg_bucket(param)']:
  assert part in source,'Native gradient bucket contract changed'
 assert not optimizer._coalesce_grad_reduction and optimizer.reduce_scatter
 assert optimizer.reduce_and_partition_stream is not None
 stats=collections.defaultdict(float)
 def ready(self,param):
  result=original(param)
  dtype=self.get_param_comm_dtype(param);bucket=self.ipg_buckets[dtype]
  if bucket.elements>self.reduce_bucket_size:
   # In the original implementation this necessarily flushes alone when the
   # next param arrives. Submit the identical bucket now instead of waiting.
   assert len(bucket.params)==1 and bucket.params[0] is param
   assert bucket.elements==param.ds_numel
   stats['oversized_elements']+=bucket.elements
   stats['immediate_singleton_flush_calls']+=1
   start=time.monotonic()
   with torch.profiler.record_function('zero/flush_oversized_ready'):
    self._DeepSpeedZeroOptimizer_Stage3__reduce_and_partition_ipg_grads(dtype)
   stats['immediate_singleton_flush_host_seconds']+=time.monotonic()-start
  return result
 optimizer.reduce_independent_p_g_buckets_and_remove_grads=types.MethodType(ready,optimizer)
 return {'mode':'oversized singleton ready-triggered submit','native_packet_grouping_preserved':True,
         'native_source_sha256':hashlib.sha256(source.encode()).hexdigest(),
         'reduce_bucket_elements':optimizer.reduce_bucket_size,'event_limit':optimizer.max_param_reduce_events,
         'communication_dtype':str(optimizer.communication_data_type)},stats
