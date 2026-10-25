"""Label existing event operations and gradient buckets without adding NPU work."""
import collections,hashlib,inspect,itertools,types
import torch,torch_npu

def install(optimizer):
 state={'active':False,'counts':collections.Counter(),'streams':{},'source_sha256':{}}
 tokens=itertools.count(1);event_cls=torch_npu.npu.Event
 def stream_id(stream):
  if stream is None:stream=torch_npu.npu.current_stream()
  handle=(stream.device_index,int(stream.npu_stream))
  if handle not in state['streams']:state['streams'][handle]=len(state['streams'])+1
  return state['streams'][handle]
 def token(event):
  if not hasattr(event,'_dep_token'):
   event._dep_token=next(tokens);event._dep_record=0
  return event._dep_token
 for name in ['record','wait','synchronize']:
  original=getattr(event_cls,name)
  state['source_sha256']['Event.'+name]=hashlib.sha256(inspect.getsource(original).encode()).hexdigest()
  def wrapped(event,*args,_name=name,_original=original,**kwargs):
   if not state['active']:return _original(event,*args,**kwargs)
   ident=token(event)
   if _name=='record':event._dep_record+=1
   label=f'dep/E{ident}/R{event._dep_record}/{_name}'
   if _name!='synchronize':label+=f'/S{stream_id(args[0] if args else kwargs.get("stream"))}'
   state['counts'][_name]+=1
   with torch.profiler.record_function(label):return _original(event,*args,**kwargs)
  setattr(event_cls,name,wrapped)
 name='_DeepSpeedZeroOptimizer_Stage3__avg_scatter_grads';original=getattr(optimizer,name)
 def bucket(self,params,*args,**kwargs):
  if not state['active']:return original(params,*args,**kwargs)
  ids=','.join(str(p.ds_id) for p in params)
  state['counts']['bucket']+=1
  with torch.profiler.record_function('dep/bucket/'+ids):return original(params,*args,**kwargs)
 setattr(optimizer,name,types.MethodType(bucket,optimizer))
 name='reduce_independent_p_g_buckets_and_remove_grads';ready=getattr(optimizer,name)
 def hook(self,param,*args,**kwargs):
  if not state['active']:return ready(param,*args,**kwargs)
  state['counts']['host_grad_hook']+=1
  with torch.profiler.record_function(f'dep/host_grad_hook/{param.ds_id}'):return ready(param,*args,**kwargs)
 setattr(optimizer,name,types.MethodType(hook,optimizer))
 return state
