import faulthandler
faulthandler.enable()
faulthandler.dump_traceback_later(600, repeat=True)
import json,time,os,math
from pathlib import Path
import torch,torch_npu
from swift.callbacks import TrainerCallback,callbacks_map
r=Path(os.environ['SFT_RUN'])
class Audit(TrainerCallback):
 def write(self,event,**data):
  with (r/'audit.jsonl').open('a') as f:f.write(json.dumps({'event':event,'time':time.time(),**data})+'\n')
 def on_train_begin(self,args,state,control,model=None,**kw):
  ps=list(model.named_parameters()); train=[(n,p) for n,p in ps if p.requires_grad]
  assert train and all('lora_' in n for n,p in train)
  self.initial={n:p.detach().float().cpu().clone() for n,p in train}
  self.write('train_begin',trainable=sum(p.numel() for n,p in train),total=sum(p.numel() for n,p in ps),dtypes=list(set(str(p.dtype) for n,p in ps)),allocated=torch.npu.memory_allocated(),reserved=torch.npu.memory_reserved())
  torch.npu.reset_peak_memory_stats()
 def on_log(self,args,state,control,logs=None,**kw):
  self.write('log',step=state.global_step,logs=logs or {},peak_allocated=torch.npu.max_memory_allocated(),peak_reserved=torch.npu.max_memory_reserved())
 def on_pre_optimizer_step(self,args,state,control,model=None,optimizer=None,**kw):
  train=[p for p in model.parameters() if p.requires_grad]
  assert all(p.grad is None for p in model.parameters() if not p.requires_grad)
  assert all(torch.isfinite(p.grad).all().item() for p in train if p.grad is not None)
 def on_train_end(self,args,state,control,model=None,**kw):
  changed=sum(not torch.equal(self.initial[n],p.detach().float().cpu()) for n,p in model.named_parameters() if p.requires_grad)
  self.write('train_end',step=state.global_step,changed_lora_tensors=changed,peak_allocated=torch.npu.max_memory_allocated(),peak_reserved=torch.npu.max_memory_reserved())
  assert changed>0
callbacks_map['single_card_audit']=Audit
from swift.pipelines import sft_main
sft_main()
