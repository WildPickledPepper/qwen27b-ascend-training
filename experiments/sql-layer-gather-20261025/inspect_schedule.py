import inspect,ast,textwrap
from deepspeed.runtime.zero.partitioned_param_coordinator import PartitionedParameterCoordinator as C
from deepspeed.utils import set_z3_leaf_modules
def unwrap(fn):
 fn=getattr(fn,'__func__',fn)
 for _ in range(8):
  if fn.__module__!='deepspeed.utils.nvtx':break
  fn=inspect.getclosurevars(fn).nonlocals['func']
 return inspect.unwrap(fn)
print('PUBLIC_LEAF_API\n'+inspect.getsource(set_z3_leaf_modules))
source=textwrap.dedent(inspect.getsource(C));tree=ast.parse(source)
for node in ast.walk(tree):
 if isinstance(node,ast.FunctionDef) and node.name in ['_fetch_sub_module_impl','__all_gather_params_']:
  print('METHOD',node.name);print(ast.get_source_segment(source,node))
