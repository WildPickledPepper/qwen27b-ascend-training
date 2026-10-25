"""Guarded local CPUAdam fusion; never modify the installed DeepSpeed library."""
import hashlib,json,os,platform
from pathlib import Path
import deepspeed
from torch.utils.cpp_extension import load
r=Path(__file__).resolve().parent;b=r/'build.private';b.mkdir(exist_ok=True)
inc=Path(deepspeed.__file__).parent/'ops/csrc/includes';src=inc.parent/'adam/cpu_adam_impl.cpp'
expected={'cpu_adam.h':'d269bd3a7adcd2b6cd1c897efdf3ba31e3b2890d5944f6bfb9cc368ab0b3b345','cpu_adam_impl.cpp':'b9f44e950995ad056d998c068b35de6d15b7aa9a60b61149f179316131d608b5','simd.h':'525321614f3f817db98d612711b95705937ecb93264b7d345cce9fdb6c5e02be'}
paths={'cpu_adam.h':inc/'cpu_adam.h','cpu_adam_impl.cpp':src,'simd.h':inc/'simd.h'}
assert platform.machine()=='aarch64','This guarded prototype targets the tested NEON CPU path only'
for name,path in paths.items():assert hashlib.sha256(path.read_bytes()).hexdigest()==expected[name],name
h=(inc/'cpu_adam.h').read_text();cpp=src.read_text()
def replace_once(s,a,z):
 assert s.count(a)==1,a
 return s.replace(a,z)
h=replace_once(h,'class Adam_Optimizer {\npublic:', '''class Adam_Optimizer {
public:
    float fused_grad_scale = 1.0f;
    float* fused_param_base = nullptr;
    c10::BFloat16* fused_compute_base = nullptr;''')
h=replace_once(h,'    AVX_Data betta1_4;', '    AVX_Data fused_scale;\n    fused_scale.data = SIMD_SET(fused_grad_scale);\n\n    AVX_Data betta1_4;')
h=replace_once(h,'            simd_load<span>(grad_4, grads + i);','            simd_load<span>(grad_4, grads + i);\n            simd_mul<span>(grad_4, grad_4, fused_scale);')
h=replace_once(h,'            simd_store<span>(_params + i, param_4);', '''            simd_store<span>(_params + i, param_4);
            if constexpr (std::is_same_v<ds_params_precision_t, float>) {
                if (fused_compute_base != nullptr) {
                    simd_store<span>(fused_compute_base + ((_params + i) - fused_param_base), param_4);
                }
            }''')
cpp=replace_once(cpp,'                float grad = (float)grads[k];','                float grad = (float)grads[k] * fused_grad_scale;')
cpp=replace_once(cpp,'                _params[k] = param;', '''                _params[k] = param;
                if constexpr (std::is_same_v<ds_params_precision_t, float>) {
                    if (fused_compute_base != nullptr) {
                        fused_compute_base[(_params + k) - fused_param_base] = c10::BFloat16(param);
                    }
                }''')
cpp+='''
// Per-call context; FP32 states and the original native Adam expression sequence remain.
int ds_adam_fused_step(int optimizer_id, size_t step, float lr, float beta1, float beta2,
    float epsilon, float weight_decay, bool bias_correction, torch::Tensor& params,
    torch::Tensor& grads, torch::Tensor& exp_avg, torch::Tensor& exp_avg_sq,
    float grad_scale, torch::Tensor& compute_out) {
    for (const auto& t : {params, grads, exp_avg, exp_avg_sq}) {
        TORCH_CHECK(t.device().is_cpu() && t.scalar_type()==at::kFloat && t.is_contiguous());
        TORCH_CHECK(t.numel()==params.numel());
    }
    TORCH_CHECK(compute_out.device().is_cpu() && compute_out.scalar_type()==at::kBFloat16);
    TORCH_CHECK(compute_out.is_contiguous() && compute_out.numel()==params.numel());
    TORCH_CHECK(std::isfinite(grad_scale) && grad_scale>0.0f);
    auto entry=s_optimizers.find(optimizer_id);
    TORCH_CHECK(entry!=s_optimizers.end(), "Missing isolated fused optimizer handle");
    auto opt=std::static_pointer_cast<Adam_Optimizer>(entry->second);
    TORCH_CHECK(opt->fused_param_base==nullptr, "Concurrent update of one handle is unsupported");
    opt->fused_grad_scale=grad_scale;
    opt->fused_param_base=params.data_ptr<float>();
    opt->fused_compute_base=compute_out.data_ptr<c10::BFloat16>();
    try {
        auto code=ds_adam_step(optimizer_id,step,lr,beta1,beta2,epsilon,weight_decay,
            bias_correction,params,grads,exp_avg,exp_avg_sq);
        opt->fused_grad_scale=1.0f;opt->fused_param_base=nullptr;opt->fused_compute_base=nullptr;
        return code;
    } catch (...) {
        opt->fused_grad_scale=1.0f;opt->fused_param_base=nullptr;opt->fused_compute_base=nullptr;
        throw;
    }
}
'''
# Hide native C++ symbols and rename the class to keep both extensions isolated.
h=h.replace('Adam_Optimizer','FusedAdam_Optimizer');cpp=cpp.replace('Adam_Optimizer','FusedAdam_Optimizer')
(b/'cpu_adam.h').write_text(h);(b/'cpu_adam_impl.cpp').write_text(cpp)
(b/'binding.cpp').write_text('''// Derived from Apache-2.0 DeepSpeed CPUAdam; mathematical implementation reused.
#include "cpu_adam.h"
int ds_adam_fused_step(int,size_t,float,float,float,float,float,bool,torch::Tensor&,torch::Tensor&,torch::Tensor&,torch::Tensor&,float,torch::Tensor&);
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m) {
 m.def("create_adam",&create_adam_optimizer);
 m.def("destroy_adam",&destroy_adam_optimizer);
 m.def("adam_update_fused",&ds_adam_fused_step,pybind11::call_guard<pybind11::gil_scoped_release>());
}
''')
os.environ.setdefault('MAX_JOBS','2')
flags=['-O3','-std=c++17','-g','-Wno-reorder','-fopenmp','-fstack-protector-all','-march=native','-D__NEON__','-fvisibility=hidden','-UC10_USE_GLOG']
op=load(name='cpu_adam_fused',sources=[str(b/'binding.cpp'),str(b/'cpu_adam_impl.cpp')],extra_include_paths=[str(b),str(inc)],extra_cflags=flags,extra_ldflags=['-fopenmp'],build_directory=str(b),with_cuda=False,verbose=True)
info={'built':True,'native_sources_sha256':expected,'modified_sources_sha256':{n:hashlib.sha256((b/n).read_bytes()).hexdigest() for n in ['cpu_adam.h','cpu_adam_impl.cpp','binding.cpp']},'compiler_flags':flags,'backend':'NEON, FP32 master/m/v; native SIMD expressions retained','scope':'CPU-only fusion of gradient scale and BF16 output; no installed/shared files changed'}
(r/'build-info.json').write_text(json.dumps(info,indent=2));print(json.dumps(info))
