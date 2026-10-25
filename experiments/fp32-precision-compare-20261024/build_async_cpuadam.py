"""Private build of installed CPUAdam with GIL-free update binding, math unchanged.

Pass --package pointing to installed deepspeed, --built to its existing CPUAdam
build directory and --output to a new experiment-local directory. Copies only
the binding and reuses the unchanged compiled implementation object.
"""
import argparse, hashlib, json, subprocess
from pathlib import Path
p = argparse.ArgumentParser()
for arg in ['package', 'built', 'output']: p.add_argument('--'+arg, required=True)
a = p.parse_args(); package = Path(a.package).resolve(); built = Path(a.built).resolve(); out = Path(a.output).resolve()
out.mkdir(exist_ok=True)
binding = (package/'ops/csrc/adam/cpu_adam.cpp').read_text()
old = 'm.def("adam_update", &ds_adam_step, "DeepSpeed CPU Adam update (C++)");'
assert binding.count(old) == 1
binding = binding.replace(old, 'm.def("adam_update", &ds_adam_step, "DeepSpeed CPU Adam update (C++)", pybind11::call_guard<pybind11::gil_scoped_release>());')
(out/'cpu_adam_async.cpp').write_text(binding)
ninja = (built/'build.ninja').read_text()
ninja = ninja.replace('-DTORCH_EXTENSION_NAME=cpu_adam ', '-DTORCH_EXTENSION_NAME=cpu_adam_async ')
ninja = ninja.replace(str(package/'ops/csrc/adam/cpu_adam.cpp'), str(out/'cpu_adam_async.cpp'))
lines = []
for line in ninja.splitlines():
    if line.startswith('build cpu_adam_impl.o:'): continue
    if line.startswith('build cpu_adam.so:'):
        line = 'build cpu_adam_async.so: link cpu_adam.o '+str(built/'cpu_adam_impl.o')
    if line == 'default cpu_adam.so': line = 'default cpu_adam_async.so'
    lines.append(line)
(out/'build.ninja').write_text('\n'.join(lines)+'\n')
meta = {'change': 'GIL release on adam_update binding only; installed CPUAdam implementation unchanged',
        'source_sha256': {str(f.relative_to(package)): hashlib.sha256(f.read_bytes()).hexdigest()
                          for f in [package/'ops/csrc/adam/cpu_adam.cpp', package/'ops/csrc/adam/cpu_adam_impl.cpp',
                                    package/'ops/csrc/includes/cpu_adam.h', package/'ops/csrc/includes/simd.h']}}
(out/'provenance.json').write_text(json.dumps(meta, indent=2))
subprocess.run(['ninja', '-j', '1'], cwd=out, check=True)
