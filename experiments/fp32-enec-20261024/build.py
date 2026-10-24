"""Build pinned author's FP32 kernels in an isolated directory; never run author scripts."""
import argparse, hashlib, json, os, shutil, subprocess, shlex
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument('--source', required=True)
p.add_argument('--sdk', required=True)
p.add_argument('--variant', default='ENEC-FP32-120-6-3-16')
p.add_argument('--cmake', default='cmake')
p.add_argument('--host-only', action='store_true')
a = p.parse_args()
root = Path(__file__).resolve().parent
src = Path(a.source) / 'csrc' / 'FP32' / a.variant
dst = root / 'author_variant'
if a.host_only:
    build = root / 'build'
    flags = (build / 'CMakeFiles/enec_fp32_bridge.dir/flags.make').read_text()
    values = {line.split(' = ', 1)[0]: line.split(' = ', 1)[1]
              for line in flags.splitlines() if ' = ' in line}
    link = shlex.split((build / 'CMakeFiles/enec_fp32_bridge.dir/link.txt').read_text())
    obj = next(x for x in link if x.endswith('bridge.cpp.o'))
    subprocess.run(['/usr/bin/c++', *shlex.split(values['CXX_DEFINES']),
                    *shlex.split(values['CXX_INCLUDES']), *shlex.split(values['CXX_FLAGS']),
                    '-c', str(root / 'bridge.cpp'), '-o', str(build / obj)], check=True)
    subprocess.run(link, cwd=build, check=True)
    raise SystemExit(0)
dst.mkdir(exist_ok=True)
for f in src.iterdir():
    if f.is_file():
        shutil.copy2(f, dst / f.name)
cmake = (dst / 'CMakeLists.txt').read_text()
cmake += '\nadd_library(enec_fp32_bridge SHARED "' + str(root / 'bridge.cpp') + '")\n'
cmake += 'target_include_directories(enec_fp32_bridge PRIVATE "' + str(dst) + '")\n'
cmake += 'target_compile_options(enec_fp32_bridge PRIVATE -O2 -std=c++17 -D_GLIBCXX_USE_CXX11_ABI=0)\n'
cmake += 'target_link_libraries(enec_fp32_bridge PRIVATE snec_compress snec_decompress ascendcl)\n'
(dst / 'CMakeLists.txt').write_text(cmake)
env = os.environ.copy()
env['CPLUS_INCLUDE_PATH'] = '/usr/include/c++/11:/usr/include/aarch64-linux-gnu/c++/11:/usr/include/aarch64-linux-gnu'
env['LIBRARY_PATH'] = '/usr/lib/gcc/aarch64-linux-gnu/11:/usr/lib/aarch64-linux-gnu'
subprocess.run([a.cmake, '-S', str(dst), '-B', str(root / 'build'),
                '-DASCEND_CANN_PACKAGE_PATH=' + a.sdk, '-DCMAKE_POSITION_INDEPENDENT_CODE=ON',
                '-DSOC_VERSION=Ascend910B2'], check=True, env=env)
subprocess.run([a.cmake, '--build', str(root / 'build'), '--target', 'enec_fp32_bridge', '-j', '4'], check=True, env=env)
names = ['snec_compress.cpp', 'snec_decompress.cpp', 'snec_merge.cpp', 'snec_utils.h', 'snec_device.h']
for n in names:
    assert (src / n).read_bytes() == (dst / n).read_bytes()
(root / 'build_manifest.json').write_text(json.dumps({
    'author_commit': subprocess.check_output(['git', '-C', a.source, 'rev-parse', 'HEAD'], text=True).strip(),
    'variant': a.variant, 'author_kernels_unchanged': True,
    'sha256': {n: hashlib.sha256((src / n).read_bytes()).hexdigest() for n in names}}, indent=2))
