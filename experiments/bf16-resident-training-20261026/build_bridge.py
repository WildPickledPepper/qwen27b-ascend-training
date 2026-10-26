"""Link a host-only bridge against existing, validated ENEC author static libraries."""
import argparse, shlex, subprocess, json, hashlib
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--author-build',required=True);a=p.parse_args()
root=Path(__file__).resolve().parent; b=Path(a.author_build)
flags=(b/'CMakeFiles/enec_bf16_bridge.dir/flags.make').read_text()
values={line.split(' = ',1)[0]:line.split(' = ',1)[1] for line in flags.splitlines() if ' = ' in line}
obj=root/'codec_device.o';so=root/'libonline_codec.so'
subprocess.run(['/usr/bin/c++',*shlex.split(values['CXX_INCLUDES']),*shlex.split(values['CXX_FLAGS']),
                '-c',str(root/'codec_device.cpp'),'-o',str(obj)],check=True)
link=shlex.split((b/'CMakeFiles/enec_bf16_bridge.dir/link.txt').read_text())
for i,s in enumerate(link):
    if s=='-o':link[i+1]=str(so)
    if s.endswith('bridge.cpp.o'):link[i]=str(obj)
    if s.startswith('-Wl,-soname,'):link[i]='-Wl,-soname,libonline_codec.so'
subprocess.run(link,cwd=b,check=True)
(root/'bridge-build.json').write_text(json.dumps({'host_source_sha256':hashlib.sha256((root/'codec_device.cpp').read_bytes()).hexdigest(),
    'library_sha256':hashlib.sha256(so.read_bytes()).hexdigest(),'author_kernels_modified':False},indent=2))
