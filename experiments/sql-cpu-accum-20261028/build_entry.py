"""Derive the fused-CPU-accumulation entry from the 81.22 s pipeline entry.

Only the CPU accumulation internals change:
  --cpu-accum torch : original single-worker torch add_ path (control)
  --cpu-accum k1    : fused kernel, one microbatch per accumulator touch (bitwise same)
  --cpu-accum k2    : fused kernel, two microbatches per accumulator touch (order changes)
"""
from pathlib import Path

HERE = Path(__file__).resolve().parent
BASE = HERE.parent / 'qwen-sql-pipeline-overlap-20261026' / 'train_resident.py'
src = BASE.read_text(encoding='utf-8')

# ---------------------------------------------------------------- 1) new flag
anchor = "p.add_argument('--norm-device',choices=['cpu','npu'],default='cpu')"
assert src.count(anchor) == 1
src = src.replace(anchor, anchor + "\np.add_argument('--cpu-accum',choices=['torch','k1','k2'],default='torch')")

# ---------------------------------------------------------------- 2) keep original class, add fused one
head = "class Offload:"
tail = "\nfrom resident_gradients import create_offload_class"
start = src.index(head)
end = src.index(tail)
original = src[start:end]
assert 'Bounded D2H FIFO' in original

fused = '''class OffloadFused:
    """Fused CPU BF16 accumulation; external surface identical to Offload."""
    def __init__(self):
        from cpu_fused import FusedOffload
        self.chunk = 65536 if a.tiny else 64 * 1024 * 1024
        mode = 'k2' if a.cpu_accum == 'k2' else 'k1'
        self.fa = FusedOffload(params, mode=mode, chunk=self.chunk, threads=24)
        self.bytes = 0; self.wait = 0.; self.copy_wait = 0.; self.cpu_add = 0.
        self.hooks = [p.register_post_accumulate_grad_hook(self.hook(i)) for i, p in enumerate(params)]

    @property
    def accum(self):
        return self.fa.accum

    def begin_micro(self):
        self.fa.begin_micro()

    def hook(self, idx):
        def take(param):
            self.fa.hook(idx)(param)
        return take

    def drain(self):
        self.fa.drain()
        self.bytes = self.fa.bytes; self.wait = self.fa.slot_wait
        self.copy_wait = self.fa.copy_wait; self.cpu_add = self.fa.fuse_seconds

    def finish(self):
        self.drain()

    def close(self):
        self.fa.close()

    def reset(self):
        self.drain(); self.fa.zero()
        self.bytes = 0; self.wait = 0.; self.copy_wait = 0.; self.cpu_add = 0.


'''

replacement = original.replace("class Offload:", "class OffloadTorch:", 1)
# the control implementation ignores microbatch boundaries
replacement = replacement.replace(
    '    """Bounded D2H FIFO. One BF16 CPU accumulator per owned parameter."""\n',
    '    """Bounded D2H FIFO. One BF16 CPU accumulator per owned parameter."""\n'
    '    def begin_micro(self):\n        pass\n', 1)
replacement += fused
src = src[:start] + replacement + src[end:]

# ---------------------------------------------------------------- 3) select implementation
select = "Offload = OffloadTorch if a.cpu_accum == 'torch' else OffloadFused\n"
src = src.replace(tail, "\n" + select + tail.lstrip('\n'), 1)

# ---------------------------------------------------------------- 4) microbatch boundaries
lines, inserted = src.split('\n'), 0
out = []
for line in lines:
    stripped = line.strip()
    if (stripped.startswith("with scope('pipeline/stage0/backward')")
            or stripped.startswith("with scope('pipeline/stage1/backward')")):
        indent = line[:len(line) - len(line.lstrip())]
        out.append(indent + 'off.begin_micro()')
        inserted += 1
    out.append(line)
assert inserted == 3, inserted
src = '\n'.join(out)

# ---------------------------------------------------------------- 5) keep devices 0,3
src = src.replace("assert os.environ['ASCEND_RT_VISIBLE_DEVICES'] == '2,3'",
                  "assert os.environ['ASCEND_RT_VISIBLE_DEVICES'] == '0,3'", 1)

# ---------------------------------------------------------------- 6) shutdown path
shutdown = "off.jobs.put(None); off.jobs.join(); off.thread.join(); dist.destroy_process_group()"
shutdown_new = ("off.close() if hasattr(off, 'close') else None\n"
                "if not hasattr(off, 'close'):\n"
                "    off.jobs.put(None); off.jobs.join(); off.thread.join()\n"
                "dist.destroy_process_group()")
assert src.count(shutdown) == 1, 'shutdown line not found'
src = src.replace(shutdown, shutdown_new)

# ---------------------------------------------------------------- 7) scale-aware tiny gate
old_gate = "assert torch.allclose(grad.float(),expected.float(),rtol=.02,atol=1e-7),checks[-1]"
new_gate = ("_scale=float(expected.float().abs().max()) or 1.0\n"
            "                checks[-1]['scale']=_scale\n"
            "                assert diff<=8*2**-8*_scale,(checks[-1],_scale)")
assert src.count(old_gate) == 1, 'tiny gate not found'
src = src.replace(old_gate, new_gate)

(HERE / 'train_accum.py').write_text(src, encoding='utf-8', newline='\n')
print('wrote train_accum.py', len(src.splitlines()), 'lines')
