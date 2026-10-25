#!/usr/bin/env python3
"""Generate the link-time stub libraries listed in stubs.txt.

usage: mkstubs.py <clang> <out-dir>

<clang> is the NDK's aarch64 Android clang. Each library becomes an empty shared object with the
real one's SONAME and the listed symbols (functions that return, zero-filled data), which is all
the linker reads. They are never installed or loaded: on the device the system's own libraries
are. Nothing from those libraries is copied, only the names the driver calls.
"""
import os, subprocess, sys, tempfile

here = os.path.dirname(os.path.abspath(__file__))
clang, out = sys.argv[1], sys.argv[2]
os.makedirs(out, exist_ok=True)

libs, syms = [], {}
for line in open(os.path.join(here, 'stubs.txt')):
    f = line.split()
    if not f or f[0].startswith('#'):
        continue
    if f[0] == 'lib':
        libs.append(f[1])
        syms[f[1]] = []
    else:
        lib, kind, bind, size, name = f
        syms[lib].append((kind, bind, int(size), name))

with tempfile.TemporaryDirectory() as tmp:
    for lib in libs:
        asm = ['.text']
        for kind, bind, size, name in syms[lib]:
            if kind != 'FUNC':
                continue
            asm += ['.%s "%s"' % ('weak' if bind == 'WEAK' else 'globl', name),
                    '.type "%s", @function' % name, '"%s":' % name, 'ret']
        asm.append('.data')
        for kind, bind, size, name in syms[lib]:
            if kind != 'OBJECT':
                continue
            asm += ['.%s "%s"' % ('weak' if bind == 'WEAK' else 'globl', name),
                    '.type "%s", @object' % name, '.size "%s", %d' % (name, size),
                    '.p2align 3', '"%s":' % name, '.zero %d' % size]
        src = os.path.join(tmp, 'stub.S')
        open(src, 'w').write('\n'.join(asm) + '\n')
        subprocess.run([clang, '-shared', '-nostdlib', '-Wl,-soname,' + lib, '-o',
                        os.path.join(out, lib), src], check=True)
print('%d stub libraries in %s' % (len(libs), out))
