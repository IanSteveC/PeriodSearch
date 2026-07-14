#!/usr/bin/env python3
"""Generate kernels.cpp (the baked-in OpenCL source string) from the .cl files.

The app compiles ocl_src_kernelSource at runtime -- NOT the .cl files on disk
(those are only read in _DEBUG builds, and even then only to dump
kernelSource.cl for inspection). This script keeps kernels.cpp in sync with
the .cl sources; the Makefile runs it automatically when any input changes.

Usage: gen_kernels_cpp.py <out.cpp> <input files in concatenation order...>
"""
import sys


def main():
    out_path, inputs = sys.argv[1], sys.argv[2:]
    lines = ["inline const char* ocl_src_kernelSource ="]
    for path in inputs:
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
        for line in content.split("\n")[:-1] if content.endswith("\n") else content.split("\n"):
            escaped = line.rstrip("\r").replace("\\", "\\\\").replace('"', '\\"')
            lines.append('"%s\\n"' % escaped)
    lines[-1] += ";"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
