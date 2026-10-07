"""One-off probe: what does rocprofv3 --marker-trace actually emit for ROCTX
ranges, and which dlopen/library combination makes markers visible to it?
Not part of the collector; answers an open b_traces question (native ROCTX
CSV dialect + working marker registration) before writing the real adapter.

Usage: python3 probe_rocprofv3_format.py <mode>
  mode: local   -> ctypes.CDLL("libroctx64.so") default (RTLD_LOCAL) [current marker_runtime.py behavior]
        global  -> ctypes.CDLL("libroctx64.so", mode=ctypes.RTLD_GLOBAL)
        sdk     -> ctypes.CDLL("librocprofiler-sdk-roctx.so") directly
"""
import ctypes
import os
import sys

mode = sys.argv[1]
sys.path.insert(0, os.path.dirname(__file__))

import torch  # noqa: E402

if mode == "local":
    lib = ctypes.CDLL("libroctx64.so")
elif mode == "global":
    lib = ctypes.CDLL("libroctx64.so", mode=ctypes.RTLD_GLOBAL)
elif mode == "sdk":
    lib = ctypes.CDLL("librocprofiler-sdk-roctx.so")
else:
    raise SystemExit(f"unknown mode {mode}")

push = lib.roctxRangePushA
push.argtypes = [ctypes.c_char_p]
push.restype = ctypes.c_int
pop = lib.roctxRangePop
pop.argtypes = []
pop.restype = ctypes.c_int

dev = torch.device("cuda:0")
a = torch.randn(256, 256, device=dev, dtype=torch.bfloat16)
b = torch.randn(256, 256, device=dev, dtype=torch.bfloat16)

label = f"B|scope|probe_{mode}|0|test".encode()
rc_push = push(label)
for i in range(3):
    c = a @ b
    torch.cuda.synchronize()
rc_pop = pop()

print(f"probe[{mode}] push_rc={rc_push} pop_rc={rc_pop} sum={float(c.sum())}")
