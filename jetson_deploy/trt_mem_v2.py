# =============================================================================
# trt_mem_v2.py  (v2 of trt_mem.py; runs on the Jetson with /usr/bin/python3 + tensorrt, no torch)
# v2: np.bool shim for TensorRT 8.5 (JetPack 5) with NumPy >= 1.24; rest unchanged
#
# Memory column of Table 1 (Orin memory, MiB) for every engine, measured the
# same way for all methods:
#   engine MiB      size of the serialized engine file (weights + kernels)
#   activation MiB  TensorRT device memory of one execution context
#                   (ICudaEngine.device_memory_size_v2, TRT >= 10.1; else
#                   device_memory_size)
#   footprint MiB   drop of MemAvailable (/proc/meminfo) from before loading
#                   to after: deserialize + context + I/O buffers + one
#                   inference (Orin = unified memory, so GPU allocations show
#                   up here). Includes the CUDA context of a fresh process;
#                   run each engine in its own process (this script does one
#                   engine per call) and close other GPU jobs first.
#   /usr/bin/python3 trt_mem.py --engine deploy/p2pnet_sha_512_fp32.engine
#   for E in deploy/*_fp32.engine deploy/*_fp16.engine; do /usr/bin/python3 trt_mem.py --engine $E; done
# =============================================================================

import os
import time
import ctypes
import argparse

import numpy as np
# v2: TensorRT 8.5 Python (JetPack 5) uses the alias np.bool inside trt.nptype();
# NumPy >= 1.24 removed it (AttributeError). Restore it as the builtin bool
# (identical behaviour); NumPy 2.x defines np.bool again, then nothing changes.
if not hasattr(np, "bool"):
    np.bool = bool

_ap = argparse.ArgumentParser()
_ap.add_argument("--engine", required=True)
_ap.add_argument("--settle", type=float, default=1.0, help="seconds to wait before each MemAvailable reading")
_args = _ap.parse_args()


def mem_available_mib():
    with open("/proc/meminfo") as f:
        for ln in f:
            if ln.startswith("MemAvailable:"):
                return int(ln.split()[1]) / 1024.0
    raise SystemExit("MemAvailable not in /proc/meminfo")


def cudart():
    for name in ("libcudart.so", "libcudart.so.12", "/usr/local/cuda/lib64/libcudart.so"):
        try:
            rt = ctypes.CDLL(name)
            break
        except OSError:
            continue
    else:
        raise SystemExit("libcudart.so not found")
    vp, sz, i = ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int
    rt.cudaMalloc.argtypes = [ctypes.POINTER(vp), sz]
    rt.cudaMemcpy.argtypes = [vp, vp, sz, i]
    rt.cudaDeviceSynchronize.argtypes = []
    for f in (rt.cudaMalloc, rt.cudaMemcpy, rt.cudaDeviceSynchronize):
        f.restype = i
    return rt


def main():
    time.sleep(_args.settle)
    m0 = mem_available_mib()
    import tensorrt as trt
    rt = cudart()
    with open(_args.engine, "rb") as f, trt.Runtime(trt.Logger(trt.Logger.WARNING)) as r:
        eng = r.deserialize_cuda_engine(f.read())
    if eng is None:
        raise SystemExit(f"cannot deserialize {_args.engine}")
    act = getattr(eng, "device_memory_size_v2", None)
    if act is None:
        act = eng.device_memory_size
    ctx = eng.create_execution_context()
    host = {}
    for k in range(eng.num_io_tensors):
        n = eng.get_tensor_name(k)
        h = np.zeros(tuple(eng.get_tensor_shape(n)), dtype=trt.nptype(eng.get_tensor_dtype(n)))
        d = ctypes.c_void_p()
        assert rt.cudaMalloc(ctypes.byref(d), h.nbytes) == 0
        ctx.set_tensor_address(n, d.value)
        host[n] = (h, d)
    ok = ctx.execute_async_v3(0)
    assert rt.cudaDeviceSynchronize() == 0 and ok
    time.sleep(_args.settle)
    m1 = mem_available_mib()
    print(f"[mem] {os.path.basename(_args.engine):40s} engine {os.path.getsize(_args.engine) / 2**20:7.1f} MiB | "
          f"activation {act / 2**20:7.1f} MiB | footprint {m0 - m1:7.1f} MiB (MemAvailable drop, incl. CUDA context)")


if __name__ == "__main__":
    main()
