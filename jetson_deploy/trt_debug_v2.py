# =============================================================================
# trt_debug_v2.py  (v2 of trt_debug.py; diagnostic, runs on the Jetson with /usr/bin/python3)
# v2: np.bool shim for TensorRT 8.5 (JetPack 5) with NumPy >= 1.24; rest unchanged
#
# Why: field_node_v2.py --backend trtc with the FP16 engine counted 0 heads on
# every field image (ONNX Runtime FP32 on the server: MAE 0.50 on the same
# images). Two possible causes, separated here on ONE real tile:
#   [a] numerics of the engine (FP16 overflow -> NaN / inf logits), or
#   [b] the cudart / ctypes runner of field_node_v2.py
# For every engine given (e.g. the FP16 and an FP32 build of the same ONNX)
# it prints: fraction of finite logits / points, logit range, max head score
# and the count at --thr on the first tile of each image; with onnxruntime
# installed (--onnx), the same numbers for ONNX Runtime CPU as reference.
#   FP32 engine (no --fp16) for the comparison:
#     /usr/src/tensorrt/bin/trtexec --onnx=deploy/m.onnx --saveEngine=deploy/m_fp32.engine
#   /usr/bin/python3 trt_debug.py --engines deploy/m_fp16.engine deploy/m_fp32.engine \
#       --images photo/PHOTO_4.png frames/video_lapangan_f000150.jpg
# Reading: FP32 engine counts ~ correct and FP16 NaN / 0  -> [a] numerics
#          both engines 0 with finite logits, or garbage   -> [b] runner
# =============================================================================

import os
import sys
import ctypes
import argparse

import numpy as np
# v2: TensorRT 8.5 Python (JetPack 5) uses the alias np.bool inside trt.nptype();
# NumPy >= 1.24 removed it (AttributeError). Restore it as the builtin bool
# (identical behaviour); NumPy 2.x defines np.bool again, then nothing changes.
if not hasattr(np, "bool"):
    np.bool = bool
import cv2

_ap = argparse.ArgumentParser()
_ap.add_argument("--engines", nargs="+", required=True)
_ap.add_argument("--images", nargs="+", required=True)
_ap.add_argument("--onnx", default=None, help="optional ONNX for an ONNX Runtime CPU reference")
_ap.add_argument("--size", type=int, default=512)
_ap.add_argument("--thr", type=float, default=0.5)
_args = _ap.parse_args()

MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


def first_tile(path, S):
    bgr = cv2.imread(path)
    if bgr is None:
        raise SystemExit(f"cannot read {path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    x = ((rgb - MEAN) / STD).transpose(2, 0, 1)
    t = np.zeros((1, 3, S, S), np.float32)
    h, w = min(S, x.shape[1]), min(S, x.shape[2])
    t[0, :, :h, :w] = x[:, :h, :w]
    return t


class Cudart:
    def __init__(self):
        for name in ("libcudart.so", "libcudart.so.12", "/usr/local/cuda/lib64/libcudart.so"):
            try:
                self.rt = ctypes.CDLL(name)
                break
            except OSError:
                continue
        else:
            raise SystemExit("libcudart.so not found")
        vp, sz, i = ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int
        self.rt.cudaMalloc.argtypes = [ctypes.POINTER(vp), sz]
        self.rt.cudaMemcpy.argtypes = [vp, vp, sz, i]
        self.rt.cudaDeviceSynchronize.argtypes = []
        self.rt.cudaGetLastError.argtypes = []
        for f in (self.rt.cudaMalloc, self.rt.cudaMemcpy, self.rt.cudaDeviceSynchronize, self.rt.cudaGetLastError):
            f.restype = i


def run_engine(path, cu, x):
    """synchronous, default stream: as simple as possible (separates runner bugs from numerics)."""
    import tensorrt as trt
    rt = cu.rt
    with open(path, "rb") as f, trt.Runtime(trt.Logger(trt.Logger.WARNING)) as r:
        eng = r.deserialize_cuda_engine(f.read())
    ctx = eng.create_execution_context()
    host, dev, info = {}, {}, []
    for k in range(eng.num_io_tensors):
        n = eng.get_tensor_name(k)
        shape = tuple(eng.get_tensor_shape(n))
        dt = trt.nptype(eng.get_tensor_dtype(n))
        mode = "in" if eng.get_tensor_mode(n) == trt.TensorIOMode.INPUT else "out"
        info.append(f"{n}:{mode}:{shape}:{np.dtype(dt).name}")
        host[n] = np.zeros(shape, dtype=dt)
        d = ctypes.c_void_p()
        assert rt.cudaMalloc(ctypes.byref(d), host[n].nbytes) == 0
        dev[n] = d
        ctx.set_tensor_address(n, d.value)
    host["image"][...] = x.astype(host["image"].dtype)
    assert rt.cudaMemcpy(dev["image"], host["image"].ctypes.data, host["image"].nbytes, 1) == 0
    ok = ctx.execute_async_v3(0)
    e1 = rt.cudaDeviceSynchronize()
    for n in host:
        if n != "image":
            assert rt.cudaMemcpy(host[n].ctypes.data, dev[n], host[n].nbytes, 2) == 0
    return host["logits"][0].astype(np.float32), host["points"][0].astype(np.float32), ok, e1, info


def stats(tag, lo, po, thr):
    fin_l = np.isfinite(lo).mean()
    fin_p = np.isfinite(po).mean()
    z = lo - np.nanmax(lo, -1, keepdims=True)
    e = np.exp(z)
    s = e[:, 1] / e.sum(-1)
    print(f"    {tag:24s} finite logits {fin_l:6.1%} points {fin_p:6.1%} | logits [{np.nanmin(lo):9.3f}, "
          f"{np.nanmax(lo):9.3f}] | max score {np.nanmax(s):.4f} | count@{thr} {int((s > thr).sum())}")


def main():
    cu = Cudart()
    sess = None
    if _args.onnx:
        try:
            import onnxruntime as ort
            sess = ort.InferenceSession(_args.onnx, providers=["CPUExecutionProvider"])
        except ImportError:
            print("[debug] onnxruntime not installed for this python: no ORT reference")
    for p in _args.images:
        x = first_tile(p, _args.size)
        print(f"[debug] {os.path.basename(p)} (first {_args.size}x{_args.size} tile)")
        if sess is not None:
            lo, po = sess.run(["logits", "points"], {"image": x})
            stats("onnxruntime CPU fp32", lo[0], po[0], _args.thr)
        for eng in _args.engines:
            lo, po, ok, err, info = run_engine(eng, cu, x)
            stats(os.path.basename(eng), lo, po, _args.thr)
            if not ok or err:
                print(f"      execute ok={ok} cudaDeviceSynchronize={err}")
    print("[debug] engine I/O:", "; ".join(info))


if __name__ == "__main__":
    main()
