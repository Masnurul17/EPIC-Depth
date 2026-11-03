# =============================================================================
# field_node_v7.py  (v7 of field_node.py; runs ON the sensor node: Jetson AGX Orin / Jetson Orin Nano / Raspberry Pi 5,
#                    and on a desktop / server NVIDIA GPU as the reference platform)
#
# v7: --power nvsmi  (auto-selected when there is no tegrastats but nvidia-smi
#     exists): desktop / server GPU as the reference platform.
#       `nvidia-smi --query-gpu=power.draw.instant,temperature.gpu,
#        utilization.gpu,clocks.sm,memory.used --format=csv,noheader,nounits
#        -lms <ms> -i <--gpu-index>`
#     (power.draw.instant needs a recent driver; if the field is not
#     supported, power.draw is used and the note says so). total = GPU board
#     power of that one GPU (NOT wall power, NOT the host CPU / DRAM).
#     If the RAPL counter /sys/class/powercap/intel-rapl:0/energy_uj is
#     readable (usually root only), the CPU package power is logged as an
#     extra rail CPU_PKG (not added to the total). summary.json: new
#     power.rails_mean_W (mean of every rail over the run) and
#     power.gpu_util_mean. Pin the GPU: CUDA_DEVICE_ORDER=PCI_BUS_ID
#     CUDA_VISIBLE_DEVICES=<i> ... --gpu-index <i> (same physical GPU for
#     inference and power). Everything else identical to v6.
#
# v6: `sudo -n tegrastats` (v5) runs as root, so proc.terminate() from the
#     normal user raised PermissionError at the end of the run (no summary
#     written). stop() now ends the root tegrastats with `sudo -n kill`
#     (sudo relays SIGTERM to tegrastats) and, if still alive, `sudo -n pkill`.
#     Everything else identical to v5.
#
# v5: on the Orin Nano (JetPack 5.1.1) tegrastats prints the power rails
#     (VDD_IN, VDD_CPU_GPU_CV, VDD_SOC) ONLY when run as root; as a normal user
#     the line has no "mW" fields and v4 recorded no power at all.
#     --tegrastats-sudo auto (default): probe tegrastats; without rails, use
#     `sudo -n tegrastats` (needs a cached sudo ticket: run `sudo -v` first,
#     no password prompt mid-run). A run with no power samples now prints a
#     loud warning. Everything else identical to v4.
#
# v4: np.bool shim for TensorRT 8.5 (JetPack 5, Orin Nano) with NumPy >= 1.24;
#     measurement code identical to v3
#
# v3: two guards only, measurement code identical to v2 (2026-10-13: three
#     video runs were written into the same --out and the P2PNet / APGCC
#     engines were run with --model of our EPFI ONNX -> summary.json
#     overwritten and model_meta of the wrong model):
#       - refuses an --out that already holds a summary.json (nothing overwritten)
#       - refuses an --engine whose file name does not start with the --model
#         stem (deploy/p2pnet_sha_512.onnx <-> deploy/p2pnet_sha_512_fp16.engine);
#         --allow-mismatch to override
#
# v2: + --backend trtc  TensorRT engine through the CUDA runtime via ctypes
#       (libcudart.so of JetPack): needs only the `tensorrt` Python module,
#       NOT torch. JetPack 6 ships tensorrt for /usr/bin/python3 but no torch,
#       so --backend trt of v1 could not run there. Everything else (timing,
#       counting rule, power, accuracy, outputs) is identical to field_node.py.
#   Orin:  /usr/bin/python3 field_node_v2.py --model deploy/<m>.onnx --engine deploy/<m>_fp16.engine \
#              --backend trtc --input photo frames --out runs/orin_acc
#
# Field test of the crowd-counting sensor node. Stand-alone: needs only numpy,
# opencv-python and one inference backend (no APGCC / timm / training code):
#   --backend ort   ONNX Runtime; --device cpu (Pi 5) | cuda | trt (ORT
#                   TensorRT execution provider, FP16) on the Orin
#   --backend trt   a TensorRT engine built with trtexec (Orin), TensorRT
#                   >= 8.5 name-based I/O API, buffers via torch.cuda
# Model = the ONNX (+ .json sidecar) of export_onnx_counter.py. Per frame:
#   [pre]   BGR -> RGB / 255 -> ImageNet normalisation, optional --max-side
#           resize, zero-pad to a multiple of S, cut into S x S tiles
#   [infer] one forward per tile (batch 1, the exported shape)
#   [post]  softmax, score > thr, reference point inside the real image part
#           of the tile (border tiles are zero-padded), points shifted to
#           frame coordinates; count = number of kept points
#   (same counting rule as apgcc_test_once.py tile<S> / apgcc_demo.py)
# Timing (time.perf_counter, after --warmup frames): pre / infer / post / total
# per frame, and infer per tile. Drawing / writing of visualisations happens
# after the timed part and is not included.
# Power / thermal (background thread, --power auto | tegrastats | rpi5 | none):
#   Orin  : `tegrastats --interval <ms>`; every "<RAIL> <cur>mW/<avg>mW" pair
#           is logged; total = VDD_IN if the module reports it, else the sum of
#           the reported rails (AGX Orin: VDD_GPU_SOC + VDD_CPU_CV + VIN_SR_5V0
#           + VDDQ_VDD2_1V8AO), temperatures "<name>@<t>C"
#   Pi 5  : `vcgencmd pmic_read_adc` (sum of V x I over the PMIC rails = SoC /
#           memory / IO rails, NOT the USB-C input: use an external USB-C
#           meter for wall power), `vcgencmd measure_temp`, `get_throttled`
#   --idle-sec N measures the idle power first -> dynamic power = run - idle.
#   energy / frame = mean power x mean frame time (total and dynamic).
# Accuracy (optional): for an image with a ground-truth file (same stem .txt,
# "x y" per line = PointData / field_annotate.py format, or --gt-dir), the
# count error is logged; summary MAE / RMSE overall and per GT-count bin
# (--bins, fixed BEFORE looking at results).
# Inputs: images, folders, video files, a camera index; --duration-min loops
# the inputs for a sustained (thermal) run.
# Outputs in --out: frames.csv, power.csv, summary.json (+ vis/ with --save-vis)
#   Pi 5 :  python field_node.py --model motiny_ifi_sha_512.onnx --backend ort --device cpu --threads 4 \
#               --input field/images field/clip1.mp4 --out runs/pi5_field --idle-sec 20
#   Orin :  sudo nvpmodel -m 0 && sudo jetson_clocks   (report the power mode)
#           python field_node.py --model motiny_ifi_sha_512.onnx --engine motiny_ifi_sha_512_fp16.engine \
#               --backend trt --input field/images field/clip1.mp4 --out runs/orin_field --idle-sec 20
#   sustained 30 min:  ... --input field/clip1.mp4 --duration-min 30 --out runs/orin_30min
# =============================================================================

import os
import re
import sys
import csv
import json
import glob
import time
import shutil
import argparse
import platform
import threading
import subprocess

import numpy as np
# v4: TensorRT 8.5 Python (JetPack 5) uses the alias np.bool inside trt.nptype();
# NumPy >= 1.24 removed it (AttributeError). Restore it as the builtin bool
# (identical behaviour); NumPy 2.x defines np.bool again, then nothing changes.
if not hasattr(np, "bool"):
    np.bool = bool
import cv2

_ap = argparse.ArgumentParser()
_ap.add_argument("--model", required=True, help="ONNX from export_onnx_counter.py (its .json sidecar is read)")
_ap.add_argument("--backend", choices=["ort", "trt", "trtc"], default="ort",
                 help="ort | trt (TensorRT, buffers via torch) | trtc (TensorRT, CUDA runtime via ctypes, no torch)")
_ap.add_argument("--device", choices=["cpu", "cuda", "trt"], default="cpu", help="ort execution provider")
_ap.add_argument("--engine", default=None, help="--backend trt / trtc: TensorRT engine file (trtexec --saveEngine)")
_ap.add_argument("--allow-mismatch", action="store_true", help="v3: allow an --engine not named after --model")
_ap.add_argument("--threads", type=int, default=0, help="ort CPU intra-op threads (0 = ORT default)")
_ap.add_argument("--input", nargs="+", required=True, help="images, folders, videos, or a camera index")
_ap.add_argument("--gt-dir", default=None, help="folder with <stem>.txt GT files (default: next to each image)")
_ap.add_argument("--thr", type=float, default=None, help="score threshold (default: the sidecar's, 0.5)")
_ap.add_argument("--max-side", type=int, default=0, help="downscale frames whose long side is larger (0 = keep)")
_ap.add_argument("--video-stride", type=int, default=1, help="process every k-th video frame")
_ap.add_argument("--max-frames", type=int, default=0, help="stop after N timed frames (0 = all)")
_ap.add_argument("--duration-min", type=float, default=0.0, help="loop the inputs for this long (sustained run)")
_ap.add_argument("--warmup", type=int, default=10, help="untimed frames at the start")
_ap.add_argument("--bins", type=float, nargs="+", default=[0, 100, 500, 1000, 1e9], help="GT-count bin edges")
_ap.add_argument("--power", choices=["auto", "tegrastats", "rpi5", "nvsmi", "none"], default="auto")
_ap.add_argument("--gpu-index", type=int, default=0,
                 help="v7: --power nvsmi: nvidia-smi index of the GPU that runs the model (PCI bus order)")
_ap.add_argument("--power-ms", type=int, default=500, help="power sampling interval")
_ap.add_argument("--tegrastats-sudo", choices=["auto", "yes", "no"], default="auto",
                 help="v5: run tegrastats through `sudo -n` (rails need root on JetPack 5)")
_ap.add_argument("--idle-sec", type=float, default=0.0, help="measure idle power for N s before the run")
_ap.add_argument("--save-vis", action="store_true", help="write images / video with the predicted points")
_ap.add_argument("--out", required=True)
_args = _ap.parse_args()

IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp")
VID_EXT = (".mp4", ".avi", ".mov", ".mkv")


# -----------------------------------------------------------------------------
# model
# -----------------------------------------------------------------------------
def load_meta(onnx_path):
    js = os.path.splitext(onnx_path)[0] + ".json"
    if not os.path.isfile(js):
        raise SystemExit(f"missing sidecar {js} (written by export_onnx_counter.py)")
    with open(js) as f:
        return json.load(f)


def ref_points_np(S, stride, row, line):
    """reference points of one S x S tile, identical to models.Decoder.AnchorPoints."""
    row_step, line_step = stride / row, stride / line
    sx = (np.arange(1, line + 1) - 0.5) * line_step - stride / 2
    sy = (np.arange(1, row + 1) - 0.5) * row_step - stride / 2
    sx, sy = np.meshgrid(sx, sy)
    a = np.vstack((sx.ravel(), sy.ravel())).T
    g = (S + stride - 1) // stride
    cx, cy = np.meshgrid((np.arange(g) + 0.5) * stride, (np.arange(g) + 0.5) * stride)
    shifts = np.vstack((cx.ravel(), cy.ravel())).T
    return (a[None] + shifts[:, None]).reshape(-1, 2).astype(np.float32)


class OrtRunner:
    def __init__(self, path, device, threads):
        import onnxruntime as ort
        so = ort.SessionOptions()
        if threads > 0:
            so.intra_op_num_threads = threads
        prov = {"cpu": ["CPUExecutionProvider"],
                "cuda": ["CUDAExecutionProvider", "CPUExecutionProvider"],
                "trt": [("TensorrtExecutionProvider", {"trt_fp16_enable": True,
                                                       "trt_engine_cache_enable": True,
                                                       "trt_engine_cache_path": os.path.dirname(os.path.abspath(path))}),
                        "CUDAExecutionProvider", "CPUExecutionProvider"]}[device]
        self.sess = ort.InferenceSession(path, sess_options=so, providers=prov)
        self.name = f"onnxruntime {ort.__version__} {self.sess.get_providers()[0]}"
        if device != "cpu" and self.sess.get_providers()[0] == "CPUExecutionProvider":
            print(f"[node] WARNING: requested {device} but ORT runs on CPU (providers {ort.get_available_providers()})")

    def __call__(self, x):
        lo, po = self.sess.run(["logits", "points"], {"image": x})
        return lo[0], po[0]


class TrtRunner:
    """TensorRT engine, name-based I/O (TensorRT >= 8.5, incl. 10.x); buffers are torch CUDA tensors."""

    def __init__(self, engine_path):
        import tensorrt as trt
        import torch
        self.torch = torch
        logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, "rb") as f, trt.Runtime(logger) as rt:
            self.engine = rt.deserialize_cuda_engine(f.read())
        if self.engine is None:
            raise SystemExit(f"cannot deserialize {engine_path} (built with another TensorRT version?)")
        self.ctx = self.engine.create_execution_context()
        self.stream = torch.cuda.Stream()
        dt = {trt.float32: torch.float32, trt.float16: torch.float16, trt.int32: torch.int32}
        self.buf = {}
        for i in range(self.engine.num_io_tensors):
            n = self.engine.get_tensor_name(i)
            shape = tuple(self.engine.get_tensor_shape(n))
            self.buf[n] = torch.empty(shape, dtype=dt[self.engine.get_tensor_dtype(n)], device="cuda")
            self.ctx.set_tensor_address(n, self.buf[n].data_ptr())
        self.name = f"tensorrt {trt.__version__} engine {os.path.basename(engine_path)}"

    def __call__(self, x):
        torch = self.torch
        self.buf["image"].copy_(torch.from_numpy(x))
        with torch.cuda.stream(self.stream):
            self.ctx.execute_async_v3(self.stream.cuda_stream)
        self.stream.synchronize()
        return (self.buf["logits"][0].float().cpu().numpy(), self.buf["points"][0].float().cpu().numpy())


class TrtCudartRunner:
    """v2: TensorRT engine, name-based I/O (TensorRT >= 8.5, incl. 10.x); device buffers, copies and the
    stream come from libcudart through ctypes, so no torch / pycuda / cuda-python is needed."""

    def __init__(self, engine_path):
        import ctypes
        import tensorrt as trt
        self.ct = ctypes
        rt = None
        for name in ("libcudart.so", "libcudart.so.12", "/usr/local/cuda/lib64/libcudart.so"):
            try:
                rt = ctypes.CDLL(name)
                break
            except OSError:
                continue
        if rt is None:
            raise SystemExit("libcudart.so not found (JetPack: /usr/local/cuda/lib64)")
        c_vp, c_sz, c_int = ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int
        rt.cudaMalloc.argtypes = [ctypes.POINTER(c_vp), c_sz]
        rt.cudaFree.argtypes = [c_vp]
        rt.cudaMemcpyAsync.argtypes = [c_vp, c_vp, c_sz, c_int, c_vp]
        rt.cudaStreamCreate.argtypes = [ctypes.POINTER(c_vp)]
        rt.cudaStreamSynchronize.argtypes = [c_vp]
        for f in (rt.cudaMalloc, rt.cudaFree, rt.cudaMemcpyAsync, rt.cudaStreamCreate, rt.cudaStreamSynchronize):
            f.restype = c_int
        self.rt = rt
        logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, "rb") as f, trt.Runtime(logger) as trt_rt:
            self.engine = trt_rt.deserialize_cuda_engine(f.read())
        if self.engine is None:
            raise SystemExit(f"cannot deserialize {engine_path} (built with another TensorRT version?)")
        self.ctx = self.engine.create_execution_context()
        self.stream = c_vp()
        self._chk(rt.cudaStreamCreate(ctypes.byref(self.stream)), "cudaStreamCreate")
        self.host, self.dev, self.inputs, self.outputs = {}, {}, [], []
        for i in range(self.engine.num_io_tensors):
            n = self.engine.get_tensor_name(i)
            shape = tuple(self.engine.get_tensor_shape(n))
            h = np.empty(shape, dtype=trt.nptype(self.engine.get_tensor_dtype(n)))
            d = c_vp()
            self._chk(rt.cudaMalloc(ctypes.byref(d), h.nbytes), f"cudaMalloc {n}")
            self.host[n], self.dev[n] = h, d
            self.ctx.set_tensor_address(n, d.value)
            (self.inputs if self.engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT else self.outputs).append(n)
        self.name = f"tensorrt {trt.__version__} engine {os.path.basename(engine_path)} (cudart/ctypes)"

    @staticmethod
    def _chk(err, what):
        if err != 0:
            raise RuntimeError(f"{what} failed: cudaError {err}")

    def __call__(self, x):
        H2D, D2H = 1, 2                                      # cudaMemcpyHostToDevice / DeviceToHost
        h = self.host["image"]
        np.copyto(h, x.astype(h.dtype, copy=False))
        self._chk(self.rt.cudaMemcpyAsync(self.dev["image"], h.ctypes.data, h.nbytes, H2D, self.stream), "H2D")
        if not self.ctx.execute_async_v3(self.stream.value):
            raise RuntimeError("execute_async_v3 failed")
        for n in self.outputs:
            o = self.host[n]
            self._chk(self.rt.cudaMemcpyAsync(o.ctypes.data, self.dev[n], o.nbytes, D2H, self.stream), f"D2H {n}")
        self._chk(self.rt.cudaStreamSynchronize(self.stream), "cudaStreamSynchronize")
        return self.host["logits"][0].astype(np.float32), self.host["points"][0].astype(np.float32)

    def __del__(self):
        try:
            for d in self.dev.values():
                self.rt.cudaFree(d)
        except Exception:
            pass


# -----------------------------------------------------------------------------
# counting
# -----------------------------------------------------------------------------
class Counter:
    def __init__(self, meta, runner, thr):
        self.S = meta["size"]
        self.thr = thr
        self.mean = np.array(meta["mean"], np.float32)
        self.std = np.array(meta["std"], np.float32)
        self.ref = ref_points_np(self.S, meta["stride"], meta["row"], meta["line"])
        self.run = runner

    def preprocess(self, bgr):
        if _args.max_side > 0 and max(bgr.shape[:2]) > _args.max_side:
            s = _args.max_side / max(bgr.shape[:2])
            bgr = cv2.resize(bgr, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) * (1.0 / 255.0)
        x = ((rgb - self.mean) / self.std).transpose(2, 0, 1)
        H, W = x.shape[1:]
        S = self.S
        nh, nw = -(-H // S), -(-W // S)
        xp = np.zeros((3, nh * S, nw * S), np.float32)
        xp[:, :H, :W] = x
        tiles = []
        for a in range(nh):
            for b in range(nw):
                tiles.append((a, b, min(S, H - a * S), min(S, W - b * S),
                              np.ascontiguousarray(xp[None, :, a * S:(a + 1) * S, b * S:(b + 1) * S])))
        return tiles, (H, W), bgr

    def postprocess(self, logits, points, h, w):
        z = logits - logits.max(-1, keepdims=True)
        e = np.exp(z)
        score = e[:, 1] / e.sum(-1)
        keep = (score > self.thr) & (self.ref[:, 0] < w) & (self.ref[:, 1] < h)
        return points[keep]


# -----------------------------------------------------------------------------
# power / thermal monitors
# -----------------------------------------------------------------------------
class PowerMonitor:
    def __init__(self, kind, interval_ms):
        self.kind, self.ms = kind, interval_ms
        self.samples = []                      # (t, total_W, rails dict, temps dict, extra)
        self.lock = threading.Lock()
        self.stop_ev = threading.Event()
        self.proc = None
        self.th = None
        self.note = ""

    @staticmethod
    def detect():
        if shutil.which("tegrastats"):
            return "tegrastats"
        if shutil.which("nvidia-smi"):                    # v7: desktop / server GPU
            return "nvsmi"
        if shutil.which("vcgencmd"):
            try:
                out = subprocess.run(["vcgencmd", "pmic_read_adc"], capture_output=True, text=True, timeout=3).stdout
                if "volt" in out or "current" in out:
                    return "rpi5"
            except Exception:
                pass
        return "none"

    # Orin ------------------------------------------------------------------
    @staticmethod
    def tegra_cmd(ms, mode):
        """v5: plain tegrastats if it reports power rails, else `sudo -n tegrastats`."""
        base = ["tegrastats", "--interval", str(ms)]
        if mode == "no" or os.geteuid() == 0:
            return base
        if mode == "yes":
            return ["sudo", "-n"] + base
        line = ""
        try:
            p = subprocess.Popen(["tegrastats", "--interval", "200"], stdout=subprocess.PIPE,
                                 stderr=subprocess.DEVNULL, text=True, bufsize=1)
            line = p.stdout.readline()
            p.terminate()
            p.wait(timeout=3)
        except Exception:
            pass
        if re.search(r"\d+mW/\d+mW", line):
            return base
        if subprocess.run(["sudo", "-n", "true"], capture_output=True).returncode == 0:
            return ["sudo", "-n"] + base
        print("[node] WARNING tegrastats reports no power rails as this user and `sudo -n` is not allowed: "
              "run `sudo -v` first (power will be missing)")
        return base

    def _tegra(self):
        cmd = self.tegra_cmd(self.ms, _args.tegrastats_sudo)
        self.cmd = " ".join(cmd)
        self.proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                     stderr=subprocess.DEVNULL, text=True, bufsize=1)
        for ln in self.proc.stdout:
            if self.stop_ev.is_set():
                break
            rails = {m.group(1): float(m.group(2)) / 1000.0
                     for m in re.finditer(r"([A-Z][A-Z0-9_]+)\s+(\d+)mW/(\d+)mW", ln)}
            temps = {m.group(1): float(m.group(2)) for m in re.finditer(r"(\w+)@(-?[\d.]+)C", ln)}
            if not rails:
                continue
            total = rails.get("VDD_IN", sum(rails.values()))
            with self.lock:
                self.samples.append((time.time(), total, rails, temps, {}))
        self.note = ("total = VDD_IN" if self.samples and "VDD_IN" in self.samples[-1][2]
                     else "total = sum of the tegrastats rails (module power; not wall power)")

    # v7: desktop / server GPU ---------------------------------------------
    RAPL = "/sys/class/powercap/intel-rapl:0/energy_uj"

    def _nvsmi(self):
        idx = str(_args.gpu_index)
        field = "power.draw.instant"
        probe = subprocess.run(["nvidia-smi", f"--query-gpu={field}", "--format=csv,noheader,nounits", "-i", idx],
                               capture_output=True, text=True)
        if probe.returncode != 0 or not re.match(r"^\s*[\d.]+\s*$", probe.stdout):
            field = "power.draw"                         # older driver: no .instant field
        q = f"{field},temperature.gpu,utilization.gpu,clocks.sm,memory.used,name"
        cmd = ["nvidia-smi", f"--query-gpu={q}", "--format=csv,noheader,nounits",
               "-lms", str(self.ms), "-i", idx]
        self.cmd = " ".join(cmd)
        rapl = None
        try:
            with open(self.RAPL) as f:
                rapl = (time.time(), int(f.read()))
        except Exception:
            pass
        self.note = (f"total = GPU board power ({field}) of nvidia-smi GPU {idx}; not wall power, host CPU / "
                     f"DRAM excluded" + ("; CPU_PKG = RAPL package power (extra rail, not in total)"
                                         if rapl else "; RAPL not readable (CPU package power not logged)"))
        self.proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, bufsize=1)
        for ln in self.proc.stdout:
            if self.stop_ev.is_set():
                break
            v = [x.strip() for x in ln.split(",")]
            if len(v) < 6:
                continue
            try:
                pw = float(v[0])
            except ValueError:                               # [N/A]
                continue
            t = time.time()
            rails = {"GPU": pw}
            if rapl is not None:
                try:
                    with open(self.RAPL) as f:
                        e = int(f.read())
                    if e >= rapl[1] and t > rapl[0]:         # skip the counter wrap
                        rails["CPU_PKG"] = (e - rapl[1]) / 1e6 / (t - rapl[0])
                    rapl = (t, e)
                except Exception:
                    pass
            temps = {}
            try:
                temps["gpu"] = float(v[1])
            except ValueError:
                pass
            extra = {}
            for k, x in (("gpu_util", v[2]), ("sm_clock_MHz", v[3]), ("mem_used_MiB", v[4])):
                try:
                    extra[k] = float(x)
                except ValueError:
                    pass
            extra["gpu_name"] = v[5]
            with self.lock:
                self.samples.append((t, pw, rails, temps, extra))

    # Pi 5 ------------------------------------------------------------------
    def _rpi5(self):
        self.note = "total = sum V x I of the PMIC rails (vcgencmd pmic_read_adc); not USB-C input power"
        while not self.stop_ev.is_set():
            t0 = time.time()
            try:
                out = subprocess.run(["vcgencmd", "pmic_read_adc"], capture_output=True, text=True, timeout=3).stdout
                cur, vol = {}, {}
                for m in re.finditer(r"(\S+)_A\s+current\(\d+\)=([\d.]+)A", out):
                    cur[m.group(1)] = float(m.group(2))
                for m in re.finditer(r"(\S+)_V\s+volt\(\d+\)=([\d.]+)V", out):
                    vol[m.group(1)] = float(m.group(2))
                rails = {k: cur[k] * vol[k] for k in cur if k in vol}
                tmp = subprocess.run(["vcgencmd", "measure_temp"], capture_output=True, text=True, timeout=3).stdout
                thr = subprocess.run(["vcgencmd", "get_throttled"], capture_output=True, text=True, timeout=3).stdout
                mt = re.search(r"temp=([\d.]+)", tmp)
                mh = re.search(r"throttled=(0x[0-9a-fA-F]+)", thr)
                temps = {"soc": float(mt.group(1))} if mt else {}
                extra = {"throttled": mh.group(1) if mh else "?"}
                if rails:
                    with self.lock:
                        self.samples.append((t0, sum(rails.values()), rails, temps, extra))
            except Exception as e:                       # keep the run alive; report once
                if not self.note.endswith("(errors)"):
                    self.note += f" (errors: {e})"
            self.stop_ev.wait(max(0.0, self.ms / 1000.0 - (time.time() - t0)))

    def start(self):
        if self.kind == "none":
            return self
        target = {"tegrastats": self._tegra, "nvsmi": self._nvsmi}.get(self.kind, self._rpi5)
        self.th = threading.Thread(target=target, daemon=True)
        self.th.start()
        return self

    def stop(self):
        self.stop_ev.set()
        if self.proc is not None:
            if getattr(self, "cmd", "").startswith("sudo"):
                # v6: the process runs as root; signal it through sudo
                subprocess.run(["sudo", "-n", "kill", "-TERM", str(self.proc.pid)], capture_output=True)
                try:
                    self.proc.wait(timeout=3)
                except Exception:
                    subprocess.run(["sudo", "-n", "pkill", "-TERM", "-f", "tegrastats --interval"], capture_output=True)
            else:
                self.proc.terminate()
                try:
                    self.proc.wait(timeout=3)
                except Exception:
                    self.proc.kill()
        if self.th is not None:
            self.th.join(timeout=5)

    def window(self, t0, t1):
        with self.lock:
            return [s for s in self.samples if t0 <= s[0] <= t1]


def mean_power(samples):
    return float(np.mean([s[1] for s in samples])) if samples else None


# -----------------------------------------------------------------------------
# inputs
# -----------------------------------------------------------------------------
def expand_inputs(items):
    out = []
    for it in items:
        if it.isdigit():
            out.append(("cam", it))
        elif os.path.isdir(it):
            out += [("img", p) for p in sorted(glob.glob(os.path.join(it, "*"))) if p.lower().endswith(IMG_EXT)]
        elif it.lower().endswith(VID_EXT):
            out.append(("vid", it))
        elif it.lower().endswith(IMG_EXT):
            out.append(("img", it))
        else:
            print(f"[node] skipped {it}")
    return out


def read_gt(img_path):
    stem = os.path.splitext(os.path.basename(img_path))[0]
    cands = [os.path.join(_args.gt_dir, stem + ".txt")] if _args.gt_dir else []
    cands.append(os.path.splitext(img_path)[0] + ".txt")
    for c in cands:
        if os.path.isfile(c):
            with open(c) as f:
                return sum(1 for ln in f if ln.strip())
    return None


def frames(items):
    """yields (source, frame_idx, bgr, gt) over all inputs (once)."""
    for kind, src in items:
        if kind == "img":
            bgr = cv2.imread(src)
            if bgr is None:
                print(f"[node] cannot read {src}")
                continue
            yield os.path.basename(src), 0, bgr, read_gt(src)
        else:
            cap = cv2.VideoCapture(int(src) if kind == "cam" else src)
            if not cap.isOpened():
                print(f"[node] cannot open {src}")
                continue
            i = 0
            while True:
                ok, bgr = cap.read()
                if not ok:
                    break
                if i % _args.video_stride == 0:
                    yield (f"cam{src}" if kind == "cam" else os.path.basename(src)), i, bgr, None
                i += 1
            cap.release()


def draw(bgr, pts, n, gt=None):
    vis = bgr.copy()
    r = max(2, int(round(min(vis.shape[:2]) / 250)))
    for x, y in pts:
        cv2.circle(vis, (int(x), int(y)), r, (0, 0, 255), -1, cv2.LINE_AA)
    txt = f"Count: {n}" + (f"  GT: {gt}" if gt is not None else "")
    sc = max(0.6, min(vis.shape[:2]) / 700)
    (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, sc, 2)
    cv2.rectangle(vis, (0, 0), (tw + 20, th + 20), (0, 0, 0), -1)
    cv2.putText(vis, txt, (10, th + 10), cv2.FONT_HERSHEY_SIMPLEX, sc, (255, 255, 255), 2, cv2.LINE_AA)
    return vis


def pct(a, q):
    return float(np.percentile(a, q)) if len(a) else None


# -----------------------------------------------------------------------------
def main():
    # v3 guards: no overwrite, engine <-> model (sidecar) consistency
    if os.path.isfile(os.path.join(_args.out, "summary.json")):
        raise SystemExit(f"{_args.out}/summary.json exists: choose a new --out (nothing is overwritten)")
    if _args.engine and not _args.allow_mismatch:
        stem = os.path.splitext(os.path.basename(_args.model))[0]
        if not os.path.basename(_args.engine).startswith(stem):
            raise SystemExit(f"--engine {os.path.basename(_args.engine)} is not built from --model {stem}.onnx "
                             f"(its .json sidecar is used); fix --model or pass --allow-mismatch")
    os.makedirs(_args.out, exist_ok=True)
    meta = load_meta(_args.model)
    thr = _args.thr if _args.thr is not None else meta.get("thr", 0.5)
    if _args.backend in ("trt", "trtc"):
        if not _args.engine:
            raise SystemExit(f"--backend {_args.backend} needs --engine")
        runner = TrtRunner(_args.engine) if _args.backend == "trt" else TrtCudartRunner(_args.engine)
    else:
        runner = OrtRunner(_args.model, _args.device, _args.threads)
    counter = Counter(meta, runner, thr)
    items = expand_inputs(_args.input)
    if not items:
        raise SystemExit("no inputs")

    kind = PowerMonitor.detect() if _args.power == "auto" else _args.power
    mon = PowerMonitor(kind, _args.power_ms).start()
    if kind in ("tegrastats", "nvsmi"):
        time.sleep(1.5)                                   # v5: first samples (and the sudo probe) before the run
        print(f"[node] power source: {getattr(mon, 'cmd', kind)} | samples so far {len(mon.samples)}")
    print(f"[node] {runner.name} | tile {counter.S} | thr {thr} | power: {kind} | {len(items)} input(s)")

    idle_w = None
    if _args.idle_sec > 0 and kind != "none":
        print(f"[node] idle power for {_args.idle_sec:.0f} s ...")
        t0 = time.time()
        time.sleep(_args.idle_sec)
        idle_w = mean_power(mon.window(t0, time.time()))

    vis_dir = os.path.join(_args.out, "vis")
    if _args.save_vis:
        os.makedirs(vis_dir, exist_ok=True)
    writers = {}
    rows, tile_ms = [], []
    n_seen = 0
    t_run0 = None
    deadline = None
    done = False
    while not done:
        for src, fi, bgr, gt in frames(items):
            t0 = time.perf_counter()
            tiles, (H, W), frame = counter.preprocess(bgr)
            t1 = time.perf_counter()
            outs, t_tiles = [], []
            for a, b, h, w, x in tiles:
                ta = time.perf_counter()
                lo, po = runner(x)
                t_tiles.append(time.perf_counter() - ta)
                outs.append((a, b, h, w, lo, po))
            t2 = time.perf_counter()
            pts = []
            for a, b, h, w, lo, po in outs:
                p = counter.postprocess(lo, po, h, w)
                if len(p):
                    p = p.copy()
                    p[:, 0] += b * counter.S
                    p[:, 1] += a * counter.S
                    pts.append(p)
            pts = np.concatenate(pts, 0) if pts else np.zeros((0, 2), np.float32)
            t3 = time.perf_counter()
            n_seen += 1
            if n_seen <= _args.warmup:
                continue
            if t_run0 is None:
                t_run0 = time.time()
                deadline = t_run0 + _args.duration_min * 60 if _args.duration_min > 0 else None
            n = int(len(pts))
            rows.append(dict(source=src, frame=fi, height=H, width=W, n_tiles=len(tiles), count=n,
                             gt="" if gt is None else gt, err="" if gt is None else n - gt,
                             pre_ms=(t1 - t0) * 1e3, infer_ms=(t2 - t1) * 1e3, post_ms=(t3 - t2) * 1e3,
                             total_ms=(t3 - t0) * 1e3, wall=time.time()))
            tile_ms += [t * 1e3 for t in t_tiles]
            if _args.save_vis:                                   # after the timed part
                vis = draw(frame, pts, n, gt)
                if fi == 0 and not src.lower().endswith(VID_EXT) and not src.startswith("cam"):
                    cv2.imwrite(os.path.join(vis_dir, os.path.splitext(src)[0] + "_pred.jpg"), vis)
                else:
                    if src not in writers:
                        writers[src] = cv2.VideoWriter(os.path.join(vis_dir, os.path.splitext(src)[0] + "_pred.mp4"),
                                                       cv2.VideoWriter_fourcc(*"mp4v"), 25, (vis.shape[1], vis.shape[0]))
                    writers[src].write(vis)
            if len(rows) % 50 == 0:
                m = np.mean([r["total_ms"] for r in rows[-50:]])
                print(f"[node] {len(rows)} frames | last-50 mean {m:.1f} ms ({1000 / m:.2f} FPS) | count {n}")
            if (_args.max_frames and len(rows) >= _args.max_frames) or (deadline and time.time() >= deadline):
                done = True
                break
        if deadline is None:
            done = True
    t_run1 = time.time()
    mon.stop()
    for w in writers.values():
        w.release()
    if not rows:
        raise SystemExit("no timed frames (fewer inputs than --warmup?)")

    # ---- files
    with open(os.path.join(_args.out, "frames.csv"), "w", newline="") as f:
        cw = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        cw.writeheader()
        cw.writerows(rows)
    with open(os.path.join(_args.out, "power.csv"), "w", newline="") as f:
        cw = csv.writer(f)
        cw.writerow(["t", "total_W", "rails_W", "temps_C", "extra"])
        for s in mon.samples:
            cw.writerow([f"{s[0]:.3f}", f"{s[1]:.4f}", json.dumps(s[2]), json.dumps(s[3]), json.dumps(s[4])])

    # ---- summary
    tot = np.array([r["total_ms"] for r in rows])
    run_samples = mon.window(t_run0, t_run1)
    run_w = mean_power(run_samples)
    frame_s = float(tot.mean()) / 1e3
    temps = [max(s[3].values()) for s in run_samples if s[3]]
    throttled = sorted({s[4].get("throttled") for s in run_samples if s[4].get("throttled")})
    rail_names = sorted({k for s in run_samples for k in s[2]})                       # v7
    rails_mean = {k: float(np.mean([s[2][k] for s in run_samples if k in s[2]])) for k in rail_names}
    gu = [s[4]["gpu_util"] for s in run_samples if "gpu_util" in s[4]]
    gpu_name = next((s[4]["gpu_name"] for s in run_samples if "gpu_name" in s[4]), None)
    acc = {}
    g = [(r["count"], r["gt"]) for r in rows if r["gt"] != ""]
    if g:
        p, t = np.array([x[0] for x in g], float), np.array([x[1] for x in g], float)
        e = p - t
        acc["all"] = dict(n=len(g), mean_gt=float(t.mean()), mae=float(np.abs(e).mean()),
                          rmse=float(np.sqrt((e ** 2).mean())), bias=float(e.mean()))
        for lo, hi in zip(_args.bins[:-1], _args.bins[1:]):
            m = (t >= lo) & (t < hi)
            if m.any():
                key = f"[{int(lo)},{'inf' if hi >= 1e9 else int(hi)})"
                acc[key] = dict(n=int(m.sum()), mean_gt=float(t[m].mean()), mae=float(np.abs(e[m]).mean()),
                                rmse=float(np.sqrt((e[m] ** 2).mean())), bias=float(e[m].mean()))
    summ = dict(
        env=dict(host=platform.node(), machine=platform.machine(), python=platform.python_version(),
                 backend=runner.name, model=os.path.basename(_args.model), model_meta=meta, thr=thr,
                 max_side=_args.max_side, warmup=_args.warmup, video_stride=_args.video_stride),
        frames=len(rows), run_seconds=t_run1 - t_run0,
        latency_ms=dict(mean=float(tot.mean()), p50=pct(tot, 50), p90=pct(tot, 90), p99=pct(tot, 99),
                        pre=float(np.mean([r["pre_ms"] for r in rows])),
                        infer=float(np.mean([r["infer_ms"] for r in rows])),
                        post=float(np.mean([r["post_ms"] for r in rows])),
                        per_tile_infer=float(np.mean(tile_ms)), tiles_per_frame=float(np.mean([r["n_tiles"] for r in rows]))),
        fps=1000.0 / float(tot.mean()),
        power=dict(source=kind, note=mon.note, n_samples=len(run_samples), idle_W=idle_w, run_W=run_w,
                   dynamic_W=(run_w - idle_w) if (run_w is not None and idle_w is not None) else None,
                   energy_J_per_frame=(run_w * frame_s) if run_w is not None else None,
                   dynamic_energy_J_per_frame=((run_w - idle_w) * frame_s)
                   if (run_w is not None and idle_w is not None) else None,
                   max_temp_C=max(temps) if temps else None, throttled_flags=throttled,
                   rails_mean_W=rails_mean, gpu_util_mean=float(np.mean(gu)) if gu else None,   # v7
                   gpu_name=gpu_name, cmd=getattr(mon, "cmd", None)),
        accuracy=acc)
    with open(os.path.join(_args.out, "summary.json"), "w") as f:
        json.dump(summ, f, indent=2)
    L = summ["latency_ms"]
    print(f"\n[node] {len(rows)} frames | {L['mean']:.1f} ms/frame (p90 {L['p90']:.1f}) = {summ['fps']:.2f} FPS | "
          f"{L['tiles_per_frame']:.1f} tiles x {L['per_tile_infer']:.1f} ms")
    P = summ["power"]
    if P["run_W"] is not None:
        print(f"[node] power {P['run_W']:.2f} W" + (f" (idle {P['idle_W']:.2f}, dynamic {P['dynamic_W']:.2f})"
                                                     if P["idle_W"] is not None else "")
              + f" | {P['energy_J_per_frame']:.3f} J/frame | max temp {P['max_temp_C']} C | {P['note']}")
    elif kind != "none":
        print("[node] WARNING no power samples recorded (v5): check `sudo -v` / --tegrastats-sudo "
              "(nvsmi: nvidia-smi / --gpu-index); latency and accuracy above are still valid")
    if P["rails_mean_W"]:
        print("[node] rails mean: " + ", ".join(f"{k} {v:.2f} W" for k, v in P["rails_mean_W"].items())
              + (f" | GPU util {P['gpu_util_mean']:.0f}%" if P["gpu_util_mean"] is not None else "")
              + (f" | {P['gpu_name']}" if P["gpu_name"] else ""))
    for k, v in acc.items():
        print(f"[node] acc {k:>12s}: n={v['n']:3d} meanGT {v['mean_gt']:7.1f} MAE {v['mae']:7.2f} "
              f"RMSE {v['rmse']:7.2f} bias {v['bias']:+7.2f}")
    print(f"[node] -> {_args.out}/summary.json, frames.csv, power.csv")


if __name__ == "__main__":
    main()
