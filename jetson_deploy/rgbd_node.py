# =============================================================================
# rgbd_node.py  (new; runs ON the Jetson next to field_node_v7.py)
#
# RGB-D counting node: one TensorRT (or ONNX Runtime) forward per frame of the
# ONNX written by export_onnx_rgbd.py (inputs: raw BGR + raw depth in mm).
# Power / thermal logging, the TensorRT runner and the counting rule are those
# of field_node_v7.py (imported, not copied), so the numbers are measured
# exactly like the RGB results of the paper.
# Sources (--source):
#   pairs      colour frames + depth/<stem>.png (rs_record.py --depth,
#              zed_record.py, prep_shtrgbd.py); optional GT <stem>.txt next to
#              the frame or in --gt-dir -> MAE / RMSE. This is the measurement
#              protocol of the paper (frames from disk, depth acquisition NOT
#              included: the time / energy of the counter only).
#   realsense  live Intel RealSense D435: colour + z16 aligned to colour
#              (depth is computed on the camera ASIC, powered over USB).
#   zed        live Stereolabs ZED: left image + MEASURE.DEPTH (mm). The
#              stereo depth is computed by the ZED SDK ON THE JETSON GPU, so
#              in this mode the logged power includes depth computation; report
#              it as "sensor + counter" and the pairs run as "counter only".
#   --no-depth feeds depth = 0 (RGB-only fallback of the same engine).
#   Density counters (IADM) run too: ONNX from baselines/rgbd_baselines.py export --raw
#   (same two raw inputs, output "density", count = sum over the real area).
#   --duration-min T repeats the pairs for T minutes (stable power); MAE uses each
#   labelled frame once.
# Per frame: fit to the engine size (keep aspect, pad colour with the ImageNet
# mean, depth with 0), infer, keep score > thr with the reference point inside
# the real area, map points back to the source frame. Timing: capture (live
# only), pre, infer, post. Outputs in --out: frames.csv, power.csv, summary.json.
#   sudo nvpmodel -m 0 && sudo jetson_clocks; sudo -v
#   /usr/bin/python3 rgbd_node.py --model deploy/motiny_epfid_shtrgbd_544x960.onnx \
#       --engine deploy/motiny_epfid_shtrgbd_544x960_fp16.engine --backend trtc \
#       --source pairs --input field_rgbd/frames --out runs/orin_rgbd_pairs_fp16 --idle-sec 20
# =============================================================================

import os
import sys
import csv
import json
import glob
import time
import platform
import argparse

import numpy as np
if not hasattr(np, "bool"):
    np.bool = bool
import cv2

_ap = argparse.ArgumentParser()
_ap.add_argument("--model", required=True, help="ONNX of export_onnx_rgbd.py (.json sidecar next to it)")
_ap.add_argument("--engine", default=None)
_ap.add_argument("--backend", choices=["ort", "trtc"], default="trtc")
_ap.add_argument("--device", choices=["cpu", "cuda"], default="cuda", help="--backend ort")
_ap.add_argument("--source", choices=["pairs", "realsense", "zed"], default="pairs")
_ap.add_argument("--input", nargs="*", default=[], help="pairs: colour frames / folders")
_ap.add_argument("--gt-dir", default=None)
_ap.add_argument("--no-depth", action="store_true")
_ap.add_argument("--thr", type=float, default=None)
_ap.add_argument("--warmup", type=int, default=10)
_ap.add_argument("--max-frames", type=int, default=0)
_ap.add_argument("--duration-min", type=float, default=0.0)
_ap.add_argument("--cam-wh", type=int, nargs=2, default=[640, 480], help="realsense stream size")
_ap.add_argument("--cam-fps", type=int, default=30)
_ap.add_argument("--zed-res", default="HD1080", help="ZED: HD2K / HD1080 / HD720 / VGA")
_ap.add_argument("--zed-depth-mode", default="PERFORMANCE", help="ZED: PERFORMANCE / QUALITY / ULTRA / NEURAL")
_ap.add_argument("--power", choices=["auto", "tegrastats", "nvsmi", "none"], default="auto")
_ap.add_argument("--power-ms", type=int, default=500)
_ap.add_argument("--tegrastats-sudo", choices=["auto", "yes", "no"], default="auto")
_ap.add_argument("--gpu-index", type=int, default=0)
_ap.add_argument("--idle-sec", type=float, default=0.0)
_ap.add_argument("--out", required=True)
_args = _ap.parse_args()

# field_node_v7 parses its own argv at import: give it a valid dummy command line,
# then point the options its PowerMonitor reads at ours.
_saved = sys.argv
sys.argv = [_saved[0], "--model", _args.model, "--input", "x", "--out", _args.out]
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import field_node_v7 as FN  # noqa: E402
sys.argv = _saved
FN._args.tegrastats_sudo = _args.tegrastats_sudo
FN._args.gpu_index = _args.gpu_index
import rgbd_depth_utils as DU  # noqa: E402


class TrtRGBD(FN.TrtCudartRunner):
    """two inputs (image, depth); everything else as field_node_v7.TrtCudartRunner."""

    def run(self, feeds):
        H2D, D2H = 1, 2
        for n, x in feeds.items():
            h = self.host[n]
            np.copyto(h, x.astype(h.dtype, copy=False))
            self._chk(self.rt.cudaMemcpyAsync(self.dev[n], h.ctypes.data, h.nbytes, H2D, self.stream), f"H2D {n}")
        if not self.ctx.execute_async_v3(self.stream.value):
            raise RuntimeError("execute_async_v3 failed")
        for n in self.outputs:
            o = self.host[n]
            self._chk(self.rt.cudaMemcpyAsync(o.ctypes.data, self.dev[n], o.nbytes, D2H, self.stream), f"D2H {n}")
        self._chk(self.rt.cudaStreamSynchronize(self.stream), "cudaStreamSynchronize")
        return {n: self.host[n].astype(np.float32) for n in self.outputs}

    def __call__(self, image, depth):
        o = self.run({"image": image, "depth": depth})
        if "density" in o:                                   # density counter (IADM, rgbd_baselines.py --raw)
            return o["density"], None
        return o["logits"][0], o["points"][0]


class OrtRGBD:
    def __init__(self, path, device):
        import onnxruntime as ort
        prov = ["CUDAExecutionProvider", "CPUExecutionProvider"] if device == "cuda" else ["CPUExecutionProvider"]
        self.sess = ort.InferenceSession(path, providers=prov)
        self.outs = [o.name for o in self.sess.get_outputs()]
        self.name = f"onnxruntime {ort.__version__} {self.sess.get_providers()[0]}"

    def __call__(self, image, depth):
        o = dict(zip(self.outs, self.sess.run(None, {"image": image, "depth": depth})))
        if "density" in o:
            return o["density"], None
        return o["logits"][0], o["points"][0]


def ref_points(H, W, stride, row, line):
    row_step, line_step = stride / row, stride / line
    sx = (np.arange(1, line + 1) - 0.5) * line_step - stride / 2
    sy = (np.arange(1, row + 1) - 0.5) * row_step - stride / 2
    sx, sy = np.meshgrid(sx, sy)
    a = np.vstack((sx.ravel(), sy.ravel())).T
    gh, gw = (H + stride - 1) // stride, (W + stride - 1) // stride
    cx, cy = np.meshgrid((np.arange(gw) + 0.5) * stride, (np.arange(gh) + 0.5) * stride)
    shifts = np.vstack((cx.ravel(), cy.ravel())).T
    return (a[None] + shifts[:, None]).reshape(-1, 2).astype(np.float32)


# ----------------------------------------------------------------------------- sources
def pairs_source(items):
    files = []
    for it in items:
        if os.path.isdir(it):
            files += sorted(p for p in glob.glob(os.path.join(it, "*")) if p.lower().endswith((".png", ".jpg", ".jpeg"))
                            and not p.endswith("_depth.png"))
        else:
            files += sorted(glob.glob(it))
    if not files:
        raise SystemExit("--source pairs: no frames")
    while True:
        for p in files:
            bgr = cv2.imread(p)
            dp = DU.depth_path_for(p)
            z = DU.read_depth_png(dp) if dp else None
            yield os.path.basename(p), p, bgr, z, 0.0
        if _args.duration_min <= 0:
            return


def realsense_source():
    import pyrealsense2 as rs
    W, H = _args.cam_wh
    pipe, cfg = rs.pipeline(), rs.config()
    cfg.enable_stream(rs.stream.color, W, H, rs.format.bgr8, _args.cam_fps)
    cfg.enable_stream(rs.stream.depth, W, H, rs.format.z16, _args.cam_fps)
    prof = pipe.start(cfg)
    scale = prof.get_device().first_depth_sensor().get_depth_scale()
    align = rs.align(rs.stream.color)
    i = 0
    try:
        while True:
            t0 = time.perf_counter()
            fs = align.process(pipe.wait_for_frames())
            c, d = fs.get_color_frame(), fs.get_depth_frame()
            if not c or not d:
                continue
            bgr = np.asanyarray(c.get_data()).copy()
            z = DU.mm_to_m(np.asanyarray(d.get_data()).astype(np.float32) * scale * 1000.0)
            i += 1
            yield f"rs_{i:06d}", None, bgr, z, time.perf_counter() - t0
    finally:
        pipe.stop()


def zed_source():
    import pyzed.sl as sl
    cam = sl.Camera()
    ip = sl.InitParameters()
    ip.camera_resolution = getattr(sl.RESOLUTION, _args.zed_res)
    ip.camera_fps = _args.cam_fps
    ip.depth_mode = getattr(sl.DEPTH_MODE, _args.zed_depth_mode)
    ip.coordinate_units = sl.UNIT.MILLIMETER
    ip.depth_maximum_distance = DU.Z_MAX_M * 1000.0
    err = cam.open(ip)
    if err != sl.ERROR_CODE.SUCCESS:
        raise SystemExit(f"ZED open failed: {err}")
    rt, img, dep = sl.RuntimeParameters(), sl.Mat(), sl.Mat()
    i = 0
    try:
        while True:
            t0 = time.perf_counter()
            if cam.grab(rt) != sl.ERROR_CODE.SUCCESS:
                continue
            cam.retrieve_image(img, sl.VIEW.LEFT)
            cam.retrieve_measure(dep, sl.MEASURE.DEPTH)
            bgr = img.get_data()[:, :, :3].copy()
            z = DU.mm_to_m(dep.get_data().copy())
            i += 1
            yield f"zed_{i:06d}", None, bgr, z, time.perf_counter() - t0
    finally:
        cam.close()


def read_gt(path):
    if path is None:
        return None
    stem = os.path.splitext(os.path.basename(path))[0]
    for d in ([_args.gt_dir] if _args.gt_dir else []) + [os.path.dirname(path),
                                                          os.path.join(os.path.dirname(os.path.dirname(path)), "gt")]:
        g = os.path.join(d, stem + ".txt")
        if os.path.isfile(g):
            return sum(1 for ln in open(g) if len(ln.strip().split()) >= 2)
    return None


# ----------------------------------------------------------------------------- main
def main():
    if os.path.isfile(os.path.join(_args.out, "summary.json")):
        raise SystemExit(f"{_args.out} already holds a summary.json: choose a new --out")
    os.makedirs(_args.out, exist_ok=True)
    meta = json.load(open(os.path.splitext(_args.model)[0] + ".json"))
    H, W = meta["hw"]
    density = meta.get("kind") == "density"                     # IADM: count = sum of the density map
    if density and not meta.get("raw_inputs"):
        raise SystemExit("density ONNX must be exported with rgbd_baselines.py export --raw (raw BGR + depth mm)")
    thr = _args.thr if _args.thr is not None else meta.get("thr", 0.5)
    if _args.backend == "trtc":
        if not _args.engine:
            raise SystemExit("--backend trtc needs --engine")
        if not os.path.basename(_args.engine).startswith(os.path.splitext(os.path.basename(_args.model))[0]):
            raise SystemExit("--engine is not named after --model (same guard as field_node_v3+)")
        run = TrtRGBD(_args.engine)
    else:
        run = OrtRGBD(_args.model, _args.device)
    ref = None if density else ref_points(H, W, meta["stride"], meta["row"], meta["line"])
    src = {"pairs": lambda: pairs_source(_args.input), "realsense": realsense_source, "zed": zed_source}[_args.source]()

    kind = FN.PowerMonitor.detect() if _args.power == "auto" else _args.power
    pm = FN.PowerMonitor(kind, _args.power_ms).start()
    idle = None
    if _args.idle_sec > 0 and kind != "none":
        t0 = time.time()
        time.sleep(_args.idle_sec)
        idle = FN.mean_power(pm.window(t0, time.time()))
    rows, n, t_start, t_end = [], 0, None, None
    t_limit = time.time() + 60 * _args.duration_min if _args.duration_min > 0 else None
    for name, path, bgr, z, t_cap in src:
        t0 = time.perf_counter()
        img, zf, s, (h, w) = DU.fit_frame(bgr, None if _args.no_depth else z, (H, W))
        image = np.ascontiguousarray(img.transpose(2, 0, 1)[None])
        depth = np.ascontiguousarray((zf * 1000.0)[None, None].astype(np.float32))
        t1 = time.perf_counter()
        lo, po = run(image, depth)
        t2 = time.perf_counter()
        if density:                                                # sum over the real (unpadded) area
            dm = lo.reshape(lo.shape[-2:])
            cnt = float(dm[:int(np.ceil(h * dm.shape[0] / H)), :int(np.ceil(w * dm.shape[1] / W))].sum())
        else:
            e = np.exp(lo - lo.max(-1, keepdims=True))
            sc = e[:, 1] / e.sum(-1)
            keep = (sc > thr) & (ref[:, 0] < w) & (ref[:, 1] < h)
            pts = po[keep] / s                                     # source-frame coordinates
            cnt = int(keep.sum())
        t3 = time.perf_counter()
        n += 1
        if n == _args.warmup + 1:
            t_start = time.time()
        if n > _args.warmup:
            gt = read_gt(path)
            rows.append(dict(frame=name, count=cnt, gt="" if gt is None else gt,
                             capture_ms=1000 * t_cap, pre_ms=1000 * (t1 - t0), infer_ms=1000 * (t2 - t1),
                             post_ms=1000 * (t3 - t2), total_ms=1000 * (t3 - t0),
                             valid_depth=float((zf[:h, :w] > 0).mean())))
            t_end = time.time()
        if _args.max_frames and len(rows) >= _args.max_frames:
            break
        if t_limit and time.time() > t_limit:
            break
    pm.stop()
    if not rows:
        raise SystemExit("no timed frames (fewer frames than --warmup?)")
    run_pw = FN.mean_power(pm.window(t_start, t_end))
    with open(os.path.join(_args.out, "frames.csv"), "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    with open(os.path.join(_args.out, "power.csv"), "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["t", "total_W", "rails", "temps"])
        for smp in pm.samples:
            wr.writerow([f"{smp[0]:.3f}", f"{smp[1]:.3f}", json.dumps(smp[2]), json.dumps(smp[3])])
    m = lambda k: float(np.mean([r[k] for r in rows]))
    frame_ms = m("total_ms") + (m("capture_ms") if _args.source != "pairs" else 0.0)
    seen, ev = set(), []
    for r in rows:                                             # each labelled frame once (frames repeat with --duration-min)
        if r["gt"] != "" and r["frame"] not in seen:
            seen.add(r["frame"])
            ev.append(r["count"] - r["gt"])
    summ = dict(model=os.path.basename(_args.model), engine=os.path.basename(_args.engine or ""), runner=run.name,
                source=_args.source, depth_used=not _args.no_depth, hw=[H, W], thr=thr, frames=len(rows),
                ms=dict(capture=m("capture_ms"), pre=m("pre_ms"), infer=m("infer_ms"), post=m("post_ms"),
                        counter_total=m("total_ms"), frame_incl_capture=frame_ms),
                fps_counter=1000.0 / m("total_ms"),
                power=dict(kind=kind, note=pm.note, run_W=run_pw, idle_W=idle, samples=len(pm.samples)),
                energy_J_per_frame_counter=(run_pw * m("total_ms") / 1000.0) if run_pw else None,
                energy_note=("pairs: counter only (depth read from disk)" if _args.source == "pairs" else
                             "live: power includes the camera pipeline running on the Jetson "
                             "(ZED: stereo depth on the GPU); energy = power x counter time"),
                accuracy=dict(n=len(ev), MAE=float(np.mean(np.abs(ev))) if ev else None,
                              RMSE=float(np.sqrt(np.mean(np.square(ev)))) if ev else None),
                host=platform.node(), meta=meta)
    with open(os.path.join(_args.out, "summary.json"), "w") as f:
        json.dump(summ, f, indent=2)
    print(json.dumps({k: summ[k] for k in ("frames", "ms", "fps_counter", "power", "energy_J_per_frame_counter",
                                           "accuracy")}, indent=2))


if __name__ == "__main__":
    main()
