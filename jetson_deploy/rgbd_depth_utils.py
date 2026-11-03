# =============================================================================
# rgbd_depth_utils.py  (new; numpy + OpenCV only, no torch)
#
# One definition of "depth" for every script of the RGB-D counter, so the
# dataset preparation, training, the test script and the Jetson node treat
# depth identically:
#   storage   16-bit PNG, millimetres, 0 = invalid / no measurement
#             (= the format rs_record.py writes for the RealSense D435 and
#              zed_record.py writes for the ZED; ShanghaiTechRGBD .mat depth is
#              converted to it by prep_shtrgbd.py)
#   layout    <folder>/<stem>.png  (colour)   <sibling>/depth/<stem>.png (depth)
#             e.g. frames/video_5_000030.png -> depth/video_5_000030.png
#             also accepted: <stem>_depth.png next to the colour frame
#   valid     Z_MIN_M < z < Z_MAX_M   (ShanghaiTechRGBD: ZED stereo, 0-20 m;
#             -999 and > 20 000 mm are invalid in the original files)
#   network   d = z / Z_MAX_M on valid pixels, 0 elsewhere; v = valid (0 / 1)
#             -> two extra input channels [d, v] after the normalised RGB
#   resizing  "valid-aware area": area-average z*v and v separately, then
#             z = (z v) / v where v > 0.5, else invalid. Invalid pixels never
#             leak into valid ones (a plain resize would blend 0 m into the
#             heads at the border of a hole).
# =============================================================================

import os

import numpy as np
import cv2

Z_MIN_M = 0.1          # below: no measurement (ZED / D435 minimum range is larger anyway)
Z_MAX_M = 20.0         # ShanghaiTechRGBD valid range (Lian et al., CVPR 2019); also the normaliser
PAD_BGR = (103.53, 116.28, 123.675)   # ImageNet mean * 255 in BGR: pads to exactly 0 after normalisation


def depth_path_for(img_path, must_exist=False):
    """colour frame path -> depth PNG path (sibling 'depth' folder, same stem), or None."""
    d, f = os.path.split(os.path.abspath(img_path))
    stem = os.path.splitext(f)[0]
    cands = [os.path.join(os.path.dirname(d), "depth", stem + ".png"),
             os.path.join(d, stem + "_depth.png")]
    for c in cands:
        if os.path.isfile(c):
            return c
    if must_exist:
        raise FileNotFoundError(f"no depth for {img_path} (looked for {cands})")
    return None


def read_depth_png(path):
    """16-bit PNG (mm) -> float32 metres, 0 where invalid."""
    d = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if d is None:
        raise FileNotFoundError(path)
    if d.dtype != np.uint16 or d.ndim != 2:
        raise ValueError(f"{path}: expected a single-channel 16-bit PNG in millimetres, got {d.dtype} {d.shape}")
    z = d.astype(np.float32) * 1e-3
    z[~valid_mask(z)] = 0.0
    return z


def write_depth_png(path, z_m):
    """float32 metres (0 / nan = invalid) -> 16-bit PNG in millimetres (0 = invalid)."""
    z = np.nan_to_num(np.asarray(z_m, np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    z[~valid_mask(z)] = 0.0
    mm = np.clip(np.round(z * 1000.0), 0, 65535).astype(np.uint16)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    if not cv2.imwrite(path, mm):
        raise IOError(f"cannot write {path}")


def valid_mask(z_m):
    return (z_m > Z_MIN_M) & (z_m < Z_MAX_M)


def mm_to_m(depth_mm):
    """raw millimetres (any dtype; ShanghaiTechRGBD uses -999 for 'out of range') -> metres, 0 = invalid."""
    z = np.asarray(depth_mm, np.float32) * 1e-3
    z = np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)
    z[~valid_mask(z)] = 0.0
    return z


def resize_depth(z_m, out_wh):
    """valid-aware resize to (W, H). Downscaling: area average of the valid pixels only;
    upscaling: nearest neighbour (no invented depth between measurements)."""
    W, H = int(out_wh[0]), int(out_wh[1])
    h, w = z_m.shape
    if (w, h) == (W, H):
        return z_m.astype(np.float32, copy=True)
    v = valid_mask(z_m).astype(np.float32)
    if W <= w and H <= h:
        zv = cv2.resize(z_m * v, (W, H), interpolation=cv2.INTER_AREA)
        vv = cv2.resize(v, (W, H), interpolation=cv2.INTER_AREA)
        z = np.where(vv > 0.5, zv / np.maximum(vv, 1e-6), 0.0).astype(np.float32)
    else:
        z = cv2.resize(z_m * v, (W, H), interpolation=cv2.INTER_NEAREST).astype(np.float32)
    z[~valid_mask(z)] = 0.0
    return z


def depth_channels(z_m):
    """metres -> [d, v] float32 (2, H, W): d = z / Z_MAX on valid pixels, v = validity."""
    v = valid_mask(z_m).astype(np.float32)
    d = (z_m / Z_MAX_M).astype(np.float32) * v
    return np.stack([d, v], 0)


def fit_frame(bgr, z_m, out_hw):
    """resize a colour frame (and its depth) to fit inside out_hw = (H, W) keeping the aspect
    ratio, pad bottom / right: colour with PAD_BGR (normalises to 0, like the training collate),
    depth with 0 (= invalid, like the training collate). Returns (bgr_out float32 0..255,
    z_out, scale, (h, w)) where (h, w) is the real image area inside the padded frame."""
    H, W = out_hw
    h0, w0 = bgr.shape[:2]
    s = min(H / h0, W / w0)
    h, w = int(round(h0 * s)), int(round(w0 * s))
    h, w = min(h, H), min(w, W)
    img = bgr if (h, w) == (h0, w0) else cv2.resize(bgr, (w, h), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)
    out = np.empty((H, W, 3), np.float32)
    out[:] = np.array(PAD_BGR, np.float32)
    out[:h, :w] = img
    zo = np.zeros((H, W), np.float32)
    if z_m is not None:
        zo[:h, :w] = resize_depth(z_m, (w, h))
    return out, zo, s, (h, w)


def depth_at_points(z_m, pts, r=2):
    """median valid depth in a (2r+1)^2 window around each (x, y) point; 0 if none is valid."""
    out = np.zeros(len(pts), np.float32)
    H, W = z_m.shape
    for i, (x, y) in enumerate(np.asarray(pts, np.float32).reshape(-1, 2)):
        xi, yi = int(round(x)), int(round(y))
        win = z_m[max(0, yi - r):min(H, yi + r + 1), max(0, xi - r):min(W, xi + r + 1)]
        win = win[valid_mask(win)]
        out[i] = float(np.median(win)) if win.size else 0.0
    return out
