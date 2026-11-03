#!/usr/bin/env python
# rgbd_table5.py: Table V (efficiency on the Jetsons) and Fig. 4 (accuracy vs energy) for the RGB-D models,
# from the summary.json files of rgbd_node.py and the trtexec logs. No model, no GPU.
#
# Run folders (rgbd_node.py --out) named <device>_<model>_<prec>[_nodepth]:
#   device  agx | nano      prec  fp16 | fp32      e.g. runs_jetson/agx_epfid_fp16, runs_jetson/nano_iadm_fp32
# trtexec logs (optional, GPU ms): <engine file name>.log in --trt-dir, written by
#   trtexec --loadEngine=X.engine --iterations=500 --avgRuns=100 2>&1 | tee X.engine.log
# Extra rows (optional, e.g. RGB models of the paper's Table V with their RealSense MAE): CSV with columns
#   device,model,prec,dmae16,gpu_ms,frame_ms,fps,W,J,gmacs,mae
# Outputs in --out: table5_rgbd.md, table5_rgbd.csv, fig4_rgbd.png / .pdf
#   python rgbd_table5.py --runs runs_jetson/* --trt-dir deploy --out runs_jetson/table5
import os
import re
import csv
import json
import glob
import argparse

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--runs", nargs="+", required=True)
ap.add_argument("--trt-dir", default=None)
ap.add_argument("--extra", default=None)
ap.add_argument("--out", required=True)
ap.add_argument("--n-label", default="139 frames")
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)

NAMES = {"epfid": "Ours (EPIC-Depth)", "apgccd": "APGCC + depth", "p2pd": "P2PNet + depth", "iadm": "IADM",
         "epfi": "Ours (EPIC, RGB)", "apgcc": "APGCC", "p2p": "P2PNet"}
ORDER = ["iadm", "p2pd", "apgccd", "epfid"]


def gpu_ms(engine):
    if not a.trt_dir or not engine:
        return None
    for p in (os.path.join(a.trt_dir, engine + ".log"), os.path.join(a.trt_dir, os.path.splitext(engine)[0] + ".log")):
        if os.path.isfile(p):
            m = re.findall(r"GPU Compute Time:.*?mean = ([0-9.]+) ms", open(p, errors="ignore").read())
            if m:
                return float(m[-1])
    return None


R = {}
for d in a.runs:
    js = os.path.join(d, "summary.json")
    if not os.path.isfile(js):
        continue
    m = re.match(r"(agx|nano)_(.+?)_(fp16|fp32)(_nodepth)?$", os.path.basename(os.path.normpath(d)))
    if not m:
        print(f"[skip] {d}: name is not <agx|nano>_<model>_<fp16|fp32>[_nodepth]")
        continue
    dev, key, prec, nod = m.groups()
    s = json.load(open(js))
    key = key + ("_off" if nod else "")
    R[(dev, key, prec)] = dict(
        mae=s["accuracy"]["MAE"], frame_ms=s["ms"]["counter_total"], fps=s["fps_counter"],
        W=s["power"]["run_W"], J=s["energy_J_per_frame_counter"], gpu=gpu_ms(s.get("engine")),
        gmacs=s["meta"].get("gmacs_per_frame"), params=s["meta"].get("params_M"), n=s["accuracy"]["n"])

rows = []
for (dev, key, prec), r in R.items():
    if prec != "fp16" and (dev, key, "fp16") in R:
        continue                                                  # fp32 only enters through dMAE16
    f32 = R.get((dev, key, "fp32"))
    base = key[:-4] if key.endswith("_off") else key
    name = NAMES.get(base, base) + (", depth off" if key.endswith("_off") else "")
    rows.append(dict(device=dev, key=key, model=name, prec=prec.upper(),
                     dmae16=(r["mae"] - f32["mae"]) if (prec == "fp16" and f32 and r["mae"] is not None) else None,
                     gpu_ms=r["gpu"], frame_ms=r["frame_ms"], fps=r["fps"], W=r["W"], J=r["J"],
                     gmacs=r["gmacs"], params=r["params"], mae=r["mae"], n=r["n"], src="measured"))
if a.extra:
    for e in csv.DictReader(open(a.extra)):
        f = lambda k: float(e[k]) if e.get(k) not in (None, "", "-") else None
        rows.append(dict(device=e["device"], key=e["model"], model=e["model"], prec=e["prec"], dmae16=f("dmae16"),
                         gpu_ms=f("gpu_ms"), frame_ms=f("frame_ms"), fps=f("fps"), W=f("W"), J=f("J"),
                         gmacs=f("gmacs"), params=f("params") if "params" in e else None, mae=f("mae"), n=None,
                         src="extra"))
if not rows:
    raise SystemExit("no summary.json found")

with open(os.path.join(a.out, "table5_rgbd.csv"), "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)

fmt = lambda v, p=2: "–" if v is None else f"{v:.{p}f}"
fs = lambda v: "–" if v is None else f"{v:+.2f}"
rank = lambda r: (ORDER.index(r["key"].replace("_off", "")) if r["key"].replace("_off", "") in ORDER else -1,
                  r["key"].endswith("_off"))
models = sorted({r["model"]: r for r in rows}.values(), key=rank)
by = {(r["device"], r["model"]): r for r in rows}
L = ["| Method | GMACs | AGX Prec. | ΔMAE16 | GPU ms | ms/frame | FPS | W | J/frame "
     "| Nano Prec. | ΔMAE16 | GPU ms | ms/frame | FPS | W | J/frame |",
     "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
for mr in models:
    cells = [mr["model"], fmt(mr["gmacs"], 1)]
    for dev in ("agx", "nano"):
        r = by.get((dev, mr["model"]))
        cells += ([r["prec"], fs(r["dmae16"]), fmt(r["gpu_ms"]), fmt(r["frame_ms"], 1), fmt(r["fps"], 1),
                   fmt(r["W"], 1), fmt(r["J"], 3)] if r else ["–"] * 7)
    L.append("| " + " | ".join(cells) + " |")
L.append(f"\nAccuracy for ΔMAE16: {a.n_label} with ground truth, FP16 engine minus FP32 engine. "
         "W: module power (AGX: sum of rails; Nano: VDD_IN). J/frame = W × ms/frame (counter only, depth read from disk).")
L.append("\n| Method | device | field MAE (FP16 engine) | params (M) |\n|---|---|---|---|")
for r in sorted(rows, key=lambda r: (r["device"], rank(r))):
    L.append(f"| {r['model']} | {r['device']} | {fmt(r['mae'])} | {fmt(r['params'], 2)} |")
open(os.path.join(a.out, "table5_rgbd.md"), "w").write("\n".join(L) + "\n")
print("\n".join(L))

# ---------------------------------------------------------------- Fig. 4
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm, colors

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})
devs = [d for d in ("agx", "nano") if any(r["device"] == d and r["J"] and r["mae"] is not None for r in rows)]
title = {"agx": "Jetson AGX Orin (MAXN, TensorRT 10.3)", "nano": "Jetson Orin Nano (15 W, TensorRT 8.5)"}
norm = colors.Normalize(0, 100)
fig, axs = plt.subplots(1, len(devs), figsize=(4.2 * len(devs), 3.6), squeeze=False)
area = lambda g: 2.2 * (g or 25)
for ax, dev in zip(axs[0], devs):
    pts = [r for r in rows if r["device"] == dev and r["J"] and r["mae"] is not None and not r["key"].endswith("_off")]
    for r in sorted(pts, key=lambda r: -(r["gmacs"] or 0)):
        ours = r["key"].startswith("epfi")
        ax.scatter(r["J"], r["mae"], s=area(r["gmacs"]), c=[cm.viridis(norm(r["fps"] or 0))],
                   edgecolors="#c62828" if ours else "#333", linewidths=2.0 if ours else 0.6, zorder=3)
        ax.annotate(r["model"], (r["J"], r["mae"]), xytext=(6, 6), textcoords="offset points", fontsize=8,
                    color="#c62828" if ours else "#222", fontweight="bold" if ours else "normal")
    ax.set_title(title[dev], fontsize=9)
    ax.set_xlabel("Energy per frame (J)  ← better")
    ax.grid(alpha=0.3)
    ax.margins(0.25)
axs[0][0].set_ylabel(f"Field MAE, RGB-D camera ({a.n_label})  ← better")
sm = cm.ScalarMappable(norm=norm, cmap="viridis")
fig.colorbar(sm, ax=axs[0].tolist(), label="Throughput (FPS)", fraction=0.03)
for g in (25, 100, 200):
    axs[0][0].scatter([], [], s=area(g), c="#ccc", edgecolors="#555", label=f"{g} GMACs per frame")
fig.legend(*axs[0][0].get_legend_handles_labels(), loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.45, -0.02))
fig.subplots_adjust(bottom=0.27, wspace=0.3, right=0.86)
for e in ("png", "pdf"):
    fig.savefig(os.path.join(a.out, f"fig4_rgbd.{e}"), dpi=300, bbox_inches="tight")
print(f"[table5] -> {a.out}/table5_rgbd.md, table5_rgbd.csv, fig4_rgbd.png")
