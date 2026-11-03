# =============================================================================
# field_summary.py  (new; run anywhere with the runs/ folders copied together)
#
# One table from the summary.json files of field_node.py (Orin, Pi 5, our
# model and the baselines): backend, frames, latency (mean / p90 / per tile),
# FPS, power (run / idle / dynamic), energy per frame, max temperature,
# throttling, and accuracy overall + per GT bin when GT was present.
# Nothing is recomputed: the numbers are those written on the device.
#   python field_summary.py runs/*/summary.json --md field_table.md --csv field_table.csv
# =============================================================================

import os
import csv
import json
import glob
import argparse

_ap = argparse.ArgumentParser()
_ap.add_argument("summaries", nargs="+", help="summary.json files (globs allowed)")
_ap.add_argument("--md", default=None)
_ap.add_argument("--csv", default=None)
_args = _ap.parse_args()


def f(v, d=2):
    return "-" if v is None else f"{v:.{d}f}"


def main():
    files = sorted(sum([glob.glob(p) for p in _args.summaries], []))
    if not files:
        raise SystemExit("no summary.json")
    bins = []
    rows = []
    for p in files:
        s = json.load(open(p))
        acc = s.get("accuracy", {})
        for k in acc:
            if k != "all" and k not in bins:
                bins.append(k)
        rows.append((p, s))
    hdr = ["run", "host", "backend", "model", "GMACs/tile", "frames", "ms/frame", "p90 ms", "tiles", "ms/tile",
           "FPS", "W run", "W idle", "W dyn", "J/frame", "max C", "throttled", "MAE all (n)"] + \
          [f"MAE {b}" for b in bins]
    table = []
    for p, s in rows:
        L, P, A, E = s["latency_ms"], s["power"], s.get("accuracy", {}), s["env"]
        meta = E.get("model_meta", {})
        a = A.get("all")
        r = [os.path.basename(os.path.dirname(os.path.abspath(p))), E.get("host", "?"), E.get("backend", "?"),
             f"{meta.get('encoder', '?')}+{meta.get('decoder', '?')}", f(meta.get("gmacs_per_tile")),
             str(s["frames"]), f(L["mean"], 1), f(L["p90"], 1), f(L["tiles_per_frame"], 1), f(L["per_tile_infer"], 1),
             f(s["fps"]), f(P.get("run_W")), f(P.get("idle_W")), f(P.get("dynamic_W")),
             f(P.get("energy_J_per_frame"), 3), f(P.get("max_temp_C"), 1),
             ",".join(P.get("throttled_flags") or []) or "-",
             "-" if not a else f"{a['mae']:.2f} ({a['n']})"]
        r += ["-" if b not in A else f"{A[b]['mae']:.2f} ({A[b]['n']})" for b in bins]
        table.append(r)
    w = [max(len(h), *(len(t[i]) for t in table)) for i, h in enumerate(hdr)]
    print("  ".join(h.ljust(w[i]) for i, h in enumerate(hdr)))
    for t in table:
        print("  ".join(c.ljust(w[i]) for i, c in enumerate(t)))
    if _args.md:
        with open(_args.md, "w") as fo:
            fo.write("| " + " | ".join(hdr) + " |\n|" + "---|" * len(hdr) + "\n")
            for t in table:
                fo.write("| " + " | ".join(t) + " |\n")
        print(f"[summary] wrote {_args.md}")
    if _args.csv:
        with open(_args.csv, "w", newline="") as fo:
            cw = csv.writer(fo)
            cw.writerow(hdr)
            cw.writerows(table)
        print(f"[summary] wrote {_args.csv}")


if __name__ == "__main__":
    main()
