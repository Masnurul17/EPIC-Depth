#!/usr/bin/env bash
# run_jetson_rgbd.sh: Table V measurements of the RGB-D counters on a Jetson (same protocol as the paper).
#   bash run_jetson_rgbd.sh agx                 # all models found in rgbd_deploy/
#   bash run_jetson_rgbd.sh agx epfid           # only some models
#   bash run_jetson_rgbd.sh nano
# Needs in this folder: rgbd_node.py, rgbd_depth_utils.py, field_node_v7.py (or a later field_node_v*.py),
#   rgbd_deploy/<model>_rgbd_480x640.onnx + .json, rs_test_gt/frames (+ GT .txt) and rs_test_gt/depth.
# Per model: FP16 and FP32 engines (FP32 with --noTF32), trtexec GPU time (<engine>.log),
#   rgbd_node.py: 139 labelled frames looped for 1 min, 20 s idle power first -> runs_jetson/<dev>_<model>_<prec>.
# EPFI-D also runs with --no-depth (same FP16 engine). Finished runs are skipped, so the script can be re-run.
set -u
D=${1:?give the device: agx or nano}; shift
MODELS=${*:-"epfid apgccd p2pd iadm"}
TRTEXEC=$(command -v trtexec || echo /usr/src/tensorrt/bin/trtexec)
PY=/usr/bin/python3
mkdir -p runs_jetson logs_rgbd
if [ ! -f field_node_v7.py ]; then
  V=$(ls field_node_v*.py 2>/dev/null | sort -V | tail -1)
  [ -z "$V" ] && { echo "no field_node_v*.py in $(pwd)"; exit 1; }
  sed -i "s/field_node_v7/${V%.py}/g" rgbd_node.py
  echo "[jetson] rgbd_node.py now imports ${V%.py}"
fi
[ -d rs_test_gt/frames ] && [ -d rs_test_gt/depth ] || { echo "rs_test_gt/frames and rs_test_gt/depth missing"; exit 1; }
sudo -v

node() {   # model onnx engine out [extra args]
  local m=$1 O=$2 E=$3 R=$4; shift 4
  if [ -f "$R/summary.json" ]; then echo "[jetson] $R done"; else
    $PY rgbd_node.py --model "$O.onnx" --engine "$E" --backend trtc --source pairs --input rs_test_gt/frames \
      --duration-min 1 --idle-sec 20 --out "$R" "$@" > "logs_rgbd/$(basename "$R").log" 2>&1 \
      || { echo "[jetson] NODE FAILED: $R (see logs_rgbd/$(basename "$R").log)"; tail -5 "logs_rgbd/$(basename "$R").log"; return; }
  fi
  $PY - "$R/summary.json" <<'EOF'
import json, sys
s = json.load(open(sys.argv[1]))
a, p = s["accuracy"], s["power"]
print(f"   {sys.argv[1].split('/')[-2]:28s} MAE {a['MAE']:.3f} (n {a['n']}) | {s['ms']['counter_total']:.2f} ms/frame "
      f"| {s['fps_counter']:.1f} FPS | {p['run_W'] or 0:.2f} W | {s['energy_J_per_frame_counter'] or 0:.3f} J/frame")
EOF
}

for m in $MODELS; do
  O=rgbd_deploy/${m}_rgbd_480x640
  [ -f "$O.onnx" ] || { echo "[jetson] skip $m: no $O.onnx"; continue; }
  for p in fp16 fp32; do
    E=${O}_$p.engine
    if [ ! -f "$E" ]; then
      F=$([ $p = fp16 ] && echo "--fp16" || echo "--noTF32")
      echo "[jetson] building $E"
      $TRTEXEC --onnx="$O.onnx" $F ${TRT_EXTRA:-} --saveEngine="$E" > "logs_rgbd/build_${D}_${m}_$p.log" 2>&1 \
        || { echo "[jetson] BUILD FAILED $m $p"; tail -5 "logs_rgbd/build_${D}_${m}_$p.log"; continue; }
    fi
    [ -f "$E.log" ] || $TRTEXEC --loadEngine="$E" --iterations=500 --avgRuns=100 > "$E.log" 2>&1
    echo "[jetson] $m $p trtexec: $(grep -m1 -o 'GPU Compute Time:.*mean = [0-9.]* ms' "$E.log")"
    node "$m" "$O" "$E" "runs_jetson/${D}_${m}_$p"
  done
  [ "$m" = epfid ] && [ -f "${O}_fp16.engine" ] && node "$m" "$O" "${O}_fp16.engine" "runs_jetson/${D}_epfid_fp16_nodepth" --no-depth
done
echo "[jetson] results in runs_jetson/ ; copy back: runs_jetson/ and rgbd_deploy/*.engine.log"
