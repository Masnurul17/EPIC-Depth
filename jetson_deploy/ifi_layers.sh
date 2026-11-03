#!/usr/bin/env bash
# ifi_layers.sh (Jetson AGX): which part of the IFI decoder breaks the FP16 engine of APGCC + depth?
# Builds FP16 engines in which one group of layers is forced to FP32 (trtexec --precisionConstraints=obey),
# then counts the 139 labelled frames once with each engine (rgbd_node.py) and prints the MAE.
#   all16  : plain FP16 (reference failure, MAE ~7.84)          all32 : plain FP32 (reference, MAE ~4.69)
#   dec32  : decoder FP32, encoder + depth branch FP16          enc32 : decoder FP16, everything else FP32
#   gs32   : only the GridSample layers of the decoder FP32     pe32  : only the Sin/Cos position code FP32
#   mlp32  : only the MatMul/Gemm/Conv layers of the decoder FP32
#   If dec32 passes and enc32 fails, the failure is in IFI; gs32 / pe32 / mlp32 then name the operation.
# Usage (in ~/Documents/onnx_inference_test/edge, tmux):  bash ifi_layers.sh [model]     (default apgccd)
#   MAXN first:  sudo nvpmodel -m 0 && sudo jetson_clocks
set -u
m=${1:-apgccd}
O=rgbd_deploy/${m}_rgbd_480x640
W=runs_ifi/$m; mkdir -p $W logs_rgbd
TRTEXEC=$(command -v trtexec || echo /usr/src/tensorrt/bin/trtexec)
PY=${PY:-python3}
[ -f "$O.onnx" ] || { echo "no $O.onnx"; exit 1; }

# ---- 1. split the ONNX nodes into groups (names = TensorRT layer names) ----
$PY - "$O.onnx" "$W" <<'EOF'
import sys, onnx, collections
p, W = sys.argv[1], sys.argv[2]
mdl = onnx.load(p)
try:
    mdl = onnx.shape_inference.infer_shapes(mdl)
except Exception as e:
    print("[ifi] shape inference failed:", e)
g = mdl.graph
# element type of every tensor; only layers whose outputs are all float may be forced to FP32
# (TensorRT refuses a precision on layers that compute indices or shapes)
ty = {v.name: v.type.tensor_type.elem_type for v in list(g.value_info) + list(g.input) + list(g.output)}
ty.update({t.name: t.data_type for t in g.initializer})
FLT = {onnx.TensorProto.FLOAT, onnx.TensorProto.FLOAT16}
SKIP = {"Shape", "Constant", "ConstantOfShape", "Range", "Cast", "Equal", "Greater", "Less", "Not",
        "And", "Or", "NonZero", "Where", "Size"}
allf = [n for n in g.node if n.name and n.op_type not in SKIP]
nodes = [n for n in allf if n.output and all(ty.get(o) in FLT for o in n.output)]
print(f"[ifi] {len(nodes)} float layers kept, {len(allf) - len(nodes)} index/shape/unknown-type layers left free")
pref = collections.Counter(n.name.strip("/").split("/")[0] for n in nodes)
print("[ifi] top-level prefixes:", dict(pref.most_common(8)))
dec = [n for n in nodes if "decoder" in n.name.lower()]
if not dec:
    sys.exit("[ifi] no node name contains 'decoder': set the prefix by hand (see the prefixes above)")
rest = [n for n in nodes if n not in dec]
grp = {"dec32": dec, "enc32": rest,
       "gs32": [n for n in dec if n.op_type == "GridSample"],
       "pe32": [n for n in dec if n.op_type in ("Sin", "Cos")],
       "mlp32": [n for n in dec if n.op_type in ("MatMul", "Gemm", "Conv")]}
for k, v in grp.items():
    open(f"{W}/{k}.txt", "w").write(",".join(f"{n.name}:fp32" for n in v))
    print(f"[ifi] {k:6s} {len(v):4d} layers  ops {dict(collections.Counter(n.op_type for n in v).most_common(6))}")
EOF
[ -f $W/dec32.txt ] || exit 1

# ---- 2. build ----
build() {   # name  extra-flags...
  local k=$1; shift; local E=$W/$k.engine
  [ -f $E ] && return
  echo "[ifi] building $k"
  $TRTEXEC --onnx="$O.onnx" "$@" ${TRT_EXTRA:-} --saveEngine=$E > $W/build_$k.log 2>&1 \
    || { echo "[ifi] BUILD FAILED $k"; grep -iE "error|unsupported" $W/build_$k.log | tail -4; }
}
build all16 --fp16
build all32 --noTF32
for k in dec32 enc32 gs32 pe32 mlp32; do
  L=$(cat $W/$k.txt); [ -n "$L" ] || { echo "[ifi] $k: no layers, skipped"; continue; }
  build $k --fp16 --noTF32 --precisionConstraints=obey --layerPrecisions="$L" --layerOutputTypes="$L"
done

# ---- 3. count the 139 frames once per engine ----
for k in all32 all16 dec32 enc32 gs32 pe32 mlp32; do
  E=$W/$k.engine; R=$W/run_$k
  [ -f $E ] || continue
  [ -f $R/summary.json ] || $PY rgbd_node.py --model "$O.onnx" --engine $E --backend trtc --source pairs \
      --input rs_test_gt/frames --max-frames 139 --power none --out $R > $W/node_$k.log 2>&1 \
      || { echo "[ifi] NODE FAILED $k"; tail -3 $W/node_$k.log; continue; }
  [ -f $E.log ] || $TRTEXEC --loadEngine=$E --iterations=300 --avgRuns=100 > $E.log 2>&1
  $PY - $R/summary.json $k "$(grep -m1 -o 'mean = [0-9.]* ms' $E.log | head -1)" <<'EOF'
import json, sys
s = json.load(open(sys.argv[1])); a = s["accuracy"]
mae = a.get("MAE", a.get("all", {}).get("mae"))
print(f"[ifi] {sys.argv[2]:6s} MAE {mae:.3f} (n {a.get('n', '?')}) | GPU {sys.argv[3]}")
EOF
done
echo "[ifi] send back: the [ifi] lines above (and $W/build_*.log tails if a build failed)"
