# =============================================================================
# trt_build_engine_v2.py  (v2 of trt_build_engine.py, 2026-10-18; desktop / server GPU)
#
# v2: pip installed TensorRT 11.4 on the server; TensorRT 11 removed
#     BuilderFlag.FP16 (all networks strongly typed, FP16 only through an
#     FP16 ONNX, e.g. ModelOpt AutoCast) and its Myelin backend failed on the
#     EPFI ONNX (MyelinCheckException, logical memory size). The Jetsons use
#     TensorRT 10.3 / 8.5 with the FP16 builder flag (= trtexec --fp16), so the
#     server must use TensorRT 10.x (>= 10.8 for Blackwell, < 11) to compare
#     like with like. v2 refuses TensorRT >= 11 with a clear message and
#     writes the TensorRT version into a <engine>.txt next to the engine.
#
# Builds a TensorRT engine from an ONNX with the TensorRT Python API, for
# machines where TensorRT comes from pip (`tensorrt-cu12`) and trtexec is not
# installed. Same settings as the trtexec builds on the Jetsons: static
# shape of the ONNX (1 x 3 x 512 x 512), FP32 or --fp16 (BuilderFlag.FP16,
# I/O tensors stay FP32), workspace limit --ws MiB. The engine is tied to
# this GPU + TensorRT version; nothing is overwritten (an existing --out is
# refused). Blackwell (RTX 50xx, sm_120) needs TensorRT >= 10.8, driver >= R570.
#   python trt_build_engine_v2.py --onnx deploy/p2pnet_sha_512.onnx --out deploy_rtx5090/p2pnet_sha_512_fp16.engine --fp16
# =============================================================================

import os
import time
import argparse

import tensorrt as trt

_ap = argparse.ArgumentParser()
_ap.add_argument("--onnx", required=True)
_ap.add_argument("--out", required=True, help="engine file (must not exist)")
_ap.add_argument("--fp16", action="store_true")
_ap.add_argument("--ws", type=int, default=4096, help="workspace limit (MiB)")
_args = _ap.parse_args()


def main():
    if os.path.exists(_args.out):
        raise SystemExit(f"{_args.out} exists: kept (nothing is overwritten)")
    major = int(trt.__version__.split(".")[0])
    if major >= 11:
        raise SystemExit(f"tensorrt {trt.__version__}: use TensorRT 10.x (pip install \"tensorrt-cu12>=10.8,<11\"); "
                         "11.x has no BuilderFlag.FP16 (strongly typed only), not comparable with the Jetson builds")
    os.makedirs(os.path.dirname(os.path.abspath(_args.out)), exist_ok=True)
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(0)              # TensorRT 10: explicit batch is the only mode
    parser = trt.OnnxParser(network, logger)
    if not parser.parse_from_file(_args.onnx):
        for i in range(parser.num_errors):
            print("[build] parser:", parser.get_error(i))
        raise SystemExit(f"cannot parse {_args.onnx}")
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, _args.ws << 20)
    if _args.fp16:
        config.set_flag(trt.BuilderFlag.FP16)
    t0 = time.time()
    ser = builder.build_serialized_network(network, config)
    if ser is None:
        raise SystemExit("build FAILED (see TensorRT messages above)")
    with open(_args.out, "wb") as f:
        f.write(ser)
    with open(_args.out + ".txt", "w") as f:                        # v2: provenance
        f.write(f"tensorrt {trt.__version__}\nonnx {os.path.abspath(_args.onnx)}\n"
                f"precision {'fp16' if _args.fp16 else 'fp32'}\nworkspace_MiB {_args.ws}\n")
    print(f"[build] tensorrt {trt.__version__} | {os.path.basename(_args.onnx)} | "
          f"{'fp16' if _args.fp16 else 'fp32'} | {time.time() - t0:.0f} s -> {_args.out} "
          f"({os.path.getsize(_args.out) / 2**20:.1f} MiB)")


if __name__ == "__main__":
    main()
