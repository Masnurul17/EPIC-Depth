# Deploy EPIC-Depth ke Jetson AGX Orin dan Orin Nano

## 1. Di server (folder `APGCC/apgcc`, conda `floodseg`)

Ekspor ONNX (satu forward 480×640; input BGR mentah + depth mentah dalam mm; opset 16):

```bash
# TAG = tag checkpoint yang dipakai di makalah (lihat "ckpt" EPIC-Depth di models_rs_gpu.json)
SYN="OUT=/home/mna/mna-hdd/PointDataRGBDSyn_s050 HO=ho10 TAG=<TAG> DSNAME=SHTRGBDSYN EPOCHS=40 LR_DROP=30"
env $SYN bash run_rgbd_B_1024.sh export full      # EPIC-Depth: rewrite NCHW + LayerNorm aman FP16 (epfid_deploy.py)
env $SYN bash run_rgbd_B_1024.sh export apgccd    # pembanding VGG (tanpa rewrite)
env $SYN bash run_rgbd_B_1024.sh export p2pd
```

Hasil: `deploy/full_rgbd_480x640.onnx`, `deploy/apgccd_...`, `deploy/p2pd_...` (+ `.json`).
`run_jetson_rgbd.sh` mencari `epfid_rgbd_480x640.onnx`, jadi ganti nama `full_` menjadi `epfid_`
(file .onnx dan .json), lalu salin ke folder `rgbd_deploy/` di Jetson.
Ekspor sudah mengecek ONNX terhadap PyTorch pada 10 frame.

## 2. Isi folder kerja di Jetson

```
edge/
  rgbd_node.py  rgbd_depth_utils.py  field_node_v7.py  run_jetson_rgbd.sh  rgbd_table5.py  table5_extra_rgb.csv
  rgbd_deploy/epfid_rgbd_480x640.onnx (+ .json), apgccd_..., p2pd_..., iadm_...
  rs_test_gt/frames/*.png (+ GT .txt)   rs_test_gt/depth/*.png
```

Salin dari server, misalnya:

```bash
scp -r jetson_deploy/* nvidia-agx@100.120.140.102:~/Documents/onnx_inference_test/edge/
scp -r rgbd_deploy field_pairs/rs_test_gt nvidia-agx@100.120.140.102:~/Documents/onnx_inference_test/edge/
```

Python yang dipakai: `/usr/bin/python3` (JetPack: `tensorrt`, NumPy, OpenCV). Bukan conda.

## 3. AGX Orin (MAXN, TensorRT 10.3)

```bash
sudo nvpmodel -m 0 && sudo jetson_clocks
bash run_jetson_rgbd.sh agx                 # semua model di rgbd_deploy/
bash run_jetson_rgbd.sh agx epfid           # hanya EPIC-Depth
/usr/bin/python3 rgbd_table5.py --runs runs_jetson/* --trt-dir rgbd_deploy --out runs_jetson/table5
```

Per model: engine FP16 dan FP32 (`--noTF32`), waktu GPU trtexec, lalu `rgbd_node.py` memutar 139 frame berlabel
selama 1 menit setelah 20 s daya idle. EPIC-Depth juga dijalankan dengan `--no-depth` (engine FP16 yang sama).

## 4. Orin Nano (mode 15 W, TensorRT 8.5)

```bash
sudo nvpmodel -q                             # cek nomor mode 15 W, lalu: sudo nvpmodel -m <nomor>
sudo jetson_clocks
TRT_EXTRA="--tacticSources=-CUBLAS,-CUBLAS_LT,-CUDNN" bash run_jetson_rgbd.sh nano
```

`TRT_EXTRA` wajib di TRT 8.5: tanpa itu build engine gagal.

## 5. Uji IFI per lapisan (APGCC + depth, FP16)

```bash
PY=/usr/bin/python3 bash ifi_layers.sh apgccd     # perlu: pip install --user onnx
```

## 6. Ambil hasil

Salin `runs_jetson/` dan `rgbd_deploy/*.engine.log` kembali ke server.

Bobot (`best_ema.pth`), ONNX, dan engine tidak ada di paket ini: semuanya ada di server
(`output_SHTRGBD/motiny_epfid_full_c512_ho20_ema_<TAG>/`). Engine harus dibangun ulang di tiap perangkat.
