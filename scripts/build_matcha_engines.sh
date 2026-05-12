#!/usr/bin/env bash
# Build Matcha TTS TRT engines (encoder + estimator + vocos) on Jetson.
# TensorRT 10.3, Ampere SM 8.7. Single workspace size per device tier.
#
# Per-device workspace (override via env):
#   Nano 8GB  (default): WS=512    (encoder/vocos), ENC_WS keeps encoder, EST_WS=1024 for estimator if RAM allows
#   NX 16GB           :  WS=2048
#   AGX 32GB          :  WS=4096
#
# Precision is fixed by the original Matcha tuning: encoder/estimator BF16
# (QK^T overflow guard) with attention/Softmax/LayerNorm/InstanceNorm forced
# to FP32 via --precisionConstraints=obey; vocos FP16.
# Shape ranges are fixed too — do not change.
#
# Output filenames match the defaults expected by
# app/backends/jetson/matcha_trt.py:
#   encoder      → matcha_encoder_s64_bf16.engine
#   estimator    → matcha_estimator_n1_m600_bf16.engine
#   vocos        → vocos_fp16.engine            (renamed from vocos_m600_fp16.engine
#                                               to match VOCOS_ENGINE default)
#
# Usage:
#   bash scripts/build_matcha_engines.sh
#   WS=2048 ONNX_DIR=/opt/models/matcha-icefall-zh-en/onnx \
#     OUT_DIR=/opt/models/matcha-icefall-zh-en/engines \
#     bash scripts/build_matcha_engines.sh

set -euo pipefail

ONNX_DIR="${ONNX_DIR:-/opt/models/matcha-icefall-zh-en/onnx}"
OUT_DIR="${OUT_DIR:-/opt/models/matcha-icefall-zh-en/engines}"
WS="${WS:-512}"
EST_WS="${EST_WS:-${WS}}"

TRTEXEC="${TRTEXEC:-/usr/src/tensorrt/bin/trtexec}"

ENCODER_NAME="${ENCODER_NAME:-matcha_encoder_s64_bf16.engine}"
ESTIMATOR_NAME="${ESTIMATOR_NAME:-matcha_estimator_n1_m600_bf16.engine}"
VOCOS_NAME="${VOCOS_NAME:-vocos_fp16.engine}"

mkdir -p "${OUT_DIR}"

echo "=== Matcha TRT Engine Build ==="
echo "ONNX dir:   ${ONNX_DIR}"
echo "Engine dir: ${OUT_DIR}"
echo "Workspace:  ${WS} MiB  (estimator: ${EST_WS} MiB)"
echo ""

# ── Encoder (BF16 + FP32 attention obey) ──────────────────────────────────
ENC_OUT="${OUT_DIR}/${ENCODER_NAME}"
echo "=== Building encoder → ${ENC_OUT}"
"${TRTEXEC}" \
  --onnx="${ONNX_DIR}/matcha_encoder_s64_trt.onnx" \
  --saveEngine="${ENC_OUT}" \
  --bf16 --stronglyTyped --precisionConstraints=obey \
  --minShapes=x:1x1,x_length:1,noise_scale:1,length_scale:1 \
  --optShapes=x:1x64,x_length:1,noise_scale:1,length_scale:1 \
  --maxShapes=x:1x64,x_length:1,noise_scale:1,length_scale:1 \
  --memPoolSize=workspace:"${WS}" \
  2>&1 | tee "${OUT_DIR}/encoder_build.log"
ls -lh "${ENC_OUT}"
md5sum "${ENC_OUT}"
echo ""

# ── Estimator (BF16 + FP32 attention obey, N=1 frozen) ────────────────────
EST_OUT="${OUT_DIR}/${ESTIMATOR_NAME}"
echo "=== Building estimator → ${EST_OUT}"
"${TRTEXEC}" \
  --onnx="${ONNX_DIR}/matcha_estimator_n1_m600_trt.onnx" \
  --saveEngine="${EST_OUT}" \
  --bf16 --stronglyTyped --precisionConstraints=obey \
  --minShapes=z:1x80x72,mu:1x80x72,mask:1x1x72,noise_like:1x80x72 \
  --optShapes=z:1x80x256,mu:1x80x256,mask:1x1x256,noise_like:1x80x256 \
  --maxShapes=z:1x80x600,mu:1x80x600,mask:1x1x600,noise_like:1x80x600 \
  --memPoolSize=workspace:"${EST_WS}" \
  2>&1 | tee "${OUT_DIR}/estimator_build.log"
ls -lh "${EST_OUT}"
md5sum "${EST_OUT}"
echo ""

# ── Vocos (FP16) ──────────────────────────────────────────────────────────
VOCOS_OUT="${OUT_DIR}/${VOCOS_NAME}"
echo "=== Building vocos → ${VOCOS_OUT}"
"${TRTEXEC}" \
  --onnx="${ONNX_DIR}/vocos_m600_trt.onnx" \
  --saveEngine="${VOCOS_OUT}" \
  --fp16 --stronglyTyped \
  --minShapes=mels:1x80x72 \
  --optShapes=mels:1x80x256 \
  --maxShapes=mels:1x80x600 \
  --memPoolSize=workspace:"${WS}" \
  2>&1 | tee "${OUT_DIR}/vocos_build.log"
ls -lh "${VOCOS_OUT}"
md5sum "${VOCOS_OUT}"
echo ""

echo "=== All Matcha engines built ==="
