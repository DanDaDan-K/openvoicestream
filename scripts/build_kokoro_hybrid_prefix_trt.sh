#!/usr/bin/env bash
set -euo pipefail

# Build the TensorRT-friendly Kokoro encoder prefix for the CPU/TRT hybrid path.
#
# Prerequisites:
#   1. Kokoro sherpa ONNX package is present at MODEL_DIR.
#   2. scripts/split_kokoro_hybrid.py has access to onnx in the active Python env.
#   3. This runs on the target Jetson/Orin host or in a Jetson container with
#      host TensorRT mounted at /usr/src/tensorrt.
#
# The full sherpa Kokoro ONNX cannot be built as one TensorRT engine on TRT 10.3
# because the tail graph contains unsupported STFT/dynamic-shape regions. This
# script only builds the validated prefix and leaves the suffix on CPU ORT.

MODEL_DIR="${MODEL_DIR:-/opt/models/kokoro-multi-lang-v1_0}"
OUT_DIR="${OUT_DIR:-${MODEL_DIR}/hybrid}"
TRTEXEC="${TRTEXEC:-/usr/src/tensorrt/bin/trtexec}"
PYTHON="${PYTHON:-python3}"
CUT_OUTPUT="${CUT_OUTPUT:-/encoder/Cast_2_output_0}"
ENGINE_NAME="${ENGINE_NAME:-kokoro_prefix_encoder_dyn4_128_fp16.engine}"

mkdir -p "${OUT_DIR}"

"${PYTHON}" scripts/split_kokoro_hybrid.py \
  --model "${MODEL_DIR}/model.onnx" \
  --out-dir "${OUT_DIR}" \
  --cut-output "${CUT_OUTPUT}"

"${TRTEXEC}" \
  --onnx="${OUT_DIR}/kokoro_prefix_encoder.onnx" \
  --saveEngine="${OUT_DIR}/${ENGINE_NAME}" \
  --fp16 \
  --minShapes=tokens:1x4,style:1x256,speed:1 \
  --optShapes=tokens:1x64,style:1x256,speed:1 \
  --maxShapes=tokens:1x128,style:1x256,speed:1 \
  --memPoolSize=workspace:512
