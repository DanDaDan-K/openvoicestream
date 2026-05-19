#!/usr/bin/env bash
# Build Kokoro v1.0 TensorRT engine on Jetson Orin.
#
# Status: this is currently a reproducer/debug tool. The upstream sherpa
# Kokoro v1.0 ONNX contains an STFT node that TensorRT 10.3 cannot parse as a
# whole graph. A successful product path needs graph surgery or a custom STFT
# plugin, not a direct full-graph trtexec build.
#
# The sherpa/kokoro ONNX model uses:
#   input_ids: int64   [1, N]      with BOS/EOS pad token 0
#   style:     float32 [1, 256]    selected from voices.bin by token count
#   speed:     float32 [1]
#
# Build on the target Jetson, not WSL. WSL/x86 can prepare/export ONNX, but
# TensorRT engines are tied to GPU arch + TensorRT/CUDA versions.

set -euo pipefail

MODEL_DIR="${MODEL_DIR:-/opt/models/kokoro-multi-lang-v1_0}"
ONNX_PATH="${ONNX_PATH:-${ONNX:-${MODEL_DIR}/model.onnx}}"
OUT_DIR="${OUT_DIR:-${MODEL_DIR}/engines}"
ENGINE_NAME="${ENGINE_NAME:-kokoro_fp16.engine}"
WS="${WS:-512}"
TRTEXEC="${TRTEXEC:-/usr/src/tensorrt/bin/trtexec}"

MIN_SEQ="${MIN_SEQ:-4}"
OPT_SEQ="${OPT_SEQ:-96}"
MAX_SEQ="${MAX_SEQ:-512}"

mkdir -p "${OUT_DIR}"

if [[ ! -x "${TRTEXEC}" ]]; then
  echo "trtexec not found or not executable: ${TRTEXEC}" >&2
  exit 1
fi
if [[ ! -f "${ONNX_PATH}" ]]; then
  echo "Kokoro ONNX not found: ${ONNX_PATH}" >&2
  exit 1
fi

OUT="${OUT_DIR}/${ENGINE_NAME}"
echo "=== Kokoro TRT Engine Build ==="
echo "ONNX:      ${ONNX_PATH}"
echo "Engine:    ${OUT}"
echo "Workspace: ${WS} MiB"
echo "Seq range: ${MIN_SEQ}/${OPT_SEQ}/${MAX_SEQ}"

"${TRTEXEC}" \
  --onnx="${ONNX_PATH}" \
  --saveEngine="${OUT}" \
  --fp16 \
  --minShapes=input_ids:1x"${MIN_SEQ}",style:1x256,speed:1 \
  --optShapes=input_ids:1x"${OPT_SEQ}",style:1x256,speed:1 \
  --maxShapes=input_ids:1x"${MAX_SEQ}",style:1x256,speed:1 \
  --memPoolSize=workspace:"${WS}" \
  2>&1 | tee "${OUT_DIR}/kokoro_build.log"

ls -lh "${OUT}"
md5sum "${OUT}"
echo "=== Kokoro engine built ==="
