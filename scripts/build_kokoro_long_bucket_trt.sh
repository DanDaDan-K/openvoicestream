#!/usr/bin/env bash
set -euo pipefail

cd "${SPEECH_DIR:-/opt/speech}"

KOKORO_MODEL_DIR="${KOKORO_LONG_MODEL_DIR:-/opt/models/kokoro-multi-lang-v1_0}"
KOKORO_OUT_DIR="${KOKORO_LONG_OUT_DIR:-${KOKORO_MODEL_DIR}/engines}"

for engine in \
  kokoro_decoder_backbone_dyn256_512_fp16.engine \
  kokoro_generator_source_dyn512_1024_bf16.engine \
  kokoro_generator_rest_preexp_dyn256_512_fp16.engine
do
  echo "==== build ${engine} ===="
  date
  ENGINE_NAME="${engine}" \
    MODEL_DIR="${KOKORO_MODEL_DIR}" \
    OUT_DIR="${KOKORO_OUT_DIR}" \
    ./scripts/build_kokoro_split_generator_trt.sh
done

date
echo "DONE"
