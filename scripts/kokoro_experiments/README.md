# Kokoro Experiment Scripts

These scripts are retained for graph surgery, equivalence checks, and one-off
TensorRT investigations. They are not part of the product startup path.

Product-facing Kokoro entry points remain in `scripts/`:

- `build_kokoro_split_generator_trt.sh`
- `build_kokoro_long_bucket_trt.sh`
- `probe_kokoro_long_bucket.py`
- `verify_tts_asr_roundtrip.py`

When an experiment becomes a required build step, move it back to `scripts/`
and reference it from the relevant profile or build script.
