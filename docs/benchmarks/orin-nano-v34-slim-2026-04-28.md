# Orin Nano 8GB — Qwen3 ASR+TTS feasibility test

**Date**: 2026-04-28
**Image**: `jetson-voice-speech:v3.4-slim` (sha256:99f9493f...)
**Device**: orin-nano (tailscale 100.92.125.65), JetPack 6.x, 7.4GB iGPU shared RAM

## Verdict

❌ **Qwen3 ASR + TTS does NOT fit on Orin Nano 8GB.**

ASR loads fine, TTS load OOM-kills the container (`OOMKilled=true`, exit 137).

## What was tried

| Run | Config | Result |
|---|---|---|
| 1 | default (`CP_POOL_SIZE=2`, `TTS_NUM_THREADS=4`) | OOM during TTS CP pool slot 1 creation |
| 2 | `CP_POOL_SIZE=1`, `TTS_NUM_THREADS=2`, `STREAMING_ASR_NUM_THREADS=2` | OOM right after vocoder load (before warmup) |

Both runs:
- ASR loaded successfully (15-18s, warmup 36-37s)
- ASR offline transcribe verified working (1s audio → 2 tokens)
- TTS loaded talker engine, CP KV engine, vocoder engine
- Container killed by host OOM before TTS warmup completed

## Verified ASR perf on Nano (single transcribe, 1s audio)

| Stage | Nano | Orin NX (reference) |
|---|---:|---:|
| encoder | 46.3 ms | ~37 ms |
| prefill | 57.3 ms | ~40 ms |
| decode | 104.6 ms | ~176 ms |
| d2h | 0.66 ms | ~2 ms |

Decode faster than NX is misleading — only 2 tokens emitted (warmup utterance). For real workloads expect Nano to be 1.3-1.5× slower per stage due to lower SM count.

## Cross-device TRT engine

Engines built on Orin NX deserialize on Nano with `[TRT-WARN] Using an engine plan file across different models of devices` but **do work** (no rebuild required). Ampere SM 8.7 cubin compatibility holds across NX/Nano variants.

## Memory budget (estimate, peak)

- Qwen3 ASR: ~1.5 GB (encoder ORT + decoder TRT + workspace)
- Qwen3 TTS: ~3.5+ GB (talker BF16 + CP KV + vocoder + workspace + activations during warmup)
- Container/system overhead: ~1 GB
- **Estimated peak: ~6 GB** in 7.4 GB available shared iGPU memory → tight, gets killed when warmup activations push past

## Options for future

| Option | Note |
|---|---|
| `LANGUAGE_MODE=zh_en` (paraformer + matcha) | Untested but should fit easily (~1.5GB total models, both small) — recommend as Nano default |
| Drop ASR-only or TTS-only mode | Would need code change in `main.py` to skip pre-load of one |
| Quantize Qwen3 TTS to INT8 | Significant work, may hurt quality |
| Recommend Orin NX 16GB minimum for Qwen3 dual mode | Realistic conclusion |

## Files

- Models on Nano: `/home/harvest/voice_test/models/{qwen3-asr-v2,qwen3-tts}` (15.4 GB) — keep for future tests
- Overlay on Nano: `/home/harvest/voice_test/app_overlay/` (2.8 MB)
- Image: `jetson-voice-speech:v3.4-slim` (1.49 GB) — already cached, registry pull also works

Cleanup done: failed container removed; models + overlay kept for `LANGUAGE_MODE=zh_en` retest.
