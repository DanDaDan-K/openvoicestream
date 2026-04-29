# Handover 2026-04-29 — TTS OOM Mitigation Done; INT8 Talker Next

## TL;DR
Multilanguage mode (ASR + TTS) on Orin Nano 8GB **almost works**:
- All vocab pruning + load reorder + CUDA Graph gates committed (`c270b18`, `3cb6cab`, `eca4bf8`)
- Pipeline loads, first `Synth` starts
- Still OOM at first synthesize call (vocoder lazy ctx + Talker first inference need ~250-400 MB more)

**Last gap can only be closed by Talker weight quantization** (BF16 870 MB → INT8 ~440 MB or INT4 ~220 MB). Path A (TRT INT8 W8A16) chosen for next test.

---

## Session 2026-04-29 Outcomes

### Commits on main
| Commit | Title | Validates |
|---|---|---|
| `c270b18` | feat(asr): vocab pruning Phase C — Python indirection + 128-aligned lm_head | ASR pruning shipped, ~389 MB saved |
| `3cb6cab` | feat(tts): vocab pruning C++ patches — text_embed indirection | TTS pruning shipped, runtime loads 139MB instead of 593MB |
| `eca4bf8` | perf(tts,asr): OOM mitigation suite for Orin Nano 8GB multilanguage | Pipeline loads + first synth starts; vocoder lazy ctx OOM |

### Validated memory savings (from MEM instrumentation)
- ASR encoder TRT native (instead of ORT CUDA EP): ~150-200 MB
- ASR decoder no CUDA Graph: ~100 MB
- Talker no CUDA Graph: ~150-300 MB at decode time
- Vocoder warmup defer (env-gated): ~150-300 MB at vocoder load
- Vocoder defer_context: ~200-300 MB at vocoder load
- Talker_init reorder (run when ~1 GB free, not 80 MB): unblocks talker_init OOM
- cp_embed CPU/GPU split: avoids 360 MB peak, frees 415 MB for cp_kv
- CP_GRAPH_WARMUP=none: ~80 MB at cp_kv load
- ASR vocab pruning embed + lm_head: ~389 MB
- TTS vocab pruning text_embed: ~454 MB
- **Total cumulative: ~1.6 GB freed vs original**

### Last MEM sequence (with ALL fixes)
```
asr_start=6022 MB
after_encoder=5414 MB (-608 ASR encoder TRT)
after_embed=5342 MB (-72)
after_decoder=3088 MB (-2254 ASR decoder TRT, no CUDA Graph)
asr_ready=2997 MB
tts_start=2975 MB
tts_after_tokenizer=2989 MB
before_ort_load=2918 MB
after_ort_load=2629 MB (-289 ORT runtime + pruned text_embed 139MB)
before_talker_weights=2629 MB
after_talker_weights=1083 MB (-1546 Talker BF16, streamed)
before_talker_init=1083 MB
after_talker_init=625 MB (-458 Talker exec context, dual-profile)
before_cp_embed_cpu=625 MB
after_cp_embed_cpu=460 MB (-165 CPU 120MB bin + Python overhead)
before_vocoder_weights=460 MB
after_vocoder_weights=172 MB (-288 vocoder weights, context deferred)
before_cp_kv_load=172 MB
after_cp_kv_load=47 MB (-125 CP KV pool, no warmup)
→ Pipeline ready, codec_embed loaded
→ First Synth call → OOM (vocoder context lazy create + Talker first run)
```

**Gap to close**: ~250-400 MB more headroom needed for synthesize.

---

## Required Container Env (for memory-tight setup)

```bash
docker run -d --name voice_nano_test --network host --runtime nvidia --gpus all \
    -e CP_POOL_SIZE=1 \
    -e ASR_ENCODER_BACKEND=trt_native \
    -e ASR_VOCAB_PRUNED=1 \
    -e ASR_DECODER_CUDA_GRAPH=0 \           # NEW (eca4bf8)
    -e TTS_VOCAB_PRUNED=1 \
    -e TTS_TALKER_CUDA_GRAPH=0 \            # NEW (eca4bf8)
    -e CP_GRAPH_WARMUP=none \               # CHANGED from full
    -e LAZY_TTS=1 \
    -e SKIP_ASR_WARMUP=1 \
    -e LANGUAGE_MODE=multilanguage \
    -e LD_LIBRARY_PATH=/usr/local/lib/python3.10/dist-packages/onnxruntime/capi:/host-cuda:/host-nvidia-libs:/host-libs \
    # ^ /host-cuda BEFORE /host-libs (cublas resolution fix)
    ... (volumes same as before) \
    jetson-voice-speech:v3.4-slim
```

---

## Footguns Hit This Session

1. **`/lib/aarch64-linux-gnu/libcublasLt.so.12` empty dir on Nano host** — caused container to fail loading cuBLAS via ONNX runtime. Workaround: `LD_LIBRARY_PATH` puts `/host-cuda` first. Permanent fix needs sudo: `sudo rmdir + ln -s /usr/local/cuda/lib64/libcublasLt.so.12`.
2. **`nm -D` doesn't show pybind11 class symbols on stripped Release .so** — symbols ARE there (use `strings` instead). Wasted hours debugging "missing TRTASREncoder" that wasn't actually missing.
3. **Slim runtime container has no compiler** — `.so` rebuild MUST happen on Nano host (via `/home/harvest/jetson-voice-cpp/build_cmake/` flow). In-container build fails (no cmake/g++).
4. **Docker file-over-dir mount silently dropped** — can't shadow a host directory with a file mount; must fix host-side.
5. **`fleet exec --sudo --` syntax** — requires `--` between options and command.
6. **Device-level OOM possible** — when system pushes too hard, kernel kills SSH and triggers reboot. Be careful with iterative load testing.
7. **TRT engine deserialize peak = 2-6x file size** — from runtime context + workspace overhead, not just weights. CUDA Graph capture adds another 100-300 MB.

---

## Next Session: INT8 Talker Path

### Goal
Build INT8-weight Talker engine (~440 MB instead of 870 MB), close the ~250-400 MB final OOM gap.

### Approach: TRT W8A16
- **Weights INT8** (50% smaller)
- **Activations BF16** (Qwen3 attention QK^T must stay BF16, FP16/INT8 overflow per `feedback_bf16_attention.md`)
- Use TRT `IInt8EntropyCalibrator2` with calibration data (~500 representative text→audio samples)
- Per-channel quantization (not per-tensor, less error)

### Steps
1. **Calibration dataset**: 500 samples, mix zh/en/ja/ko/etc, run through original BF16 Talker to capture activations
2. **trtexec build**: `--int8 --calib=<bin> --layerPrecisions=*attention*:bf16` to keep attention BF16
3. **Smoke**: 5 prompts, INT8 vs BF16 mel L2 (target < 1%) + perceptual A/B listen
4. **Memory measurement**: confirm Talker engine ~440 MB, total Pipeline fits with ~200 MB margin
5. **End-to-end TTS smoke**: assert http=200 + audio > 5KB

### Risks
- Voice clone quality may degrade more than basic synthesis (fine-grained features)
- Rare phonemes (multilingual edge cases) sensitive to quant
- Per-step latency: +10-20 ms (Jetson INT8 reformat overhead, per `feedback_jetson_int8_small_batch.md`)

### Fallback
- INT4 GGUF + llama.cpp (per `khimaros/qwen3-tts.cpp`) — mentioned in handover-2026-04-28 as last resort
- Or relegate Nano to ASR-only, multilanguage TTS only on NX/AGX

---

## Artifacts (current state)

### Mac repo
- All commits on `main` (latest `eca4bf8`)
- Spec docs: `vocab-pruning-2026-04-28.md`, `vocab-pruning-phase-c-2026-04-28.md`, `tts-vocab-pruning-2026-04-28.md`, `talker-streamreader-2026-04-29.md`, this handover
- Tests: `tests/test_vocab_pruning_indirection.py`

### Nano `/home/harvest/voice_test/`
- `models/qwen3-asr-v2/`: padded BF16 engine, FP32 fallback, pruned bin, token_map, sidecar
- `models/qwen3-tts/onnx/`: pruned text_embed_fp16_pruned.bin, token_map.bin
- `app_overlay/`: deployed .so md5 `cb3231426e538ae4c7639370ff58bf88` (with all C++ fixes)
- Backups: `qwen3_speech_engine.so.bak.*` (multiple safe restore points)
- Build dir: `/home/harvest/voice_test/src/cpp_fresh/build/` (last successful host build)

### WSL `/home/harve/qwen3-vocab-pruning/`
- All offline pruning artifacts (not changed this session)

---

## Status Snapshot (end of session)

| Capability | Mode | Status |
|---|---|---|
| ASR multilanguage | Single | ✅ Works (TRT native encoder + padded BF16 decoder + vocab pruning) |
| TTS multilanguage | Single | ❌ Pipeline loads, synth OOM. Need INT8 Talker. |
| ASR + TTS coexist | Multilanguage | ❌ Same as TTS |
| ASR-only mode | Various | ✅ Plenty of headroom |
| ASR + TTS coexist | Single-lang (matcha-icefall) | ✅ Works (different smaller models) |
