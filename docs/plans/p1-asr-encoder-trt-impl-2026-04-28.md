# P1 Implementation Spec: ASR encoder ORT → TRT

Date: 2026-04-28
Goal: Move Qwen3-ASR encoder from ORT CUDA EP to native TRT to free 200-400 MB GPU RAM, enabling Orin Nano 8GB multilanguage mode.

Builds on `docs/plans/p0b-p1-memory-opt-2026-04-28.md` Section B. This doc adds concrete file:line, acceptance criteria, ordering, and prohibitions for implementation dispatch.

## Why now

P0a (embed_tokens FP16 in-place) ✅ shipped. P0b (TRT IStreamReaderV2) ✅ shipped. SKIP_ASR_WARMUP ✅ shipped. After all three, Nano 8GB still OOMs at TTS Talker engine load with sys avail 1781 MB but TTS needs ~3 GB.

Per-step instrumented data (Nano, multilanguage mode):
- ASR encoder load (ORT CUDA EP): consumes **1226 MB system RAM**
- ASR ready total: 5777 MB system used (4154 MB above baseline)
- TTS still needs ~1 GB more than available

P1 directly attacks the 1226 MB ORT encoder allocation. Goal: bring ASR ready down by 200-400 MB so TTS has room.

## Concrete file map

### Files to ADD

- `benchmark/export_asr_encoder_trt.sh` — wrapper around `trtexec` to build the TRT engine from existing `encoder_fp16.onnx`. Modeled on `scripts/build_asr_decoder_engine.sh:1`. Produce `asr_encoder_fp16.engine` next to `encoder_fp16.onnx`. Two profiles:
  - profile 0 (streaming): mel `[1,128,40-200]`, opt 100, max 200
  - profile 1 (offline): mel `[1,128,200-3000]`, opt 1000, max 3000

### Files to MODIFY

- `benchmark/cpp/tts_trt_engine.h` — add class `TRTASREncoder` declaration. **Note: cannot reuse `TRTASRPrefillEngine` (line 652) — its API takes token IDs + positions + features, encoder needs mel→features**. Only the file-scope helper `LoadEngineStreaming` (cpp line 112) is reusable. Public API (matches Section B):
  ```
  TRTASREncoder(const std::string& engine_path, int max_mel_frames, int max_out_frames, int hidden_dim = 1024);
  // returns audio_features [1, Tp, 1024]; copies to host vector
  std::vector<float> run(const float* mel_ptr, int n_frames, int& out_T);
  ```

- `benchmark/cpp/tts_trt_engine.cpp` — `TRTASREncoder` implementation. Use `LoadEngineStreaming(runtime_.get(), engine_path)`. Allocate `d_mel_` and `d_audio_features_` to max sizes. Switch optimization profile based on input size (streaming < 200 frames → profile 0, offline → profile 1).

- `benchmark/cpp/tts_binding.cpp` — find existing `PYBIND11_MODULE(qwen3_speech_engine, m) { ... }` block (around `:52-56`) and add `py::class_<TRTASREncoder>(m, "TRTASREncoder").def(py::init<const std::string&, int, int, int>()).def("run", ...)`. Without this explicit registration the Python `qwen3_speech_engine.TRTASREncoder` lookup fails.

- `app/backends/qwen3_asr.py` — refactor encoder loading at lines 821-830 to handle 3 backends with **explicit early return** (otherwise ORT path still runs and the memory benefit is lost). Pseudo:
  ```python
  encoder_backend = os.environ.get("ASR_ENCODER_BACKEND", "ort_cuda").lower()
  # Accepted values: "ort_cuda" (default, current), "ort_trt" (existing P1-Path-A),
  #                  "trt_native" (this P1: native TRT, lowest memory)
  if encoder_backend == "trt_native":
      engine_path = os.path.join(_BASE, "asr_encoder_fp16.engine")
      if os.path.exists(engine_path):
          import qwen3_speech_engine
          trt_enc = qwen3_speech_engine.TRTASREncoder(engine_path, 3000, 750, 1024)
          self._encoder = _TRTEncoderAdapter(trt_enc)
          logger.info("Encoder loaded (TRT native): %s", engine_path)
          # ↓ MUST: explicit return / skip ORT loop ↓
      else:
          logger.warning("trt_native requested but %s missing; falling back to ORT", engine_path)
          encoder_backend = "ort_cuda"
  if encoder_backend in ("ort_cuda", "ort_trt"):
      # existing ORT path (lines 821-830 unchanged)
  ```
  Plus add `_TRTEncoderAdapter` class at module level. The adapter implements `.run(output_names, feeds)` returning `[features]`. Call sites verified to pass only `{"mel": mel}` — no other feeds needed.

  **Verified call sites**:
  - `app/backends/qwen3_asr.py:898` (was 883) — warm-up
  - `app/backends/qwen3_asr.py:993` (was 976) — `_transcribe_python`
  - `app/backends/qwen3_asr.py:410` — streaming context
  All pass `{"mel": <array>}` only; adapter signature `run(output_names, feeds) → [features]` is sufficient.

### Files to NOT touch

- `app/backends/qwen3_trt.py` — TTS, not in scope
- `app/main.py` — already SKIP_ASR_WARMUP-aware
- `Dockerfile` — no new deps
- `docker-compose*.yml` — no
- `app/backends/sherpa*.py` — different mode, irrelevant

## Acceptance criteria (must verify all)

1. **Numerical diagnostic** (ORT vs TRT, **diagnostic only — does NOT gate**): on same test wav, feed identical mel to both. Report `max_abs` and `mean_abs`. Reference benchmark only — silent quality regression is caught by criterion 2 below, not numeric tolerance.

2. **Transcript parity** (**THIS GATES**): 5 wavs from `tests/asr_real_wav_eval/` (or `/home/harvest/bench/*.wav` if test set absent). End-to-end decode with TRT encoder. CER vs ORT-encoder transcript baseline ≤ 5%.

3. **Latency** (**relative gate**): per-chunk encoder latency on 0.4s streaming chunk N=20 (drop first 5 warmup):
   - TRT median ≤ ORT median × 1.20 (i.e., ≤ 20% regression)
   - Report absolute ms numbers alongside (e.g., "TRT 17.2 ms vs ORT 16.5 ms").
   - On 30s offline mel N=10: TRT ≤ ORT × 1.20.
   - Also report **D2H copy overhead** of `_TRTEncoderAdapter` (numpy round-trip) separately. If > 2 ms, flag for follow-up.

4. **Memory on NX 16GB** (gate before Nano test):
   - **Primary metric**: `tegrastats RAM` system-wide measurement before vs after restart.
     - Sample baseline (current ORT prod) for 30s, take median.
     - Restart with `ASR_ENCODER_BACKEND=trt_native`, sample after `[MEM:asr_ready]` for 30s, take median.
     - Required reduction: ≥ 200 MB system used.
   - **Secondary**: instrumented `[MEM:after_encoder]` sys MemAvailable shows ≥ 200 MB more free with TRT vs ORT.
   - Note: process RSS does NOT capture ORT CUDA EP arena (GPU-side); use tegrastats and `[MEM:*]` `sys MemAvailable` as the truth.
   - prod /health 200 + 5min stable.

5. **Memory on Nano 8GB** (final goal):
   - container with all envs + new engine: full TTS load completes (no OOM).
   - tegrastats peak < 7400 MB (≥ 200 MB margin from 7620 ceiling).
   - /health 200.
   - one ASR + one TTS smoke request both succeed end-to-end.

6. **Profile choice** (decide based on measurement, not assumption):
   - Build single profile (mel `[1,128,40-3000]`, opt 200) AND dual profile variant.
   - Benchmark streaming chunk latency on both. If single ≤ dual + 1ms, use single.
   - Otherwise use dual. Reason: single profile is simpler, may be enough.

7. **Phase A shape/dtype smoke** (early sanity):
   - Adapter `.run(["audio_features"], {"mel": np.zeros((1,128,40), np.float32)})` returns `[arr]` with shape `(1, T, 1024)`, dtype float32, finite values.
   - Same with `(1,128,3000)`.

## 禁区 + dispatch 护栏

**🚨 ANTI-RECURSION (for deepseek-pro implementer)**:
- 你 **就是** rescue worker 本人，**不要** 再 dispatch 任何 claude-rescue / opencode-rescue / codex-rescue 子任务。所有命令你自己用 Bash + Read + Edit 工具直接跑。
- 你 **不要发明新的 build 流程**。本仓库唯一合法 build 入口：
  - C++ build: `benchmark/cpp/build.sh` (在 NX 上跑)
  - TRT engine build: `scripts/build_asr_decoder_engine.sh` 模板（你新写一个 `scripts/build_asr_encoder_engine.sh` 镜像它的结构）
- 你 **不要自己改 trtexec 命令**自创 minShape/maxShape/optShape 的组合，照 spec 写。

**禁止操作**:
- ❌ Do not break existing ORT path (must remain default; trt_native is opt-in via `ASR_ENCODER_BACKEND=trt_native`)
- ❌ Do not touch TTS code (qwen3_trt.py / qwen3_speech_engine TTS classes)
- ❌ Do not touch docker-compose / Dockerfile
- ❌ Do not stop NX prod for >5 min (use compose `restart speech` not `up -d --force-recreate`); **trtexec build on NX 必须先 `docker compose stop speech`** (memory: prod 自动 restart 会抢 RAM 把 trtexec 打死)，build 完立即 `docker compose start speech`
- ❌ Do not edit code on remote (orin-nx / orin-nano) — local Mac edit first, scp sync (per memory `feedback_local_first_then_sync.md`)
- ❌ Do not commit / push without main thread review
- ❌ Do not use `--no-verify` git
- ❌ Do not delete `.bak.before_p0b` or any other backup files

## Implementation order (sub-phases)

Phase A (1-2h): Build engine + verify numerical parity

A.1 Write `benchmark/export_asr_encoder_trt.sh` locally.
A.2 scp to NX, run on NX (prod stopped for build to free RAM, like ASR decoder build).
A.3 Write standalone Python parity test in `tests/test_p1_encoder_parity.py`: load both ORT and new TRT (via TRT Python API directly, no C++ binding needed yet), feed same mel, assert thresholds.
A.4 If parity fails, abort and report which op disagrees (codex spec flagged dynamic slicing risk).

Phase B (3-5h): C++ binding

B.1 Add `TRTASREncoder` class to .h + .cpp on Mac.
B.2 Add pybind11 wrapper.
B.3 scp to NX, build .so via existing `benchmark/cpp/build.sh`.
B.4 Smoke: pure Python script imports new class, feeds mel, gets output, compares vs Phase A standalone.

Phase C (1-2h): Integrate into qwen3_asr.py

C.1 Add `_TRTEncoderAdapter` + branch on `ASR_ENCODER_BACKEND=trt_native` at line 821.
C.2 Sync to NX overlay, restart prod with `ASR_ENCODER_BACKEND=trt_native`, verify [MEM:*] checkpoints.
C.3 Run end-to-end ASR test (transcript + latency).
C.4 If MEM saving < 150 MB on NX, abort — won't help Nano either.

Phase D (1h): Nano integration

D.1 scp engine + .so + .py to Nano.
D.2 Run with all envs + trt_native, instrumentation on, tegrastats running.
D.3 If passes /health → smoke ASR+TTS → write report `docs/benchmarks/p1-nano-2026-04-28.md`.
D.4 If still OOM → write report explaining how close, recommend abandon Nano multilanguage.

## Risks ranked

1. **TRT can't ingest encoder ONNX**: dynamic slicing / reshape patterns Codex flagged. Likely catch: padding ops or attention mask creation may use ops TRT doesn't support fully. Mitigation: try `trtexec --verbose`; if fail, may need ONNX surgery (extra 1-2d).
2. **FP16 numerical drift**: encoder is FP16 already in ORT, TRT FP16 should match. If not, fallback to FP32 (larger engine but trustworthy).
3. **Streaming dispatch cost**: profile 0/1 switch overhead. Mitigation: measure both single wide profile and split; pick lower latency.
4. **Build OOM on NX**: ASR decoder build needed prod stopped. Likely same here. Already learned pattern.
5. **End-to-end CER regression**: TRT FP16 may shift attention precision marginally. Acceptance threshold 5% allows small drift.

## Wall-time estimate

**Conditional on whether ONNX surgery is required (Risk #1)**:

| Sub-phase | Best case | Realistic | Worst case (ONNX surgery) |
|---|---:|---:|---:|
| A: trtexec build + parity test | 1h | 2h | 1-2d |
| B: C++ + binding + build | 3h | 5h | same |
| C: Integration into qwen3_asr.py + NX validation | 1h | 2h | same |
| D: Nano integration + verification | 1h | 1h | same |
| **Total** | **6h** | **10h** | **2-3d** |

Risk #1 (TRT can't ingest encoder ONNX) is medium-high — will know within first 30 min of Phase A.

## Out of scope (defer)

- Removing ORT entirely from container image (separate cleanup)
- Fully removing onnxruntime python package (depends on encoder TRT being default)
- TTS Talker max_seq cut (independent track)
- Weight streaming / context swap (other side of design space)
