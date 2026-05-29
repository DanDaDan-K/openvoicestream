# TTS Cohort Batch D-2 Implementation Spec

### 1. qwen3OmniTTSRuntime batch readiness audit

The fork exposes `handleAudioGeneration(std::vector<TalkerGenerationRequest> const&...)` and a single-request wrapper at `cpp/runtime/qwen3OmniTTSRuntime.h:211-218`; implementation starts at `cpp/runtime/qwen3OmniTTSRuntime.cpp:1405`. `runTalkerGenerationLoop` is declared at `cpp/runtime/qwen3OmniTTSRuntime.h:440-444` and implemented at `cpp/runtime/qwen3OmniTTSRuntime.cpp:1791-1795`. `executeTalkerPrefillStep` is declared at `cpp/runtime/qwen3OmniTTSRuntime.h:368-369` and implemented at `cpp/runtime/qwen3OmniTTSRuntime.cpp:1114-1116`. `PerBatchTalkerState` fields are at `cpp/runtime/qwen3OmniTTSRuntime.h:398-409`.

`executeTalkerPrefillStep` supports `batchSize > 1` at the engine-call level: it requires 3D `[batchSize, seqLen, hiddenSize]`, rejects `batchSize > mMaxBatchSize`, fills per-batch context lengths, reshapes logits, and calls `executePrefillStep` once at `cpp/runtime/qwen3OmniTTSRuntime.cpp:1119-1166`. Standalone TTS does not yet assemble one padded prefill cohort; it loops per request and calls prefill per lane at `cpp/runtime/qwen3OmniTTSRuntime.cpp:1443-1467`. The Omni path does padded batched prefill at `cpp/runtime/qwen3OmniTTSRuntime.cpp:1664-1693`.

Sampling parameters come from `requests[0]`: temperature/topK/topP/repetition penalty at `cpp/runtime/qwen3OmniTTSRuntime.cpp:1427-1437`, matching the header note at `cpp/runtime/qwen3OmniTTSRuntime.h:202-204`. Per-request data still includes messages/template flags at `cpp/runtime/qwen3OmniTTSRuntime.cpp:1445-1454`, speaker/language at `cpp/runtime/qwen3OmniTTSRuntime.cpp:1355-1388`, max length at `cpp/runtime/qwen3OmniTTSRuntime.cpp:1499-1508`, and streaming hooks at `cpp/runtime/qwen3OmniTTSRuntime.cpp:1513-1533`.

Finished lanes idle while the batch continues: finished lanes are skipped in CodePredictor/residual at `cpp/runtime/qwen3OmniTTSRuntime.cpp:1821-1825`, logit adjustment at `cpp/runtime/qwen3OmniTTSRuntime.cpp:1898-1902`, and state update at `cpp/runtime/qwen3OmniTTSRuntime.cpp:1928-1933`; the loop runs until `unfinished == 0` or `globalFrame == maxFrames` at `cpp/runtime/qwen3OmniTTSRuntime.cpp:1819`.

Per-call moved buffer: `codecHiddensBuffer` is local at `cpp/runtime/qwen3OmniTTSRuntime.cpp:1419-1425` and documented at `cpp/runtime/qwen3OmniTTSRuntime.h:659-662`. Still shared: `mSharedExecContextMemory` at `cpp/runtime/qwen3OmniTTSRuntime.h:578-579`, prefill/Talker/CodePredictor workspaces at `cpp/runtime/qwen3OmniTTSRuntime.h:618-658`, and streaming scratch tensors at `cpp/runtime/qwen3OmniTTSRuntime.cpp:861-876`.

### 2. micro-batcher design

Place the micro-batcher in OVS Python for D-2. OVS already owns request construction, segmentation, profile defaults, speaker fields, and `WorkerIO` submission at `app/backends/jetson/trt_edge_llm_tts.py:852-897` and `app/backends/jetson/trt_edge_llm_tts.py:982-1095`. The C++ worker currently processes one JSON line inline, builds one request, and calls one runtime request at `qwen3_tts_worker.cpp:325-337` and `qwen3_tts_worker.cpp:373-570`.

Use `EDGE_LLM_TTS_COHORT_WINDOW_MS=8` default, configurable 0-20 ms. Flush on full N, timeout, same-params cohort complete, or low-latency streaming wait beyond the window. Cohort key must include talker/predictor sampling params, repetition penalty, EOS offset, seed, language, speaker, stream/chunk settings. Heterogeneous requests go to separate cohorts or batch=1 fallback because runtime sampling uses `requests[0]`.

Run one cohort at a time against one runtime. `WorkerIO` admits N logical requests via semaphore at `app/core/worker_io.py:78-83`, but shared runtime workspaces require serial cohort execution. Batch=1 fallback uses existing `_worker_io.request(request)` call sites at `app/backends/jetson/trt_edge_llm_tts.py:896-897` and `app/backends/jetson/trt_edge_llm_tts.py:1095`.

Cancel: OVS calls `worker_io.cancel(req_id)` on `GeneratorExit` at `app/backends/jetson/trt_edge_llm_tts.py:1156-1167`; `WorkerIO.cancel` writes `{"type":"cancel","id":...}` at `app/core/worker_io.py:284-314`. Worker cohort support must map cancel ids to lane flags and wire `TalkerGenerationRequest::shouldCancel` from `cpp/runtime/qwen3OmniTTSRuntime.h:162-163`; runtime finishes that lane at `cpp/runtime/qwen3OmniTTSRuntime.cpp:1949-1958`.

### 3. Worker-side changes (if any)

Today the worker is single-request: `buildRequest` returns one `TalkerGenerationRequest` at `qwen3_tts_worker.cpp:223-247`; `handleAudioGeneration(request, ...)` is called at `qwen3_tts_worker.cpp:568-570`. It is a single-threaded stdin loop at `qwen3_tts_worker.cpp:325-337`; optional chunk threading is disabled by `asyncCode2Wav = false` at `qwen3_tts_worker.cpp:341`.

Needed: accept a cohort JSON envelope, build `std::vector<TalkerGenerationRequest>`, call the vector overload, and demux `batchRvqCodes`/`numFramesPerSample` from `cpp/runtime/qwen3OmniTTSRuntime.h:171-178`. [unverified] The read worker references `CodecFrameCallback`, `rvqCodes`, and `numFrames` at `qwen3_tts_worker.cpp:568-664`, which do not match the read header contract; align branches before coding.

### 4. OVS service layer

`concurrency_capability` reads env/profile concurrency and advertises `max_concurrent=n` at `app/backends/jetson/trt_edge_llm_tts.py:487-517`. `__init__` stores `_worker_concurrency` at `app/backends/jetson/trt_edge_llm_tts.py:536-549`; `_ensure_worker` refreshes it and creates `WorkerIO` at `app/backends/jetson/trt_edge_llm_tts.py:816-823`.

Insert the batcher before `_synthesize_worker` submission at `app/backends/jetson/trt_edge_llm_tts.py:852-897` and `_generate_streaming_single` submission at `app/backends/jetson/trt_edge_llm_tts.py:982-1095`. Capability `max_concurrent=N` should mean admitted logical TTS requests cohorted onto one serial runtime, not simultaneous runtime calls.

### 5. Engine rebuild requirements

Runtime max batch is `min(Talker, CodePredictor)` at `cpp/runtime/qwen3OmniTTSRuntime.cpp:376-380`; `LLMEngineRunner` reads `builder_config.max_batch_size` at `cpp/runtime/llmEngineRunner.cpp:891-894`. Exact deployed TTS max batch was not found [unverified]. Rebuild Talker with `max_batch_size=N`; CodePredictor can remain batch=1 because runtime calls it per lane. Export entry point: `scripts/export_qwen3_tts_onnx.sh:176-230`. Historical native build entry: `scripts/build_qwen3_nx_native_engines.sh:31-48`; CP profile is batch 1 at `scripts/build_qwen3_tts_cp_engine.py:71-78`.

### 6. Fork change list

- `cpp/runtime/qwen3OmniTTSRuntime.{h,cpp}`: no required fork change if using existing vector API and accepting per-lane standalone prefill. Estimated 0 lines. Debt: Low.
- Optional standalone padded batched prefill, porting Omni assembly from `cpp/runtime/qwen3OmniTTSRuntime.cpp:1664-1693`. Estimated 80-140 lines. Debt: Medium.
- Optional max-batch getter for capability validation. Estimated 10-20 lines. Debt: Low.

Principle: minimal additive; keep scheduling, grouping, timeout, and cancellation in product/OVS.

### 7. Acceptance criteria

- Audio parity: batch=1 RVQ MD5 equals baseline, or WAV perceptual/hash tolerance with fixed seed.
- Cohort TTFA: N=2 median first chunk <= single-request median + 20 ms + cohort window.
- Mixed-batch degradation: short lane total latency <= 1.35x single baseline, else split by text length.
- Per-request cancel: one lane emits terminal `cancelled`; peers continue; no stale chunks after cancel.
- N=2 memory budget on Orin NX: peak ASR+TTS stays within 16GB target; fail closed to batch=1 if over budget.

### 8. Phased implementation plan

1. Protocol alignment, 0.5-1 day: make worker match the read runtime header; prove single-request build.
2. Cohort envelope, 1 day: add worker `requests:[...]`, vector build/call, per-lane events; buildable with N=1.
3. OVS batcher, 1-1.5 days: same-param queues, 8 ms timeout, full-N flush, batch=1 fallback, serial executor.
4. Cancel lane wiring, 1 day: map cancel ids to lane flags and connect `shouldCancel`.
5. Engine/profile, 0.5-1 day: rebuild Talker `max_batch_size=2`, publish env/profile guard.
6. Validation, 1 day: parity, TTFA, mixed-length, cancel/disconnect, memory soak.

### 9. Risks and decision points for the main thread

- Decide whether D-2 requires true standalone batched prefill now; decode is batched, standalone prefill is still per-lane.
- Resolve worker/header mismatch before implementation.
- Decide whether `max_concurrent` means admitted logical requests or actual runtime parallelism.
- Start with N=2; larger N increases workspace/KV memory and idle loss.
- If streaming cohorts harm TTFA, cohort only non-streaming or identical low-latency streaming settings.
