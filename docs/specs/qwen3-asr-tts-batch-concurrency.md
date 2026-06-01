> Path note (post-restructure): the product service moved `app/`→`server/`
> (`app/main.py`→`server/main.py`, `app/core/`→`server/core/`). Backend
> implementations cited below as `app/backends/...` (jetson/rk/cpu) now live in the
> `voxedge` package (`voxedge.backends.*`); those `app/backends/...` paths
> are kept verbatim only to preserve the original line-anchored references — map
> them to the corresponding `voxedge` module when implementing.

# Qwen3 ASR/TTS Batch Concurrency Migration Spec

## 1. Goals and Constraints

Goal: enable concurrent streaming for qwen3 ASR and qwen3 TTS CustomVoice by using the official batch surface (`activeBatchSize=N`) instead of slot-pooling multiple single-session runtimes. Constraints: minimize memory, preserve single-session output quality, avoid source-level divergence where upstream already supports batching, and keep scheduling policy outside the EdgeLLM fork.

Batch is preferable to a slot pool because weights and one TensorRT execution context are shared while only per-batch KV grows. With a slot pool of N contexts, every live slot carries its own activation workspace/context pressure. Phase A measured encoder activation at about 352 MB/context and decoder KV at about 28 MB/slot. At N=4, slot-pool memory is approximately `4 * 352 + 4 * 28 = 1520 MB` beyond weights. Official batch is approximately `352 + 4 * 28 = 464 MB`, saving about 944 MB before any duplicated runtime buffers. This is the decisive difference on Orin Nano 8GB and still material on Orin NX 16GB.

## 2. Architecture Layers (Phase C)

Ownership has four layers:

1. EdgeLLM fork (`/Users/harvest/project/TensorRT-Edge-LLM`): expose additive batched prefill APIs and reuse existing official batch runner code. `LLMEngineRunner::executePrefillStep` already derives `activeBatchSize` from `inputsEmbeds` and packs `[activeBatchSize,...]` tensors at `cpp/runtime/llmEngineRunner.cpp:1262-1396`. `HybridCacheManager::resetForNewSequences` accepts `reuseKVCacheLengths` shaped `[batchSize]` and commits per-lane lengths at `cpp/runtime/hybridCacheManager.cpp:247-285`. `StreamChannel` is already per-slot (`cpp/runtime/streaming.h:94-102`, `cpp/runtime/streaming.cpp:160-165`) and should remain seeed-unmodified.
2. Product worker (`/Users/harvest/project/qwen3-edgellm-jetson`): own micro-batch scheduling and per-session state. Current ASR worker has one `AsrSessionState` (`native/edgellm_voice_worker/qwen3_asr_worker.cpp:124-170`) and directly calls `runStreamingHop` (`qwen3_asr_worker.cpp:1017-1018`).
3. OVS service (`/Users/harvest/project/seeed-local-voice`): own admission, session mapping, and backpressure. ASR currently binds `WorkerIO(..., concurrency=1)` at `app/backends/jetson/trt_edge_llm_asr.py:483-488`; TTS exposes configurable worker concurrency at `app/backends/jetson/trt_edge_llm_tts.py:488-517` and creates `WorkerIO` with that value at `trt_edge_llm_tts.py:820-823`.
4. Deployment/build layer: rebuild engines with `max_batch_size=N`; do not fake concurrency above engine capacity.

ASR flow:

```text
OVS ASR stream sessions
  -> admission map session_id -> worker request
  -> product ASR micro-batch window
  -> EdgeLLM appendPrefillEmbedsBatched(activeBatchSize=N)
  -> LLMEngineRunner executePrefillStep [N,maxChunkLen,H]
  -> per-lane decode / StreamChannel / JSON partials
```

TTS flow:

```text
OVS TTS stream requests
  -> WorkerIO request_id demux + concurrency cap
  -> product/CustomVoice batch window
  -> EdgeLLM executeTalkerPrefillStep [N,maxSeq,H]
  -> runTalkerGenerationLoop PerBatchTalkerState[N]
  -> per-request audio chunks / done events
```

## 3. ASR Batch Enablement Changes

Add an EdgeLLM API:

```cpp
bool appendPrefillEmbedsBatched(
    std::vector<SpecDecodeInferenceContext*> const& contexts,
    std::vector<Tensor const*> const& audioEmbedsDeltas,
    std::vector<int32_t> const& audioIndexBases,
    std::vector<std::vector<int32_t>> const& tokenSliceDeltas,
    cudaStream_t stream);
```

Packing contract: choose `N=contexts.size()` and `maxChunkLen=max(tokenSliceDeltas[b].size())`. Stage token ids as `[N,maxChunkLen]` with padding ignored by `contextLengths[N]`; stage embeddings as `[N,maxChunkLen,H]`; set `contextLengths[b]=chunkLen[b]`; bind logits as `[N,vocab]`. Each lane updates only its own `context.tokenIds[b]`, `effectivePrefillLengths[b]`, current KV length snapshot, and append status.

Current single-lane blocker is verified at `cpp/runtime/llmInferenceSpecDecodeRuntime.cpp:2465-2474`: it checks `context.activeBatchSize == 1` and fixes `kBatchIdx=0`. Slot-0-only KV read is at `llmInferenceSpecDecodeRuntime.cpp:2495-2505`; tensor shapes are fixed to `[1,...]` and `[1,vocab]` at `llmInferenceSpecDecodeRuntime.cpp:2541-2544`; CPU multimodal packing is single-row at `llmInferenceSpecDecodeRuntime.cpp:2566-2575`; `executePrefillStep` is already the official batched runner at `llmInferenceSpecDecodeRuntime.cpp:2591-2592`.

Add `PerBatchAsrState`, modeled after `PerBatchTalkerState` (`cpp/runtime/qwen3OmniTTSRuntime.h:398-408`): fields should include session id, context pointer, `audioIndexBase`, `chunkLen`, `currentKvLen`, `finished`, `appendStatus`, emitted tokens/text cursor, and stream channel pointer. Refactor old `appendPrefillEmbeds(...)` as an N=1 wrapper that constructs one-lane vectors and calls `appendPrefillEmbedsBatched(...)` for backward compatibility.

Fork files: `cpp/runtime/llmInferenceSpecDecodeRuntime.h` additive declaration; `cpp/runtime/llmInferenceSpecDecodeRuntime.cpp` additive batched implementation plus wrapper modification; tests/examples additive. No changes should be made to `llmEngineRunner`, `HybridCacheManager`, or streaming unless a verified bug appears.

## 4. TTS Batch Enablement Changes

Phase A found `executeTalkerPrefillStep` as the batchSize==1 gap. In the current checkout, the method already validates `[batchSize,seqLen,H]`, checks `batchSize <= mMaxBatchSize`, resets KV with `{batchSize}`, and writes per-batch context lengths at `cpp/runtime/qwen3OmniTTSRuntime.cpp:1114-1165`; the older binary hard-block is therefore [unverified] in this branch. Remaining work is to eliminate single-lane caller paths and standardize on the official batched prefill path.

Specific change points: the non-Omni request path still loops per batch and calls `executeTalkerPrefillStep(mTalkerInputEmbeds, ...)` inside the lane loop at `qwen3OmniTTSRuntime.cpp:1439-1467`. The Omni path already assembles padded `[activeBatchSize,maxOutSeqLen,H]` and calls `executeTalkerPrefillStep(..., perBatchSeqLen)` once at `qwen3OmniTTSRuntime.cpp:1685-1693`. `runTalkerGenerationLoop` already accepts `states`, `activeBatchSize`, and `prefillSeqLens` at `qwen3OmniTTSRuntime.cpp:1791-1795`, skips finished lanes at `qwen3OmniTTSRuntime.cpp:1821-1825`, and extracts per-batch hidden states using `prefillSeqLens[b]` at `qwen3OmniTTSRuntime.cpp:1833-1844`.

Shared pattern with ASR: padded prefill inputs, per-lane context lengths, per-lane state, logical eviction/finished flags rather than cache compaction, and product-level micro-batch admission.

## 5. Worker Micro-Batch Scheduler

Use static admission plus a small time window. ASR chunks are configured in OVS at `stream_chunk_sec` and used to derive hop samples at `trt_edge_llm_asr.py:1188-1191`; default config reads `stream_chunk_sec` at `trt_edge_llm_asr.py:219-222`. Start with a 10-20 ms ASR hop window: it is small versus 400-500 ms audio chunks but enough to coalesce simultaneous streams. TTS should start with 5-15 ms because TTFA is more user-visible and frame production is continuous.

Algorithm: maintain `pending_hops` keyed by session/request id. On first ready hop, arm a timer. Flush when `pending.size()==N`, the timer fires, or a final/end hop arrives. Build a batch from ready lanes only; for each lane carry `reuseKVCacheLengths[b]` and `contextLengths[b]`. Different progress is legal because `HybridCacheManager` supports per-batch variable reuse lengths (`hybridCacheManager.cpp:247-285`). Finished lanes are removed from admission; do not compact live KV inside a running batch, just omit finished lanes from the next batch.

Backpressure: OVS should admit at most `N` active ASR sessions per worker and at most `N` active TTS streams per runtime. ASR’s current Python cap is hardcoded to one (`trt_edge_llm_asr.py:164-170`, `trt_edge_llm_asr.py:483-488`); replace with `OVS_ASR_WORKER_CONCURRENCY=N` only after worker multi-session support lands. TTS already has `OVS_TTS_WORKER_CONCURRENCY` (`trt_edge_llm_tts.py:495-517`, `trt_edge_llm_tts.py:1084-1095`). When full, return retryable busy/429 at service level or queue for one window only, then fail fast.

## 6. Engine Rebuild

Batch>N requires engine rebuild. ASR thinker build script defaults `MAX_BATCH=1` and passes `--maxBatchSize "$MAX_BATCH"` at `/Users/harvest/project/qwen3-edgellm-jetson/scripts/build_qwen3_asr_thinker_engine.sh:24-26` and `:91-96`. TTS native engine script is `/Users/harvest/project/qwen3-edgellm-jetson/scripts/build_qwen3_nx_native_engines.sh`; talker build is command-file based at `:31-36`, CP at `:38-48`, and Code2Wav at `:50-61`. Talker is the relevant rebuild target for batched prefill/decode; CP may remain batch=1 if it is intentionally called per lane.

Target N=4 for Orin NX 16GB first. For Orin Nano 8GB, validate N=2 then N=4 only if peak memory stays inside budget with OVS, ASR, TTS, and audio buffers resident.

## 7. Memory and Performance Model

Memory model: `batch=N = shared weights + one activation workspace + N * KV + scheduler buffers`. For N=4, ASR incremental memory is about 464 MB from the Phase A numbers. Slot-pool N=4 is about 1520 MB and worsens fragmentation because each context owns activation memory. This is why MOSS-style slot-pooling is the wrong default for qwen3: it buys isolation by spending memory that qwen3 needs for larger context and TTS buffers.

Performance model: batching improves GPU occupancy and amortizes launch overhead, but the static window adds queueing latency. ASR TTFT impact should be bounded by the window (10-20 ms) plus any head-of-line lane with longer chunk packing. TTS TTFA is more sensitive; keep the window below 15 ms and flush immediately for the first request if no peer arrives. Throughput should be measured as simultaneous streams sustained without underrun, not only single-request RTF.

## 8. Phased Implementation Plan

Phase 1: ASR fork API. Add `appendPrefillEmbedsBatched`, keep N=1 wrapper, add unit/spike parity comparing N=1 old path and N=1 wrapper. Acceptance: identical CER on fixed fixtures, no regression in single-stream TTFT, no source changes outside fork runtime/tests.

Phase 2: ASR product scheduler and OVS admission. Convert `qwen3_asr_worker.cpp` from one `AsrSessionState` to a session map plus hop queue; replace direct `runStreamingHop` with async micro-batch dispatch. Acceptance: N=4 concurrent streams, per-session partial/final order preserved, 30 minute stability, busy behavior verified.

Phase 3: TTS batched prefill cleanup. Route all CustomVoice request paths through the padded batched prefill path; retain per-lane CP if desired. Acceptance: audio quality parity, no frame-order mixup, N=4 TTFA within target, cancellation still per request.

Phase 4: rebuild and deploy engines. Acceptance: config reports `max_batch_size=N`, N=4 memory peak recorded on NX, Nano target signed off separately.

Risks: static window latency; KV capacity exhaustion at N; head-of-line blocking from large padded prefill; engine rebuild regression; TTS current branch mismatch with Phase A hard-block [unverified].

## 9. Merge-Debt Impact

Low debt: additive ASR `appendPrefillEmbedsBatched`, N=1 wrapper, tests. This is a reasonable upstream PR because official runner and cache manager already expose batch semantics.

Medium debt: modifying ASR worker to multi-session. It is product-specific scheduling and should stay out of EdgeLLM.

Low/medium debt: TTS caller cleanup because `PerBatchTalkerState` and batched prefill scaffolding already exist (`qwen3OmniTTSRuntime.h:398-408`, `qwen3OmniTTSRuntime.cpp:1685-1693`). Avoid touching streaming primitives.

High debt if chosen: slot-pool fallback. It conflicts with the memory goal and should remain a diagnostic fallback only.

## 10. Decision Points for Main Thread

1. Choose target N: recommend N=4 for Orin NX, N=2 pilot for Orin Nano.
2. Choose initial windows: recommend ASR 15 ms, TTS 10 ms, both tunable by env.
3. Decide sequencing: recommend ASR first, then TTS cleanup, because ASR currently has the explicit single-session product bottleneck.
4. Decide backpressure contract: fail-fast 429/busy versus one-window queue.
5. Decide engine matrix: NX-only first or rebuild Nano/NX together.
