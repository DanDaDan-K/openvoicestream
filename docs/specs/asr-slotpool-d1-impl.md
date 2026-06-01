> Path note (post-restructure): the product service moved `app/`→`server/`
> (`app/main.py`→`server/main.py`, `app/core/`→`server/core/`). Backend
> implementations cited below as `app/backends/...` (jetson/rk/cpu) now live in the
> `voxedge` package (`voxedge.backends.*`); those `app/backends/...` paths
> are kept verbatim only to preserve the original line-anchored references — map
> them to the corresponding `voxedge` module when implementing.

# ASR Slot-Pool D1 Implementation Spec

## 1. Shared-engine path in TRT-Edge-LLM fork

Ground truth from `/Users/harvest/project/TensorRT-Edge-LLM`:

- `LLMEngineRunner` has one public constructor today: `LLMEngineRunner(std::filesystem::path const& enginePath, std::filesystem::path const& configPath, ..., cudaStream_t stream)` at `cpp/runtime/llmEngineRunner.h:102`.
- The implementation of that constructor starts at `cpp/runtime/llmEngineRunner.cpp:105`.
- Public runner APIs include `getRequiredContextMemorySize()` at `cpp/runtime/llmEngineRunner.h:112`, `setContextMemory(...)` at `cpp/runtime/llmEngineRunner.h:120`, `getRopeCosSinCacheTensor()` at `cpp/runtime/llmEngineRunner.h:125`, `getCacheManager()` at `cpp/runtime/llmEngineRunner.h:129`, `getEngineConfig()` at `cpp/runtime/llmEngineRunner.h:133`, `setLmHeadWeight(...)` at `cpp/runtime/llmEngineRunner.h:144`, `getActiveLoraWeightsName()` at `cpp/runtime/llmEngineRunner.h:217`, `getAvailableLoraWeights()` at `cpp/runtime/llmEngineRunner.h:222`, and `getTensorDataType(...)` at `cpp/runtime/llmEngineRunner.h:250`. There is no `getEngine()` accessor in the read header.
- The runner owns `mRuntime`, `mEngine`, and `mTRTExecutionContext` at `cpp/runtime/llmEngineRunner.h:253`, `cpp/runtime/llmEngineRunner.h:254`, and `cpp/runtime/llmEngineRunner.h:255`. `mEngine` is `std::unique_ptr<nvinfer1::ICudaEngine>` today.
- The runner header has no CUDA stream data member in its private member block; streams are passed to execution APIs such as `executePrefillStep(..., cudaStream_t stream)` at `cpp/runtime/llmEngineRunner.h:159` and `executeVanillaDecodingStep(..., cudaStream_t stream)` at `cpp/runtime/llmEngineRunner.h:174`.
- The current load path creates `mRuntime` at `cpp/runtime/llmEngineRunner.cpp:138`, deserializes the engine at `cpp/runtime/llmEngineRunner.cpp:146`, applies weight streaming at `cpp/runtime/llmEngineRunner.cpp:152`, and creates a user-managed execution context at `cpp/runtime/llmEngineRunner.cpp:157`.
- Runtime-level shared execution-context memory is allocated in `LLMInferenceSpecDecodeRuntime::initializeCommon`: it reads each engine's required context memory at `cpp/runtime/llmInferenceSpecDecodeRuntime.cpp:484`, allocates `mSharedExecContextMemory` at `cpp/runtime/llmInferenceSpecDecodeRuntime.cpp:491`, and passes it to the base runner at `cpp/runtime/llmInferenceSpecDecodeRuntime.cpp:493`.
- `LLMInferenceSpecDecodeRuntime` has two constructors today, both taking `engineDir` and `multimodalEngineDir` strings: Eagle mode at `cpp/runtime/llmInferenceSpecDecodeRuntime.h:157` and vanilla mode at `cpp/runtime/llmInferenceSpecDecodeRuntime.h:169`. Their implementations call `initializeCommon(...)` at `cpp/runtime/llmInferenceSpecDecodeRuntime.cpp:95` and `cpp/runtime/llmInferenceSpecDecodeRuntime.cpp:102`.
- `mBaseEngineRunner` is `std::unique_ptr<LLMEngineRunner>` at `cpp/runtime/llmInferenceSpecDecodeRuntime.h:573`. `initializeCommon` currently creates it through a `loadBaseEngine` lambda returning `std::unique_ptr<LLMEngineRunner>` at `cpp/runtime/llmInferenceSpecDecodeRuntime.cpp:120`, with `std::make_unique<LLMEngineRunner>(...)` at `cpp/runtime/llmInferenceSpecDecodeRuntime.cpp:124`; vanilla mode assigns it at `cpp/runtime/llmInferenceSpecDecodeRuntime.cpp:188`.

Minimal code changes:

1. Change `LLMEngineRunner::mEngine` from `std::unique_ptr<nvinfer1::ICudaEngine>` to `std::shared_ptr<nvinfer1::ICudaEngine>` at `cpp/runtime/llmEngineRunner.h:254`. Keep `mRuntime` ownership at `cpp/runtime/llmEngineRunner.h:253` for the deserializing constructor, because TensorRT runtime lifetime must remain valid for the deserialized engine.
2. In `cpp/runtime/llmEngineRunner.cpp`, replace the assignment at `cpp/runtime/llmEngineRunner.cpp:146` with construction of a `std::shared_ptr<nvinfer1::ICudaEngine>` using the same raw pointer returned by `deserializeCudaEngine(...)` at `cpp/runtime/llmEngineRunner.cpp:147`. Keep `rt::applyWeightStreamingBudget(mEngine.get(), ...)` at `cpp/runtime/llmEngineRunner.cpp:152`.
3. Add a public `std::shared_ptr<nvinfer1::ICudaEngine> getEngine() const noexcept` accessor near `getEngineConfig()` at `cpp/runtime/llmEngineRunner.h:133`. Implement it in `cpp/runtime/llmEngineRunner.cpp` near the existing simple runner accessors [unverified: accessor definitions were not opened in this run; header insertion point is verified].
4. Add an `LLMEngineRunner` constructor overload in `cpp/runtime/llmEngineRunner.h` immediately after the existing constructor at `cpp/runtime/llmEngineRunner.h:102`. The overload should accept `(std::shared_ptr<nvinfer1::ICudaEngine> engine, std::filesystem::path const& configPath, std::unordered_map<std::string, std::string> const& loraWeightsMap, cudaStream_t stream)`.
5. Implement the shared-engine overload near the current constructor implementation at `cpp/runtime/llmEngineRunner.cpp:105`. It must reuse the existing config parse/`initializeConfigFromJson` path, skip the deserialize block at `cpp/runtime/llmEngineRunner.cpp:136` through `cpp/runtime/llmEngineRunner.cpp:147`, assign the supplied shared engine, then create its own `mTRTExecutionContext` with `createExecutionContext(ExecutionContextAllocationStrategy::kUSER_MANAGED)` as the current constructor does at `cpp/runtime/llmEngineRunner.cpp:157`.
6. Add an `LLMInferenceSpecDecodeRuntime` constructor path in `cpp/runtime/llmInferenceSpecDecodeRuntime.h` near the vanilla constructor at `cpp/runtime/llmInferenceSpecDecodeRuntime.h:169` that accepts a shared base `ICudaEngine`. In `cpp/runtime/llmInferenceSpecDecodeRuntime.cpp`, add a matching constructor near `cpp/runtime/llmInferenceSpecDecodeRuntime.cpp:102` and route it through a new `initializeCommonWithSharedBaseEngine(...)` or an expanded `initializeCommon(...)`.
7. In the shared-engine runtime path, replace the `loadBaseEngine` creation site at `cpp/runtime/llmInferenceSpecDecodeRuntime.cpp:120` through `cpp/runtime/llmInferenceSpecDecodeRuntime.cpp:124` for the base engine only. Construct `mBaseEngineRunner` with the new shared-engine `LLMEngineRunner` overload and still set `mBaseEngineConfig = mBaseEngineRunner->getEngineConfig()` as vanilla mode does at `cpp/runtime/llmInferenceSpecDecodeRuntime.cpp:190`.
8. Keep the per-runtime context-memory allocation and `setContextMemory` flow at `cpp/runtime/llmInferenceSpecDecodeRuntime.cpp:481` through `cpp/runtime/llmInferenceSpecDecodeRuntime.cpp:493`. Sharing the `ICudaEngine` must not share `IExecutionContext` or context memory between ASR slots.

## 2. Encoder single-context serial + decoder per-slot in product worker layer

Current product worker facts from `/Users/harvest/project/qwen3-edgellm-jetson/native/edgellm_voice_worker/qwen3_asr_worker.cpp`:

- `main()` creates one CUDA stream at `qwen3_asr_worker.cpp:1268` and one `rt::LLMInferenceSpecDecodeRuntime` at `qwen3_asr_worker.cpp:1273`.
- The runtime internally loads the multimodal audio runner from `multimodalEngineDir + "/audio"` at `cpp/runtime/llmInferenceSpecDecodeRuntime.cpp:441` and `cpp/runtime/llmInferenceSpecDecodeRuntime.cpp:442`.
- The streaming hop path builds one `LLMGenerationRequest` at `qwen3_asr_worker.cpp:651` and calls `runtime.handleRequest(...)` at `qwen3_asr_worker.cpp:662`.
- The legacy one-shot path also calls `runtime.handleRequest(...)` at `qwen3_asr_worker.cpp:1187`.

D1 design:

- Keep one shared audio encoder path initially and guard it with a worker-level mutex. This avoids relying on unverified audio-runner multi-context safety.
- Create N decoder/thinker slots, each with its own `LLMInferenceSpecDecodeRuntime`, `IExecutionContext`, runtime tensors, KV/cache manager state, and CUDA stream.
- Use the shared-engine constructor path from section 1 so these N thinker slots share one deserialized `ICudaEngine` while retaining independent execution contexts.
- Route each admitted ASR session to one slot for its full begin/chunk/end lifecycle. Do not move a live session between slots.

## 3. Worker slot pool: `AsrSessionState` -> N slots + `acquireSlot`/`releaseSlot`

Current worker facts:

- The worker explicitly documents single-session behavior at `qwen3_asr_worker.cpp:78`.
- `AsrSessionState` is defined at `qwen3_asr_worker.cpp:124`. It contains `sessionId`, `lastActivity`, and `active` at `qwen3_asr_worker.cpp:126` through `qwen3_asr_worker.cpp:128`; streaming parameters at `qwen3_asr_worker.cpp:131` through `qwen3_asr_worker.cpp:137`; mel accumulation at `qwen3_asr_worker.cpp:142` through `qwen3_asr_worker.cpp:144`; decoded text state at `qwen3_asr_worker.cpp:150` through `qwen3_asr_worker.cpp:151`; session full text at `qwen3_asr_worker.cpp:154`; audio timing and segment count at `qwen3_asr_worker.cpp:158` through `qwen3_asr_worker.cpp:160`; and VAD PCM accumulation at `qwen3_asr_worker.cpp:169`.
- `freeSession(...)` resets a session by assignment at `qwen3_asr_worker.cpp:174` through `qwen3_asr_worker.cpp:176`.
- The single-session guard is `if (session.active)` in `handleBegin(...)` at `qwen3_asr_worker.cpp:729`, which emits `session_already_active` at `qwen3_asr_worker.cpp:731`.
- `main()` owns exactly one `AsrSessionState session` at `qwen3_asr_worker.cpp:1285`.
- Idle timeout checks the single `session.active` at `qwen3_asr_worker.cpp:1294` and resets that session at `qwen3_asr_worker.cpp:1304`.
- The stdin loop dispatches `begin`, `chunk`, and `end` to handlers with the single `session` at `qwen3_asr_worker.cpp:1380` through `qwen3_asr_worker.cpp:1392`.

D1 design:

- Introduce `AsrSlot { int slotId; AsrSessionState session; std::unique_ptr<rt::LLMInferenceSpecDecodeRuntime> runtime; cudaStream_t stream; bool inUse; }`.
- Replace the single `AsrSessionState session` at `qwen3_asr_worker.cpp:1285` with `std::vector<AsrSlot> slots`.
- Add `acquireSlot(sessionId)`:
  - If `sessionId` already maps to an active slot, return that slot.
  - If a free slot exists, mark it active, initialize its `AsrSessionState`, and return it.
  - If no free slot exists, emit a structured busy response such as `{"event":"error","ok":false,"error":"too_many_asr_sessions","id":...}`.
- Add `releaseSlot(slotId)` that calls `freeSession(...)` as currently implemented at `qwen3_asr_worker.cpp:174` through `qwen3_asr_worker.cpp:176`, clears the id mapping, synchronizes or cancels outstanding work for the slot, and returns it to the free list.
- Change `handleBegin`, `handleChunk`, and `handleEnd` to take `AsrSlot&` or resolve by `id` before calling existing session logic. Preserve `no_active_session` behavior from `qwen3_asr_worker.cpp:837` through `qwen3_asr_worker.cpp:845` for unknown ids.
- Extend the idle-timeout lambda currently bound to one session at `qwen3_asr_worker.cpp:1293` through `qwen3_asr_worker.cpp:1308` to scan all slots.

Reference slot-pool pattern:

- `MossTtsNanoSlot` carries per-slot state, contexts, and stream at `cpp/runtime/mossTtsNanoRuntime.h:61` through `cpp/runtime/mossTtsNanoRuntime.h:82`.
- `MossTtsNanoRuntime` exposes `acquirePoolSlot()`, `releasePoolSlot(...)`, and `beginRequest()` at `cpp/runtime/mossTtsNanoRuntime.h:143` through `cpp/runtime/mossTtsNanoRuntime.h:145`.
- The pool stores `mPoolMutex`, `mPoolCv`, `mSlots`, and `mFreeSlots` at `cpp/runtime/mossTtsNanoRuntime.h:261` through `cpp/runtime/mossTtsNanoRuntime.h:264`.
- Slot allocation creates a non-blocking CUDA stream at `cpp/runtime/mossTtsNanoRuntime.cpp:277` and creates per-slot TRT contexts at `cpp/runtime/mossTtsNanoRuntime.cpp:318` through `cpp/runtime/mossTtsNanoRuntime.cpp:322`.
- `acquirePoolSlot()` waits for a free slot at `cpp/runtime/mossTtsNanoRuntime.cpp:365` through `cpp/runtime/mossTtsNanoRuntime.cpp:366`, pops it at `cpp/runtime/mossTtsNanoRuntime.cpp:367` through `cpp/runtime/mossTtsNanoRuntime.cpp:368`, marks/reset state at `cpp/runtime/mossTtsNanoRuntime.cpp:369` through `cpp/runtime/mossTtsNanoRuntime.cpp:374`, and clears device state at `cpp/runtime/mossTtsNanoRuntime.cpp:375` through `cpp/runtime/mossTtsNanoRuntime.cpp:381`.
- `releasePoolSlot()` clears state and returns the id to `mFreeSlots` at `cpp/runtime/mossTtsNanoRuntime.cpp:385` through `cpp/runtime/mossTtsNanoRuntime.cpp:395`.
- The RAII guard releases through `endRequest()` at `cpp/runtime/mossTtsNanoRuntime.cpp:408` through `cpp/runtime/mossTtsNanoRuntime.cpp:418`.

## 4. OVS WorkerIO concurrency=N + semaphore admission + 429

Current OVS facts from `/Users/harvest/project/seeed-local-voice`:

- `TRTEdgeLLMASRBackend.__init__` loads config at `app/backends/jetson/trt_edge_llm_asr.py:145`, keeps `_worker_lock` as a lifecycle gate at `app/backends/jetson/trt_edge_llm_asr.py:148` through `app/backends/jetson/trt_edge_llm_asr.py:155`, and documents `WorkerIO` concurrency as 1 at `app/backends/jetson/trt_edge_llm_asr.py:164` through `app/backends/jetson/trt_edge_llm_asr.py:170`.
- `_load_config()` reads the manifest at `app/backends/jetson/trt_edge_llm_asr.py:172` through `app/backends/jetson/trt_edge_llm_asr.py:178`, engine/audio paths at `app/backends/jetson/trt_edge_llm_asr.py:197` through `app/backends/jetson/trt_edge_llm_asr.py:203`, stream-mode profile values at `app/backends/jetson/trt_edge_llm_asr.py:215` through `app/backends/jetson/trt_edge_llm_asr.py:235`, mel asset paths at `app/backends/jetson/trt_edge_llm_asr.py:237` through `app/backends/jetson/trt_edge_llm_asr.py:243`, and sampling defaults at `app/backends/jetson/trt_edge_llm_asr.py:249` through `app/backends/jetson/trt_edge_llm_asr.py:252`.
- `_ensure_worker()` currently creates `WorkerIO(self._worker, concurrency=1)` at `app/backends/jetson/trt_edge_llm_asr.py:488`.
- `_worker_request(...)` relies on `WorkerIO.request(...)` and documents that `Semaphore(1)` serializes calls at `app/backends/jetson/trt_edge_llm_asr.py:504` through `app/backends/jetson/trt_edge_llm_asr.py:508`.
- `WorkerIO` itself stores `threading.Semaphore(max(1, int(concurrency)))` at `server/core/worker_io.py:83`; `request(...)` acquires it at `server/core/worker_io.py:246` and releases it at `server/core/worker_io.py:282`.
- ASR websocket admission uses `try_acquire_ws(...)` at `server/main.py:2057` through `server/main.py:2059`; the token is released at `server/main.py:2120` through `server/main.py:2124`.
- `try_acquire_ws(...)` closes saturated WS sessions with code 4429 at `server/core/session_limiter.py:296` through `server/core/session_limiter.py:328`.
- HTTP-style 429 behavior already exists for TTS streaming: `status_code=429` and `Retry-After: 5` at `server/main.py:1246` through `server/main.py:1250`.
- Streaming ASR work is currently funneled through a single-thread executor: `_get_asr_executor()` constructs `ThreadPoolExecutor(max_workers=1, ...)` at `server/main.py:496` through `server/main.py:501`; `_asr_stream_backend(...)` uses that executor for `prepare_finalize`, `finalize`, and `accept_waveform` at `server/main.py:2189`, `server/main.py:2190`, `server/main.py:2213`, `server/main.py:2214`, and `server/main.py:2230`.

D1 design:

- Add an ASR concurrency config value, e.g. `EDGE_LLM_ASR_MAX_CONCURRENT` / manifest `max_concurrent`, in `_load_config()` near the existing stream config block at `app/backends/jetson/trt_edge_llm_asr.py:215`.
- Pass that value to `WorkerIO(self._worker, concurrency=N)` at `app/backends/jetson/trt_edge_llm_asr.py:488`.
- Update comments and lifecycle assumptions in `_worker_request(...)` at `app/backends/jetson/trt_edge_llm_asr.py:504` through `app/backends/jetson/trt_edge_llm_asr.py:508`; same session id can remain per lifecycle, but multiple different session ids must be allowed concurrently.
- Override ASR `concurrency_capability(...)` in `TRTEdgeLLMASRBackend` so the resolver does not keep the ASR default N=1 from `server/core/asr_backend.py:136` through `server/core/asr_backend.py:147`.
- Increase `_get_asr_executor()` from 1 worker to N at `server/main.py:496` through `server/main.py:501`, or bypass it for nonblocking worker IO once the worker protocol is safe for concurrent sessions.
- Preserve reject-not-queue semantics: HTTP callers should receive 429 as in `server/main.py:1246` through `server/main.py:1250`; WS callers should continue to receive 4429 from `server/core/session_limiter.py:296` through `server/core/session_limiter.py:328`.

## 5. No engine rebuild needed

No TRT engine rebuild is required for D1. The desired scaling is N independent execution contexts/runtimes sharing one deserialized `ICudaEngine`, not one TRT batch with `max_batch_size=N`.

The existing worker sends one request at a time into `LLMGenerationRequest` with `batch_size` 1 in the legacy request builder at `qwen3_asr_worker.cpp:492` through `qwen3_asr_worker.cpp:498`, and the streaming hop pushes one request into `llmReq.requests` at `qwen3_asr_worker.cpp:651` through `qwen3_asr_worker.cpp:652`. D1 keeps each ASR slot as logical batch size 1.

## 6. Acceptance criteria

- Ground-truth spec citations remain valid against the files read above; no stale `shared_ptr`/constructor/getter claims are reintroduced.
- With `EDGE_LLM_ASR_MAX_CONCURRENT=1`, behavior is byte-compatible with the current single-session path except for explicitly documented busy/error wording.
- With `EDGE_LLM_ASR_MAX_CONCURRENT=N`, N concurrent ASR websocket sessions can begin, send chunks, and end without `session_already_active`.
- The N+1st session is rejected immediately, not queued behind the pool.
- Each slot owns an independent CUDA stream and `IExecutionContext`; no two active sessions share `AsrSessionState`, KV cache state, runtime tensors, or execution-context memory.
- Encoder access is serialized until audio-runner multi-context safety is separately proven.
- Worker shutdown and OVS restart release all acquired slots and wake blocked callers without leaking `WorkerIO` semaphore tokens.

## 7. Five-phase rollout

1. TRT shared-engine plumbing: change `mEngine`, add `getEngine()`, add shared-engine `LLMEngineRunner` overload, and add the shared-base runtime constructor path.
2. Product worker slot scaffolding: add `AsrSlot`, slot vector, id map, acquire/release, per-slot runtime and stream creation, while keeping N=1 by default.
3. Worker protocol concurrency: route begin/chunk/end by id, scan idle timeouts across slots, and replace `session_already_active` with pool saturation only when every slot is busy.
4. OVS admission: expose ASR max concurrency in config/profile, pass it to `WorkerIO`, update ASR concurrency capability, and align ASR executor/session-limiter ceilings.
5. Stress and soak: run N=1 parity, N concurrent lifecycle, N+1 rejection, client disconnect, worker restart, idle-timeout cleanup, and long-stream segmentation tests.

## 8. Risks

- R1 context memory isolation: current `LLMInferenceSpecDecodeRuntime` allocates one `mSharedExecContextMemory` per runtime at `cpp/runtime/llmInferenceSpecDecodeRuntime.cpp:491` and binds it at `cpp/runtime/llmInferenceSpecDecodeRuntime.cpp:493`. Slot runtimes must not share that tensor.
- R2 encoder serial bottleneck: audio runner loading is inside runtime initialization at `cpp/runtime/llmInferenceSpecDecodeRuntime.cpp:441` through `cpp/runtime/llmInferenceSpecDecodeRuntime.cpp:443`; D1 intentionally serializes encoder use until safe parallelism is verified.
- R3 slot state reset: runtime exposes `beginAsrSession(...)` at `cpp/runtime/llmInferenceSpecDecodeRuntime.h:289` and `endAsrSession(...)` at `cpp/runtime/llmInferenceSpecDecodeRuntime.h:310`, but no generic `reset()` or `resetState()` API was found in the header read. Slot reuse must explicitly reset `AsrSessionState`, runtime/session state, KV lengths, and any per-slot decoder state.
- R4 shutdown: OVS `restart_worker()` closes `WorkerIO` and kills the subprocess at `app/backends/jetson/trt_edge_llm_asr.py:553` through `app/backends/jetson/trt_edge_llm_asr.py:620`; the C++ worker must tolerate EOF and destroy all slot streams/contexts.
- R5 worker threads: the current worker is a single poll/read stdin loop at `qwen3_asr_worker.cpp:1313` through `qwen3_asr_worker.cpp:1340`, dispatching handlers inline at `qwen3_asr_worker.cpp:1380` through `qwen3_asr_worker.cpp:1392`. True overlap requires adding worker-side request execution threads or another dispatch mechanism; raising only `WorkerIO` concurrency will not create GPU overlap by itself.
