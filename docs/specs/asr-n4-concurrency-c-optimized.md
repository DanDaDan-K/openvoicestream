> Path note (post-restructure): the product service moved `app/`→`server/`
> (`app/main.py`→`server/main.py`, `app/core/`→`server/core/`). Backend
> implementations cited below as `app/backends/...` (jetson/rk/cpu) now live in the
> `voxedge` package (`voxedge.backends.*`); those `app/backends/...` paths
> are kept verbatim only to preserve the original line-anchored references — map
> them to the corresponding `voxedge` module when implementing.

## 1. Background + Decision Facts Table

This spec upgrades the Qwen3-ASR backend in `seeed-local-voice` to support `N=4` concurrent ASR sessions using the C-optimized slot-pool architecture:

- One shared encoder TensorRT execution context, mutex-serialized at enqueue time.
- Four decoder slots, each with its own TensorRT execution context, CUDA stream, and KV cache buffer.
- Python-side admission control with `asyncio.Semaphore(N)` so no more than four active ASR sessions can hold slots.

| Metric | Measured Value | Source |
|---|---:|---|
| Encoder context MiB | 352 MiB | Confirmed design facts |
| Encoder p99 latency | 10.1 ms | Confirmed design facts |
| Encoder utilization@N=4 | 16.2% at 250 ms stride | Confirmed design facts |
| Decoder context MiB | 6 MiB | Confirmed design facts |
| Per-slot KV MiB | ~28 MiB/slot | Confirmed design facts |
| Per-slot marginal MiB | ~130 MiB/slot | Confirmed design facts |
| N=4 total budget | 2533 MiB (`2143 + 3x130`) | Confirmed design facts |
| Orin Nano free headroom | Sufficient for 2533 MiB ASR budget; exact free MiB not provided in prior anchors | Confirmed design facts |

The C-optimized design is preferred over C-naive because C-naive would instantiate one encoder context per session. At `N=4`, that duplicates three extra encoder contexts and costs `3 x 352 = 1056 MiB` of avoidable activation memory. The measured encoder utilization at `N=4` is only 16.2%, so a single shared encoder context is sufficient when protected by a mutex.

The C-optimized design is also preferred over C-mutex-only. C-mutex-only protects shared execution but still risks inter-session decoder KV state corruption if all sessions share the same decoder context and KV buffers. Per-slot decoder context and KV state isolate session state while retaining the encoder memory savings.

Gotcha #15 applies directly: `gotchas-jetson.md:1186-1228` documents TensorRT runtime buffer dtype ABI mismatch on Jetson. Slot initialization must probe TensorRT binding dtype with `getTensorDataType` and size KV buffers dynamically. Do not hardcode `sizeof(half)`.

## 2. Architecture Diagram (ASCII)

```text
WS[1..4] -> asyncio.Semaphore(4) -> slot_id[0..3]
                                      |
          +---------------------------+------------------------+
          |  encoder (shared, 1 IExecutionContext)              |
          |  std::mutex -> serialize enqueueV3 calls            |
          +------------+----------------------------------------+
                       | features tensor (per-request output)
          +------------v-------------------------+
          |  decoder slot pool (N slots)         |
          |  slot[i]: IExecutionContext          |
          |           cudaStream_t               |
          |           KV cache buffer[i]         |
          +--------------------------------------+
                       | token stream
          WS response <-+
```

## 3. Changelist: Python layer

### Required entries

- `server/main.py:496-502` — leave `_asr_executor(max_workers=1)` untouched through Phase 4. Phase 5 is the only step that changes it to profile-driven `asr_max_slots`, after CER consistency passes, 30-minute sustained load reports 0 GPU errors, and RSS shows no growth. Replace the existing comment in Phase 5 with one that states encoder safety lives in the C++ `encoder_mutex_`, while Python stayed single-worker until isolation verification completed.

```python
asr_max_slots = int(profile.get("asr", {}).get("asr_max_slots", 1))
_asr_executor = ThreadPoolExecutor(
    max_workers=1,  # Keep through Phase 4; Phase 5 promotes to asr_max_slots after isolation and load gates pass.
    thread_name_prefix="asr_worker",
)
```

- `server/main.py:2107-2108` — reassess `async with get_coordinator().acquire('asr'):`. Keep it only if the coordinator represents global service admission, metrics, or cross-pipeline fairness. It must not be the only ASR concurrency guard. Slot ownership must move to the Qwen3-ASR backend semaphore so `N=4` is enforced at the resource boundary. If the coordinator is currently a hard `N=1` ASR gate, replace that behavior with backend slot acquisition.

- `server/main.py:2151` — add slot acquisition before `asr_be.create_stream(language=language)`. Convert the call path to await an async stream factory, or add an explicit async acquisition API that returns a stream with a held slot.

```python
stream = await asr_be.create_stream(language=language)
```

- `server/main.py:2164-2173` — in the reset / WebSocket reconnect path, release the old stream slot before creating the replacement stream. Use `try/finally` so a failed reconnect attempt cannot retain the old slot.

```python
if stream is not None:
    await stream.close()
stream = await asr_be.create_stream(language=language)
```

- `server/main.py:2295-2303` — in the final close path, ensure `stream.close()` always releases the slot even when cancellation or inference exceptions occur. The WebSocket handler should wrap session lifetime in `try/finally` and call `await stream.close()` exactly once.

- `app/backends/jetson/qwen3_asr.py:74` — in `Qwen3ASRBackend.__init__`, add `self._slot_sem = asyncio.Semaphore(asr_max_slots)` and a slot count field. Load `asr_max_slots` from the ASR profile with default `1` for rollback compatibility.

```python
self.asr_max_slots = int(profile.get("asr_max_slots", 1))
self._slot_sem = asyncio.Semaphore(self.asr_max_slots)
```

- `app/backends/jetson/qwen3_asr.py:1205-1215` — change `create_stream()` so it acquires the Python semaphore, calls C++ `acquire_slot()`, and passes `slot_id` to the stream/session. Because `asyncio.Semaphore.acquire()` is async, `create_stream()` should become `async def create_stream(...)`.

```python
async def create_stream(self, language=None):
    await self._slot_sem.acquire()
    slot_id = None
    try:
        slot_id = self._decoder.acquire_slot()
        return Qwen3StreamingASRStream(self, language, slot_id=slot_id)
    except Exception:
        if slot_id is not None:
            self._decoder.release_slot(slot_id)
        self._slot_sem.release()
        raise
```

- `app/backends/jetson/qwen3_asr.py:328-383` — add `slot_id` to `TrueStreamingSession` and all downstream decode calls. Add an idle timeout: if no audio chunk arrives for 30 seconds, the session triggers its own close path and releases the slot. Use `asyncio.wait_for(chunk_queue.get(), timeout=30)` in the chunk loop.

- `configs/profiles/jetson-orin-nano-8gb.json` — add `"asr_max_slots": 4` inside the existing `asr` section for the N=4 target profile.

- `configs/profiles/jetson-orin-nx.json` — add `"asr_max_slots": 4` inside the existing `asr` section.

- `configs/profiles/jetson-agx-orin.json` — add `"asr_max_slots": 4` inside the existing `asr` section.

## 4. Changelist: C++ layer

### Required entries

- source in fork — unverified [CONFIDENCE: low] `TRTDecoder` struct — add a decoder slot pool with `N` `IExecutionContext` instances, `N` `cudaStream_t` streams, and `N` KV buffer arrays. Each slot owns all mutable decoder state required for prefill and decode.

- source in fork — unverified [CONFIDENCE: low] encoder wrapper — add `std::mutex encoder_mutex_;`. Lock immediately before encoder `enqueueV3`, unlock immediately after enqueue completion or failure handling. The mutex protects the single shared encoder execution context and replaces the Python-only CUDA Graph capture lock once CUDA Graph capture is removed or proven safe.

- source in fork — unverified [CONFIDENCE: low] new API `int acquire_slot()` — return a free `slot_id` in `[0, N-1]` or throw/panic if the pool is exhausted. Exhaustion should not happen when Python `asyncio.Semaphore(N)` is correct, so this is a corruption guard rather than a blocking queue.

- source in fork — unverified [CONFIDENCE: low] new API `void release_slot(int slot_id)` — mark the slot free, reset decoder/KV state, and make the slot reusable for a later Python stream.

- source in fork — unverified [CONFIDENCE: low] all prefill / `decode_step` signatures — add `int slot_id` and route all context, CUDA stream, KV cache, scratch buffers, and plugin calls through the selected slot.

- source in fork — unverified [CONFIDENCE: low] per-slot RNG [INFERRED] — if `std::mt19937` or an equivalent RNG is present, keep RNG state per slot and reset the seed on `acquire_slot()` to avoid cross-request contamination.

- source in fork — unverified [CONFIDENCE: low] KV plugin per-slot — this is the highest-risk unknown. Patch plugin source for explicit per-slot KV buffers only. Option B mutex wrapping is eliminated because it adds latency without state isolation.

Phase 3 abort clause: if the plugin refactor is blocked because internal decoder/KV state cannot be split per slot, this spec mandates project abort back to the `ASR_MAX_SLOTS=1` profile. Do not fall back to Option B mutex wrapping; that is an implicit commitment of D2.

## 5. Interface Contract

- `slot_id`: `int`, range `[0, N-1]`, acquired by Python `asyncio.Semaphore` before C++ `acquire_slot()`, released by Python after C++ `release_slot()`.
- `acquire_slot()` must be non-blocking. Python `asyncio.Semaphore(N)` already serializes admission, so C++ should always find a free slot or panic.
- `release_slot()` must zero KV state, reset the CUDA stream, and mark the slot free.
- Session abort mid-prefill: the `stream.close()` path at `main.py:2295-2303` must call `release_slot()` even on exception. Use `try/finally` around session lifetime.
- Idle timeout: 30 seconds with no new audio chunk means the stream triggers its own `close()`, which releases the slot.

## 6. Test Plan

### Unit

- Mock `TRTDecoder` and verify the backend `asyncio.Semaphore` blocks the `N+1` request.
- Verify `release_slot()` unblocks a waiter.
- Verify idle timeout fires after 30 seconds with no chunk and calls `close()`.

### Integration

- Phase 4: run the real engine with `N=4` admitted sessions while `_asr_executor` remains `max_workers=1`.
- Feed each session 30 seconds of audio.
- Phase 4 acceptance: CER for all slots `<= baseline + 2% absolute`, 0 crash, 0 GPU error, and 30-minute sustained load pass.

### Stress

- Phase 4: run 30 minutes at sustained `N=4` admitted load.
- Monitor GPU health with `nvidia-smi -l 1`; require 0 GPU errors.
- Require RSS delta `< 50 MB/hr`.
- Phase 5: promote `_asr_executor` to `N=4` only after Phase 4 acceptance passes.
- Mix long sessions of 60 seconds with short sessions of 3 seconds.
- Phase 5 acceptance: mixed long+short TTFA `<= 1.5x` single-stream baseline, same gate as MOSS Phase B; `N=4` total throughput `>= 2.5x` single-stream baseline, conservative rather than `4x`.

### Edge

- Reconnect with 2 seconds of buffered audio; slot must be released and re-acquired cleanly.
- Abort session mid-prefill; slot must not leak.
- Trigger idle timeout during slot starvation; timed-out slot must be released before the timeout waiter gets it.

## 7. Rollback Strategy

- Set `asr_max_slots=1` in the profile. This creates `Semaphore(1)` and restores single-slot behavior without code changes.
- C++ binary compatibility: if the new Python build expects `acquire_slot()` / `release_slot()` but the old C++ build does not expose them, startup health check must raise `AttributeError` and fail fast rather than silently sharing decoder state.
- Feature branch: `asr-n4-concurrency`; merge behind profile guard `asr_max_slots`, with default `1` in all profiles until stress-validated.

## 8. Risk Table

| ID | Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|---|
| R1 | C++ KV plugin per-slot interface unknown | High | Critical | Phase 0 audit validates feasibility before any Phase 3 work begins |
| R2 | Encoder CUDA Graph capture + multi-thread | Medium | High | Keep `max_workers=1` through Phase 4; Phase 5 raises only after CER consistency PASS, 30-minute 0 GPU error, and RSS no growth |
| R3 | Encoder serialization latency spike @ N=4 | Low | Medium | Measured utilization 16% @ 250 ms stride means queue depth should be `<= 1` under normal load; add p99 monitoring |
| R4 | Long-lived session slot starvation | Medium | Medium | Idle timeout 30 s; admission queue with priority for reconnects |
| R5 | Per-slot RNG state leak (cross-request contamination) | Low | Low | Reset RNG seed on `acquire_slot()`; [INFERRED — verify in C++ fork source] |
| R6 | KV dtype ABI mismatch (gotcha #15 repeat) | Low | High | Apply `getTensorDataType` probe to KV buffer sizing in slot init; do not hardcode `sizeof(half)` |
| R7 | Phase 3 blocked | High | Critical | Roll back to `ASR_MAX_SLOTS=1` profile; do NOT use mutex wrap (D2) |

Footnote: Option B (mutex wrap) — eliminated; adds latency without isolation, replaced by D2.

## 9. Engineering Work Breakdown

| Phase | Scope | Estimate |
|---|---|---:|
| Phase 0 | Fork plugin source audit — confirm per-slot KV refactor feasibility | 2-3 days |
| Phase 1 | Python slot pool + Semaphore — keep executor=1 | 2 days |
| Phase 2 | Profile field + idle timeout — Python `asyncio.wait_for` | 1 day |
| Phase 3 | C++ plugin source refactor — per-slot KV + decoder context pool | 4-6 days |
| Phase 4 | Integration + isolation correctness verification — still executor=1; run CER consistency + 30-min sustained load | 2-3 days |
| Phase 5 | Promote `_asr_executor` to N=4 + true concurrency performance verification | 1-2 days |

Phase 5 executor promotion requires Phase 4 CER consistency PASS, 30-minute 0 GPU error, and RSS no growth.

Total: 12-17 days (sum of Phase 0-5). Phase 4 completion is the earliest production-safe milestone: isolation + fair queuing, throughput = N=1. Phase 5 completion delivers true N=4 throughput gain.

## 决策记录（2026-05-28 拍板）

| Decision | Final decision | Rationale | Historical rejected alternative |
|---|---|---|---|
| D1 | `_asr_executor(max_workers=1)` stays untouched until Phase 4 completes. Phase 5 is the only step that bumps it to `N`. | The comment at `main.py:496` explicitly says single thread is intentional due to CUDA Graph; changing that without completed isolation verification risks silent corruption. | Raising earlier was rejected; executor promotion depends on Phase 4 CER consistency PASS, 30-minute 0 GPU error, and RSS no growth. |
| D2 | Direct per-slot isolation in the EdgeLLM ASR plugin source only. | Per-slot decoder context and KV state are required for correctness; mutex wrapping serializes access but does not isolate mutable state. | Option B mutex wrapper and C-naive fallback are not live options in this spec. |
| D3 | Idle timeout is implemented in Python with `asyncio.wait_for(chunk_queue.get(), timeout=30)`. | ASR sessions are driven by the Python WS handler, and Python already owns the chunk loop. | C++ watchdog thread per slot was rejected because it adds complexity and latency sensitivity with no benefit for this use case. |
