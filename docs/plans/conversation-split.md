# Plan: split `voxedge/engine/conversation.py`

**Status:** designed, **not started**. Deliberately deferred — see "Why not now".

## Problem

`voxedge/engine/conversation.py` is ~1,600 LOC / 81 KB. It is not a god-object
(responsibilities are orthogonal and well-commented) but it is the single
highest-churn, highest-risk file in the stack: barge-in, ASR stale-final
suppression, TTS sequencing, server-loop tool dispatch, and slot-leak fixes all
live here and interlock through one shared `Session` state dict. New developers
have to read the whole file to safely touch any part of it.

## Current structure (what's inside)

```
_ToolCallAcc          tool-delta accumulator        (~small)
_SentenceBuffer       CJK+EN sentence splitter       (~small)
ConversationEngine    backend holder / factory       (~110 LOC)
Session               the orchestration loop         (~1,300 LOC)
  __init__            per-connection state dict
  _audio_loop         VAD → speech_start/end → barge-in
  _event_loop         CLIENT_* frame multiplexing
  _asr_out_task       partial polling + M1 deadline + finalize
  _on_asr_final       route to LLM or TTS
  _llm_turn_with_tools  multi-turn LLM↔tool pump (server-loop)
  _tts_out_task       sentence dequeue + generate_streaming + M3 deadline
  helpers             _bargein_tts, _open_asr_turn, _enqueue_tts_text, …
```

## Target structure

Keep `Session` as a thin coordinator that owns the state dict and spawns tasks;
extract the four loops into focused collaborators in `voxedge/engine/`:

| New module | Moves out | Surface |
|---|---|---|
| `audio_dispatcher.py` | `_audio_loop` + VAD/barge-in trigger | `run(session_state, transport, on_speech_start, on_speech_end)` |
| `client_events.py` | `_event_loop` (CLIENT_TEXT/ASR_EOS/ABORT/TOOL_*) | `run(session_state, transport, handlers)` |
| `asr_loop.py` | `_asr_out_task` + `_on_asr_final` + M1 watchdog | wraps `ASRSessionManager` |
| `tts_sequencer.py` | `_tts_out_task` + sentence buffer + M3 watchdog | `enqueue(text)`, `run()` |
| `llm_turn.py` | `_llm_turn_with_tools` + tool dispatch | `run(messages, tool_registry, ctx)` |
| `conversation.py` | `Session` (state + task spawn), `ConversationEngine` | unchanged public API |

The shared `state` dict is the coupling to break carefully: replace ad-hoc dict
keys with a typed `SessionState` dataclass so each collaborator declares exactly
which fields it reads/writes (this is where most latent bugs hide).

## Constraints that make this risky

- **The public API must not change.** `ConversationEngine(backends=…)` and the
  `Transport` contract are consumed by `server/main.py` and the tests; the split
  is pure-internal refactor.
- **`llm_barged` cooperative-cancel semantics must be preserved exactly.**
  `task.cancel()` alone is unsafe here (a cancel can be swallowed mid-dispatch);
  the flag-based barge-in is load-bearing. Any extraction must keep the same
  checkpoints.
- **Generation-ID guards** (stale-final suppression) cross the audio/asr
  boundary — splitting audio and asr loops must not reorder those checks.

## Why not now (the deferral is the decision)

This file is **shared with the production robot-arm stack** (`seeed-orin-nx`,
the one device we must not destabilise). A behavioural regression in barge-in or
slot cleanup would pass a two-device ASR/TTS smoke test and only surface as a
field hang. This refactor therefore needs its **own** validation cycle:

1. Full `voxedge/tests/` green (engine, watchdogs, server-loop, slot-leak).
2. `bench/` server-loop timing + barge-in stress, before/after parity.
3. A dedicated real-machine soak on a **non-production** device with repeated
   barge-in / multi-turn / tool-call rounds — not just "ASR works, TTS works".

Bundling it into the architecture-cleanup pass (docs + packaging) would couple a
safe change to a risky one under insufficient validation. Do it as a standalone,
bench-gated change.

## Suggested sequencing when picked up

1. Introduce `SessionState` dataclass; migrate the dict in place (no behaviour
   change), land + bench.
2. Extract `tts_sequencer.py` (most self-contained), land + bench.
3. Extract `asr_loop.py`, land + bench.
4. Extract `audio_dispatcher.py` + `client_events.py`, land + bench.
5. Extract `llm_turn.py` last (most entangled with tool dispatch + barge-in).

Each step is independently shippable and independently revertible.
